"""Jetson APE I2S routing and bounded RF push-to-talk capture."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from app.config import Settings

# Invoked with True when an utterance starts arriving and False when it ends.
ListeningCallback = Callable[[bool], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CaptureResult:
    path: Path
    duration_seconds: float
    byte_count: int


def configure_ape(settings: Settings) -> None:
    commands = (
        (f"{settings.ape_i2s_port} codec master mode", "cbm-cfm"),
        ("ADMAIF1 Mux", settings.ape_i2s_port),
    )
    for name, value in commands:
        result = subprocess.run(
            ["amixer", "-c", settings.ape_card, "cset", f"name={name}", value],
            check=False, shell=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"APE routing failed for {name}")


async def capture_ptt_pcm(
    settings: Settings,
    destination: Path,
    on_state: ListeningCallback | None = None,
) -> CaptureResult | None:
    """Capture one PTT utterance; an idle pre-PTT stream produces no turn.

    ``on_state`` -- when provided -- is awaited with True at the first captured
    audio frame (the "listening" transition) and False once the capture ends.
    The listening GPIO pin and the UI both hang off this single signal.
    """
    frame_bytes = max(1, settings.rf_mic_sample_rate * settings.rf_mic_channels * 2 // 50)
    max_bytes = int(
        settings.rf_capture_max_seconds * settings.rf_mic_sample_rate * settings.rf_mic_channels * 2
    )
    process = await asyncio.create_subprocess_exec(
        "arecord", "-q", "-D", settings.rf_mic_device,
        "-r", str(settings.rf_mic_sample_rate), "-f", "S16_LE",
        "-c", str(settings.rf_mic_channels), "-t", "raw",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    started = False
    byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert process.stdout is not None
        with destination.open("wb") as output:
            while byte_count < max_bytes:
                reader = process.stdout.read(frame_bytes)
                try:
                    frame = await reader if not started else await asyncio.wait_for(
                        reader, timeout=settings.rf_ptt_frame_timeout_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    break
                if not frame:
                    break
                if not started:
                    started = True
                    if on_state is not None:
                        await on_state(True)
                output.write(frame)
                byte_count += len(frame)
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if started and on_state is not None:
            try:
                await on_state(False)
            except Exception:
                pass

    bytes_per_second = settings.rf_mic_sample_rate * settings.rf_mic_channels * 2
    duration = byte_count / bytes_per_second
    if not started or duration < settings.rf_ptt_min_seconds:
        destination.unlink(missing_ok=True)
        return None
    return CaptureResult(destination, duration, byte_count)


def convert_rf_pcm_to_stt_wav(settings: Settings, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-f", "s16le",
            "-ar", str(settings.rf_mic_sample_rate), "-ac", str(settings.rf_mic_channels),
            "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
        ],
        check=False, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError("RF PCM conversion failed")
