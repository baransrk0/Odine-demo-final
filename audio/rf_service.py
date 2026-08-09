"""Application-owned RF/I2S capture loop."""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import shutil
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.audio.gpio import ListeningIndicator
from app.audio.i2s import (
    CaptureResult,
    capture_ptt_pcm,
    configure_ape,
    convert_rf_pcm_to_stt_wav,
)
from app.audio.listening import ListeningBroadcaster
from app.audio.storage import AudioStorage
from app.config import Settings
from app.errors import ApiError
from app.schemas import TurnCreated
from app.turns.submitter import TurnSubmitter

logger = logging.getLogger(__name__)
# Capture may receive an optional listening callback as its third argument.
Capture = Callable[..., Awaitable[CaptureResult | None]]
Convert = Callable[[Settings, Path, Path], None]
Configure = Callable[[Settings], None]


class DiscoveryPublisher(Protocol):
    async def publish(self, created: TurnCreated) -> object: ...


class RFInputService:
    def __init__(
        self,
        settings: Settings,
        storage: AudioStorage,
        submitter: TurnSubmitter,
        discovery: DiscoveryPublisher,
        indicator: ListeningIndicator | None = None,
        listening: ListeningBroadcaster | None = None,
        *,
        capture: Capture = capture_ptt_pcm,
        convert: Convert = convert_rf_pcm_to_stt_wav,
        configure: Configure = configure_ape,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._submitter = submitter
        self._discovery = discovery
        self._indicator = indicator
        self._listening = listening
        self._capture = capture
        self._convert = convert
        self._configure = configure
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._configure(self._settings)
        if self._indicator is not None:
            self._indicator.set_armed(True)
        self._task = asyncio.create_task(self._run(), name="rf-i2s-capture")

    async def stop(self) -> None:
        if self._indicator is not None:
            self._indicator.set_listening(False)
            self._indicator.set_armed(False)
        if self._listening is not None:
            try:
                await self._listening.publish(False)
            except Exception:
                logger.warning("Listening state broadcast failed on stop.")
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _emit_listening(self, active: bool) -> None:
        """Drive the GPIO pin and the UI broadcast from one capture signal."""
        if self._indicator is not None:
            self._indicator.set_listening(active)
        if self._listening is not None:
            try:
                await self._listening.publish(active)
            except Exception:
                logger.warning("Listening state broadcast failed.")

    async def _run(self) -> None:
        while True:
            turn_id = uuid4()
            directory = self._storage.turn_dir(turn_id)
            raw_path = directory / "rf-input.raw"
            wav_path = self._storage.intermediate_path(turn_id)
            try:
                captured = await self._capture(
                    self._settings, raw_path, self._emit_listening
                )
                if captured is None:
                    directory.rmdir() if directory.exists() and not any(directory.iterdir()) else None
                    continue
                await asyncio.to_thread(
                    self._convert,
                    self._settings,
                    raw_path,
                    wav_path,
                )
                created = await self._submitter.submit_normalized_wav(
                    turn_id, wav_path, source_path=raw_path,
                    duration_seconds=captured.duration_seconds, byte_count=captured.byte_count,
                    content_type="audio/wav",
                )
                try:
                    await self._discovery.publish(created)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("RF turn discovery publication failed.")
                await self._submitter.wait_until_idle()
            except asyncio.CancelledError:
                raise
            except ApiError as error:
                if error.code != "turn_in_progress":
                    logger.warning("RF turn submission failed: %s", error.code)
            except Exception:
                logger.warning("RF capture failed.")
                shutil.rmtree(directory, ignore_errors=True)
