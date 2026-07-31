"""Safe ffprobe and ffmpeg wrappers for STT-normalized audio."""

import logging
import math
from pathlib import Path
import subprocess

from app.config import Settings
from app.errors import ApiError


logger = logging.getLogger(__name__)


def _invalid_audio() -> ApiError:
    return ApiError("invalid_audio", "Ses kaydı işlenemedi, tekrar deneyin.")


def probe_duration(path: Path, max_seconds: float | None = None) -> float:
    """Return a valid input duration while enforcing the configured bounds.

    Probe the normalized WAV rather than the browser upload. MediaRecorder
    writes a live WebM segment whose duration element is never filled in, and
    ffprobe reports ``N/A`` with a zero exit status for such a container.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        reported = result.stdout.strip() if result.returncode == 0 else ""
        duration = float(reported) if reported and reported != "N/A" else None
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.warning("ffprobe could not inspect audio.")
        raise _invalid_audio() from None

    if duration is None or not math.isfinite(duration):
        logger.warning("ffprobe reported no usable duration.")
        raise _invalid_audio()
    if duration <= 0.2:
        raise ApiError("audio_too_short", "Ses kaydı çok kısa, tekrar deneyin.")

    limit = max_seconds if max_seconds is not None else Settings().audio_max_seconds
    if duration > limit:
        raise ApiError("audio_too_long", "Ses kaydı süre sınırını aşıyor.")
    return duration


def convert_to_stt_wav(source: Path, target: Path) -> None:
    """Convert browser audio to 16-kHz mono PCM WAV without exposing diagnostics."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("ffmpeg conversion failed.")
        target.unlink(missing_ok=True)
        raise _invalid_audio() from None

    if result.returncode != 0:
        logger.warning("ffmpeg conversion failed.")
        target.unlink(missing_ok=True)
        raise _invalid_audio()
