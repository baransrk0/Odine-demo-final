"""Local ALSA playback for already-synthesized WAV chunks."""

import asyncio
from pathlib import Path


class LocalPlaybackError(RuntimeError):
    """Raised when ALSA cannot play a completed TTS chunk."""


class LocalAudioPlayer:
    def __init__(self, device: str) -> None:
        self._device = device

    async def play_wav(self, path: Path) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                "aplay", "-q", "-D", self._device, str(path),
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as error:
            raise LocalPlaybackError("local playback failed") from error
        if await process.wait() != 0:
            raise LocalPlaybackError("local playback failed")
