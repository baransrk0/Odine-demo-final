"""Local ALSA playback boundary tests without requiring an ALSA device."""

from pathlib import Path

import pytest

from app.audio.playback import LocalAudioPlayer, LocalPlaybackError


class _Process:
    def __init__(self, exit_code: int) -> None:
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


async def test_local_player_uses_configured_alsa_device(tmp_path: Path, monkeypatch):
    calls: list[tuple[object, ...]] = []

    async def create_process(*args, **kwargs):
        calls.append(args)
        return _Process(0)

    monkeypatch.setattr("app.audio.playback.asyncio.create_subprocess_exec", create_process)
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFF")

    await LocalAudioPlayer("plughw:2,0").play_wav(wav)

    assert calls == [("aplay", "-q", "-D", "plughw:2,0", str(wav))]


async def test_local_player_raises_on_aplay_failure(tmp_path: Path, monkeypatch):
    async def create_process(*args, **kwargs):
        return _Process(1)

    monkeypatch.setattr("app.audio.playback.asyncio.create_subprocess_exec", create_process)
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFF")

    with pytest.raises(LocalPlaybackError):
        await LocalAudioPlayer("plughw:2,0").play_wav(wav)
