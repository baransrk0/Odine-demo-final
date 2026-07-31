"""Anonymous recent-metrics HTTP contract."""

import asyncio
from collections.abc import AsyncIterator
import logging
from pathlib import Path
import time
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from app.audio.storage import AudioStorage
from app.config import Settings
from app.main import _expiry_loop, create_app
from app.runtimes.protocols import LLMDelta


class _STT:
    ready = False

    def load(self) -> None:
        self.ready = True

    async def transcribe(self, path: Path) -> str:
        return "Özel kullanıcı konuşması"


class _TTS:
    ready = False

    def load(self) -> None:
        self.ready = True

    async def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF")


class _LLM:
    ready = False

    async def health(self) -> bool:
        self.ready = True
        return True

    async def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        yield LLMDelta(text="Özel model yanıtı.")


def _convert(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"RIFF")


def test_recent_metrics_returns_terminal_summaries_without_content(tmp_path: Path):
    """Returning turn text would violate the anonymous process-local metric contract."""
    settings = Settings(
        _env_file=None,
        stt_model_id="local-stt",
        tts_model_id="local-tts",
        llama_cpp_model="local-llm",
        # Off unless a test injects a double; the real client would probe :6006.
        intent_enabled=False,
    )
    app = create_app(
        settings=settings,
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=_STT(),
        tts=_TTS(),
        llm=_LLM(),
        validate_audio_tools=lambda: None,
        probe_audio=lambda path, max_seconds: 1.0,
        convert_audio=_convert,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/turns",
            files={"audio": ("recording.wav", b"browser-audio", "audio/wav")},
        )
        turn_id = UUID(created.json()["turn_id"])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            recent = client.get("/api/metrics/recent")
            if recent.json():
                break
            time.sleep(0.005)

    assert created.status_code == 202
    assert recent.status_code == 200
    assert len(recent.json()) == 1
    assert recent.json()[0]["outcome"] == "complete"
    assert recent.json()[0]["recording_bytes"] == len(b"browser-audio")
    serialized = recent.text
    assert str(turn_id) not in serialized
    assert "Özel kullanıcı konuşması" not in serialized
    assert "Özel model yanıtı" not in serialized


async def test_expiry_failures_are_isolated_and_the_loop_continues(caplog):
    """One storage/manager failure must not stop the other expiry or later cycles."""

    class FlakyStorage:
        def __init__(self) -> None:
            self.calls = 0

        def expire(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("/private/storage-secret")

    class FlakyManager:
        def __init__(self) -> None:
            self.calls = 0

        async def expire(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("/private/manager-secret")

    sleep_calls = 0

    async def two_cycles(_: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 2:
            raise asyncio.CancelledError("stop test loop")

    storage = FlakyStorage()
    manager = FlakyManager()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await _expiry_loop(
                storage,
                manager,
                retention_seconds=1,
                sleep=two_cycles,
            )

    assert storage.calls == 2
    assert manager.calls == 2
    assert "Audio storage expiry failed." in caplog.text
    assert "Turn event expiry failed." in caplog.text
    assert "/private/storage-secret" not in caplog.text
    assert "/private/manager-secret" not in caplog.text
