"""Admission, upload, SSE, playback, and short-lived audio API contracts."""

import asyncio
from collections.abc import AsyncIterator
import json
import logging
from pathlib import Path
import threading
import time
from uuid import UUID

from fastapi.testclient import TestClient
import pytest
import starlette.formparsers

from app.audio.storage import AudioStorage
from app.config import Settings
import app.main as main_module
from app.main import create_app
from app.runtimes.protocols import LLMDelta
from app.turns.events import TurnEventBuffer
from app.turns.manager import TurnManager
from app.turns.orchestrator import TurnContext


class _STT:
    ready = False

    def __init__(self, release: threading.Event | None = None) -> None:
        self._release = release
        self.calls = 0

    def load(self) -> None:
        self.ready = True

    async def transcribe(self, path: Path) -> str:
        self.calls += 1
        if self._release is not None:
            await asyncio.to_thread(self._release.wait)
        return "Merhaba"


class _TTS:
    ready = False

    def load(self) -> None:
        self.ready = True

    async def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-test-wave")


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
        yield LLMDelta(text="Selam!")


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "CORS_ORIGINS": "http://demo.test",
        "stt_model_id": "local-stt",
        "tts_model_id": "local-tts",
        "llama_cpp_model": "local-llm",
        "audio_retention_seconds": 60,
        "stt_timeout_seconds": 5,
        "llm_timeout_seconds": 5,
        "tts_timeout_seconds": 5,
        # Off unless a test injects a double; the real client would probe :6006.
        "intent_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _convert(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"RIFF-normalized")


def _application(
    tmp_path: Path,
    *,
    stt: _STT | None = None,
    probe_audio=lambda path, max_seconds: 1.25,
    settings: Settings | None = None,
):
    settings = settings or _settings()
    storage = AudioStorage(
        tmp_path / "audio",
        retention_seconds=settings.audio_retention_seconds,
    )
    app = create_app(
        settings=settings,
        storage=storage,
        stt=stt or _STT(),
        tts=_TTS(),
        llm=_LLM(),
        validate_audio_tools=lambda: None,
        probe_audio=probe_audio,
        convert_audio=_convert,
    )
    return app, storage


async def _asgi_post(
    app,
    *,
    body_chunks: list[bytes],
    headers: list[tuple[bytes, bytes]],
    receive_hook=None,
):
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(body_chunks) - 1,
        }
        for index, chunk in enumerate(body_chunks)
    ]
    sent: list[dict] = []

    async def receive():
        if receive_hook is not None:
            receive_hook()
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/turns",
            "raw_path": b"/api/turns",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, json.loads(body)


def _post_turn(client: TestClient):
    return client.post(
        "/api/turns",
        files={"audio": ("recording.webm", b"browser-audio", "audio/webm")},
    )


def _wait_for_terminal(app, turn_id: UUID) -> TurnContext:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        turn = app.state.turn_manager.get(turn_id)
        if turn is not None and any(
            event.name in {"complete", "failed"} for event in turn.events.snapshot()
        ):
            return turn
        time.sleep(0.005)
    raise AssertionError("turn did not reach a terminal event")


async def test_turn_manager_admits_exactly_one_concurrent_turn(tmp_path: Path):
    """Removing the admission lock could admit both racing requests."""
    manager = TurnManager(retention_seconds=60)
    turns = [
        TurnContext(
            turn_id=UUID(int=index),
            input_path=tmp_path / f"{index}.wav",
            events=TurnEventBuffer(UUID(int=index)),
        )
        for index in (1, 2)
    ]

    admitted = await asyncio.gather(*(manager.try_create(turn) for turn in turns))

    assert sorted(admitted) == [False, True]


async def test_declared_oversized_multipart_is_rejected_without_reading_body(
    tmp_path: Path,
):
    """Trusting endpoint validation would read/spool a declared oversized body."""
    app, _ = _application(
        tmp_path,
        settings=_settings(audio_max_bytes=8),
    )
    receive_calls = 0

    def mark_receive() -> None:
        nonlocal receive_calls
        receive_calls += 1

    status, body = await _asgi_post(
        app,
        body_chunks=[b"must-not-be-read"],
        headers=[
            (b"content-type", b"multipart/form-data; boundary=voice"),
            (b"content-length", str(128 * 1024).encode()),
        ],
        receive_hook=mark_receive,
    )

    assert status == 413
    assert receive_calls == 0
    assert body == {
        "code": "audio_too_large",
        "message": "Ses kaydı boyut sınırını aşıyor.",
    }


async def test_chunked_oversized_multipart_is_rejected_before_upload_spooling(
    tmp_path: Path,
    monkeypatch,
):
    """Missing Content-Length must not bypass the raw ASGI request-body bound."""
    app, _ = _application(
        tmp_path,
        settings=_settings(audio_max_bytes=8),
    )
    boundary = b"voice-boundary"
    multipart = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="audio"; filename="x.webm"\r\n'
        + b"Content-Type: audio/webm\r\n\r\n"
        + (b"x" * (80 * 1024))
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )
    spool_calls = 0
    real_spooled_file = starlette.formparsers.SpooledTemporaryFile

    def track_spooling(*args, **kwargs):
        nonlocal spool_calls
        spool_calls += 1
        return real_spooled_file(*args, **kwargs)

    monkeypatch.setattr(
        starlette.formparsers,
        "SpooledTemporaryFile",
        track_spooling,
    )
    status, body = await _asgi_post(
        app,
        body_chunks=[
            multipart[:32 * 1024],
            multipart[32 * 1024 : 64 * 1024],
            multipart[64 * 1024 :],
        ],
        headers=[
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
            (b"transfer-encoding", b"chunked"),
        ],
    )

    assert status == 413
    assert spool_calls == 0
    assert body["code"] == "audio_too_large"


def test_exact_audio_field_limit_still_applies_with_multipart_overhead(
    tmp_path: Path,
):
    """The raw overhead allowance must not weaken the exact uploaded-file limit."""
    app, storage = _application(
        tmp_path,
        settings=_settings(audio_max_bytes=8),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/turns",
            files={"audio": ("x.webm", b"123456789", "audio/webm")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "audio_too_large"
    assert list(storage.root.iterdir()) == []


def test_disallowed_actual_origin_is_rejected_before_upload_or_runtime_work(
    tmp_path: Path,
    monkeypatch,
):
    """A safelisted multipart POST must not bypass origin enforcement."""
    probe_calls = 0
    spool_calls = 0
    stt = _STT()
    real_spooled_file = starlette.formparsers.SpooledTemporaryFile

    def track_spooling(*args, **kwargs):
        nonlocal spool_calls
        spool_calls += 1
        return real_spooled_file(*args, **kwargs)

    def probe(path: Path, max_seconds: float) -> float:
        nonlocal probe_calls
        probe_calls += 1
        return 1.0

    monkeypatch.setattr(
        starlette.formparsers,
        "SpooledTemporaryFile",
        track_spooling,
    )
    app, storage = _application(tmp_path, stt=stt, probe_audio=probe)

    with TestClient(app) as client:
        response = client.post(
            "/api/turns",
            headers={"Origin": "https://attacker.example"},
            files={"audio": ("x.webm", b"browser-audio", "audio/webm")},
        )

    assert response.status_code == 403
    assert response.json() == {
        "code": "cors_not_allowed",
        "message": "Bu kaynaktan erişime izin verilmiyor.",
    }
    assert spool_calls == 0
    assert probe_calls == 0
    assert stt.calls == 0
    assert list(storage.root.iterdir()) == []


def test_second_active_turn_is_rejected_and_its_upload_is_deleted(tmp_path: Path):
    """Non-atomic admission or rejected-file retention leaks work and audio."""
    release = threading.Event()
    app, storage = _application(tmp_path, stt=_STT(release))

    with TestClient(app) as client:
        first = _post_turn(client)
        second = _post_turn(client)
        directories = list(storage.root.iterdir())
        release.set()

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json() == {
        "code": "turn_in_progress",
        "message": "Mevcut yanıt tamamlanıyor.",
    }
    assert len(directories) == 1


def test_turn_sse_replays_only_events_after_last_event_id(tmp_path: Path):
    """Ignoring Last-Event-ID would duplicate already-played audio after reconnect."""
    app, _ = _application(tmp_path)

    with TestClient(app) as client:
        created = _post_turn(client)
        turn_id = UUID(created.json()["turn_id"])
        turn = _wait_for_terminal(app, turn_id)
        audio_event = next(
            event for event in turn.events.snapshot() if event.name == "audio_ready"
        )

        replay = client.get(
            created.json()["events_url"],
            headers={"Last-Event-ID": str(audio_event.event_id)},
        )

    assert replay.status_code == 200
    assert replay.headers["content-type"].startswith("text/event-stream")
    assert "event: metrics" in replay.text
    assert "event: complete" in replay.text
    assert "event: audio_ready" not in replay.text


def test_browser_codec_parameter_is_normalized_before_audio_validation(
    tmp_path: Path,
):
    """Passing MediaRecorder's codec parameter through would reject valid WebM."""
    app, _ = _application(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/turns",
            files={
                "audio": (
                    "recording.webm",
                    b"browser-audio",
                    "audio/webm;codecs=opus",
                )
            },
        )

    assert response.status_code == 202


def test_audio_routes_are_wav_and_missing_audio_is_safe_404(tmp_path: Path):
    """Serving a wrong type or leaking filesystem details breaks browser playback."""
    app, storage = _application(tmp_path)

    with TestClient(app) as client:
        created = _post_turn(client)
        turn_id = UUID(created.json()["turn_id"])
        _wait_for_terminal(app, turn_id)

        chunk = client.get(f"/api/audio/{turn_id}/0.wav")
        complete = client.get(f"/api/audio/{turn_id}.wav")
        storage.chunk_path(turn_id, 0).unlink()
        expired = client.get(f"/api/audio/{turn_id}/0.wav")
        unknown = client.get(f"/api/audio/{UUID(int=999)}.wav")

    assert chunk.status_code == 200
    assert chunk.headers["content-type"] == "audio/wav"
    assert chunk.content == b"RIFF-test-wave"
    assert complete.status_code == 200
    assert complete.headers["content-type"] == "audio/wav"
    assert expired.status_code == 404
    assert expired.json() == {
        "code": "audio_not_found",
        "message": "Ses kaydı bulunamadı veya süresi doldu.",
    }
    assert unknown.status_code == 404
    assert unknown.json() == expired.json()
    assert str(storage.root) not in expired.text


def test_complete_audio_rejects_a_truncated_tail_after_partial_expiry(
    tmp_path: Path,
):
    """Serving sequence 1 after sequence 0 expires would return a truncated answer."""
    app, storage = _application(tmp_path)

    with TestClient(app) as client:
        created = _post_turn(client)
        turn_id = UUID(created.json()["turn_id"])
        turn = _wait_for_terminal(app, turn_id)
        turn.produced_sequences.add(1)
        storage.chunk_path(turn_id, 1).write_bytes(b"RIFF-later")
        storage.chunk_path(turn_id, 0).unlink()

        response = client.get(f"/api/audio/{turn_id}.wav")

    assert response.status_code == 404
    assert response.json()["code"] == "audio_not_found"


def test_playback_accepts_produced_sequence_once_and_clamps_client_time(
    tmp_path: Path,
):
    """Accepting arbitrary/repeated offsets would corrupt first-playback latency."""
    app, _ = _application(tmp_path)

    with TestClient(app) as client:
        created = _post_turn(client)
        turn_id = UUID(created.json()["turn_id"])
        turn = _wait_for_terminal(app, turn_id)
        assert turn.started_ns is not None
        age_ms = (time.perf_counter_ns() - turn.started_ns) / 1_000_000

        first = client.post(
            f"/api/turns/{turn_id}/playback",
            json={"sequence": 0, "client_offset_ms": age_ms + 10_000},
        )
        repeated = client.post(
            f"/api/turns/{turn_id}/playback",
            json={"sequence": 0, "client_offset_ms": 999_999},
        )
        metric = client.get("/api/metrics/recent").json()[-1]
        unknown_sequence = client.post(
            f"/api/turns/{turn_id}/playback",
            json={"sequence": 9, "client_offset_ms": 10},
        )
        negative = client.post(
            f"/api/turns/{turn_id}/playback",
            json={"sequence": 0, "client_offset_ms": -1},
        )

    assert first.status_code == 204
    assert repeated.status_code == 204
    assert age_ms <= metric["first_audio_started_ms"] < age_ms + 500
    assert unknown_sequence.status_code == 404
    assert unknown_sequence.json()["code"] == "audio_not_found"
    assert negative.status_code == 422
    assert negative.json() == {
        "code": "invalid_request",
        "message": "İstek bilgileri geçersiz.",
    }


def test_upload_errors_never_expose_internal_diagnostics(tmp_path: Path):
    """Raw probe exceptions must not reveal server paths or exception text."""

    def fail_probe(path: Path, max_seconds: float) -> float:
        raise RuntimeError("/private/models/secret.wav")

    app, _ = _application(tmp_path, probe_audio=fail_probe)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = _post_turn(client)

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "İşlem tamamlanamadı, tekrar deneyin.",
    }
    assert "/private/models" not in response.text


async def test_rejected_upload_close_failure_does_not_skip_directory_cleanup(
    tmp_path: Path,
    caplog,
):
    """A failing UploadFile.close must not leave rejected audio on disk."""
    directory = tmp_path / "rejected-secret-path"
    directory.mkdir()
    (directory / "upload.webm").write_bytes(b"private")

    class CloseFails:
        async def close(self) -> None:
            raise OSError("/private/close-secret")

    with caplog.at_level(logging.WARNING):
        await main_module._finish_request_upload(
            CloseFails(),
            directory,
            remove_directory=True,
        )

    assert not directory.exists()
    assert "Rejected upload close failed." in caplog.text
    assert "/private/close-secret" not in caplog.text
    assert str(directory) not in caplog.text


async def test_rejected_upload_cleanup_finishes_before_cancellation_propagates(
    tmp_path: Path,
):
    """Cancellation must not interrupt close-and-delete finalization."""
    directory = tmp_path / "rejected"
    directory.mkdir()
    (directory / "upload.webm").write_bytes(b"private")
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class SlowClose:
        async def close(self) -> None:
            close_started.set()
            await allow_close.wait()

    cleanup = asyncio.create_task(
        main_module._finish_request_upload(
            SlowClose(),
            directory,
            remove_directory=True,
        )
    )
    await asyncio.wait_for(close_started.wait(), timeout=0.1)
    cleanup.cancel("request cancelled")
    await asyncio.sleep(0)
    remained_owned = not cleanup.done()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await cleanup

    assert remained_owned is True
    assert caught.value.args == ("request cancelled",)
    assert not directory.exists()


async def test_rejected_upload_deletion_failure_is_logged_without_details(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    """Deletion errors must be observable without exposing paths or exception text."""
    directory = tmp_path / "rejected-secret-path"
    directory.mkdir()

    class Closed:
        async def close(self) -> None:
            return None

    def fail_delete(path: Path) -> None:
        raise OSError("/private/delete-secret")

    monkeypatch.setattr(main_module.shutil, "rmtree", fail_delete)
    with caplog.at_level(logging.WARNING):
        await main_module._finish_request_upload(
            Closed(),
            directory,
            remove_directory=True,
        )

    assert directory.exists()
    assert "Rejected upload deletion failed." in caplog.text
    assert "/private/delete-secret" not in caplog.text
    assert str(directory) not in caplog.text
