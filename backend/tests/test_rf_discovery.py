"""Replay-safe RF turn discovery without physical audio hardware."""

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from app.audio.i2s import CaptureResult
from app.audio.rf_discovery import RFDiscoveryBuffer, RFDiscoveryHeartbeat
from app.audio.rf_service import RFInputService
from app.audio.storage import AudioStorage
from app.config import Settings
from app.errors import ApiError
from app.main import create_app
from app.runtimes.protocols import LLMDelta
from app.schemas import TurnCreated


def _created(value: int) -> TurnCreated:
    turn_id = UUID(int=value)
    return TurnCreated(
        turn_id=turn_id,
        events_url=f"/api/turns/{turn_id}/events",
    )


class _Speech:
    ready = True

    async def transcribe(self, path: Path) -> str:
        return "Merhaba"

    async def synthesize(self, text: str, output_path: Path) -> None:
        output_path.write_bytes(b"RIFF")


class _LLM:
    ready = True

    async def health(self) -> bool:
        return True

    async def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        yield LLMDelta(text="Merhaba!")


class _NoopRFService:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FiniteDiscovery:
    def __init__(self, event) -> None:
        self.event = event
        self.cursor: int | None = None

    async def subscribe(self, last_event_id: int | None = None):
        self.cursor = last_event_id
        yield self.event


class _CaptureSequence:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.blocked = asyncio.Event()
        self.calls = 0

    async def __call__(self, settings: Settings, path: Path):
        self.calls += 1
        if self.calls == 1:
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            if isinstance(self.outcome, CaptureResult):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"raw")
            return self.outcome
        self.blocked.set()
        await asyncio.Event().wait()


class _Submitter:
    def __init__(self, outcome: TurnCreated | BaseException) -> None:
        self.outcome = outcome
        self.wait_started = asyncio.Event()
        self.turn_id = None

    async def submit_normalized_wav(self, turn_id, path, **kwargs):
        self.turn_id = turn_id
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return TurnCreated(
            turn_id=turn_id,
            events_url=f"/api/turns/{turn_id}/events",
        )

    async def wait_until_idle(self) -> None:
        self.wait_started.set()
        await asyncio.Event().wait()


class _FailingDiscovery:
    def __init__(self) -> None:
        self.calls = 0

    async def publish(self, created: TurnCreated) -> None:
        self.calls += 1
        raise RuntimeError("discovery unavailable")


async def test_discovery_assigns_monotonic_ids_and_replays_only_latest_turn():
    discovery = RFDiscoveryBuffer(
        retention_seconds=60,
        heartbeat_seconds=0.01,
    )

    first = await discovery.publish(_created(1))
    second = await discovery.publish(_created(2))
    replay = discovery.subscribe(last_event_id=0)

    assert first.event_id == 1
    assert second.event_id == 2
    assert await anext(replay) == second
    await replay.aclose()


async def test_discovery_cursor_suppresses_duplicate_and_yields_heartbeat():
    discovery = RFDiscoveryBuffer(
        retention_seconds=60,
        heartbeat_seconds=0.01,
    )
    published = await discovery.publish(_created(3))
    replay = discovery.subscribe(last_event_id=published.event_id)

    assert await anext(replay) == RFDiscoveryHeartbeat()
    await replay.aclose()


async def test_discovery_does_not_replay_expired_latest_turn():
    now = 100.0
    discovery = RFDiscoveryBuffer(
        retention_seconds=10,
        heartbeat_seconds=0.01,
        clock=lambda: now,
    )
    await discovery.publish(_created(4))
    now = 111.0
    replay = discovery.subscribe(last_event_id=0)

    assert await anext(replay) == RFDiscoveryHeartbeat()
    await replay.aclose()


async def test_discovery_sse_contract_contains_exact_turn_created_payload():
    discovery = RFDiscoveryBuffer(retention_seconds=60)
    published = await discovery.publish(_created(5))
    sse = published.as_sse()

    assert sse["event"] == "turn_created"
    assert sse["id"] == "1"
    assert json.loads(sse["data"]) == {
        "event_id": 1,
        "turn_id": "00000000-0000-0000-0000-000000000005",
        "events_url": (
            "/api/turns/00000000-0000-0000-0000-000000000005/events"
        ),
    }


def _app(
    tmp_path: Path,
    *,
    audio_input_mode: str,
    discovery=None,
):
    settings = Settings(
        _env_file=None,
        audio_input_mode=audio_input_mode,
        local_audio_playback=False,
        stt_model_id="local-stt",
        tts_model_id="local-tts",
        llama_cpp_model="local-llm",
        # Off unless a test injects a double; the real client would probe :6006.
        intent_enabled=False,
    )
    return create_app(
        settings=settings,
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=_Speech(),
        tts=_Speech(),
        llm=_LLM(),
        validate_audio_tools=lambda: None,
        rf_discovery=discovery,
        rf_service_factory=lambda *args: _NoopRFService(),
    )


def test_rf_discovery_endpoint_is_disabled_in_browser_mode(tmp_path: Path):
    app = _app(tmp_path, audio_input_mode="browser")

    with TestClient(app) as client:
        response = client.get("/api/rf/turns/events")

    assert response.status_code == 409
    assert response.json() == {
        "code": "rf_input_disabled",
        "message": "RF ses girişi etkin değil.",
    }


def test_rf_discovery_endpoint_rejects_invalid_cursor(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("app.main.shutil.which", lambda tool: f"/bin/{tool}")
    app = _app(
        tmp_path,
        audio_input_mode="rf_i2s",
        discovery=_FiniteDiscovery(None),
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/rf/turns/events",
            headers={"Last-Event-ID": "invalid"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "message": "İstek bilgileri geçersiz.",
    }


async def test_rf_discovery_endpoint_streams_exact_event_and_cursor(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("app.main.shutil.which", lambda tool: f"/bin/{tool}")
    discovery = RFDiscoveryBuffer(retention_seconds=60)
    published = await discovery.publish(_created(6))
    finite = _FiniteDiscovery(published)
    app = _app(
        tmp_path,
        audio_input_mode="rf_i2s",
        discovery=finite,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/rf/turns/events",
            headers={"Last-Event-ID": "4"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert finite.cursor == 4
    assert "event: turn_created" in response.text
    assert "id: 1" in response.text
    assert (
        '"turn_id":"00000000-0000-0000-0000-000000000006"'
        in response.text
    )


def _rf_settings() -> Settings:
    return Settings(
        _env_file=None,
        audio_input_mode="rf_i2s",
        local_audio_playback=False,
    )


def _convert(settings: Settings, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"RIFF")


async def test_rf_service_announces_successful_admission(tmp_path: Path):
    discovery = RFDiscoveryBuffer(retention_seconds=60)
    capture = _CaptureSequence(
        CaptureResult(Path("unused"), duration_seconds=1.0, byte_count=32000)
    )
    submitter = _Submitter(_created(7))
    service = RFInputService(
        _rf_settings(),
        AudioStorage(tmp_path / "audio", retention_seconds=60),
        submitter,
        discovery,
        capture=capture,
        convert=_convert,
        configure=lambda settings: None,
    )

    service.start()
    replay = discovery.subscribe(last_event_id=0)
    announced = await asyncio.wait_for(anext(replay), timeout=1)
    await service.stop()
    await replay.aclose()

    assert announced.payload.turn_id == submitter.turn_id
    assert announced.payload.events_url == (
        f"/api/turns/{announced.payload.turn_id}/events"
    )


async def test_rf_service_does_not_announce_unadmitted_capture(
    tmp_path: Path,
):
    for outcome in (
        None,
        RuntimeError("conversion failed"),
        ApiError("turn_in_progress", "Mevcut yanıt tamamlanıyor.", 409),
    ):
        discovery = RFDiscoveryBuffer(retention_seconds=60)
        capture_outcome = (
            CaptureResult(
                Path("unused"),
                duration_seconds=1.0,
                byte_count=32000,
            )
            if outcome is not None
            else None
        )
        capture = _CaptureSequence(capture_outcome)
        submitter_outcome = (
            outcome if isinstance(outcome, ApiError) else _created(8)
        )

        def convert(settings: Settings, source: Path, target: Path) -> None:
            if isinstance(outcome, RuntimeError):
                raise outcome
            _convert(settings, source, target)

        service = RFInputService(
            _rf_settings(),
            AudioStorage(
                tmp_path / str(type(outcome).__name__),
                retention_seconds=60,
            ),
            _Submitter(submitter_outcome),
            discovery,
            capture=capture,
            convert=convert,
            configure=lambda settings: None,
        )

        service.start()
        await asyncio.wait_for(capture.blocked.wait(), timeout=1)
        replay = discovery.subscribe(last_event_id=0)
        try:
            await asyncio.wait_for(anext(replay), timeout=0.02)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("unadmitted RF capture was announced")
        finally:
            await service.stop()
            await replay.aclose()


async def test_discovery_failure_does_not_skip_active_turn_wait(
    tmp_path: Path,
):
    capture = _CaptureSequence(
        CaptureResult(Path("unused"), duration_seconds=1.0, byte_count=32000)
    )
    submitter = _Submitter(_created(9))
    discovery = _FailingDiscovery()
    service = RFInputService(
        _rf_settings(),
        AudioStorage(tmp_path / "audio", retention_seconds=60),
        submitter,
        discovery,
        capture=capture,
        convert=_convert,
        configure=lambda settings: None,
    )

    service.start()
    await asyncio.wait_for(submitter.wait_started.wait(), timeout=1)
    await service.stop()

    assert discovery.calls == 1
