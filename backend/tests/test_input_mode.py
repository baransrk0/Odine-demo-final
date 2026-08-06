"""Runtime input-mode switching and RF service lifecycle."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.audio.input_mode import InputModeController
from app.audio.storage import AudioStorage
from app.config import Settings
from app.main import create_app


class _FakeRFService:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


def test_start_launches_rf_only_in_device_mode():
    services: list[_FakeRFService] = []

    def factory() -> _FakeRFService:
        service = _FakeRFService()
        services.append(service)
        return service

    browser = InputModeController(mode="browser", rf_factory=factory)
    browser.start()
    assert services == []

    device = InputModeController(mode="rf_i2s", rf_factory=factory)
    device.start()
    assert len(services) == 1 and services[0].started == 1


async def test_set_mode_starts_and_stops_rf_service():
    services: list[_FakeRFService] = []
    controller = InputModeController(
        mode="browser",
        rf_factory=lambda: services.append(_FakeRFService()) or services[-1],
    )

    await controller.set_mode("rf_i2s")
    assert controller.mode == "rf_i2s"
    assert services[-1].started == 1

    await controller.set_mode("browser")
    assert controller.mode == "browser"
    assert services[-1].stopped == 1


async def test_failed_rf_start_leaves_mode_unchanged():
    def boom() -> None:
        raise RuntimeError("APE routing failed")

    controller = InputModeController(
        mode="browser",
        rf_factory=lambda: (_ for _ in ()).throw(AssertionError("unreached")),
        validate=boom,
    )

    with pytest.raises(RuntimeError):
        await controller.set_mode("rf_i2s")
    assert controller.mode == "browser"


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

    async def stream_answer(self, transcript, system_prompt=None):
        if False:
            yield None


class _NoopRFService:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _app(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        audio_input_mode="browser",
        stt_model_id="local-stt",
        tts_model_id="local-tts",
        llama_cpp_model="local-llm",
        intent_enabled=False,
    )
    return create_app(
        settings=settings,
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=_Speech(),
        tts=_Speech(),
        llm=_LLM(),
        validate_audio_tools=lambda: None,
        rf_service_factory=lambda *args: _NoopRFService(),
    )


def test_input_endpoint_switches_mode_and_enables_rf(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.main.shutil.which", lambda tool: f"/bin/{tool}")
    app = _app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/api/audio/input").json() == {"mode": "browser"}
        # RF streams disabled while in browser mode.
        assert client.get("/api/rf/turns/events").status_code == 409

        switched = client.post("/api/audio/input", json={"mode": "rf_i2s"})
        assert switched.status_code == 200
        assert switched.json() == {"mode": "rf_i2s"}
        assert client.get("/api/health").json()["audio_input_mode"] == "rf_i2s"

        back = client.post("/api/audio/input", json={"mode": "browser"})
        assert back.json() == {"mode": "browser"}
        assert client.get("/api/health").json()["audio_input_mode"] == "browser"


def test_input_endpoint_reports_rf_start_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "app.main._validate_hardware_audio_tools",
        lambda settings: (_ for _ in ()).throw(RuntimeError("no arecord")),
    )
    app = _app(tmp_path)

    with TestClient(app) as client:
        response = client.post("/api/audio/input", json={"mode": "rf_i2s"})

    assert response.status_code == 503
    assert response.json()["code"] == "rf_start_failed"
