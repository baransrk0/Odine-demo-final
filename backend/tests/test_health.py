"""Readiness and browser-origin contracts for the FastAPI application."""

from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from app.audio.storage import AudioStorage
from app.config import Settings
from app.main import create_app
from app.runtimes.protocols import LLMDelta


class _SpeechRuntime:
    def __init__(self, *, ready_after_load: bool) -> None:
        self.ready = False
        self._ready_after_load = ready_after_load

    def load(self) -> None:
        self.ready = self._ready_after_load

    async def transcribe(self, path: Path) -> str:
        return "Merhaba"

    async def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF")


class _LLMRuntime:
    def __init__(self, *, healthy: bool) -> None:
        self.ready = False
        self._healthy = healthy

    async def health(self) -> bool:
        self.ready = self._healthy
        return self.ready

    async def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        yield LLMDelta(text="Merhaba!")


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "CORS_ORIGINS": "http://demo.test",
        "stt_model_id": "local-stt",
        "tts_model_id": "local-tts",
        "llama_cpp_model": "local-llm",
        # Off unless a test injects a double; the real client would probe :6006.
        "intent_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


class _IntentRuntime:
    def __init__(self, *, healthy: bool) -> None:
        self.ready = False
        self._healthy = healthy

    async def health(self) -> bool:
        self.ready = self._healthy
        return self.ready

    async def classify(self, text: str, labels):
        raise AssertionError("health checks must not classify")


def _application(
    tmp_path: Path,
    *,
    speech_ready: bool,
    llm_ready: bool,
    settings: Settings | None = None,
    intent: object | None = None,
):
    stt = _SpeechRuntime(ready_after_load=speech_ready)
    tts = _SpeechRuntime(ready_after_load=speech_ready)
    llm = _LLMRuntime(healthy=llm_ready)
    return create_app(
        settings=settings or _settings(),
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=stt,
        tts=tts,
        llm=llm,
        intent=intent,
        validate_audio_tools=lambda: None,
    )


def test_an_unreachable_classifier_is_reported_without_failing_the_pipeline(
    tmp_path: Path,
):
    """Losing routing costs answer quality, so it must not read as an outage."""
    app = _application(
        tmp_path,
        speech_ready=True,
        llm_ready=True,
        settings=_settings(intent_enabled=True),
        intent=_IntentRuntime(healthy=False),
    )

    with TestClient(app) as client:
        payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["intent_ready"] is False
    assert payload["intent_enabled"] is True


def test_a_reachable_classifier_is_probed_at_startup(tmp_path: Path):
    """A classifier that is never probed would show as down for the whole session."""
    app = _application(
        tmp_path,
        speech_ready=True,
        llm_ready=True,
        settings=_settings(intent_enabled=True),
        intent=_IntentRuntime(healthy=True),
    )

    with TestClient(app) as client:
        payload = client.get("/api/health").json()

    assert payload["intent_ready"] is True
    assert "6006" not in client.get("/api/health").text


def test_health_is_ok_only_when_every_runtime_is_ready(tmp_path: Path):
    """Dropping any readiness check would incorrectly advertise a usable pipeline."""
    app = _application(tmp_path, speech_ready=True, llm_ready=True)

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "stt_ready": True,
        "tts_ready": True,
        "llm_ready": True,
        "intent_ready": False,
        "intent_enabled": False,
        "llm_base_url": "configured",
        "audio_input_mode": "browser",
        "local_audio_playback": False,
        "audio_output_mode": "browser",
        "audio_output_device": "plughw:2,0",
    }
    assert "127.0.0.1" not in response.text


def test_empty_model_configuration_starts_in_degraded_mode(tmp_path: Path):
    """Treating blank model IDs as fatal would prevent deliberate staged setup."""
    app = _application(
        tmp_path,
        speech_ready=False,
        llm_ready=False,
        settings=_settings(
            stt_model_id="",
            tts_model_id="",
            llama_cpp_model="",
            llama_cpp_base_url="http://private-llm.internal:8080",
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "stt_ready": False,
        "tts_ready": False,
        "llm_ready": False,
        "intent_ready": False,
        "intent_enabled": False,
        "llm_base_url": "configured",
        "audio_input_mode": "browser",
        "local_audio_playback": False,
        "audio_output_mode": "browser",
        "audio_output_device": "plughw:2,0",
    }
    assert "private-llm.internal" not in response.text


def test_cors_allows_only_the_configured_browser_origin(tmp_path: Path):
    """Reflecting arbitrary origins would expose this LAN-only API cross-origin."""
    app = _application(tmp_path, speech_ready=True, llm_ready=True)

    with TestClient(app) as client:
        allowed = client.options(
            "/api/turns",
            headers={
                "Origin": "http://demo.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        rejected = client.options(
            "/api/turns",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://demo.test"
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers
    assert rejected.json() == {
        "code": "cors_not_allowed",
        "message": "Bu kaynaktan erişime izin verilmiyor.",
    }


def test_openapi_and_interactive_documentation_routes_are_disabled(tmp_path: Path):
    """Leaving generated schemas or consoles enabled expands the LAN API surface."""
    app = _application(tmp_path, speech_ready=True, llm_ready=True)

    with TestClient(app) as client:
        responses = [
            client.get(path)
            for path in (
                "/openapi.json",
                "/docs",
                "/redoc",
                "/docs/oauth2-redirect",
            )
        ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
