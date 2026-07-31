"""Production frontend asset and SPA fallback contracts."""

from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from app.audio.storage import AudioStorage
from app.config import Settings
from app.main import create_app
from app.runtimes.protocols import LLMDelta


class _SpeechRuntime:
    ready = True

    async def transcribe(self, path: Path) -> str:
        return "Merhaba"

    async def synthesize(self, text: str, output_path: Path) -> None:
        output_path.write_bytes(b"RIFF")


class _LLMRuntime:
    ready = True

    async def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        yield LLMDelta(text="Merhaba!")


def _application(tmp_path: Path, frontend_dist: Path):
    return create_app(
        settings=Settings(
            _env_file=None,
            stt_model_id="local-stt",
            tts_model_id="local-tts",
            llama_cpp_model="local-llm",
            # Off unless a test injects a double; the real client would probe :6006.
            intent_enabled=False,
        ),
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=_SpeechRuntime(),
        tts=_SpeechRuntime(),
        llm=_LLMRuntime(),
        validate_audio_tools=lambda: None,
        frontend_dist=frontend_dist,
    )


def test_serves_vite_assets_and_spa_fallback_from_injected_directory(
    tmp_path: Path,
):
    """Dropping either mount would break production assets or client-side routes."""
    frontend_dist = tmp_path / "frontend-dist"
    assets = frontend_dist / "assets"
    assets.mkdir(parents=True)
    (frontend_dist / "index.html").write_text(
        '<main id="root">Orin demo</main>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text(
        "globalThis.__orinDemo = true;",
        encoding="utf-8",
    )
    app = _application(tmp_path, frontend_dist)

    with TestClient(app) as client:
        root = client.get("/")
        nested_route = client.get("/turn/live")
        asset = client.get("/assets/app.js")

    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert root.text == '<main id="root">Orin demo</main>'
    assert nested_route.status_code == 200
    assert nested_route.text == root.text
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("text/javascript")
    assert asset.text == "globalThis.__orinDemo = true;"


def test_spa_fallback_never_shadows_api_or_disabled_docs(tmp_path: Path):
    """A broad catch-all must not turn API misses or disabled consoles into HTML."""
    frontend_dist = tmp_path / "frontend-dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text(
        '<main id="root">Orin demo</main>',
        encoding="utf-8",
    )
    app = _application(tmp_path, frontend_dist)

    with TestClient(app) as client:
        health = client.get("/api/health")
        missing_api = client.get("/api/unknown")
        disabled_responses = {
            path: client.get(path)
            for path in (
                "/docs",
                "/docs/internal",
                "/docs/oauth2-redirect/callback",
                "/redoc",
                "/redoc/internal",
                "/openapi.json",
                "/openapi.json/schema",
            )
        }

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert missing_api.status_code == 404
    assert missing_api.json()["code"] == "not_found"
    assert {
        path: response.status_code
        for path, response in disabled_responses.items()
    } == {
        "/docs": 404,
        "/docs/internal": 404,
        "/docs/oauth2-redirect/callback": 404,
        "/redoc": 404,
        "/redoc/internal": 404,
        "/openapi.json": 404,
        "/openapi.json/schema": 404,
    }
    assert all(
        response.json()["code"] == "not_found"
        for response in disabled_responses.values()
    )
