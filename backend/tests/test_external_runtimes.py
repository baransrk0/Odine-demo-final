"""Contracts for the dependency-free Orin speech runtime adapters."""

import builtins
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import respx
from httpx import Response

from app.config import Settings
from app.runtimes.factory import build_speech_runtimes
from app.runtimes.piper import PiperTTS
from app.runtimes.whisper_cpp import WhisperCppSTT


@pytest.mark.asyncio
async def test_whisper_cpp_posts_a_turkish_wav_and_trims_transcript(tmp_path: Path):
    """Changing the HTTP form contract would break the deployed whisper-server."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF test wav")
    runtime = WhisperCppSTT("http://whisper.test", timeout_seconds=3)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("http://whisper.test/inference").mock(
            return_value=Response(200, json={"text": "  Merhaba dünya. \n"})
        )
        transcript = await runtime.transcribe(audio)

    assert transcript == "Merhaba dünya."
    body = route.calls[0].request.content
    assert b'name="file"' in body
    assert b'name="language"' in body
    assert b'tr' in body
    assert b'name="response_format"' in body
    assert b'json' in body


@pytest.mark.asyncio
async def test_whisper_cpp_rejects_invalid_json_response(tmp_path: Path):
    """A successful but malformed server reply must not look like an empty transcript."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF test wav")
    runtime = WhisperCppSTT("http://whisper.test", timeout_seconds=3)

    with respx.mock(assert_all_called=True) as router:
        router.post("http://whisper.test/inference").mock(
            return_value=Response(200, json={"unexpected": "payload"})
        )
        with pytest.raises(RuntimeError, match="invalid response"):
            await runtime.transcribe(audio)


@pytest.mark.asyncio
async def test_piper_writes_requested_wav_with_text_on_standard_input(tmp_path: Path):
    """Changing the Piper invocation would silently prevent audio chunks from being created."""
    output = tmp_path / "answer.wav"
    runner = Mock()
    process = AsyncMock()
    process.communicate.return_value = (b"", b"")
    process.returncode = 0

    async def create_process(*args, **kwargs):
        output.write_bytes(b"RIFF piper wav")
        runner(*args, **kwargs)
        return process

    runtime = PiperTTS(
        binary="/usr/local/bin/piper",
        model_path="/models/tr_TR.onnx",
        process_factory=create_process,
    )

    await runtime.synthesize("Merhaba.", output)

    args, kwargs = runner.call_args
    assert args == (
        "/usr/local/bin/piper",
        "--model",
        "/models/tr_TR.onnx",
        "--output_file",
        str(output),
    )
    assert kwargs["stdin"] is not None
    assert output.read_bytes().startswith(b"RIFF")
    process.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_piper_removes_partial_output_after_nonzero_exit(tmp_path: Path):
    output = tmp_path / "answer.wav"

    async def create_process(*args, **kwargs):
        output.write_bytes(b"partial")
        process = AsyncMock()
        process.communicate.return_value = (b"", b"model load failed")
        process.returncode = 1
        return process

    runtime = PiperTTS(
        binary="/usr/local/bin/piper",
        model_path="/models/tr_TR.onnx",
        process_factory=create_process,
    )

    with pytest.raises(RuntimeError, match="Piper synthesis failed"):
        await runtime.synthesize("Merhaba.", output)

    assert output.exists() is False


class _BlockingProcess:
    def __init__(self, *, stop_on_terminate: bool) -> None:
        self.returncode = None
        self.stop_on_terminate = stop_on_terminate
        self.communicate_started = asyncio.Event()
        self.release_communicate = asyncio.Event()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self._stopped = asyncio.Event()

    async def communicate(self, data):
        self.communicate_started.set()
        await self.release_communicate.wait()
        return b"", b""

    def terminate(self):
        self.terminate_calls += 1
        if self.stop_on_terminate:
            self.returncode = -15
            self._stopped.set()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9
        self._stopped.set()

    async def wait(self):
        self.wait_calls += 1
        await self._stopped.wait()
        return self.returncode


@pytest.mark.asyncio
async def test_piper_cancellation_terminates_and_reaps_process(tmp_path: Path):
    output = tmp_path / "answer.wav"
    process = _BlockingProcess(stop_on_terminate=True)

    async def create_process(*args, **kwargs):
        output.write_bytes(b"partial")
        return process

    runtime = PiperTTS(
        binary="/usr/local/bin/piper",
        model_path="/models/tr_TR.onnx",
        process_factory=create_process,
    )
    running = asyncio.create_task(runtime.synthesize("Merhaba.", output))
    await process.communicate_started.wait()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == 1
    assert output.exists() is False


@pytest.mark.asyncio
async def test_piper_outer_timeout_terminates_process_and_removes_output(
    tmp_path: Path,
):
    """The orchestrator's wait_for timeout must not leave a Piper child behind."""
    output = tmp_path / "answer.wav"
    process = _BlockingProcess(stop_on_terminate=True)

    async def create_process(*args, **kwargs):
        output.write_bytes(b"partial")
        return process

    runtime = PiperTTS(
        binary="/usr/local/bin/piper",
        model_path="/models/tr_TR.onnx",
        process_factory=create_process,
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            runtime.synthesize("Merhaba.", output),
            timeout=0.01,
        )

    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert output.exists() is False


@pytest.mark.asyncio
async def test_piper_kills_process_that_ignores_terminate(tmp_path: Path):
    output = tmp_path / "answer.wav"
    process = _BlockingProcess(stop_on_terminate=False)

    async def create_process(*args, **kwargs):
        output.write_bytes(b"partial")
        return process

    runtime = PiperTTS(
        binary="/usr/local/bin/piper",
        model_path="/models/tr_TR.onnx",
        process_factory=create_process,
        terminate_grace_seconds=0.01,
    )
    running = asyncio.create_task(runtime.synthesize("Merhaba.", output))
    await process.communicate_started.wait()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert output.exists() is False


def test_external_backend_selection_avoids_huggingface_runtime_imports(monkeypatch):
    """Selecting deployed services must not import Torch on the broken Jetson install."""
    settings = Settings(
        _env_file=None,
        stt_backend="whisper_cpp",
        tts_backend="piper",
        whisper_cpp_base_url="http://127.0.0.1:8080",
        piper_binary="/home/odine/.local/bin/piper",
        piper_model_path="/home/odine/piper-models/tr_TR.onnx",
    )

    real_import = builtins.__import__

    def reject_huggingface_dependencies(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers"}:
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_huggingface_dependencies)
    stt, tts = build_speech_runtimes(settings)

    assert isinstance(stt, WhisperCppSTT)
    assert isinstance(tts, PiperTTS)
