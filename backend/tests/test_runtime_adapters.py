"""Contract tests for warm, model-agnostic Hugging Face runtime adapters."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from app.runtimes.stt import HuggingFaceSTT
from app.runtimes.tts import HuggingFaceTTS, SynthesizedAudio


def test_empty_model_ids_do_not_load_or_mark_adapters_ready():
    """An unconfigured deployment must neither load nor advertise STT/TTS."""
    stt_loader = Mock()
    tts_loader = Mock()

    stt = HuggingFaceSTT(model_id="", loader=stt_loader)
    tts = HuggingFaceTTS(model_id="", loader=tts_loader)

    stt.load()
    tts.load()

    stt_loader.assert_not_called()
    tts_loader.assert_not_called()
    assert stt.ready is False
    assert tts.ready is False


def test_blank_model_ids_with_injected_pipelines_are_not_deployment_ready():
    """Test-only injection must not make an unconfigured deployment look ready."""
    stt = HuggingFaceSTT(model_id="", pipeline=Mock())
    tts = HuggingFaceTTS(model_id="", pipeline=Mock())

    stt.load()
    tts.load()

    assert stt.ready is False
    assert tts.ready is False


async def test_stt_passes_turkish_transcription_contract_and_trims_result(
    tmp_path: Path,
):
    """Changing STT language/task or returning untrimmed text would break callers."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"placeholder")
    pipeline = Mock(return_value={"text": "  Merhaba, İzmir!  \n"})
    runtime = HuggingFaceSTT(model_id="", pipeline=pipeline)

    transcript = await runtime.transcribe(audio)

    assert transcript == "Merhaba, İzmir!"
    pipeline.assert_called_once_with(
        str(audio),
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )


async def test_stt_normalizes_whitespace_only_transcript_to_empty(tmp_path: Path):
    """Whitespace speech output must remain an explicit empty transcript."""
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"placeholder")
    runtime = HuggingFaceSTT(model_id="", pipeline=Mock(return_value={"text": " \t\n "}))

    assert await runtime.transcribe(audio) == ""


def test_stt_loader_receives_mapped_device_dtype_and_local_only_flag():
    """Changing loader configuration could silently load the wrong precision/device."""
    pipeline = Mock()
    loader = Mock(return_value=pipeline)
    runtime = HuggingFaceSTT(
        model_id="configured-at-deployment",
        loader=loader,
        device="cpu",
        dtype="float32",
    )

    runtime.load()

    loader.assert_called_once_with(
        "configured-at-deployment",
        device=-1,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    assert runtime.ready is True


async def test_stt_propagates_inference_exception(tmp_path: Path):
    """Inference failure must not be disguised as a successful empty transcript."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"placeholder")
    runtime = HuggingFaceSTT(
        model_id="",
        pipeline=Mock(side_effect=RuntimeError("inference failed")),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        await runtime.transcribe(audio)


async def test_tts_writes_mono_wav_with_pipeline_sample_rate(tmp_path: Path):
    """Changing audio shape or sample rate would make returned chunks play incorrectly."""
    output = tmp_path / "0.wav"
    pipeline = Mock(
        return_value=SynthesizedAudio(
            samples=np.array([0.0, 0.25, -0.25], dtype=np.float32),
            sample_rate=22_050,
        )
    )
    runtime = HuggingFaceTTS(model_id="", pipeline=pipeline)

    await runtime.synthesize("Merhaba.", output)

    assert output.read_bytes().startswith(b"RIFF")
    info = sf.info(output)
    assert info.samplerate == 22_050
    assert info.channels == 1
    pipeline.assert_called_once_with("Merhaba.")


@pytest.mark.parametrize("sample_rate_key", ["sampling_rate", "sample_rate"])
async def test_tts_normalizes_transformers_mapping_output(
    tmp_path: Path, sample_rate_key: str
):
    """Raw text-to-audio output must write before model-specific normalization exists."""
    output = tmp_path / f"{sample_rate_key}.wav"
    pipeline = Mock(
        return_value={
            "audio": np.array([0.0, 0.25, -0.25], dtype=np.float32),
            sample_rate_key: 16_000,
        }
    )
    runtime = HuggingFaceTTS(model_id="", pipeline=pipeline)

    await runtime.synthesize("Merhaba.", output)

    assert sf.info(output).samplerate == 16_000
    assert sf.info(output).channels == 1


def test_tts_loader_receives_mapped_device_dtype_and_local_only_flag():
    """Changing TTS loader configuration could silently load a network model or wrong dtype."""
    loader = Mock(return_value=Mock())
    runtime = HuggingFaceTTS(
        model_id="configured-at-deployment",
        loader=loader,
        device="cuda:1",
        dtype="bfloat16",
    )

    runtime.load()

    loader.assert_called_once_with(
        "configured-at-deployment",
        device=1,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    assert runtime.ready is True


async def test_tts_propagates_inference_exception(tmp_path: Path):
    """A TTS pipeline failure must remain visible to the orchestration error path."""
    runtime = HuggingFaceTTS(
        model_id="",
        pipeline=Mock(side_effect=RuntimeError("synthesis failed")),
    )

    with pytest.raises(RuntimeError, match="synthesis failed"):
        await runtime.synthesize("Merhaba.", tmp_path / "0.wav")
