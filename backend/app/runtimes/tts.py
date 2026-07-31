"""Warm, injectable Hugging Face text-to-speech adapter."""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.runtimes.stt import _pipeline_device, _torch_dtype


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """Normalized output supplied by a model-specific TTS pipeline adapter."""

    samples: np.ndarray
    sample_rate: int


TTSResult = SynthesizedAudio | Mapping[str, object]
TTSPipeline = Callable[[str], TTSResult]
TTSLoader = Callable[..., TTSPipeline]


class HuggingFaceTTS:
    """Run one already-loaded TTS pipeline and write each chunk independently."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        loader: TTSLoader | None = None,
        pipeline: TTSPipeline | None = None,
        local_files_only: bool = True,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._loader = loader or _load_pipeline
        self._pipeline = pipeline
        self._local_files_only = local_files_only
        self.ready = bool(self._model_id.strip() and pipeline is not None)

    def load(self) -> None:
        """Load a configured pipeline from the local Hugging Face cache only."""
        if self._pipeline is not None:
            self.ready = bool(self._model_id.strip())
            return
        if not self._model_id.strip():
            self.ready = False
            return

        self._pipeline = self._loader(
            self._model_id,
            device=_pipeline_device(self._device),
            torch_dtype=_torch_dtype(self._dtype),
            local_files_only=self._local_files_only,
        )
        self.ready = True

    async def synthesize(self, text: str, output_path: Path) -> None:
        """Synthesize and persist one WAV chunk on a worker thread."""
        pipeline = self._require_pipeline()
        await asyncio.to_thread(_synthesize, pipeline, text, output_path)

    def _require_pipeline(self) -> TTSPipeline:
        if self._pipeline is None:
            raise RuntimeError("TTS runtime is not ready.")
        return self._pipeline


def _load_pipeline(
    model_id: str,
    *,
    device: int,
    torch_dtype: torch.dtype,
    local_files_only: bool,
) -> TTSPipeline:
    """Build a generic Transformers TTS pipeline without choosing a model."""
    from transformers import pipeline

    return pipeline(
        "text-to-speech",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
        model_kwargs={"local_files_only": local_files_only},
    )


def _synthesize(pipeline: TTSPipeline, text: str, output_path: Path) -> None:
    with torch.inference_mode():
        audio = _normalize_audio(pipeline(text))

    if audio.sample_rate <= 0:
        raise ValueError("TTS sample rate must be positive.")

    samples = np.asarray(audio.samples)
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples = samples[:, 0]
    if samples.ndim != 1:
        raise ValueError("TTS samples must be mono.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, audio.sample_rate, format="WAV", subtype="PCM_16")


def _normalize_audio(output: TTSResult) -> SynthesizedAudio:
    """Normalize standard Transformers text-to-audio mappings at the boundary."""
    if isinstance(output, SynthesizedAudio):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("TTS pipeline result must be SynthesizedAudio or a mapping.")

    samples = output.get("audio")
    sample_rate = output.get("sampling_rate", output.get("sample_rate"))
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("TTS pipeline result must contain an integer sample rate.")
    return SynthesizedAudio(samples=np.asarray(samples), sample_rate=sample_rate)
