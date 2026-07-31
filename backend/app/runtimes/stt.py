"""Warm, injectable Hugging Face speech-to-text adapter."""

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


STTPipeline = Callable[..., Mapping[str, Any]]
STTLoader = Callable[..., STTPipeline]


class HuggingFaceSTT:
    """Run one already-loaded ASR pipeline without selecting a concrete model."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        loader: STTLoader | None = None,
        pipeline: STTPipeline | None = None,
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

    async def transcribe(self, path: Path) -> str:
        """Transcribe one normalized audio file on a worker thread."""
        pipeline = self._require_pipeline()
        return await asyncio.to_thread(_transcribe, pipeline, path)

    def _require_pipeline(self) -> STTPipeline:
        if self._pipeline is None:
            raise RuntimeError("STT runtime is not ready.")
        return self._pipeline


def _load_pipeline(
    model_id: str,
    *,
    device: int,
    torch_dtype: torch.dtype,
    local_files_only: bool,
) -> STTPipeline:
    """Build the generic Transformers ASR pipeline without model-specific logic."""
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
        model_kwargs={"local_files_only": local_files_only},
    )


def _transcribe(pipeline: STTPipeline, path: Path) -> str:
    with torch.inference_mode():
        result = pipeline(
            str(path),
            generate_kwargs={"language": "turkish", "task": "transcribe"},
        )

    text = result.get("text")
    if not isinstance(text, str):
        raise TypeError("STT pipeline result must contain text.")
    return text.strip()


def _pipeline_device(device: str) -> int:
    """Map configuration labels to the device identifiers Transformers expects."""
    normalized = device.strip().lower()
    if normalized == "cpu":
        return -1
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        try:
            return int(normalized.removeprefix("cuda:"))
        except ValueError as error:
            raise ValueError(f"Unsupported device: {device}") from error
    raise ValueError(f"Unsupported device: {device}")


def _torch_dtype(dtype: str) -> torch.dtype:
    """Map a safe settings string to a concrete torch dtype."""
    value = getattr(torch, dtype, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return value
