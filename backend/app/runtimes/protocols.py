"""Structural interfaces shared by runtime adapters and orchestration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Protocol


@dataclass(frozen=True, slots=True)
class LLMDelta:
    """A streamed answer fragment with optional server telemetry."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    """One classifier verdict plus the full distribution it was chosen from."""

    label: str
    confidence: float
    scores: Mapping[str, float] = field(default_factory=dict)


class STTRuntime(Protocol):
    """Asynchronous speech-to-text runtime contract."""

    ready: bool

    async def transcribe(self, path: Path) -> str:
        """Transcribe one validated audio file."""
        ...


class TTSRuntime(Protocol):
    """Asynchronous text-to-speech runtime contract."""

    ready: bool

    async def synthesize(self, text: str, output_path: Path) -> None:
        """Synthesize text into the given output file."""
        ...


class IntentRuntime(Protocol):
    """Zero-shot intent classification runtime contract."""

    ready: bool

    async def classify(
        self,
        text: str,
        labels: Sequence[str],
    ) -> IntentPrediction:
        """Score the candidate labels against one transcript."""
        ...


class LLMRuntime(Protocol):
    """Streaming language-model runtime contract."""

    ready: bool

    def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """Stream answer fragments in server order under the routed agent prompt."""
        ...
