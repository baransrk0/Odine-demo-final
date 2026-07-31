"""Deterministic runtime doubles shared by turn-orchestration tests."""

from pathlib import Path

import pytest

from collections.abc import Sequence

from app.runtimes.protocols import IntentPrediction, LLMDelta


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event() -> None:
    """Keep SSE endpoint tests isolated across TestClient event loops."""
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


class FakeSTT:
    ready = True

    def __init__(self, transcript: str = "Merhaba") -> None:
        self.transcript = transcript
        self.calls: list[Path] = []

    async def transcribe(self, path: Path) -> str:
        self.calls.append(path)
        return self.transcript


class FakeLLM:
    ready = True

    def __init__(
        self,
        deltas: tuple[LLMDelta, ...] | None = None,
    ) -> None:
        self.deltas = deltas or (
            LLMDelta(text="Birinci cümle. "),
            LLMDelta(text="İkinci cümle."),
        )
        self.calls: list[str] = []
        self.system_prompts: list[str | None] = []

    async def stream_answer(self, transcript: str, system_prompt: str | None = None):
        self.calls.append(transcript)
        self.system_prompts.append(system_prompt)
        for delta in self.deltas:
            yield delta


class FakeTTS:
    ready = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str, output_path: Path) -> None:
        self.calls.append(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF")


class FakeIntentClassifier:
    """A zero-shot service that always returns one scripted verdict."""

    def __init__(
        self,
        label: str = "sohbet",
        confidence: float = 0.9,
        *,
        ready: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.ready = ready
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def classify(self, text: str, labels: Sequence[str]) -> IntentPrediction:
        self.calls.append((text, tuple(labels)))
        if self.error is not None:
            raise self.error
        return IntentPrediction(
            label=self.label,
            confidence=self.confidence,
            scores={self.label: self.confidence},
        )


@pytest.fixture
def fake_intent() -> FakeIntentClassifier:
    return FakeIntentClassifier()


@pytest.fixture
def fake_stt() -> FakeSTT:
    return FakeSTT()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_tts() -> FakeTTS:
    return FakeTTS()
