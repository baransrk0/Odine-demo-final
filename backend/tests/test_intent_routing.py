"""How the routing decision reaches the language model, the clock, and the metrics."""

from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.audio.storage import AudioStorage
from app.config import Settings
from app.intents.engine import IntentEngine
from app.intents.taxonomy import load_taxonomy
from app.schemas import Stage
from app.turns.events import TurnEventBuffer
from app.turns.metrics import RecentMetrics
from app.turns.orchestrator import TurnContext, TurnOrchestrator

from conftest import FakeIntentClassifier, FakeLLM, FakeSTT, FakeTTS


FIXED_TIME = datetime(2026, 7, 31, 9, 30)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "llama_cpp_model": "local-llm",
        "metrics_capacity": 3,
        "stt_timeout_seconds": 5,
        "llm_timeout_seconds": 5,
        "tts_timeout_seconds": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _make_turn(storage: AudioStorage, turn_id: UUID = UUID(int=7)) -> TurnContext:
    stt_input = storage.intermediate_path(turn_id)
    stt_input.parent.mkdir(parents=True, exist_ok=True)
    stt_input.write_bytes(b"RIFF")
    return TurnContext(
        turn_id=turn_id,
        input_path=stt_input,
        events=TurnEventBuffer(turn_id),
    )


def _orchestrator(
    tmp_path: Path,
    *,
    transcript: str,
    classifier: FakeIntentClassifier | None,
    settings: Settings | None = None,
    routed: bool = True,
):
    configured = settings or _settings()
    storage = AudioStorage(tmp_path, retention_seconds=60)
    llm = FakeLLM()
    tts = FakeTTS()
    engine = (
        IntentEngine(
            taxonomy=load_taxonomy(),
            classifier=classifier,
            clock=lambda: FIXED_TIME,
        )
        if routed
        else None
    )
    orchestrator = TurnOrchestrator(
        settings=configured,
        storage=storage,
        stt=FakeSTT(transcript),
        llm=llm,
        tts=tts,
        recent_metrics=RecentMetrics(configured),
        intent=engine,
    )
    return orchestrator, storage, llm, tts


def _events(turn: TurnContext, name: str):
    return [event for event in turn.events.snapshot() if event.name == name]


async def test_routed_agent_prompt_reaches_the_language_model(tmp_path: Path):
    """Classifying and then ignoring the result would make the whole layer decorative."""
    orchestrator, storage, llm, _ = _orchestrator(
        tmp_path,
        transcript="Turnike nasıl uygulanır?",
        classifier=FakeIntentClassifier(label="ilk yardım", confidence=0.9),
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    assert turn.intent is not None
    assert turn.intent.agent == "medikal"
    assert llm.system_prompts == [_settings().agent_system_prompts["medikal"]]
    assert "sağlık asistanısın" in llm.system_prompts[0]


async def test_the_intent_event_and_stage_are_published_before_generation(tmp_path: Path):
    """The operator sees which agent answered; a silent route is undebuggable."""
    orchestrator, storage, _, _ = _orchestrator(
        tmp_path,
        transcript="Fonetik alfabede a b c nasıl söylenir?",
        classifier=FakeIntentClassifier(label="telsiz ve raporlama", confidence=0.77),
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    stages = [event.payload.stage for event in _events(turn, "state")]
    assert stages[:3] == [Stage.TRANSCRIBING, Stage.CLASSIFYING, Stage.GENERATING]

    published = _events(turn, "intent")
    assert len(published) == 1
    payload = published[0].payload
    # Six classifier labels answer through one agent: the operator is shown the
    # narrow label that won, and the broad agent that will answer.
    assert (payload.label, payload.agent, payload.source) == (
        "telsiz ve raporlama",
        "savaş yönergeleri",
        "classifier",
    )
    assert payload.confidence == 0.77
    assert payload.function_call is False


async def test_a_clock_question_is_answered_and_synthesized_without_the_model(
    tmp_path: Path,
):
    """The function-call path must still produce audio, just never a model call."""
    orchestrator, storage, llm, tts = _orchestrator(
        tmp_path,
        transcript="Saat kaç?",
        classifier=FakeIntentClassifier(label="sohbet", confidence=0.99),
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    assert llm.calls == []
    assert turn.answer == "Şu an saat 09:30"
    assert tts.calls == ["Şu an saat 09:30"]
    assert turn.produced_sequences == {0}
    assert _events(turn, "intent")[0].payload.function_call is True
    assert _events(turn, "complete")[0].payload.answer == "Şu an saat 09:30"


async def test_a_degraded_classifier_still_completes_the_turn(tmp_path: Path):
    """An intent outage must not be able to fail a turn that STT and the model can serve."""
    orchestrator, storage, llm, _ = _orchestrator(
        tmp_path,
        transcript="Turnike nasıl uygulanır?",
        classifier=FakeIntentClassifier(error=RuntimeError("service down")),
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    assert _events(turn, "complete")
    assert turn.intent is not None
    assert turn.intent.source == "unavailable"
    assert llm.system_prompts == [_settings().agent_system_prompts["sohbet"]]


async def test_metrics_record_the_routing_decision(tmp_path: Path):
    """Intent accuracy cannot be measured on the device without these fields."""
    orchestrator, storage, _, _ = _orchestrator(
        tmp_path,
        transcript="Yüz yirmi beşin yüzde yirmisi kaç?",
        classifier=FakeIntentClassifier(label="matematik", confidence=0.64),
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    assert turn.metrics is not None
    assert turn.metrics.intent_label == "matematik"
    assert turn.metrics.intent_agent == "matematik"
    assert turn.metrics.intent_source == "classifier"
    assert turn.metrics.intent_confidence == 0.64
    assert turn.metrics.intent_ms is not None


async def test_disabling_routing_leaves_the_original_pipeline_untouched(tmp_path: Path):
    """The layer must be removable; a demo cannot be blocked on the intent service."""
    orchestrator, storage, llm, _ = _orchestrator(
        tmp_path,
        transcript="Saat kaç?",
        classifier=None,
        routed=False,
    )
    turn = _make_turn(storage)

    await orchestrator.run(turn)

    assert _events(turn, "intent") == []
    assert Stage.CLASSIFYING not in [
        event.payload.stage for event in _events(turn, "state")
    ]
    assert llm.system_prompts == [None]
    assert turn.metrics is not None
    assert turn.metrics.intent_label is None
