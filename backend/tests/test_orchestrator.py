"""Concurrency and lifecycle contracts for streamed voice turns."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from app.audio.storage import AudioStorage
from app.config import Settings
from app.errors import ApiError
from app.runtimes.protocols import LLMDelta
from app.schemas import Stage, TurnMetrics, TurnStatus
from app.turns.events import EventHeartbeat, TurnEventBuffer
from app.turns.metrics import RecentMetrics
from app.turns.orchestrator import TurnContext, TurnOrchestrator


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "metrics_capacity": 3,
        "stt_timeout_seconds": 0.2,
        "llm_timeout_seconds": 0.2,
        "tts_timeout_seconds": 0.2,
        "stt_device": "cuda",
        "tts_device": "cuda",
        "stt_dtype": "float16",
        "tts_dtype": "float16",
    }
    values.update(overrides)
    return Settings(**values)


def _make_turn(
    storage: AudioStorage,
    turn_id: UUID = UUID(int=1),
    release_lease: Callable[[], object] | None = None,
) -> TurnContext:
    source = storage.upload_path(turn_id, ".webm")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    stt_input = storage.intermediate_path(turn_id)
    stt_input.write_bytes(b"RIFF")
    return TurnContext(
        turn_id=turn_id,
        input_path=stt_input,
        events=TurnEventBuffer(turn_id),
        source_path=source,
        recording_duration_seconds=1.25,
        recording_bytes=6,
        recording_content_type="audio/webm",
        upload_ms=4.0,
        release_lease=release_lease,
    )


def _events(turn: TurnContext, name: str):
    return [event for event in turn.events.snapshot() if event.name == name]


def _terminal_events(turn: TurnContext):
    return [
        event
        for event in turn.events.snapshot()
        if event.name in {"complete", "failed"}
    ]


async def test_event_buffer_replays_after_id_then_follows_without_duplicates():
    """Using >= instead of > during replay would replay an audio chunk."""
    turn_id = UUID(int=10)
    events = TurnEventBuffer(turn_id)
    state = await events.publish("state", stage=Stage.TRANSCRIBING)
    audio = await events.publish(
        "audio_ready",
        sequence=0,
        text="Merhaba.",
        audio_url=f"/api/audio/{turn_id}/0.wav",
    )

    subscriber = events.subscribe(last_event_id=state.event_id)
    assert (await anext(subscriber)).event_id == audio.event_id

    next_event = asyncio.create_task(anext(subscriber))
    metrics = await events.publish("metrics", metrics=TurnMetrics())
    assert await next_event == metrics

    complete = await events.publish(
        "complete",
        transcript="Merhaba",
        answer="Merhaba.",
        metrics=TurnMetrics(outcome=TurnStatus.COMPLETE),
    )
    assert (await anext(subscriber)).event_id == complete.event_id
    with pytest.raises(StopAsyncIteration):
        await anext(subscriber)

    reconnect = events.subscribe(last_event_id=audio.event_id)
    replayed = [event async for event in reconnect]
    assert [event.name for event in replayed] == ["metrics", "complete"]
    assert [event.event_id for event in events.snapshot()] == [1, 2, 3, 4]


async def test_event_buffer_emits_comment_heartbeat_while_idle():
    """Removing the idle timeout would leave a quiet SSE connection unprobed."""
    subscriber = TurnEventBuffer(
        UUID(int=11), heartbeat_seconds=0.001
    ).subscribe()

    heartbeat = await asyncio.wait_for(anext(subscriber), timeout=0.1)

    assert heartbeat == EventHeartbeat()
    await subscriber.aclose()


def test_recent_metrics_are_bounded_anonymous_and_playback_is_set_once():
    """Unbounded/content-bearing records or repeat playback writes violate privacy."""
    recent = RecentMetrics(_settings(metrics_capacity=2))
    first_id, second_id, third_id = UUID(int=1), UUID(int=2), UUID(int=3)
    recent.append(first_id, TurnMetrics(transcript_chars=7))
    recent.append(second_id, TurnMetrics(answer_chars=12))
    recent.append(third_id, TurnMetrics(answer_chars=19))

    assert [item.answer_chars for item in recent.snapshot()] == [12, 19]
    assert recent.set_first_audio_started(third_id, 81.5) is True
    assert recent.set_first_audio_started(third_id, 999.0) is False
    assert recent.snapshot()[-1].first_audio_started_ms == 81.5
    serialized = "".join(item.model_dump_json() for item in recent.snapshot())
    assert "gizli konuşma" not in serialized
    metric_keys = recent.snapshot()[-1].model_dump().keys()
    assert "transcript" not in metric_keys
    assert "answer" not in metric_keys


async def test_blank_stt_fails_without_calling_llm_or_tts(
    tmp_path: Path, fake_llm, fake_tts
):
    """Whitespace STT must not enter generation or synthesis."""
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    released = 0

    async def release() -> None:
        nonlocal released
        released += 1

    turn.release_lease = release
    stt = type(
        "BlankSTT",
        (),
        {"ready": True, "transcribe": lambda self, path: _async_value("  \n")},
    )()
    recent = RecentMetrics(_settings())
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    await orchestrator.run(turn)

    assert fake_llm.calls == []
    assert fake_tts.calls == []
    assert _terminal_events(turn)[0].payload.error.code == "stt_empty"
    assert len(_terminal_events(turn)) == 1
    assert turn.metrics is not None
    assert turn.metrics.outcome is TurnStatus.FAILED
    assert turn.metrics.error_stage is Stage.TRANSCRIBING
    assert len(recent.snapshot()) == 1
    assert released == 1
    assert not turn.source_path.exists()
    assert not turn.input_path.exists()


async def _async_value(value):
    return value


async def test_first_sentence_reaches_tts_before_llm_stream_completes(
    tmp_path: Path, fake_stt, fake_tts
):
    """Serializing LLM then TTS would leave synthesis idle at this gate."""
    first_yielded = asyncio.Event()
    release_second = asyncio.Event()
    llm_completed = asyncio.Event()

    class GatedLLM:
        ready = True

        async def stream_answer(self, transcript, system_prompt=None):
            yield LLMDelta(text="Birinci cümle. ")
            first_yielded.set()
            await release_second.wait()
            yield LLMDelta(text="İkinci cümle.")
            llm_completed.set()

    tts_started = asyncio.Event()
    original_synthesize = fake_tts.synthesize

    async def observed_synthesize(text, output_path):
        tts_started.set()
        await original_synthesize(text, output_path)

    fake_tts.synthesize = observed_synthesize
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=GatedLLM(),
        tts=fake_tts,
        recent_metrics=RecentMetrics(_settings()),
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(first_yielded.wait(), timeout=0.1)
    await asyncio.wait_for(tts_started.wait(), timeout=0.1)

    assert llm_completed.is_set() is False
    assert _events(turn, "answer_delta")[0].payload.text == "Birinci cümle."
    release_second.set()
    await running


async def test_success_synthesizes_ordered_sentences_and_final_tail(
    tmp_path: Path, fake_stt, fake_tts
):
    """Dropping finish(), using parallel TTS, or renumbering chunks breaks playback."""

    class TelemetryLLM:
        ready = True

        async def stream_answer(self, transcript, system_prompt=None):
            yield LLMDelta(text="Birinci cümle. ")
            yield LLMDelta(
                text="İkinci cümle. Son parça",
                prompt_tokens=8,
                completion_tokens=5,
                tokens_per_second=12.5,
            )

    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    recent = RecentMetrics(_settings())
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=TelemetryLLM(),
        tts=fake_tts,
        recent_metrics=recent,
    )

    await orchestrator.run(turn)

    assert fake_tts.calls == [
        "Birinci cümle.",
        " İkinci cümle.",
        " Son parça",
    ]
    assert [event.payload.sequence for event in _events(turn, "audio_ready")] == [
        0,
        1,
        2,
    ]
    assert turn.produced_sequences == {0, 1, 2}
    assert turn.answer == "Birinci cümle. İkinci cümle. Son parça"
    assert turn.metrics is not None
    assert turn.metrics.sentence_count == 3
    assert turn.metrics.audio_chunk_count == 3
    assert turn.metrics.llm_prompt_tokens == 8
    assert turn.metrics.llm_completion_tokens == 5
    assert turn.metrics.llm_tokens_per_second == 12.5
    assert turn.metrics.first_audio_started_ms is None
    assert len(_terminal_events(turn)) == 1
    assert _terminal_events(turn)[0].name == "complete"
    assert not turn.source_path.exists()
    assert not turn.input_path.exists()
    assert [path.read_bytes() for path in sorted(
        storage.turn_dir(turn.turn_id).glob("chunks/*.wav")
    )] == [b"RIFF", b"RIFF", b"RIFF"]
    assert len(recent.snapshot()) == 1


async def test_tts_failure_preserves_text_and_emits_one_failed_terminal(
    tmp_path: Path, fake_stt
):
    """A synthesis exception must not erase text or be mislabeled complete."""

    class OneDeltaLLM:
        ready = True

        async def stream_answer(self, transcript, system_prompt=None):
            yield LLMDelta(text="Birinci cümle. İkinci cümle.")

    class FailingTTS:
        ready = True

        async def synthesize(self, text, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"partial")
            raise RuntimeError("internal synthesis detail")

    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=OneDeltaLLM(),
        tts=FailingTTS(),
        recent_metrics=RecentMetrics(_settings()),
    )

    await orchestrator.run(turn)

    terminal = _terminal_events(turn)
    assert len(terminal) == 1
    assert terminal[0].name == "failed"
    assert terminal[0].payload.error.code == "tts_failed"
    assert "internal synthesis detail" not in terminal[0].payload.error.message
    assert turn.transcript == "Merhaba"
    assert turn.answer == "Birinci cümle. İkinci cümle."
    assert turn.metrics.error_stage is Stage.SYNTHESIZING
    assert list(storage.turn_dir(turn.turn_id).glob("chunks/*.wav")) == []


async def test_llm_failure_drains_an_already_queued_sentence(
    tmp_path: Path, fake_stt
):
    """Canceling the worker on LLM failure would silently lose queued audio."""
    synthesis_started = asyncio.Event()
    allow_synthesis = asyncio.Event()

    class FailingLLM:
        ready = True

        async def stream_answer(self, transcript, system_prompt=None):
            yield LLMDelta(text="Korunacak cümle.")
            await synthesis_started.wait()
            raise ApiError("llm_timeout", "Yanıt üretilemedi, tekrar deneyin.", 504)

    class GatedTTS:
        ready = True

        async def synthesize(self, text, output_path):
            synthesis_started.set()
            await allow_synthesis.wait()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF")

    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=FailingLLM(),
        tts=GatedTTS(),
        recent_metrics=RecentMetrics(_settings()),
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(synthesis_started.wait(), timeout=0.1)
    assert running.done() is False
    allow_synthesis.set()
    await running

    assert [event.payload.sequence for event in _events(turn, "audio_ready")] == [0]
    assert storage.chunk_path(turn.turn_id, 0).exists()
    assert _terminal_events(turn)[0].payload.error.code == "llm_timeout"
    assert turn.metrics.timed_out is True
    assert turn.metrics.audio_chunk_count == 1


@pytest.mark.parametrize(
    ("failure", "expected_code", "timed_out"),
    [
        (asyncio.TimeoutError(), "tts_timeout", True),
        (RuntimeError("CUDA out of memory"), "gpu_out_of_memory", False),
    ],
)
async def test_tts_timeout_or_oom_is_an_explicit_terminal_failure(
    tmp_path: Path, fake_stt, fake_llm, failure, expected_code, timed_out
):
    """Timeout/OOM must never look like a successful turn with a missing chunk."""

    class BrokenTTS:
        ready = True

        async def synthesize(self, text, output_path):
            raise failure

    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=BrokenTTS(),
        recent_metrics=RecentMetrics(_settings()),
    )

    await orchestrator.run(turn)

    terminal = _terminal_events(turn)[0]
    assert terminal.name == "failed"
    assert terminal.payload.error.code == expected_code
    assert turn.metrics.timed_out is timed_out
    assert turn.metrics.sentence_count >= 1
    assert turn.metrics.audio_chunk_count == 0


async def test_cancellation_cleans_up_releases_and_records_one_terminal(
    tmp_path: Path, fake_llm, fake_tts
):
    """External cancellation must not leak the active lease or temp audio."""
    transcription_started = asyncio.Event()
    never_release = asyncio.Event()
    released = 0

    class BlockingSTT:
        ready = True

        async def transcribe(self, path):
            transcription_started.set()
            await never_release.wait()

    async def release() -> None:
        nonlocal released
        released += 1

    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage, release_lease=release)
    recent = RecentMetrics(_settings())
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=BlockingSTT(),
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(transcription_started.wait(), timeout=0.1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert released == 1
    assert not turn.source_path.exists()
    assert not turn.input_path.exists()
    assert len(_terminal_events(turn)) == 1
    assert _terminal_events(turn)[0].payload.error.code == "turn_cancelled"
    assert len(recent.snapshot()) == 1


async def test_cancellation_during_metrics_publication_finishes_finalization(
    tmp_path: Path, fake_stt, fake_llm, fake_tts
):
    """Canceling the parent at metrics publication must not abort the finalizer."""
    publication_started = asyncio.Event()
    allow_publication = asyncio.Event()
    released = 0

    class GatedMetricsEvents(TurnEventBuffer):
        async def publish(self, name, payload=None, **fields):
            if name == "metrics":
                publication_started.set()
                await allow_publication.wait()
            return await super().publish(name, payload, **fields)

    async def release() -> None:
        nonlocal released
        released += 1

    settings = _settings()
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage, release_lease=release)
    turn.events = GatedMetricsEvents(turn.turn_id)
    recent = RecentMetrics(settings)
    orchestrator = TurnOrchestrator(
        settings=settings,
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(publication_started.wait(), timeout=0.1)
    running.cancel()
    await asyncio.sleep(0)
    remained_owned = not running.done()
    allow_publication.set()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert remained_owned is True
    assert len(recent.snapshot()) == 1
    assert len(_events(turn, "metrics")) == 1
    assert len(_terminal_events(turn)) == 1
    assert released == 1


async def test_cancellation_during_lease_release_finishes_release(
    tmp_path: Path, fake_stt, fake_llm, fake_tts
):
    """Canceling the parent while releasing must not leak the active-turn lease."""
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    released = 0

    async def release() -> None:
        nonlocal released
        release_started.set()
        await allow_release.wait()
        released += 1

    settings = _settings()
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage, release_lease=release)
    recent = RecentMetrics(settings)
    orchestrator = TurnOrchestrator(
        settings=settings,
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(release_started.wait(), timeout=0.1)
    running.cancel()
    await asyncio.sleep(0)
    remained_owned = not running.done()
    allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert remained_owned is True
    assert len(recent.snapshot()) == 1
    assert len(_events(turn, "metrics")) == 1
    assert len(_terminal_events(turn)) == 1
    assert released == 1


async def test_original_cancellation_wins_when_release_later_fails(
    tmp_path: Path, fake_stt, fake_llm, fake_tts
):
    """A finalizer error must not replace an already-observed owner cancellation."""
    release_started = asyncio.Event()
    allow_release_failure = asyncio.Event()
    release_failure_reached = asyncio.Event()

    async def failing_release() -> None:
        release_started.set()
        await allow_release_failure.wait()
        release_failure_reached.set()
        raise RuntimeError("internal release failure")

    settings = _settings()
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage, release_lease=failing_release)
    recent = RecentMetrics(settings)
    orchestrator = TurnOrchestrator(
        settings=settings,
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    running = asyncio.create_task(orchestrator.run(turn))
    await asyncio.wait_for(release_started.wait(), timeout=0.1)
    running.cancel("owner-shutdown")
    allow_release_failure.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await running

    assert caught.value.args == ("owner-shutdown",)
    assert release_failure_reached.is_set() is True
    assert len(recent.snapshot()) == 1
    assert len(_events(turn, "metrics")) == 1
    assert len(_terminal_events(turn)) == 1


async def test_cleanup_failure_still_records_terminal_metric_and_releases(
    tmp_path: Path, fake_stt, fake_llm, fake_tts
):
    """A filesystem cleanup error must not skip observable terminal cleanup."""
    released = 0

    class CleanupFailingStorage(AudioStorage):
        def cleanup_terminal(self, turn_id):
            raise OSError("/private/runtime-audio/secret")

    async def release() -> None:
        nonlocal released
        released += 1

    settings = _settings()
    storage = CleanupFailingStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage, release_lease=release)
    recent = RecentMetrics(settings)
    orchestrator = TurnOrchestrator(
        settings=settings,
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    await orchestrator.run(turn)

    terminal = _terminal_events(turn)
    assert len(terminal) == 1
    assert terminal[0].name == "failed"
    assert terminal[0].payload.error.code == "cleanup_failed"
    assert "/private/runtime-audio/secret" not in terminal[0].payload.error.message
    assert turn.metrics.outcome is TurnStatus.FAILED
    assert turn.metrics.error_stage is Stage.FAILED
    assert len(recent.snapshot()) == 1
    assert len(_events(turn, "metrics")) == 1
    assert released == 1


async def test_metrics_include_only_safe_anonymous_configuration(
    tmp_path: Path, fake_stt, fake_llm, fake_tts
):
    """Persisting model/content/path settings would expose deployment secrets."""
    sensitive_values = (
        "https://user:password@private-host/internal",
        "/private/models/assistant-secret.gguf",
        "private/stt-model-secret",
        "private/tts-model-secret",
        "gizli sistem promptu",
        "/private/device-secret",
        "secret-dtype",
    )
    settings = _settings(
        llama_cpp_base_url=sensitive_values[0],
        llama_cpp_model=sensitive_values[1],
        stt_model_id=sensitive_values[2],
        tts_model_id=sensitive_values[3],
        turkish_system_prompt=sensitive_values[4],
        stt_device=sensitive_values[5],
        tts_device="cuda:1",
        stt_dtype=sensitive_values[6],
        tts_dtype="float32",
        llm_max_tokens=321,
        stt_timeout_seconds=0.11,
        llm_timeout_seconds=0.12,
        tts_timeout_seconds=0.13,
    )
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    recent = RecentMetrics(settings)
    orchestrator = TurnOrchestrator(
        settings=settings,
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=recent,
    )

    await orchestrator.run(turn)

    assert turn.metrics.configuration.model_dump() == {
        "stt_device": "configured",
        "tts_device": "cuda:1",
        "stt_dtype": "configured",
        "tts_dtype": "float32",
        "llm_max_tokens": 321,
        "stt_timeout_seconds": 0.11,
        "llm_timeout_seconds": 0.12,
        "tts_timeout_seconds": 0.13,
        "tts_queue_capacity": 4,
    }
    assert turn.metrics.device == "stt=configured;tts=cuda:1"
    assert turn.metrics.dtype == "stt=configured;tts=float32"
    serialized = recent.snapshot()[0].model_dump_json()
    for sensitive in sensitive_values:
        assert sensitive not in serialized
    assert turn.transcript not in serialized
    assert turn.answer not in serialized


async def test_elapsed_metrics_use_perf_counter_ns_not_wall_clock(
    tmp_path: Path, fake_stt, fake_llm, fake_tts, monkeypatch
):
    """Changing elapsed timing away from perf_counter_ns breaks exact durations."""
    import app.turns.orchestrator as orchestrator_module

    class StepClock:
        calls = 0

        def __call__(self) -> int:
            self.calls += 1
            return self.calls * 1_000_000

    def wall_clock_is_not_a_duration_source():
        raise AssertionError("wall clock used for duration")

    clock = StepClock()
    monkeypatch.setattr(orchestrator_module.time, "perf_counter_ns", clock)
    monkeypatch.setattr(
        orchestrator_module.time,
        "time",
        wall_clock_is_not_a_duration_source,
    )
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn = _make_turn(storage)
    orchestrator = TurnOrchestrator(
        settings=_settings(),
        storage=storage,
        stt=fake_stt,
        llm=fake_llm,
        tts=fake_tts,
        recent_metrics=RecentMetrics(_settings()),
    )

    await orchestrator.run(turn)

    assert clock.calls >= 8
    assert turn.metrics.stt_ms == 1.0
    assert turn.metrics.total_ms == clock.calls - 1
