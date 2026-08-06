"""Concurrent STT, streamed LLM, and strictly ordered TTS turn execution."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
from pathlib import Path
import re
import time
from uuid import UUID

from app.audio.output import AudioOutputController
from app.audio.storage import AudioStorage
from app.audio.playback import LocalAudioPlayer, LocalPlaybackError
from app.config import Settings
from app.errors import ApiError
from app.intents.engine import IntentEngine, IntentResult
from app.runtimes.protocols import LLMRuntime, STTRuntime, TTSRuntime
from app.schemas import (
    ErrorBody,
    SafeConfigurationSummary,
    Stage,
    TurnMetrics,
    TurnStatus,
)
from app.turns.events import TurnEventBuffer
from app.turns.metrics import RecentMetrics
from app.turns.sentences import TurkishSentenceBuffer


@dataclass(frozen=True, slots=True)
class SentenceJob:
    """One immutable synthesis unit with its playback sequence."""

    sequence: int
    text: str


_QUEUE_STOP = object()
QueueItem = SentenceJob | object
LeaseRelease = Callable[[], Awaitable[None] | None]
_SAFE_DEVICE = re.compile(r"(?:cpu|mps|cuda(?::\d+)?)")
_SAFE_DTYPES = frozenset(
    {
        "auto",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
    }
)


@dataclass(slots=True)
class TurnContext:
    """Mutable, in-memory state owned by one admitted turn."""

    turn_id: UUID
    input_path: Path
    events: TurnEventBuffer
    source_path: Path | None = None
    recording_duration_seconds: float | None = None
    recording_bytes: int | None = None
    recording_content_type: str | None = None
    upload_ms: float | None = None
    started_ns: int | None = None
    release_lease: LeaseRelease | None = None
    sentence_queue: asyncio.Queue[QueueItem] = field(
        default_factory=lambda: asyncio.Queue(maxsize=4),
        repr=False,
    )
    produced_sequences: set[int] = field(default_factory=set)
    transcript: str = ""
    intent: IntentResult | None = None
    answer: str = ""
    metrics: TurnMetrics | None = None
    first_audio_started_ms: float | None = None


@dataclass(slots=True)
class _Timing:
    stt_ns: int | None = None
    intent_ns: int | None = None
    llm_ns: int | None = None
    tts_ns: int | None = None
    first_sentence_ready_ms: float | None = None
    sentence_count: int = 0
    audio_chunk_count: int = 0
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_tokens_per_second: float | None = None
    published_answer: str = ""


class _StageFailure(Exception):
    def __init__(
        self,
        stage: Stage,
        code: str,
        message: str,
        *,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.timed_out = timed_out


class TurnOrchestrator:
    """Execute one turn while preserving streamed text and ordered audio."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage: AudioStorage,
        stt: STTRuntime,
        llm: LLMRuntime,
        tts: TTSRuntime,
        recent_metrics: RecentMetrics,
        intent: IntentEngine | None = None,
        local_player: LocalAudioPlayer | None = None,
        output: "AudioOutputController | None" = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._recent_metrics = recent_metrics
        self._intent = intent
        self._local_player = local_player
        self._output = output

    def _playback_player(self) -> "LocalAudioPlayer | None":
        """The player for the active output route (controller wins if present)."""
        if self._output is not None:
            return self._output.current_player()
        return self._local_player

    async def run(self, turn: TurnContext) -> TurnContext:
        """Run the pipeline and always leave one terminal event and metric."""
        if turn.sentence_queue.maxsize <= 0:
            raise ValueError("turn sentence queue must be bounded")
        if not turn.sentence_queue.empty():
            raise ValueError("turn sentence queue must start empty")

        if turn.started_ns is None:
            turn.started_ns = time.perf_counter_ns()
        timing = _Timing()
        current_stage = Stage.TRANSCRIBING
        failure: _StageFailure | None = None
        cancelled: asyncio.CancelledError | None = None
        finalization_error: BaseException | None = None
        succeeded = False

        try:
            await turn.events.publish("state", stage=Stage.TRANSCRIBING)
            stt_started_ns = time.perf_counter_ns()
            try:
                turn.transcript = await asyncio.wait_for(
                    self._stt.transcribe(turn.input_path),
                    timeout=self._settings.stt_timeout_seconds,
                )
            finally:
                timing.stt_ns = time.perf_counter_ns() - stt_started_ns

            if not turn.transcript.strip():
                raise _StageFailure(
                    Stage.TRANSCRIBING,
                    "stt_empty",
                    "Konuşma anlaşılamadı, tekrar deneyin.",
                )

            await turn.events.publish("transcript", text=turn.transcript)

            if self._intent is not None:
                current_stage = Stage.CLASSIFYING
                await turn.events.publish("state", stage=Stage.CLASSIFYING)
                await self._classify(turn, timing)

            current_stage = Stage.GENERATING
            await turn.events.publish("state", stage=Stage.GENERATING)
            await self._run_streaming_pipeline(turn, timing)
            succeeded = True
        except asyncio.CancelledError as error:
            cancelled = error
            failure = _StageFailure(
                current_stage,
                "turn_cancelled",
                "İşlem iptal edildi.",
            )
        except _StageFailure as error:
            failure = error
        except Exception as error:
            failure = self._map_error(error, current_stage)
        finally:
            finalizer = asyncio.create_task(
                self._finalize(
                    turn=turn,
                    timing=timing,
                    succeeded=succeeded,
                    failure=failure,
                    current_stage=current_stage,
                )
            )
            finalization_cancellation, finalization_error = (
                await self._await_owned_finalizer(finalizer)
            )
            if cancelled is None:
                cancelled = finalization_cancellation

        if cancelled is not None:
            raise cancelled
        if finalization_error is not None:
            raise finalization_error
        return turn

    async def _finalize(
        self,
        *,
        turn: TurnContext,
        timing: _Timing,
        succeeded: bool,
        failure: _StageFailure | None,
        current_stage: Stage,
    ) -> None:
        try:
            try:
                self._storage.cleanup_terminal(turn.turn_id)
            except Exception:
                if succeeded:
                    succeeded = False
                    failure = _StageFailure(
                        Stage.FAILED,
                        "cleanup_failed",
                        "İşlem tamamlanamadı, tekrar deneyin.",
                    )

            turn.metrics = self._build_metrics(
                turn=turn,
                timing=timing,
                succeeded=succeeded,
                failure=failure,
            )
            self._recent_metrics.append(turn.turn_id, turn.metrics)
            await turn.events.publish("metrics", metrics=turn.metrics)
            if succeeded:
                await turn.events.publish(
                    "complete",
                    transcript=turn.transcript,
                    answer=turn.answer,
                    audio_url=None,
                    metrics=turn.metrics,
                )
            else:
                terminal_failure = failure or _StageFailure(
                    current_stage,
                    "turn_failed",
                    "İşlem tamamlanamadı, tekrar deneyin.",
                )
                await turn.events.publish(
                    "failed",
                    error=ErrorBody(
                        code=terminal_failure.code,
                        message=terminal_failure.message,
                    ),
                    metrics=turn.metrics,
                )
        finally:
            await self._release(turn.release_lease)

    @staticmethod
    async def _await_owned_finalizer(
        finalizer: asyncio.Task[None],
    ) -> tuple[asyncio.CancelledError | None, BaseException | None]:
        cancellation: asyncio.CancelledError | None = None
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception:
                break

        try:
            finalizer.result()
        except BaseException as error:
            return cancellation, error
        return cancellation, None

    async def _run_streaming_pipeline(
        self,
        turn: TurnContext,
        timing: _Timing,
    ) -> None:
        worker = asyncio.create_task(self._tts_worker(turn, timing))
        llm_started_ns = time.perf_counter_ns()
        producer = asyncio.create_task(self._consume_llm(turn, timing))
        producer_failure: Exception | None = None

        try:
            done, _ = await asyncio.wait(
                {producer, worker},
                timeout=self._settings.llm_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
                producer_failure = _StageFailure(
                    Stage.GENERATING,
                    "llm_timeout",
                    "Yanıt üretilemedi, tekrar deneyin.",
                    timed_out=True,
                )
            else:
                if worker.done():
                    await self._raise_worker_result(worker)
                try:
                    producer.result()
                except Exception as error:
                    producer_failure = error

            timing.llm_ns = time.perf_counter_ns() - llm_started_ns
            await self._finish_worker(turn.sentence_queue, worker)
            if producer_failure is not None:
                if isinstance(producer_failure, _StageFailure):
                    raise producer_failure
                raise self._map_error(producer_failure, Stage.GENERATING)
        finally:
            if timing.llm_ns is None:
                timing.llm_ns = time.perf_counter_ns() - llm_started_ns
            for task in (producer, worker):
                if not task.done():
                    task.cancel()
            await asyncio.gather(producer, worker, return_exceptions=True)
            self._discard_pending_jobs(turn.sentence_queue)

    async def _classify(self, turn: TurnContext, timing: _Timing) -> None:
        """Route the transcript to an agent and announce the decision."""
        engine = self._intent
        if engine is None:
            return

        started_ns = time.perf_counter_ns()
        try:
            intent = await engine.classify(turn.transcript)
        finally:
            timing.intent_ns = time.perf_counter_ns() - started_ns

        turn.intent = intent
        await turn.events.publish(
            "intent",
            label=intent.label,
            agent=intent.agent,
            source=intent.source,
            confidence=intent.confidence,
            function_call=intent.function_answer is not None,
        )

    def _agent_prompt(self, intent: IntentResult | None) -> str | None:
        """Return the routed agent's prebuilt prompt, or None for the shared one."""
        if intent is None:
            return None
        return self._settings.agent_system_prompts.get(intent.agent)

    async def _consume_llm(self, turn: TurnContext, timing: _Timing) -> None:
        intent = turn.intent
        if intent is not None and intent.function_answer:
            # The evaluation set marks these as function calls: the answer is
            # already known, and asking a model for the time can only produce a
            # confidently wrong one. Hand it straight to the synthesis queue.
            turn.answer = intent.function_answer
            await self._publish_and_enqueue(
                turn,
                timing,
                SentenceJob(sequence=0, text=intent.function_answer),
            )
            return

        buffer = TurkishSentenceBuffer()
        next_sequence = 0

        async for delta in self._llm.stream_answer(
            turn.transcript,
            self._agent_prompt(intent),
        ):
            if delta.prompt_tokens is not None:
                timing.llm_prompt_tokens = delta.prompt_tokens
            if delta.completion_tokens is not None:
                timing.llm_completion_tokens = delta.completion_tokens
            if delta.tokens_per_second is not None:
                timing.llm_tokens_per_second = delta.tokens_per_second
            if not delta.text:
                continue

            turn.answer += delta.text
            for sentence in buffer.push(delta.text):
                await self._publish_and_enqueue(
                    turn,
                    timing,
                    SentenceJob(sequence=next_sequence, text=sentence),
                )
                next_sequence += 1

        for sentence in buffer.finish():
            await self._publish_and_enqueue(
                turn,
                timing,
                SentenceJob(sequence=next_sequence, text=sentence),
            )
            next_sequence += 1

    async def _publish_and_enqueue(
        self,
        turn: TurnContext,
        timing: _Timing,
        job: SentenceJob,
    ) -> None:
        timing.sentence_count += 1
        timing.published_answer += job.text
        if timing.first_sentence_ready_ms is None:
            timing.first_sentence_ready_ms = self._elapsed_ms(
                turn.started_ns,
                time.perf_counter_ns(),
            )
        await turn.events.publish(
            "answer_delta",
            text=job.text,
            answer=timing.published_answer,
        )
        await turn.sentence_queue.put(job)

    async def _tts_worker(self, turn: TurnContext, timing: _Timing) -> None:
        expected_sequence = 0
        synthesis_state_published = False

        while True:
            item = await turn.sentence_queue.get()
            try:
                if item is _QUEUE_STOP:
                    return
                if not isinstance(item, SentenceJob):
                    raise _StageFailure(
                        Stage.SYNTHESIZING,
                        "tts_failed",
                        "Metin yanıtı hazır, ses üretilemedi.",
                    )
                if item.sequence != expected_sequence:
                    raise _StageFailure(
                        Stage.SYNTHESIZING,
                        "tts_failed",
                        "Metin yanıtı hazır, ses üretilemedi.",
                    )
                if not synthesis_state_published:
                    await turn.events.publish("state", stage=Stage.SYNTHESIZING)
                    synthesis_state_published = True

                output_path = self._storage.chunk_path(turn.turn_id, item.sequence)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tts_started_ns = time.perf_counter_ns()
                try:
                    await asyncio.wait_for(
                        self._tts.synthesize(item.text, output_path),
                        timeout=self._settings.tts_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    output_path.unlink(missing_ok=True)
                    raise
                except Exception as error:
                    output_path.unlink(missing_ok=True)
                    raise self._map_error(error, Stage.SYNTHESIZING)
                finally:
                    elapsed_ns = time.perf_counter_ns() - tts_started_ns
                    timing.tts_ns = (timing.tts_ns or 0) + elapsed_ns

                if not output_path.is_file():
                    raise _StageFailure(
                        Stage.SYNTHESIZING,
                        "tts_failed",
                        "Metin yanıtı hazır, ses üretilemedi.",
                    )
                playback_player = self._playback_player()
                if playback_player is not None:
                    try:
                        await playback_player.play_wav(output_path)
                    except LocalPlaybackError as error:
                        raise _StageFailure(
                            Stage.SYNTHESIZING,
                            "local_playback_failed",
                            "Yerel kulaklıkta ses çalınamadı.",
                        ) from error
                await turn.events.publish(
                    "audio_ready",
                    sequence=item.sequence,
                    text=item.text,
                    audio_url=self._storage.chunk_url(turn.turn_id, item.sequence),
                )
                turn.produced_sequences.add(item.sequence)
                timing.audio_chunk_count += 1
                expected_sequence += 1
            finally:
                turn.sentence_queue.task_done()

    async def _finish_worker(
        self,
        queue: asyncio.Queue[QueueItem],
        worker: asyncio.Task[None],
    ) -> None:
        joined = asyncio.create_task(queue.join())
        try:
            await asyncio.wait(
                {joined, worker},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker.done():
                await self._raise_worker_result(worker)
            await joined
            await queue.put(_QUEUE_STOP)
            await worker
            await queue.join()
        finally:
            if not joined.done():
                joined.cancel()
            await asyncio.gather(joined, return_exceptions=True)

    @staticmethod
    async def _raise_worker_result(worker: asyncio.Task[None]) -> None:
        try:
            worker.result()
        except asyncio.CancelledError:
            raise
        except _StageFailure:
            raise
        except Exception as error:
            raise _StageFailure(
                Stage.SYNTHESIZING,
                "tts_failed",
                "Metin yanıtı hazır, ses üretilemedi.",
            ) from error
        raise _StageFailure(
            Stage.SYNTHESIZING,
            "tts_failed",
            "Metin yanıtı hazır, ses üretilemedi.",
        )

    @staticmethod
    def _discard_pending_jobs(queue: asyncio.Queue[QueueItem]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                queue.task_done()

    def _build_metrics(
        self,
        *,
        turn: TurnContext,
        timing: _Timing,
        succeeded: bool,
        failure: _StageFailure | None,
    ) -> TurnMetrics:
        now_ns = time.perf_counter_ns()
        intent = turn.intent
        stt_device = self._safe_device(self._settings.stt_device)
        tts_device = self._safe_device(self._settings.tts_device)
        stt_dtype = self._safe_dtype(self._settings.stt_dtype)
        tts_dtype = self._safe_dtype(self._settings.tts_dtype)
        return TurnMetrics(
            timestamp=datetime.now(timezone.utc),
            outcome=TurnStatus.COMPLETE if succeeded else TurnStatus.FAILED,
            recording_duration_seconds=turn.recording_duration_seconds,
            recording_bytes=turn.recording_bytes,
            recording_content_type=turn.recording_content_type,
            upload_ms=turn.upload_ms,
            stt_ms=self._to_ms(timing.stt_ns),
            intent_ms=self._to_ms(timing.intent_ns),
            llm_ms=self._to_ms(timing.llm_ns),
            tts_ms=self._to_ms(timing.tts_ns),
            total_ms=self._elapsed_ms(turn.started_ns, now_ns),
            first_sentence_ready_ms=timing.first_sentence_ready_ms,
            first_audio_started_ms=turn.first_audio_started_ms,
            sentence_count=timing.sentence_count,
            audio_chunk_count=timing.audio_chunk_count,
            transcript_chars=len(turn.transcript),
            answer_chars=len(turn.answer),
            llm_prompt_tokens=timing.llm_prompt_tokens,
            llm_completion_tokens=timing.llm_completion_tokens,
            llm_tokens_per_second=timing.llm_tokens_per_second,
            intent_label=intent.label if intent is not None else None,
            intent_agent=intent.agent if intent is not None else None,
            intent_source=intent.source if intent is not None else None,
            intent_confidence=intent.confidence if intent is not None else None,
            error_stage=failure.stage if failure is not None else None,
            error_code=failure.code if failure is not None else None,
            timed_out=failure.timed_out if failure is not None else False,
            device=f"stt={stt_device};tts={tts_device}",
            dtype=f"stt={stt_dtype};tts={tts_dtype}",
            configuration=SafeConfigurationSummary(
                stt_device=stt_device,
                tts_device=tts_device,
                stt_dtype=stt_dtype,
                tts_dtype=tts_dtype,
                llm_max_tokens=self._settings.llm_max_tokens,
                stt_timeout_seconds=self._settings.stt_timeout_seconds,
                llm_timeout_seconds=self._settings.llm_timeout_seconds,
                tts_timeout_seconds=self._settings.tts_timeout_seconds,
                tts_queue_capacity=turn.sentence_queue.maxsize,
            ),
        )

    @staticmethod
    def _map_error(error: Exception, stage: Stage) -> _StageFailure:
        if isinstance(error, _StageFailure):
            return error
        if isinstance(error, ApiError):
            return _StageFailure(
                stage,
                error.code,
                error.message,
                timed_out=error.code.endswith("_timeout"),
            )
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            code = {
                Stage.TRANSCRIBING: "stt_timeout",
                Stage.CLASSIFYING: "intent_timeout",
                Stage.GENERATING: "llm_timeout",
                Stage.SYNTHESIZING: "tts_timeout",
            }.get(stage, "turn_timeout")
            return _StageFailure(
                stage,
                code,
                TurnOrchestrator._safe_message(stage),
                timed_out=True,
            )
        if TurnOrchestrator._is_out_of_memory(error):
            return _StageFailure(
                stage,
                "gpu_out_of_memory",
                TurnOrchestrator._safe_message(stage),
            )
        code = {
            Stage.TRANSCRIBING: "stt_failed",
            Stage.CLASSIFYING: "intent_failed",
            Stage.GENERATING: "llm_failed",
            Stage.SYNTHESIZING: "tts_failed",
        }.get(stage, "turn_failed")
        return _StageFailure(stage, code, TurnOrchestrator._safe_message(stage))

    @staticmethod
    def _safe_message(stage: Stage) -> str:
        if stage is Stage.TRANSCRIBING:
            return "Konuşma anlaşılamadı, tekrar deneyin."
        if stage is Stage.CLASSIFYING:
            return "Soru yönlendirilemedi, tekrar deneyin."
        if stage is Stage.GENERATING:
            return "Yanıt üretilemedi, tekrar deneyin."
        if stage is Stage.SYNTHESIZING:
            return "Metin yanıtı hazır, ses üretilemedi."
        return "İşlem tamamlanamadı, tekrar deneyin."

    @staticmethod
    def _is_out_of_memory(error: Exception) -> bool:
        class_name = error.__class__.__name__.lower()
        return "outofmemory" in class_name or "out of memory" in str(error).lower()

    @staticmethod
    def _safe_device(value: str) -> str:
        normalized = value.strip().lower()
        return normalized if _SAFE_DEVICE.fullmatch(normalized) else "configured"

    @staticmethod
    def _safe_dtype(value: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized in _SAFE_DTYPES else "configured"

    @staticmethod
    async def _release(release: LeaseRelease | None) -> None:
        if release is None:
            return
        result = release()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _to_ms(duration_ns: int | None) -> float | None:
        if duration_ns is None:
            return None
        return max(duration_ns, 0) / 1_000_000

    @staticmethod
    def _elapsed_ms(start_ns: int | None, end_ns: int) -> float | None:
        if start_ns is None:
            return None
        return max(end_ns - start_ns, 0) / 1_000_000
