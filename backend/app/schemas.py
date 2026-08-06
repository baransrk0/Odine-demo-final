"""Stable HTTP and SSE payload contracts for voice-assistant turns."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class Stage(str, Enum):
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    CLASSIFYING = "classifying"
    GENERATING = "generating"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    COMPLETE = "complete"
    FAILED = "failed"


class TurnStatus(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class ErrorBody(BaseModel):
    code: str
    message: str


class ListeningPayload(BaseModel):
    """SSE payload announcing whether the RF capture is actively listening."""

    event_id: int = Field(ge=1)
    listening: bool


class SafeConfigurationSummary(BaseModel):
    stt_device: str
    tts_device: str
    stt_dtype: str
    tts_dtype: str
    llm_max_tokens: int
    stt_timeout_seconds: float
    llm_timeout_seconds: float
    tts_timeout_seconds: float
    tts_queue_capacity: int


class TurnMetrics(BaseModel):
    timestamp: datetime | None = None
    outcome: TurnStatus | None = None
    recording_duration_seconds: float | None = None
    recording_bytes: int | None = None
    recording_content_type: str | None = None
    upload_ms: float | None = None
    stt_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    total_ms: float | None = None
    first_sentence_ready_ms: float | None = None
    first_audio_started_ms: float | None = None
    sentence_count: int | None = None
    audio_chunk_count: int | None = None
    transcript_chars: int | None = None
    answer_chars: int | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_tokens_per_second: float | None = None
    intent_label: str | None = None
    intent_agent: str | None = None
    intent_source: str | None = None
    intent_confidence: float | None = None
    intent_ms: float | None = None
    error_stage: Stage | None = None
    error_code: str | None = None
    timed_out: bool = False
    device: str | None = None
    dtype: str | None = None
    configuration: SafeConfigurationSummary | None = None


class TurnCreated(BaseModel):
    turn_id: UUID
    events_url: str


class RFDiscoveryPayload(TurnCreated):
    event_id: int = Field(ge=1)


class EventPayload(BaseModel):
    turn_id: UUID
    event_id: int = Field(ge=1)


class StatePayload(EventPayload):
    stage: Stage


class TranscriptPayload(EventPayload):
    text: str


class IntentPayload(EventPayload):
    label: str
    agent: str
    source: str
    confidence: float | None = None
    function_call: bool = False


class AnswerDeltaPayload(EventPayload):
    text: str
    answer: str


class AudioReadyPayload(EventPayload):
    sequence: int = Field(ge=0)
    text: str
    audio_url: str


class MetricsPayload(EventPayload):
    metrics: TurnMetrics


class CompletePayload(EventPayload):
    transcript: str
    answer: str
    audio_url: str | None = None
    metrics: TurnMetrics


class FailedPayload(EventPayload):
    error: ErrorBody
    metrics: TurnMetrics | None = None


class PlaybackReport(BaseModel):
    sequence: int = Field(ge=0)
    client_offset_ms: float = Field(ge=0)
