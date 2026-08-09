"""Stable HTTP and SSE payload contracts for audio uploads and forwarding."""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Stage(str, Enum):
    UPLOADING = "uploading"
    NORMALIZING = "normalizing"
    NOTIFYING = "notifying"
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


class AudioInputState(BaseModel):
    """The current audio-input source."""

    mode: Literal["browser", "rf_i2s"]


class AudioInputRequest(BaseModel):
    """Request to switch audio input at runtime."""

    mode: Literal["browser", "rf_i2s"]


class OutputDevice(BaseModel):
    """A selectable ALSA playback target on the device."""

    value: str
    label: str


class AudioOutputState(BaseModel):
    """The current audio-output routing."""

    mode: Literal["browser", "device"]
    device: str


class AudioOutputRequest(BaseModel):
    """Request to switch audio output at runtime."""

    mode: Literal["browser", "device"]
    device: str | None = None


class AudioOutputsResponse(BaseModel):
    """The current routing plus the device's enumerated outputs."""

    current: AudioOutputState
    devices: list[OutputDevice]


class TurnMetrics(BaseModel):
    timestamp: datetime | None = None
    outcome: TurnStatus | None = None
    recording_duration_seconds: float | None = None
    recording_bytes: int | None = None
    recording_content_type: str | None = None
    upload_ms: float | None = None
    normalize_ms: float | None = None
    notify_ms: float | None = None
    total_ms: float | None = None
    error_stage: Stage | None = None
    error_code: str | None = None
    timed_out: bool = False
    device: str | None = None
    dtype: str | None = None


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


class MetricsPayload(EventPayload):
    metrics: TurnMetrics


class CompletePayload(EventPayload):
    metrics: TurnMetrics


class FailedPayload(EventPayload):
    error: ErrorBody
    metrics: TurnMetrics | None = None


class PlaybackReport(BaseModel):
    sequence: int = Field(ge=0)
    client_offset_ms: float = Field(ge=0)
