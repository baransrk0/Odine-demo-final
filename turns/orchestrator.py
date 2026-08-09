"""Minimal turn context used for upload admission and event retention."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.schemas import TurnMetrics
from app.turns.events import TurnEventBuffer


LeaseRelease = Callable[[], Awaitable[None] | None]


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
    normalize_ms: float | None = None
    notify_ms: float | None = None
    started_ns: int | None = None
    release_lease: LeaseRelease | None = None
    metrics: TurnMetrics | None = None
