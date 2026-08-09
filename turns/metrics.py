"""Bounded, anonymous, process-local turn metrics."""

from collections import deque
from dataclasses import dataclass
from uuid import UUID

from app.config import Settings
from app.schemas import TurnMetrics


@dataclass(slots=True)
class _MetricEntry:
    turn_id: UUID
    metrics: TurnMetrics


class RecentMetrics:
    """Keep only upload and forwarding summaries for the most recent turns."""

    def __init__(self, settings: Settings | int) -> None:
        capacity = (
            settings.metrics_capacity if isinstance(settings, Settings) else settings
        )
        if capacity <= 0:
            raise ValueError("metrics capacity must be positive")
        self._entries: deque[_MetricEntry] = deque(maxlen=capacity)

    def append(self, turn_id: UUID, metrics: TurnMetrics) -> None:
        """Store a defensive copy without transcript or answer content."""
        self._entries.append(
            _MetricEntry(turn_id=turn_id, metrics=metrics.model_copy(deep=True))
        )

    def snapshot(self) -> list[TurnMetrics]:
        """Return independent records in oldest-to-newest order."""
        return [entry.metrics.model_copy(deep=True) for entry in self._entries]
