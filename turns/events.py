"""Replayable, per-turn event delivery for reconnect-safe SSE streams."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.schemas import CompletePayload, EventPayload, FailedPayload, MetricsPayload, StatePayload


EVENT_MODELS: dict[str, type[EventPayload]] = {
    "state": StatePayload,
    "metrics": MetricsPayload,
    "complete": CompletePayload,
    "failed": FailedPayload,
}
TERMINAL_EVENTS = frozenset({"complete", "failed"})


@dataclass(frozen=True, slots=True)
class TurnEvent:
    """One validated event retained for the lifetime of its turn."""

    name: str
    payload: EventPayload

    @property
    def event_id(self) -> int:
        return self.payload.event_id

    def as_sse(self) -> dict[str, str]:
        """Return the mapping accepted by SSE response implementations."""
        return {
            "event": self.name,
            "id": str(self.event_id),
            "data": self.payload.model_dump_json(),
        }


@dataclass(frozen=True, slots=True)
class EventHeartbeat:
    """An SSE comment that probes an otherwise idle connection."""

    comment: str = "heartbeat"

    def as_sse(self) -> dict[str, str]:
        return {"comment": self.comment}


class TurnEventBuffer:
    """Retain ordered events and replay only those after a subscriber cursor."""

    def __init__(
        self,
        turn_id: UUID,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.turn_id = turn_id
        self.heartbeat_seconds = heartbeat_seconds
        self._events: list[TurnEvent] = []
        self._condition = asyncio.Condition()
        self._next_event_id = 1
        self._terminal_event_id: int | None = None

    async def publish(
        self,
        name: str,
        payload: Mapping[str, Any] | BaseModel | None = None,
        **fields: Any,
    ) -> TurnEvent:
        """Validate, assign an ID to, retain, and announce one event."""
        model_type = EVENT_MODELS.get(name)
        if model_type is None:
            raise ValueError(f"unsupported turn event: {name}")

        values: dict[str, Any]
        if payload is None:
            values = {}
        elif isinstance(payload, BaseModel):
            values = payload.model_dump()
        else:
            values = dict(payload)
        values.update(fields)

        async with self._condition:
            if self._terminal_event_id is not None:
                raise RuntimeError("cannot publish after a terminal event")

            event_id = self._next_event_id
            values["turn_id"] = self.turn_id
            values["event_id"] = event_id
            event = TurnEvent(name=name, payload=model_type(**values))
            self._events.append(event)
            self._next_event_id += 1
            if name in TERMINAL_EVENTS:
                self._terminal_event_id = event_id
            self._condition.notify_all()
            return event

    def snapshot(self) -> tuple[TurnEvent, ...]:
        """Return the currently retained events without exposing mutable storage."""
        return tuple(self._events)

    async def subscribe(
        self,
        last_event_id: int | None = None,
    ) -> AsyncIterator[TurnEvent | EventHeartbeat]:
        """Replay events after the cursor, then follow live events until terminal."""
        cursor = max(last_event_id or 0, 0)

        while True:
            ready: tuple[TurnEvent, ...] = ()
            terminal_replayed = False
            heartbeat_due = False

            async with self._condition:
                ready = tuple(
                    event for event in self._events if event.event_id > cursor
                )
                if not ready:
                    terminal_replayed = (
                        self._terminal_event_id is not None
                        and cursor >= self._terminal_event_id
                    )
                    if not terminal_replayed:
                        try:
                            await asyncio.wait_for(
                                self._condition.wait(),
                                timeout=self.heartbeat_seconds,
                            )
                        except asyncio.TimeoutError:
                            heartbeat_due = True

            if terminal_replayed:
                return
            if heartbeat_due:
                yield EventHeartbeat()
                continue
            if not ready:
                continue

            for event in ready:
                cursor = event.event_id
                yield event
