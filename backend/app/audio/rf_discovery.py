"""Replay-safe discovery of the latest RF-created voice turn."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
import time

from app.schemas import RFDiscoveryPayload, TurnCreated


@dataclass(frozen=True, slots=True)
class RFDiscoveryEvent:
    payload: RFDiscoveryPayload

    @property
    def event_id(self) -> int:
        return self.payload.event_id

    def as_sse(self) -> dict[str, str]:
        return {
            "event": "turn_created",
            "id": str(self.event_id),
            "data": self.payload.model_dump_json(),
        }


@dataclass(frozen=True, slots=True)
class RFDiscoveryHeartbeat:
    comment: str = "heartbeat"

    def as_sse(self) -> dict[str, str]:
        return {"comment": self.comment}


class RFDiscoveryBuffer:
    """Retain only the latest fresh RF turn announcement."""

    def __init__(
        self,
        retention_seconds: float,
        heartbeat_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.retention_seconds = retention_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        self._condition = asyncio.Condition()
        self._latest: tuple[RFDiscoveryEvent, float] | None = None
        self._next_event_id = 1

    async def publish(self, created: TurnCreated) -> RFDiscoveryEvent:
        async with self._condition:
            event = RFDiscoveryEvent(
                RFDiscoveryPayload(
                    event_id=self._next_event_id,
                    turn_id=created.turn_id,
                    events_url=created.events_url,
                )
            )
            self._latest = (event, self._clock())
            self._next_event_id += 1
            self._condition.notify_all()
            return event

    async def subscribe(
        self,
        last_event_id: int | None = None,
    ) -> AsyncIterator[RFDiscoveryEvent | RFDiscoveryHeartbeat]:
        cursor = max(last_event_id or 0, 0)

        while True:
            event: RFDiscoveryEvent | None = None
            heartbeat_due = False
            async with self._condition:
                event = self._fresh_latest_after(cursor)
                if event is None:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=self.heartbeat_seconds,
                        )
                    except asyncio.TimeoutError:
                        heartbeat_due = True

            if heartbeat_due:
                yield RFDiscoveryHeartbeat()
                continue
            if event is None:
                continue
            cursor = event.event_id
            yield event

    def _fresh_latest_after(
        self,
        cursor: int,
    ) -> RFDiscoveryEvent | None:
        if self._latest is None:
            return None
        event, published_at = self._latest
        if self._clock() - published_at >= self.retention_seconds:
            return None
        if event.event_id <= cursor:
            return None
        return event
