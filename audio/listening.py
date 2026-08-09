"""Live broadcast of RF capture "listening" state to connected UIs.

The physical GPIO pin and this broadcaster are two consumers of the same signal
-- the moment ``capture_ptt_pcm`` starts receiving audio -- so the UI indicator
and the hardware line can never disagree. The browser is not on the Orin, so it
learns listening state here over SSE rather than by reading the pin.

Retains only the latest state (like ``RFDiscoveryBuffer``): a reconnecting
subscriber gets the current value, not a replay of every toggle.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.schemas import ListeningPayload


@dataclass(frozen=True, slots=True)
class ListeningEvent:
    payload: ListeningPayload

    @property
    def event_id(self) -> int:
        return self.payload.event_id

    def as_sse(self) -> dict[str, str]:
        return {
            "event": "listening",
            "id": str(self.event_id),
            "data": self.payload.model_dump_json(),
        }


@dataclass(frozen=True, slots=True)
class ListeningHeartbeat:
    comment: str = "heartbeat"

    def as_sse(self) -> dict[str, str]:
        return {"comment": self.comment}


class ListeningBroadcaster:
    """Fan out the latest listening state to every SSE subscriber."""

    def __init__(self, heartbeat_seconds: float = 15.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.heartbeat_seconds = heartbeat_seconds
        self._condition = asyncio.Condition()
        self._latest: ListeningEvent | None = None
        self._state = False
        self._next_event_id = 1

    async def publish(self, listening: bool) -> ListeningEvent | None:
        """Announce a state change. Repeated identical states are ignored."""
        async with self._condition:
            if self._latest is not None and self._state == listening:
                return None
            event = ListeningEvent(
                ListeningPayload(
                    event_id=self._next_event_id,
                    listening=listening,
                )
            )
            self._state = listening
            self._latest = event
            self._next_event_id += 1
            self._condition.notify_all()
            return event

    async def subscribe(
        self,
        last_event_id: int | None = None,
    ) -> AsyncIterator[ListeningEvent | ListeningHeartbeat]:
        cursor = max(last_event_id or 0, 0)

        while True:
            event: ListeningEvent | None = None
            heartbeat_due = False
            async with self._condition:
                if self._latest is not None and self._latest.event_id > cursor:
                    event = self._latest
                else:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=self.heartbeat_seconds,
                        )
                    except asyncio.TimeoutError:
                        heartbeat_due = True

            if heartbeat_due:
                yield ListeningHeartbeat()
                continue
            if event is None:
                continue
            cursor = event.event_id
            yield event
