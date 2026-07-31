"""Shared admission of a normalized WAV into the existing turn pipeline."""

import time
from pathlib import Path
from uuid import UUID

from app.audio.storage import AudioStorage
from app.errors import ApiError
from app.schemas import TurnCreated
from app.turns.events import TurnEventBuffer
from app.turns.manager import TurnManager
from app.turns.orchestrator import TurnContext, TurnOrchestrator


class TurnSubmitter:
    def __init__(self, storage: AudioStorage, manager: TurnManager, orchestrator: TurnOrchestrator) -> None:
        self._storage = storage
        self._manager = manager
        self._orchestrator = orchestrator

    async def submit_normalized_wav(
        self, turn_id: UUID, input_path: Path, *, source_path: Path | None,
        duration_seconds: float, byte_count: int, content_type: str | None,
        started_ns: int | None = None,
    ) -> TurnCreated:
        started = started_ns or time.perf_counter_ns()
        turn = TurnContext(
            turn_id=turn_id, input_path=input_path, source_path=source_path,
            events=TurnEventBuffer(turn_id), recording_duration_seconds=duration_seconds,
            recording_bytes=byte_count, recording_content_type=content_type,
            upload_ms=max(time.perf_counter_ns() - started, 0) / 1_000_000,
            started_ns=started,
        )
        if not await self._manager.try_create(turn):
            raise ApiError("turn_in_progress", "Mevcut yanıt tamamlanıyor.", 409)
        self._manager.start(turn, self._orchestrator.run(turn))
        return TurnCreated(turn_id=turn_id, events_url=f"/api/turns/{turn_id}/events")

    async def wait_until_idle(self) -> None:
        await self._manager.wait_until_idle()
