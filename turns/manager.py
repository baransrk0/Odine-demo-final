"""Atomic admission and application-owned lifecycle for voice turns."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
import inspect
import time
from uuid import UUID

from app.turns.orchestrator import LeaseRelease, TurnContext


@dataclass(slots=True)
class _ManagedTurn:
    turn: TurnContext
    terminal_at: float | None = None


class TurnManager:
    """Own one active turn while retaining terminal state for SSE replay."""

    def __init__(
        self,
        retention_seconds: int = 600,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self._retention_seconds = retention_seconds
        self._lock = asyncio.Lock()
        self._active_turn_id: UUID | None = None
        self._turns: dict[UUID, _ManagedTurn] = {}
        self._tasks: set[asyncio.Task[TurnContext | None]] = set()

    async def try_create(self, turn: TurnContext) -> bool:
        """Atomically acquire the single active-turn lease."""
        async with self._lock:
            if self._active_turn_id is not None or turn.turn_id in self._turns:
                return False

            previous_release = turn.release_lease
            turn.release_lease = self._release_callback(
                turn.turn_id,
                previous_release,
            )
            self._active_turn_id = turn.turn_id
            self._turns[turn.turn_id] = _ManagedTurn(turn=turn)
            return True

    def start(
        self,
        turn: TurnContext,
        execution: Awaitable[TurnContext],
    ) -> asyncio.Task[TurnContext | None]:
        """Start one admitted execution as an application-owned task."""
        managed = self._turns.get(turn.turn_id)
        if managed is None or managed.turn is not turn:
            if inspect.iscoroutine(execution):
                execution.close()
            raise RuntimeError("turn must be admitted before it can start")

        task = asyncio.create_task(
            self._run_owned(turn.turn_id, execution),
            name=f"voice-turn-{turn.turn_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def get(self, turn_id: UUID) -> TurnContext | None:
        """Return retained turn state without exposing manager bookkeeping."""
        managed = self._turns.get(turn_id)
        return managed.turn if managed is not None else None

    async def expire(self, now: float | None = None) -> None:
        """Drop terminal event buffers once their retention window has elapsed."""
        current = time.monotonic() if now is None else now
        async with self._lock:
            expired = [
                turn_id
                for turn_id, managed in self._turns.items()
                if managed.terminal_at is not None
                and managed.terminal_at + self._retention_seconds <= current
            ]
            for turn_id in expired:
                self._turns.pop(turn_id, None)

    async def wait_until_idle(self) -> None:
        """Wait for the single active-turn lease to become available."""
        while True:
            async with self._lock:
                if self._active_turn_id is None:
                    return
            await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        """Cancel and await every application-owned turn before process exit."""
        tasks = tuple(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel("application shutdown")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _release_callback(
        self,
        turn_id: UUID,
        previous_release: LeaseRelease | None,
    ) -> LeaseRelease:
        async def release() -> None:
            await self._release(turn_id)
            if previous_release is None:
                return
            result = previous_release()
            if inspect.isawaitable(result):
                await result

        return release

    async def _release(self, turn_id: UUID) -> None:
        async with self._lock:
            if self._active_turn_id == turn_id:
                self._active_turn_id = None

    async def _run_owned(
        self,
        turn_id: UUID,
        execution: Awaitable[TurnContext],
    ) -> TurnContext | None:
        try:
            return await execution
        finally:
            async with self._lock:
                managed = self._turns.get(turn_id)
                if managed is not None:
                    managed.terminal_at = time.monotonic()
                if self._active_turn_id == turn_id:
                    self._active_turn_id = None

    def _task_finished(self, task: asyncio.Task[TurnContext | None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        task.exception()
