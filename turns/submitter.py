"""Shared admission of a normalized WAV plus external orchestrator notification."""

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

from app.audio.storage import AudioStorage
from app.config import Settings
from app.errors import ApiError
from app.schemas import Stage, TurnCreated, TurnMetrics, TurnStatus
from app.turns.events import TurnEventBuffer
from app.turns.manager import TurnManager
from app.turns.orchestrator import TurnContext
from app.turns.metrics import RecentMetrics


_ORCHESTRATOR_TIMEOUT_SECONDS = 5.0


class TurnSubmitter:
    def __init__(
        self,
        storage: AudioStorage,
        manager: TurnManager,
        recent_metrics: RecentMetrics,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._manager = manager
        self._recent_metrics = recent_metrics
        self._settings = settings

    async def submit_normalized_wav(
        self,
        turn_id: UUID,
        input_path: Path,
        *,
        source_path: Path | None,
        duration_seconds: float,
        byte_count: int,
        content_type: str | None,
        normalize_ms: float,
        started_ns: int | None = None,
    ) -> TurnCreated:
        started = started_ns or time.perf_counter_ns()
        turn = TurnContext(
            turn_id=turn_id,
            input_path=input_path,
            source_path=source_path,
            events=TurnEventBuffer(turn_id),
            recording_duration_seconds=duration_seconds,
            recording_bytes=byte_count,
            recording_content_type=content_type,
            upload_ms=max(time.perf_counter_ns() - started, 0) / 1_000_000,
            normalize_ms=normalize_ms,
            started_ns=started,
        )
        if not await self._manager.try_create(turn):
            raise ApiError("turn_in_progress", "Mevcut yanıt tamamlanıyor.", 409)
        self._manager.start(turn, self._run_turn(turn))
        return TurnCreated(turn_id=turn_id, events_url=f"/api/turns/{turn_id}/events")

    async def wait_until_idle(self) -> None:
        await self._manager.wait_until_idle()

    async def _run_turn(self, turn: TurnContext) -> TurnContext:
        notify_started_ns = time.perf_counter_ns()
        await turn.events.publish("state", stage=Stage.NOTIFYING)
        await self._notify_orchestrator(turn)
        turn.notify_ms = max(time.perf_counter_ns() - notify_started_ns, 0) / 1_000_000

        total_ms = (
            max(time.perf_counter_ns() - turn.started_ns, 0) / 1_000_000
            if turn.started_ns is not None
            else None
        )
        metrics = TurnMetrics(
            timestamp=datetime.now(timezone.utc),
            outcome=TurnStatus.COMPLETE,
            recording_duration_seconds=turn.recording_duration_seconds,
            recording_bytes=turn.recording_bytes,
            recording_content_type=turn.recording_content_type,
            upload_ms=turn.upload_ms,
            normalize_ms=turn.normalize_ms,
            notify_ms=turn.notify_ms,
            total_ms=total_ms,
            device=self._settings.speaker_device if self._settings.local_audio_playback else None,
            dtype=None,
        )
        turn.metrics = metrics
        self._recent_metrics.append(turn.turn_id, metrics)
        await turn.events.publish("metrics", metrics=metrics)
        await turn.events.publish("complete", metrics=metrics)
        return turn

    async def _notify_orchestrator(self, turn: TurnContext) -> None:
        base_url = self._settings.orchestrator_base_url.strip()
        if not base_url:
            return

        audio_name = (
            turn.source_path.name
            if turn.source_path is not None
            else turn.input_path.name
        )
        try:
            async with httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(_ORCHESTRATOR_TIMEOUT_SECONDS),
            ) as client:
                response = await client.get("/instances")
                response.raise_for_status()
                instance_ids = self._parse_instance_ids(response.json())
                if not instance_ids:
                    create_response = await client.post(
                        "/instances",
                        json={
                            "template_id": await self._fetch_default_template_id(),
                            "reference": "audio-upload",
                        },
                    )
                    create_response.raise_for_status()
                    instance_ids.extend(self._parse_instance_ids(create_response.json()))
                for instance_id in instance_ids:
                    await client.post(
                        f"/instances/{instance_id}/inputs",
                        json={"audio_path_field": audio_name},
                    )
        except Exception:
            return

    async def _fetch_default_template_id(self) -> str:
        cached = getattr(self, "default_template_id", None)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(
            base_url=self._settings.orchestrator_base_url.strip().rstrip("/"),
            timeout=httpx.Timeout(_ORCHESTRATOR_TIMEOUT_SECONDS),
        ) as client:
            response = await client.get("/templates")
            response.raise_for_status()
            for template_dict in response.json():
                if template_dict.get("name") == self._settings.default_template_name:
                    template_id = str(template_dict["id"])
                    self.default_template_id = template_id
                    return template_id
            raise RuntimeError(
                f"Could not find template with name '{self._settings.default_template_name}' in orchestrator templates"
            )

    @staticmethod
    def _parse_instance_ids(payload: object) -> list[str]:
        if isinstance(payload, dict):
            instances = payload.get("instances")
            if instances is None:
                instances = [payload]
        else:
            instances = payload

        if not isinstance(instances, list):
            return []

        ids: list[str] = []
        for instance in instances:
            if isinstance(instance, dict):
                value = instance.get("instance_id", instance.get("id"))
            else:
                value = instance
            if value is not None:
                ids.append(str(value))
        return ids
