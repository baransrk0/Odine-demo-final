"""Shared admission of a normalized WAV into the existing turn pipeline."""

import time
from pathlib import Path
from uuid import UUID

import httpx

from app.audio.storage import AudioStorage
from app.config import Settings
from app.errors import ApiError
from app.schemas import TurnCreated
from app.turns.events import TurnEventBuffer
from app.turns.manager import TurnManager
from app.turns.orchestrator import TurnContext, TurnOrchestrator


_ORCHESTRATOR_TIMEOUT_SECONDS = 5.0


class TurnSubmitter:
    def __init__(
        self,
        storage: AudioStorage,
        manager: TurnManager,
        orchestrator: TurnOrchestrator,
    ) -> None:
        self._settings = Settings()
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
        await self._notify_orchestrator(turn)
        return TurnCreated(turn_id=turn_id, events_url=f"/api/turns/{turn_id}/events")

    async def wait_until_idle(self) -> None:
        await self._manager.wait_until_idle()

    async def _fetch_default_template_id(self) -> str:
        """
        Fetches template_id for ABTK-NG from the orchestrator.
        * It will be invoked if voice is received and there's no running instance.
        """
        if (tid := getattr(self, "default_template_id", None)) is not None:
            return tid
        async with httpx.AsyncClient(
            base_url=self._settings.orchestrator_base_url.strip().rstrip("/"),
            timeout=httpx.Timeout(_ORCHESTRATOR_TIMEOUT_SECONDS),
        ) as client:
            response = await client.get("/templates")
            response.raise_for_status()
            default_template_id = None
            for template_dict in response.json():
                if template_dict["name"] == self._settings.default_template_name:
                    default_template_id = template_dict["id"]
                    break
            if default_template_id is None:
                raise RuntimeError(f"Could not find template with name '{self._settings.default_template_name}' in orchestrator templates")
            self.default_template_id = default_template_id
            return default_template_id

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
                    # Launch a new instance.
                    create_response = await client.post(
                        "/instances",
                        json={
                            "template_id": await self._fetch_default_template_id(),
                            "reference": "stt-invoked"
                        }
                    )
                    create_response.raise_for_status()
                    # Add newly created instance's ID to the list.
                    instance_ids.extend(self._parse_instance_ids(create_response.json()))
                # Notify rach running instance individually.
                for instance_id in instance_ids:
                    await client.post(
                        f"/instances/{instance_id}/inputs",
                        json={"audio_path_field": audio_name},
                    )
        except Exception:
            return

    @staticmethod
    def _parse_instance_ids(payload: object) -> list[str]:
        if isinstance(payload, dict):
            instances = payload.get("instances")
            if instances is None:
                instances = [payload]   # Assume it's a single object.
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
