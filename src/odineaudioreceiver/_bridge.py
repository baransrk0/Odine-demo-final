"""OdineAPA bridge — delovers audio file's name when it's ready."""

from typing import Optional
import httpx

from odineaudioreceiver.config import Settings

WAITING_FOR_AUDIO = 0


async def deliver_audio_to_apa(settings: Optional[Settings], *, audio_path: str) -> str:
    """Notify the latest created instance of FSM, otherwise create a new one.

    audio_path should be accessible by OdineAPA (e.g. './audio/voice.wav')

    Returns: instance_id of the notified instance.
    """
    settings = settings if settings is not None else Settings()
    ODINEAPA_URL = settings.odineapa_url
    AUTH = settings.odineapa_auth
    TEMPLATE_ID = settings.odine_template_id
    inputs = {
        "audio_packet_received": True,  # Phi_01 · f1
        "audio_channel": "K2_VOICE",  # Phi_01 · f2
        "audio_path": audio_path,  # AUDIO_BUFFERED'da gerçek Whisper STT'yi tetikler
    }
    async with httpx.AsyncClient(base_url=ODINEAPA_URL, auth=AUTH, timeout=10.0) as api:
        r = await api.get(
            "/instances/", params={"status": "active", "template_id": TEMPLATE_ID}
        )
        r.raise_for_status()
        waiting = [i for i in r.json() if i["current_state"] == WAITING_FOR_AUDIO]
        if waiting:
            newest = max(waiting, key=lambda i: i["created_at"])
            push = await api.post(
                f"/instances/{newest['id']}/inputs", json={"inputs": inputs}
            )
            if push.status_code not in (
                404,
                409,
            ):  # 404/409: bu arada kapanmış → yenisini aç
                push.raise_for_status()
                return newest["id"]
        launched = await api.post(
            "/instances/",
            json={
                "template_id": TEMPLATE_ID,
                "reference": "instance created upon received audio",
            },
        )
        launched.raise_for_status()
        new_id = launched.json()["id"]
        push = await api.post(f"/instances/{new_id}/inputs", json={"inputs": inputs})
        push.raise_for_status()
        return new_id
