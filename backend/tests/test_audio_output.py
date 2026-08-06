"""Runtime audio-output routing without ALSA hardware."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.audio.output import AudioOutputController, list_output_devices
from app.audio.storage import AudioStorage
from app.config import Settings
from app.main import create_app


_APLAY_L = """**** List of PLAYBACK Hardware Devices ****
card 0: APE [NVIDIA Jetson APE], device 0: I2S2 []
card 2: Headphones [USB Headphones], device 0: USB Audio [USB Audio]
"""


def test_list_output_devices_parses_aplay(monkeypatch):
    def runner(command, **kwargs):
        assert command == ["aplay", "-l"]
        return SimpleNamespace(returncode=0, stdout=_APLAY_L, stderr="")

    devices = list_output_devices(runner=runner)

    values = [device.value for device in devices]
    assert "plughw:2,0" in values
    usb = next(device for device in devices if device.value == "plughw:2,0")
    assert "USB Headphones" in usb.label


def test_list_output_devices_survives_missing_aplay():
    def runner(command, **kwargs):
        raise OSError("aplay not found")

    assert list_output_devices(runner=runner) == []


def test_controller_switches_between_browser_and_device():
    built: list[str] = []
    controller = AudioOutputController(
        mode="browser",
        device="plughw:2,0",
        player_factory=lambda device: built.append(device) or object(),
    )

    assert controller.current_player() is None

    controller.configure(mode="device")
    assert controller.current_player() is not None
    assert built == ["plughw:2,0"]

    controller.configure(device="plughw:3,0")
    assert built[-1] == "plughw:3,0"

    controller.configure(mode="browser")
    assert controller.current_player() is None


def _speech():
    class _Speech:
        ready = True

        async def transcribe(self, path: Path) -> str:
            return "Merhaba"

        async def synthesize(self, text: str, output_path: Path) -> None:
            output_path.write_bytes(b"RIFF")

    return _Speech()


class _LLM:
    ready = True

    async def health(self) -> bool:
        return True

    async def stream_answer(self, transcript, system_prompt=None):
        if False:
            yield None


def _app(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        audio_input_mode="browser",
        stt_model_id="local-stt",
        tts_model_id="local-tts",
        llama_cpp_model="local-llm",
        intent_enabled=False,
    )
    return create_app(
        settings=settings,
        storage=AudioStorage(tmp_path / "audio", retention_seconds=60),
        stt=_speech(),
        tts=_speech(),
        llm=_LLM(),
        validate_audio_tools=lambda: None,
    )


def test_output_endpoints_report_and_switch_live(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "app.main.list_output_devices",
        lambda: [],
    )
    app = _app(tmp_path)

    with TestClient(app) as client:
        listed = client.get("/api/audio/outputs")
        assert listed.status_code == 200
        assert listed.json()["current"] == {
            "mode": "browser",
            "device": "plughw:2,0",
        }

        switched = client.post(
            "/api/audio/output",
            json={"mode": "device", "device": "plughw:2,0"},
        )
        assert switched.status_code == 200
        assert switched.json() == {"mode": "device", "device": "plughw:2,0"}

        health = client.get("/api/health").json()
        assert health["audio_output_mode"] == "device"
        assert health["audio_output_device"] == "plughw:2,0"

        back = client.post("/api/audio/output", json={"mode": "browser"})
        assert back.json()["mode"] == "browser"


def test_output_endpoint_rejects_blank_device(tmp_path: Path):
    app = _app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/audio/output",
            json={"mode": "device", "device": "   "},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
