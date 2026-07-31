from app.config import Settings
from pydantic import ValidationError
import pytest


def test_model_ids_default_to_empty(monkeypatch):
    """Missing model configuration must keep runtimes safely unconfigured."""
    for key in ("STT_MODEL_ID", "TTS_MODEL_ID", "LLAMA_CPP_MODEL"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.stt_model_id == ""
    assert settings.tts_model_id == ""
    assert settings.llama_cpp_model == ""
    assert settings.models_configured is False


def test_origins_are_explicit():
    """Comma-separated allowed origins are exposed as distinct CORS origins."""
    settings = Settings(
        _env_file=None,
        CORS_ORIGINS="http://localhost:5173,http://orin.local:8000",
    )

    assert settings.cors_origins == [
        "http://localhost:5173",
        "http://orin.local:8000",
    ]


def test_rf_i2s_defaults_are_explicit(monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_MODE", "rf_i2s")
    settings = Settings(_env_file=None)

    assert settings.audio_input_mode == "rf_i2s"
    assert settings.rf_mic_device == "hw:APE,0"
    assert settings.rf_mic_sample_rate == 8000
    assert settings.rf_mic_channels == 2


def test_rf_minimum_duration_cannot_exceed_capture_limit(monkeypatch):
    monkeypatch.setenv("RF_PTT_MIN_SECONDS", "31")
    monkeypatch.setenv("RF_CAPTURE_MAX_SECONDS", "30")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
