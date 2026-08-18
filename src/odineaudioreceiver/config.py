"""Environment-backed, safe configuration for the voice assistant."""

from functools import cached_property
from pathlib import Path
from typing import Literal
import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings whose model identifiers are intentionally blank by default."""

    model_config = SettingsConfigDict(env_file="audio_config.env", extra="ignore")

    base_directory: str = "audio_recordings"

    odineapa_url: str = "http://127.0.0.1:8000"
    odineapa_auth: tuple[str, str] = (os.getenv("ODINEAPA_USER"), os.getenv("ODINEAPA_PASSWORD"))
    odine_template_id: str = "d4e5f6a7-0000-0000-0000-000000000201"
    
    audio_max_seconds: float = Field(default=30.0, gt=0)
    audio_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    audio_retention_seconds: int = Field(default=600, gt=0)
    cors_origins_raw: str = Field(
        default="http://localhost:5173", validation_alias="CORS_ORIGINS"
    )
    metrics_capacity: int = Field(default=50, gt=0)
    audio_input_mode: Literal["browser", "rf_i2s"] = "browser"
    local_audio_playback: bool = False
    speaker_device: str = "plughw:2,0"
    rf_mic_device: str = "hw:APE,0"
    rf_mic_sample_rate: int = Field(default=8000, gt=0)
    rf_mic_channels: int = Field(default=2, gt=0)
    rf_capture_max_seconds: float = Field(default=30.0, gt=0)
    rf_ptt_frame_timeout_ms: int = Field(default=300, gt=0)
    rf_ptt_min_seconds: float = Field(default=0.30, gt=0)
    ape_card: str = "APE"
    ape_i2s_port: str = "I2S2"
    # GPIO "listening" indicator (Jetson.GPIO). Off by default so laptop/dev and
    # CI never touch hardware. The listening pin is driven HIGH while an
    # utterance is actively being captured; the optional armed pin marks the
    # capture loop as alive.
    gpio_listening_enabled: bool = False
    gpio_listening_pin: int = Field(default=0, ge=0)
    gpio_armed_pin: int = Field(default=0, ge=0)
    gpio_active_high: bool = True
    gpio_mode: Literal["BOARD", "BCM"] = "BOARD"

    @model_validator(mode="after")
    def validate_audio_hardware(self) -> "Settings":
        fields = (
            self.speaker_device,
            self.rf_mic_device,
            self.ape_card,
            self.ape_i2s_port,
        )
        if any(not value.strip() for value in fields):
            raise ValueError("Audio hardware identifiers must not be blank.")
        if self.rf_ptt_min_seconds > self.rf_capture_max_seconds:
            raise ValueError("rf_ptt_min_seconds must not exceed rf_capture_max_seconds.")
        if self.gpio_listening_enabled and self.gpio_listening_pin <= 0:
            raise ValueError(
                "gpio_listening_pin must be set when gpio_listening_enabled is true."
            )
        return self

    @cached_property
    def cors_origins(self) -> list[str]:
        """Return the configured, non-empty CORS origins in declaration order."""
        return [
            origin.strip()
            for origin in self.cors_origins_raw.split(",")
            if origin.strip()
        ]
