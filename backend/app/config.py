"""Environment-backed, safe configuration for the voice assistant."""

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from app.intents.taxonomy import Taxonomy


class Settings(BaseSettings):
    """Runtime settings whose model identifiers are intentionally blank by default."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama_cpp_base_url: str = "http://127.0.0.1:8080"
    llama_cpp_model: str = ""
    stt_backend: Literal["huggingface", "whisper_cpp"] = "huggingface"
    tts_backend: Literal["huggingface", "piper"] = "huggingface"
    stt_model_id: str = ""
    tts_model_id: str = ""
    whisper_cpp_base_url: str = "http://127.0.0.1:8080"
    whisper_cpp_inference_path: str = "/inference"
    piper_binary: str = ""
    piper_model_path: str = ""
    stt_device: str = "cuda"
    tts_device: str = "cuda"
    stt_dtype: str = "float16"
    tts_dtype: str = "float16"
    audio_max_seconds: float = Field(default=30.0, gt=0)
    audio_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    audio_retention_seconds: int = Field(default=600, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    stt_timeout_seconds: float = Field(default=45.0, gt=0)
    tts_timeout_seconds: float = Field(default=45.0, gt=0)
    cors_origins_raw: str = Field(
        default="http://localhost:5173", validation_alias="CORS_ORIGINS"
    )
    metrics_capacity: int = Field(default=50, gt=0)
    turkish_system_prompt: str = "Kısa, açık ve yalnızca Türkçe yanıt ver."
    knowledge_base_enabled: bool = True
    knowledge_base_path: str = ""
    intent_enabled: bool = True
    intent_base_url: str = "http://127.0.0.1:6006"
    intent_classify_path: str = "/classify"
    intent_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    intent_timeout_seconds: float = Field(default=5.0, gt=0)
    # Blank keeps the evaluation set's own guven_esigi; set it only to override.
    intent_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=256, gt=0)
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
        return self

    @cached_property
    def system_prompt(self) -> str:
        """Return the base instruction plus the approved reference answers.

        Cached so the string stays byte-identical across turns; llama-server
        only reuses its prefill cache when the prompt prefix does not change.
        """
        from app.knowledge import build_system_prompt, load_reference_answers

        if not self.knowledge_base_enabled:
            return self.turkish_system_prompt

        return build_system_prompt(
            self.turkish_system_prompt,
            load_reference_answers(self.knowledge_base_file),
        )

    @cached_property
    def agent_system_prompts(self) -> dict[str, str]:
        """Return one constant prompt per routable agent, keyed by agent name.

        Built once for the same reason as `system_prompt`: llama-server only
        reuses a prefill cache when the prompt prefix is byte-identical, and each
        agent needs its own stable prefix. Agents answered by a local function
        are omitted -- their reference answers are placeholders, and folding
        "Şu an saat HH:MM" into a prompt teaches the model to say exactly that.

        An agent collects the answers of *every* label routing to it, not just
        the first. Labels and agents are not 1:1: six labels share the `savaş
        yönergeleri` agent, so stopping at the first one left that prompt
        carrying 9 of its 30 records and the model inventing answers for the
        other 21 -- a silent regression, since the prompt still looked well
        formed.
        """
        from app.knowledge import (
            ReferenceAnswer,
            agent_instruction,
            answers_for_label,
            build_system_prompt,
            load_reference_answers,
        )

        answers = (
            load_reference_answers(self.knowledge_base_file)
            if self.knowledge_base_enabled
            else []
        )

        by_agent: dict[str, list[ReferenceAnswer]] = {}
        for label in self.taxonomy.labels:
            if label.function_call:
                continue
            by_agent.setdefault(label.agent, []).extend(
                answers_for_label(label.name, answers)
            )

        return {
            agent: build_system_prompt(
                agent_instruction(agent, self.turkish_system_prompt),
                agent_answers,
            )
            for agent, agent_answers in by_agent.items()
        }

    @cached_property
    def taxonomy(self) -> "Taxonomy":
        """Return the intent routing table read from the evaluation set."""
        from app.intents.taxonomy import load_taxonomy

        return load_taxonomy(self.knowledge_base_file)

    @property
    def knowledge_base_file(self) -> Path | None:
        """Return the configured evaluation-set override, or None for the shipped one."""
        return Path(self.knowledge_base_path) if self.knowledge_base_path else None

    @cached_property
    def cors_origins(self) -> list[str]:
        """Return the configured, non-empty CORS origins in declaration order."""
        return [
            origin.strip()
            for origin in self.cors_origins_raw.split(",")
            if origin.strip()
        ]

    @property
    def models_configured(self) -> bool:
        """Whether all runtimes have a deliberately configured model identifier."""
        stt_configured = (
            self.stt_model_id.strip()
            if self.stt_backend == "huggingface"
            else self.whisper_cpp_base_url.strip()
        )
        tts_configured = (
            self.tts_model_id.strip()
            if self.tts_backend == "huggingface"
            else self.piper_binary.strip() and self.piper_model_path.strip()
        )
        return bool(self.llama_cpp_model.strip() and stt_configured and tts_configured)
