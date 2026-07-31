"""Select configured speech runtimes without importing optional HF dependencies."""

from app.config import Settings
from app.intents.engine import IntentEngine
from app.intents.taxonomy import VERBALIZATIONS
from app.runtimes.intent import ZeroShotIntentClassifier
from app.runtimes.piper import PiperTTS
from app.runtimes.protocols import STTRuntime, TTSRuntime
from app.runtimes.whisper_cpp import WhisperCppSTT


def build_stt_runtime(settings: Settings) -> STTRuntime:
    """Create the configured STT adapter with lazy HF imports."""
    if settings.stt_backend == "whisper_cpp":
        return WhisperCppSTT(
            settings.whisper_cpp_base_url,
            inference_path=settings.whisper_cpp_inference_path,
            timeout_seconds=settings.stt_timeout_seconds,
        )

    from app.runtimes.stt import HuggingFaceSTT

    return HuggingFaceSTT(
        settings.stt_model_id,
        device=settings.stt_device,
        dtype=settings.stt_dtype,
    )


def build_tts_runtime(settings: Settings) -> TTSRuntime:
    """Create the configured TTS adapter with lazy HF imports."""
    if settings.tts_backend == "piper":
        return PiperTTS(
            binary=settings.piper_binary,
            model_path=settings.piper_model_path,
        )

    from app.runtimes.tts import HuggingFaceTTS

    return HuggingFaceTTS(
        settings.tts_model_id,
        device=settings.tts_device,
        dtype=settings.tts_dtype,
    )


def build_intent_classifier(settings: Settings) -> ZeroShotIntentClassifier | None:
    """Create the zero-shot classifier client, or None when routing is disabled."""
    if not settings.intent_enabled:
        return None

    return ZeroShotIntentClassifier(
        settings.intent_base_url,
        classify_path=settings.intent_classify_path,
        model=settings.intent_model,
        verbalizations={
            label.name: label.verbalization for label in settings.taxonomy.labels
        }
        or VERBALIZATIONS,
        timeout_seconds=settings.intent_timeout_seconds,
    )


def build_intent_engine(
    settings: Settings,
    classifier: object | None = None,
) -> IntentEngine | None:
    """Create the routing engine, or None when intent routing is disabled.

    The engine is built even when the classifier is unreachable: the clock rules
    still answer without it, and everything else degrades to the default agent.
    """
    if not settings.intent_enabled:
        return None

    return IntentEngine(
        taxonomy=settings.taxonomy,
        classifier=classifier if classifier is not None else build_intent_classifier(settings),
        timeout_seconds=settings.intent_timeout_seconds,
        threshold=settings.intent_confidence_threshold,
    )


def build_speech_runtimes(settings: Settings) -> tuple[STTRuntime, TTSRuntime]:
    """Create both configured speech adapters."""
    return build_stt_runtime(settings), build_tts_runtime(settings)
