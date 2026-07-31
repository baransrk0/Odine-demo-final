"""Safe OpenAI-compatible streaming client for llama.cpp."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.errors import ApiError
from app.runtimes.protocols import LLMDelta

logger = logging.getLogger(__name__)

_UNSUPPORTED_HEALTH_STATUSES = frozenset({404, 405, 501})


class LlamaCppClient:
    """Stream chat completions without exposing request or response content."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.llama_cpp_base_url.rstrip("/")
        self._host_label = urlsplit(settings.llama_cpp_base_url).hostname or "configured"
        self.ready = False

    async def stream_answer(
        self,
        transcript: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """Yield validated content and telemetry frames from llama.cpp.

        `system_prompt` carries the routed agent's constant prompt. It must be
        one of the prebuilt strings, never assembled per turn: llama-server keys
        its prefill cache on the prompt prefix.
        """
        started_ns = time.perf_counter_ns()
        status: int | str = "unavailable"
        request = {
            "model": self._settings.llama_cpp_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or self._settings.system_prompt,
                },
                {"role": "user", "content": transcript},
            ],
            "stream": True,
            "temperature": 0.2,
            "max_tokens": self._settings.llm_max_tokens,
        }

        try:
            timeout = httpx.Timeout(self._settings.llm_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/v1/chat/completions",
                    json=request,
                ) as response:
                    status = response.status_code
                    if response.status_code != 200:
                        raise ApiError(
                            "llm_unavailable",
                            "Dil modeli şu anda kullanılamıyor.",
                            503,
                        )

                    emitted = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        if not data:
                            continue

                        try:
                            payload = json.loads(data)
                            delta = _parse_delta(payload)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            raise ApiError(
                                "llm_invalid_response",
                                "Dil modeli geçersiz bir yanıt döndürdü.",
                                502,
                            ) from None

                        if delta is not None:
                            emitted = True
                            yield delta

                    if not emitted:
                        raise ApiError(
                            "llm_empty_response",
                            "Dil modeli boş bir yanıt döndürdü.",
                            502,
                        )
        except ApiError:
            raise
        except httpx.TimeoutException:
            status = "timeout"
            raise ApiError(
                "llm_timeout",
                "Dil modeli yanıt süresini aştı.",
                504,
            ) from None
        except httpx.RequestError:
            raise ApiError(
                "llm_unavailable",
                "Dil modeli şu anda kullanılamıyor.",
                503,
            ) from None
        finally:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            logger.info(
                "llm_stream host=%s status=%s elapsed_ms=%.2f",
                self._host_label,
                status,
                elapsed_ms,
            )

    async def health(self) -> bool:
        """Probe llama.cpp readiness without returning configuration details."""
        try:
            healthy = await asyncio.wait_for(self._probe_health(), timeout=2.0)
        except (httpx.RequestError, asyncio.TimeoutError):
            healthy = False

        self.ready = healthy
        return healthy

    async def _probe_health(self) -> bool:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            response = await client.get(f"{self._base_url}/health")
            if 200 <= response.status_code < 300:
                return True
            if response.status_code in _UNSUPPORTED_HEALTH_STATUSES:
                fallback = await client.get(f"{self._base_url}/v1/models")
                return 200 <= fallback.status_code < 300
            return False


def _parse_delta(payload: Any) -> LLMDelta | None:
    if not isinstance(payload, dict):
        raise TypeError("SSE payload must be an object")

    text: str | None = None
    if "choices" in payload:
        choices = payload["choices"]
        if not isinstance(choices, list):
            raise TypeError("choices must be a list")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("choice must be an object")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise TypeError("delta must be an object")
            text = delta.get("content")
            if text is not None and not isinstance(text, str):
                raise TypeError("content must be text")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    if "usage" in payload and payload["usage"] is not None:
        usage = payload["usage"]
        if not isinstance(usage, dict):
            raise TypeError("usage must be an object")
        prompt_tokens = _optional_int(usage, "prompt_tokens")
        completion_tokens = _optional_int(usage, "completion_tokens")

    tokens_per_second: float | None = None
    if "timings" in payload and payload["timings"] is not None:
        timings = payload["timings"]
        if not isinstance(timings, dict):
            raise TypeError("timings must be an object")
        rate = timings.get("predicted_per_second")
        if rate is not None:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise TypeError("predicted_per_second must be numeric")
            tokens_per_second = float(rate)

    has_telemetry = any(
        value is not None
        for value in (prompt_tokens, completion_tokens, tokens_per_second)
    )
    if not text and not has_telemetry:
        return None

    return LLMDelta(
        text=text or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=tokens_per_second,
    )


def _optional_int(source: dict[str, Any], key: str) -> int | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value
