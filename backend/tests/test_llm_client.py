import asyncio
import json
import logging
import time

import httpx
import pytest
import respx

from app.config import Settings
from app.errors import ApiError
from app.runtimes.llm import LlamaCppClient


def _settings(base_url: str = "http://llama.test", **overrides) -> Settings:
    return Settings(
        _env_file=None,
        llama_cpp_base_url=base_url,
        llama_cpp_model="",
        llm_timeout_seconds=5,
        # Off by default here so contract assertions stay readable; the
        # reference block has its own test below.
        **{"knowledge_base_enabled": False, **overrides},
    )


async def _collect(client: LlamaCppClient, transcript: str = "Merhaba"):
    return [delta async for delta in client.stream_answer(transcript)]


@respx.mock
async def test_stream_answer_sends_contract_and_yields_ordered_deltas():
    """Changing the request contract or dropping ordered content/usage must fail."""
    sse = "\n".join(
        [
            ": keep-alive",
            'data: {"choices":[{"delta":{"content":"Merhaba"}}]}',
            "event: message",
            'data: {"choices":[{"delta":{"content":" dünya"}}]}',
            (
                'data: {"choices":[],"usage":{"prompt_tokens":7,'
                '"completion_tokens":3},"timings":{"predicted_per_second":12.5}}'
            ),
            "data: [DONE]",
            "",
        ]
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body == {
            "model": "",
            "messages": [
                {
                    "role": "system",
                    "content": "Kısa, açık ve yalnızca Türkçe yanıt ver.",
                },
                {"role": "user", "content": "Merhaba"},
            ],
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 256,
        }
        return httpx.Response(
            200,
            text=sse,
            headers={"content-type": "text/event-stream"},
        )

    respx.post("http://llama.test/v1/chat/completions").mock(side_effect=respond)

    deltas = await _collect(LlamaCppClient(_settings()))

    assert [delta.text for delta in deltas] == ["Merhaba", " dünya", ""]
    assert deltas[0].prompt_tokens is None
    assert deltas[0].completion_tokens is None
    assert deltas[0].tokens_per_second is None
    assert deltas[2].prompt_tokens == 7
    assert deltas[2].completion_tokens == 3
    assert deltas[2].tokens_per_second == 12.5


@respx.mock
async def test_stream_answer_sends_reference_answers_and_reuses_one_prefix():
    """Dropping the reference block, or rebuilding it per turn, must fail here."""
    sent: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"tamam"}}]}\ndata: [DONE]\n',
            headers={"content-type": "text/event-stream"},
        )

    respx.post("http://llama.test/v1/chat/completions").mock(side_effect=respond)
    client = LlamaCppClient(_settings(knowledge_base_enabled=True))

    await _collect(client, "Turnike nasıl uygulanır?")
    await _collect(client, "Şok belirtileri nelerdir?")

    assert "S: Turnike nasıl uygulanır?" in sent[0]
    assert "Turnikeyi yaranın 5 ila 7 santim üstüne" in sent[0]
    assert "Kısa, açık ve yalnızca Türkçe yanıt ver." in sent[0]
    # llama-server only reuses its prefill cache while the prefix is identical.
    assert sent[0] == sent[1]


@respx.mock
async def test_stream_answer_maps_non_200_without_logging_sensitive_data(caplog):
    """An upstream HTTP failure must be safe, structured, and body-independent."""
    route = respx.post("http://llama.test/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="secret upstream body")
    )
    caplog.set_level(logging.INFO)

    with pytest.raises(ApiError) as caught:
        await _collect(LlamaCppClient(_settings()), transcript="gizli konuşma")

    assert route.called
    assert caught.value.code == "llm_unavailable"
    assert caught.value.status_code == 503
    assert "llama.test" in caplog.text
    assert "status=503" in caplog.text
    assert "gizli konuşma" not in caplog.text
    assert "secret upstream body" not in caplog.text


@respx.mock
async def test_stream_answer_rejects_malformed_json():
    """Malformed data frames may not be ignored or exposed as empty output."""
    respx.post("http://llama.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="data: {not-json}\n\ndata: [DONE]\n")
    )

    with pytest.raises(ApiError) as caught:
        await _collect(LlamaCppClient(_settings()))

    assert caught.value.code == "llm_invalid_response"
    assert caught.value.status_code == 502


@respx.mock
async def test_stream_answer_rejects_empty_stream():
    """A successful HTTP response with no usable delta must be an explicit error."""
    respx.post("http://llama.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=": keep-alive\n\ndata: [DONE]\n",
            headers={"content-type": "text/event-stream"},
        )
    )

    with pytest.raises(ApiError) as caught:
        await _collect(LlamaCppClient(_settings()))

    assert caught.value.code == "llm_empty_response"
    assert caught.value.status_code == 502


@respx.mock
async def test_stream_answer_maps_timeout():
    """Any HTTP timeout must become the stable LLM timeout contract."""
    respx.post("http://llama.test/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )

    with pytest.raises(ApiError) as caught:
        await _collect(LlamaCppClient(_settings()))

    assert caught.value.code == "llm_timeout"
    assert caught.value.status_code == 504


@respx.mock
async def test_health_uses_primary_endpoint_with_two_second_bound():
    """A supported healthy primary endpoint must avoid the fallback probe."""

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"] == {
            "connect": 2.0,
            "read": 2.0,
            "write": 2.0,
            "pool": 2.0,
        }
        return httpx.Response(200)

    primary = respx.get("http://llama.test/health").mock(side_effect=respond)
    fallback = respx.get("http://llama.test/v1/models").mock(
        return_value=httpx.Response(200)
    )

    assert await LlamaCppClient(_settings()).health() is True
    assert primary.called
    assert fallback.called is False


@respx.mock
async def test_health_falls_back_only_when_primary_is_unsupported():
    """Only a missing/unsupported health route may trigger the models endpoint."""
    respx.get("http://llama.test/health").mock(return_value=httpx.Response(404))
    fallback = respx.get("http://llama.test/v1/models").mock(
        return_value=httpx.Response(200)
    )

    assert await LlamaCppClient(_settings()).health() is True
    assert fallback.called


@respx.mock
async def test_health_fallback_shares_one_overall_two_second_deadline():
    """Two individually slow probes may not each consume a separate timeout."""

    async def delayed_unsupported(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.25)
        return httpx.Response(404)

    async def delayed_healthy(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.25)
        return httpx.Response(200)

    respx.get("http://llama.test/health").mock(side_effect=delayed_unsupported)
    respx.get("http://llama.test/v1/models").mock(side_effect=delayed_healthy)
    started = time.perf_counter()

    healthy = await LlamaCppClient(_settings()).health()
    elapsed = time.perf_counter() - started

    assert healthy is False
    assert elapsed < 2.3


@respx.mock
async def test_health_does_not_fallback_after_server_failure():
    """A real primary health failure must not be hidden by a second endpoint."""
    respx.get("http://llama.test/health").mock(return_value=httpx.Response(500))
    fallback = respx.get("http://llama.test/v1/models").mock(
        return_value=httpx.Response(200)
    )

    assert await LlamaCppClient(_settings()).health() is False
    assert fallback.called is False
