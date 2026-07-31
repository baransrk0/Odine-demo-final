# Orin Turkish Voice Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-user Turkish voice-assistant demo that runs entirely on Jetson Orin, streams a `llama-server` response sentence-by-sentence into TTS, plays each audio chunk in order, and reports stage-level latency.

**Architecture:** A FastAPI process owns the single active turn, warm STT/TTS adapters, temporary audio, in-memory metrics, and an SSE event stream. A React/TypeScript/Vite client records audio, creates a turn with multipart HTTP, follows the turn through SSE, and plays immutable per-sentence WAV chunks sequentially. The existing `llama-server` remains a separate process accessed through its OpenAI-compatible streaming endpoint.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pydantic Settings, HTTPX, Transformers/PyTorch adapters, ffmpeg, pytest/pytest-asyncio, React 18, TypeScript, Vite, Vitest, Testing Library, native MediaRecorder/EventSource/Web Audio APIs.

## Global Constraints

- All inference runs on the Orin; the Mac is only a VPN-connected browser client.
- `STT_MODEL_ID=`, `TTS_MODEL_ID=`, and `LLAMA_CPP_MODEL=` remain empty until model selection; no model name is committed or hard-coded.
- The first implementation is single-user and allows exactly one active turn.
- No RAG, Qdrant, embeddings, authentication, accounts, persistent conversation history, barge-in, fine-tuning, or concurrent turns.
- STT and TTS models load during backend startup and stay warm; a user request never downloads or loads a model.
- The LLM request contains one short Turkish system message and the current transcript only.
- Raw audio, transcript, answer text, secrets, model paths, and VPN details are not persisted or logged.
- CORS is restricted to configured demo origins; the application is not exposed to the public internet.
- The UI is served from the FastAPI origin after the Vite production build. A reverse proxy is not required for the first demo.
- SSE is the server-to-client transport for state, transcript, answer deltas, audio readiness, metrics, completion, and failure events.
- Anonymous recent metrics are process-local RAM data. Restarting the backend clears them.
- Use a 30-second/10-MiB default input limit, configurable through `AUDIO_MAX_SECONDS` and `AUDIO_MAX_BYTES`.
- Sentence completion recognizes `.`, `?`, or `!` before whitespace, stream end, or a closing quote/bracket; decimal numbers and the explicit Turkish abbreviation set are not split.
- Any GPU out-of-memory condition or repeated stage timeout fails the turn explicitly; audio chunks already produced remain available until retention expiry.

## Transport Contract

`POST /api/turns` accepts multipart `audio` and returns `202`:

```json
{"turn_id":"uuid","events_url":"/api/turns/uuid/events"}
```

`GET /api/turns/{turn_id}/events` emits named SSE events. Every payload contains `turn_id`; ordered payloads also contain monotonically increasing `event_id`.

```text
event: state
data: {"turn_id":"...","event_id":1,"stage":"transcribing"}

event: transcript
data: {"turn_id":"...","event_id":2,"text":"Merhaba"}

event: answer_delta
data: {"turn_id":"...","event_id":3,"text":"Merhaba!","answer":"Merhaba!"}

event: audio_ready
data: {"turn_id":"...","event_id":4,"sequence":0,"text":"Merhaba!","audio_url":"/api/audio/.../0.wav"}

event: metrics
data: {"turn_id":"...","event_id":5,"metrics":{...}}

event: complete
data: {"turn_id":"...","event_id":6,"transcript":"...","answer":"...","audio_url":"/api/audio/....wav","metrics":{...}}
```

`POST /api/turns/{turn_id}/playback` accepts `{"sequence":0,"client_offset_ms":1234}`. The first accepted report sets `first_audio_started_ms`; repeats are idempotent.

All API failures use:

```json
{"code":"machine_readable_code","message":"Türkçe kullanıcı mesajı"}
```

## File Map

```text
backend/
  pyproject.toml                         Python dependencies and test configuration
  app/
    main.py                              FastAPI lifespan, middleware, routes, static UI
    config.py                            Validated environment configuration
    errors.py                            Safe application errors and exception mapping
    schemas.py                           API, SSE, metric, and turn data contracts
    turns/
      manager.py                         Single-turn admission, lifecycle, cleanup
      orchestrator.py                    STT → streaming LLM → queued TTS pipeline
      events.py                          Replayable SSE event buffer
      metrics.py                         Monotonic timing and bounded recent summaries
      sentences.py                       Incremental Turkish sentence segmentation
    runtimes/
      protocols.py                       STT, LLM, and TTS interfaces
      stt.py                             Warm Hugging Face STT adapter
      llm.py                             Streaming llama.cpp HTTP adapter
      tts.py                             Warm Hugging Face TTS adapter
    audio/
      validation.py                      Upload size/duration/format validation
      conversion.py                      ffmpeg conversion to STT WAV
      storage.py                         Per-turn temporary and expiring audio files
  tests/
    conftest.py                          Fake runtimes and isolated application fixture
    test_config.py
    test_health.py
    test_audio_validation.py
    test_sentences.py
    test_llm_client.py
    test_orchestrator.py
    test_turn_api.py
    test_metrics_api.py
frontend/
  package.json
  vite.config.ts
  src/
    main.tsx
    App.tsx                              Page composition and turn state reducer
    api.ts                               Multipart, SSE, playback-report client
    types.ts                             Shared frontend wire types
    recorder.ts                          MediaRecorder lifecycle and limit enforcement
    audioQueue.ts                        Strictly ordered autoplay queue
    components/
      HealthBanner.tsx
      RecorderControls.tsx
      PipelineStatus.tsx
      Conversation.tsx
      AudioResponse.tsx
      MetricsCard.tsx
    styles.css
    test/
      setup.ts
      App.test.tsx
      api.test.ts
      recorder.test.ts
      audioQueue.test.ts
scripts/
  preflight.sh                           Orin disk/CUDA/ffmpeg/llama.cpp checks
  smoke_turn.py                          Recorded-audio end-to-end smoke client
systemd/
  orin-voice-assistant.service           Backend service example
.env.example                             Empty model IDs and safe defaults
.gitignore
README.md                                Orin setup, runbook, validation, privacy
```

---

### Task 1: Project Contracts and Validated Configuration

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/errors.py`
- Create: `backend/app/schemas.py`
- Create: `backend/tests/test_config.py`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: `Settings`, `ApiError`, `Stage`, `TurnStatus`, `TurnMetrics`, `ErrorBody`, and SSE payload models used by every later backend task.
- `Settings.models_configured: bool` is true only when all three model IDs are non-empty.

- [ ] **Step 1: Add Python packaging and test configuration**

Define Python `>=3.10`, runtime dependencies `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic-settings`, `python-multipart`, `sse-starlette`, `transformers`, `torch`, `soundfile`, and test dependencies `pytest`, `pytest-asyncio`, `respx`. Configure `asyncio_mode = "auto"` and add `backend/app` as the application package.

- [ ] **Step 2: Write failing settings tests**

```python
def test_model_ids_default_to_empty(monkeypatch):
    for key in ("STT_MODEL_ID", "TTS_MODEL_ID", "LLAMA_CPP_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.stt_model_id == ""
    assert settings.tts_model_id == ""
    assert settings.llama_cpp_model == ""
    assert settings.models_configured is False


def test_origins_are_explicit():
    settings = Settings(
        _env_file=None,
        CORS_ORIGINS="http://localhost:5173,http://orin.local:8000",
    )
    assert settings.cors_origins == [
        "http://localhost:5173",
        "http://orin.local:8000",
    ]
```

- [ ] **Step 3: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_config.py -v`  
Expected: FAIL because `app.config.Settings` does not exist.

- [ ] **Step 4: Implement settings and wire contracts**

Implement `Settings` with these environment-backed fields and safe defaults:

```python
llama_cpp_base_url: str = "http://127.0.0.1:8080"
llama_cpp_model: str = ""
stt_model_id: str = ""
tts_model_id: str = ""
stt_device: str = "cuda"
tts_device: str = "cuda"
stt_dtype: str = "float16"
tts_dtype: str = "float16"
audio_max_seconds: float = 30.0
audio_max_bytes: int = 10 * 1024 * 1024
audio_retention_seconds: int = 600
llm_timeout_seconds: float = 60.0
stt_timeout_seconds: float = 45.0
tts_timeout_seconds: float = 45.0
cors_origins_raw: str = "http://localhost:5173"
metrics_capacity: int = 50
turkish_system_prompt: str = "Kısa, açık ve yalnızca Türkçe yanıt ver."
llm_max_tokens: int = 256
```

Expose a parsed `cors_origins` property. Define `Stage` as `uploading`, `transcribing`, `generating`, `synthesizing`, `playing`, `complete`, and `failed`. Define every wire payload as a Pydantic model and use nullable token telemetry rather than numeric zero defaults.

- [ ] **Step 5: Add safe environment templates**

`.env.example` must contain:

```dotenv
LLAMA_CPP_BASE_URL=http://127.0.0.1:8080
LLAMA_CPP_MODEL=
STT_MODEL_ID=
TTS_MODEL_ID=
STT_DEVICE=cuda
TTS_DEVICE=cuda
STT_DTYPE=float16
TTS_DTYPE=float16
AUDIO_MAX_SECONDS=30
AUDIO_MAX_BYTES=10485760
AUDIO_RETENTION_SECONDS=600
LLM_TIMEOUT_SECONDS=60
STT_TIMEOUT_SECONDS=45
TTS_TIMEOUT_SECONDS=45
CORS_ORIGINS=http://localhost:5173
```

Ignore `.env`, Python caches, virtual environments, test caches, `frontend/node_modules`, `frontend/dist`, runtime audio, and logs.

- [ ] **Step 6: Run tests**

Run: `cd backend && python -m pytest tests/test_config.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/app .env.example .gitignore backend/tests/test_config.py
git commit -m "chore: define voice assistant contracts"
```

---

### Task 2: Audio Validation, Conversion, and Expiring Storage

**Files:**
- Create: `backend/app/audio/__init__.py`
- Create: `backend/app/audio/validation.py`
- Create: `backend/app/audio/conversion.py`
- Create: `backend/app/audio/storage.py`
- Create: `backend/tests/test_audio_validation.py`

**Interfaces:**
- Consumes: `Settings`, `ApiError`.
- Produces: `validate_upload(content_type, byte_count)`, `probe_duration(path)`, `convert_to_stt_wav(source, target)`, and `AudioStorage`.
- `AudioStorage.chunk_url(turn_id, sequence)` returns `/api/audio/{turn_id}/{sequence}.wav`.

- [ ] **Step 1: Write failing validation and lifecycle tests**

```python
def test_rejects_oversized_audio():
    with pytest.raises(ApiError) as error:
        validate_upload("audio/webm", byte_count=11, max_bytes=10)
    assert error.value.code == "audio_too_large"


def test_storage_isolated_by_turn_and_expires(tmp_path):
    storage = AudioStorage(tmp_path, retention_seconds=1)
    first = storage.chunk_path(UUID(int=1), 0)
    second = storage.chunk_path(UUID(int=2), 0)
    assert first.parent != second.parent
    first.parent.mkdir(parents=True)
    first.write_bytes(b"RIFF")
    storage.expire(now=first.stat().st_mtime + 2)
    assert not first.exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_audio_validation.py -v`  
Expected: FAIL because audio modules do not exist.

- [ ] **Step 3: Implement bounded upload and duration validation**

Accept `audio/webm`, `audio/ogg`, `audio/wav`, `audio/mp4`, and `audio/mpeg`. Stream the multipart upload to disk in 1-MiB blocks, stop immediately when `AUDIO_MAX_BYTES` is exceeded, and reject empty input. Use:

```bash
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 INPUT
```

Reject duration `<= 0.2` seconds as `audio_too_short` and duration over the configured limit as `audio_too_long`.

- [ ] **Step 4: Implement deterministic conversion and storage**

Invoke ffmpeg without a shell:

```python
[
    "ffmpeg", "-nostdin", "-y", "-i", str(source),
    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
]
```

Capture stderr for server diagnostics but map failures to `invalid_audio`; never return command output or paths to the client. Store each upload and generated chunk below a UUID directory. Delete upload/intermediate files after a terminal turn, while retaining final chunks until expiry.

- [ ] **Step 5: Run focused tests**

Run: `cd backend && python -m pytest tests/test_audio_validation.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/audio backend/tests/test_audio_validation.py
git commit -m "feat: validate and isolate turn audio"
```

---

### Task 3: Incremental Turkish Sentence Segmentation

**Files:**
- Create: `backend/app/turns/__init__.py`
- Create: `backend/app/turns/sentences.py`
- Create: `backend/tests/test_sentences.py`

**Interfaces:**
- Produces: `TurkishSentenceBuffer.push(delta: str) -> list[str]` and `finish() -> list[str]`.
- Emitted strings preserve the exact stream text and are never emitted twice.

- [ ] **Step 1: Write the segmentation matrix**

```python
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Merhaba! Nasılsın?"], ["Merhaba!", " Nasılsın?"]),
        (["Değer 3.", "14 oldu. Sonuç iyi"], ["Değer 3.14 oldu."]),
        (["Dr. Ayşe geldi. Devam ediyor."], ["Dr. Ayşe geldi.", " Devam ediyor."]),
        (['"Hazır mısın?" Evet.'], ['"Hazır mısın?"', " Evet."]),
        (["İlk cümle.", " İkinci"], ["İlk cümle."]),
    ],
)
def test_incremental_boundaries(chunks, expected):
    buffer = TurkishSentenceBuffer()
    assert [sentence for chunk in chunks for sentence in buffer.push(chunk)] == expected


def test_finish_flushes_unpunctuated_tail_once():
    buffer = TurkishSentenceBuffer()
    buffer.push("Son parça")
    assert buffer.finish() == ["Son parça"]
    assert buffer.finish() == []
```

- [ ] **Step 2: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_sentences.py -v`  
Expected: FAIL because `TurkishSentenceBuffer` does not exist.

- [ ] **Step 3: Implement the incremental buffer**

Use an explicit abbreviation set:

```python
ABBREVIATIONS = {
    "Dr.", "Doç.", "Prof.", "Sn.", "Say.", "No.", "örn.", "vb.", "vs.",
    "bkz.", "T.C.",
}
CLOSERS = '"”’»)]}'
```

A boundary is valid when punctuation is not between digits, the current token is not an abbreviation, any closing characters are included, and the next observed character is whitespace or the stream has ended. Retain ambiguous trailing punctuation until another character arrives or `finish()` is called.

- [ ] **Step 4: Run focused tests**

Run: `cd backend && python -m pytest tests/test_sentences.py -v`  
Expected: PASS for punctuation, decimals, abbreviations, closers, chunk boundaries, and final tail.

- [ ] **Step 5: Commit**

```bash
git add backend/app/turns backend/tests/test_sentences.py
git commit -m "feat: segment streamed Turkish sentences"
```

---

### Task 4: Runtime Protocols and llama.cpp Streaming Client

**Files:**
- Create: `backend/app/runtimes/__init__.py`
- Create: `backend/app/runtimes/protocols.py`
- Create: `backend/app/runtimes/llm.py`
- Create: `backend/tests/test_llm_client.py`

**Interfaces:**
- Produces: async `STTRuntime.transcribe(path) -> str`, async `TTSRuntime.synthesize(text, output_path) -> None`, and `LLMRuntime.stream_answer(transcript) -> AsyncIterator[LLMDelta]`.
- `LLMDelta` contains `text`, nullable prompt/completion token counts, and nullable tokens/second.

- [ ] **Step 1: Write failing HTTP contract tests**

Use `respx` to assert the outgoing request:

```python
assert request.url.path == "/v1/chat/completions"
body = json.loads(request.content)
assert body == {
    "model": "",
    "messages": [
        {"role": "system", "content": "Kısa, açık ve yalnızca Türkçe yanıt ver."},
        {"role": "user", "content": "Merhaba"},
    ],
    "stream": True,
    "temperature": 0.2,
    "max_tokens": 256,
}
```

Feed SSE frames containing content deltas, a usage frame, and `[DONE]`; assert ordered deltas and nullable telemetry. Add cases for non-200 response, malformed JSON, empty stream, and timeout.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_llm_client.py -v`  
Expected: FAIL because runtime protocols/client do not exist.

- [ ] **Step 3: Implement strict SSE parsing**

Use `httpx.AsyncClient.stream()` with the configured timeout. Read only `data:` lines, stop on `[DONE]`, extract `choices[0].delta.content`, and accept usage fields only when present. Map failures to:

```text
llm_unavailable
llm_timeout
llm_invalid_response
llm_empty_response
```

Log status code, elapsed time, and safe host label; do not log messages, raw response bodies, complete URLs containing credentials, or model paths.

- [ ] **Step 4: Add health probing**

Implement `health() -> bool` against `GET /health`, falling back to `GET /v1/models` only when `/health` is unsupported. Bound this call to 2 seconds and never expose `LLAMA_CPP_BASE_URL` in the API response.

- [ ] **Step 5: Run focused tests**

Run: `cd backend && python -m pytest tests/test_llm_client.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/runtimes backend/tests/test_llm_client.py
git commit -m "feat: stream responses from llama server"
```

---

### Task 5: Warm Hugging Face STT and TTS Adapters

**Files:**
- Create: `backend/app/runtimes/stt.py`
- Create: `backend/app/runtimes/tts.py`
- Create: `backend/tests/test_runtime_adapters.py`

**Interfaces:**
- Consumes: runtime protocols and configured blank-capable model IDs.
- Produces: `HuggingFaceSTT.load()/transcribe()` and `HuggingFaceTTS.load()/synthesize()`.
- Both expose `ready: bool`; an empty model ID leaves the adapter not ready and does not contact the network.

- [ ] **Step 1: Write adapter lifecycle tests with injected loaders**

```python
def test_empty_stt_model_id_does_not_load(fake_loader):
    runtime = HuggingFaceSTT(model_id="", loader=fake_loader)
    runtime.load()
    fake_loader.assert_not_called()
    assert runtime.ready is False


async def test_tts_writes_mono_wav(fake_tts_pipeline, tmp_path):
    runtime = HuggingFaceTTS(model_id="", pipeline=fake_tts_pipeline)
    output = tmp_path / "0.wav"
    await runtime.synthesize("Merhaba.", output)
    assert output.read_bytes().startswith(b"RIFF")
```

Also test Turkish language/task arguments for STT, whitespace-only transcript normalization, dtype/device mapping, output sample rate propagation, and inference exceptions.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_runtime_adapters.py -v`  
Expected: FAIL because adapters do not exist.

- [ ] **Step 3: Implement injectable Hugging Face adapters**

Keep model-specific behavior behind adapter callables so the selected model can supply its processor/pipeline without changing orchestration. `load()` uses `local_files_only=True` by default in application startup; deployment downloads are a separate preflight step. Public `transcribe()` and `synthesize()` methods are async and run blocking model inference through `asyncio.to_thread()`. Wrap the blocking inference body with `torch.inference_mode()`. Normalize STT output with `.strip()` but do not rewrite Turkish content.

For TTS, accept pipeline output in the normalized internal form:

```python
SynthesizedAudio(samples: numpy.ndarray, sample_rate: int)
```

Write mono float/PCM WAV through `soundfile`. Do not concatenate chunks in memory.

- [ ] **Step 4: Run focused tests**

Run: `cd backend && python -m pytest tests/test_runtime_adapters.py -v`  
Expected: PASS without downloading any model.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtimes backend/tests/test_runtime_adapters.py
git commit -m "feat: add warm STT and TTS adapters"
```

---

### Task 6: Replayable Events, Metrics, and Turn Orchestration

**Files:**
- Create: `backend/app/turns/events.py`
- Create: `backend/app/turns/metrics.py`
- Create: `backend/app/turns/orchestrator.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: audio services, runtime protocols, `TurkishSentenceBuffer`, schemas.
- Produces: `TurnOrchestrator.run(turn)`, `TurnEventBuffer.publish()/subscribe(last_event_id)`, and `RecentMetrics`.
- A turn owns one bounded `asyncio.Queue[SentenceJob]`; one worker synthesizes jobs strictly by `sequence`.

- [ ] **Step 1: Define fake runtime fixtures**

Create deterministic fakes:

```python
class FakeSTT:
    ready = True
    async def transcribe(self, path): return "Merhaba"

class FakeLLM:
    ready = True
    async def stream_answer(self, transcript):
        for text in ("Birinci cümle. ", "İkinci cümle."):
            yield LLMDelta(text=text)

class FakeTTS:
    ready = True
    calls: list[str]
    async def synthesize(self, text, output_path):
        self.calls.append(text)
        output_path.write_bytes(b"RIFF")
```

- [ ] **Step 2: Write failing orchestration tests**

Assert:

- blank STT produces `stt_empty`, and LLM/TTS are not called;
- answer deltas publish before LLM completion;
- the first sentence enters TTS while the LLM fake is still blocked before its second sentence;
- `audio_ready.sequence` is exactly `0, 1, ...`;
- final unpunctuated text is synthesized;
- TTS failure preserves transcript and answer but produces terminal `tts_failed`;
- timeout/OOM produces a terminal failure and never silently drops a queued sentence;
- temporary source files are deleted and retained chunks remain;
- all durations use `time.perf_counter_ns()`, not wall-clock subtraction.

- [ ] **Step 3: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`  
Expected: FAIL because orchestration modules do not exist.

- [ ] **Step 4: Implement replayable event delivery**

Each event receives a monotonically increasing integer ID and is retained for the life of the turn. `subscribe(last_event_id)` first replays later buffered events, then follows live events. Send an SSE comment heartbeat every 15 seconds. A reconnect with `Last-Event-ID` must not duplicate audio playback events.

- [ ] **Step 5: Implement metrics**

Record:

```text
timestamp, outcome, recording duration/bytes/content type,
upload_ms, stt_ms, llm_ms, tts_ms, total_ms,
first_sentence_ready_ms, first_audio_started_ms,
sentence_count, audio_chunk_count,
transcript_chars, answer_chars,
llm_prompt_tokens, llm_completion_tokens, llm_tokens_per_second,
error_stage, error_code, timed_out,
safe device/dtype/configuration summary
```

Use a bounded `collections.deque(maxlen=settings.metrics_capacity)`. Store lengths, not content. `first_audio_started_ms` remains `null` until reported by the client.

- [ ] **Step 6: Implement the pipeline**

Run STT under its timeout. Reject empty text. Start the TTS worker before consuming LLM deltas. For every completed sentence, publish `answer_delta`, enqueue `SentenceJob(sequence, text)`, and record first-sentence time. On LLM completion call `finish()`, await `queue.join()`, then terminate the worker with a sentinel. Preserve ordered files and optionally concatenate them into the final WAV using ffmpeg's concat demuxer.

Use explicit cancellation cleanup in `finally`: cancel unfinished tasks, release the active-turn lease, remove source/intermediate files, publish one terminal event, and append one metrics record.

- [ ] **Step 7: Run focused tests**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`  
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/turns backend/tests/conftest.py backend/tests/test_orchestrator.py
git commit -m "feat: orchestrate streamed voice turns"
```

---

### Task 7: FastAPI Lifecycle and HTTP/SSE API

**Files:**
- Create: `backend/app/turns/manager.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_health.py`
- Create: `backend/tests/test_turn_api.py`
- Create: `backend/tests/test_metrics_api.py`

**Interfaces:**
- Consumes: all backend services from Tasks 1–6.
- Produces: the API routes in the spec plus `GET /api/turns/{id}/events`, chunk audio routes, and playback reporting.
- `TurnManager.try_create()` atomically rejects a second active turn.

- [ ] **Step 1: Write failing route tests**

Test:

```text
GET  /api/health
POST /api/turns
GET  /api/turns/{turn_id}/events
POST /api/turns/{turn_id}/playback
GET  /api/audio/{turn_id}/{sequence}.wav
GET  /api/audio/{turn_id}.wav
GET  /api/metrics/recent
```

Assert `409` for a second active turn, `404` for unknown/expired audio, safe Turkish error bodies, correct `audio/wav`, no real LLM URL in health, and CORS acceptance/rejection.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd backend && python -m pytest tests/test_health.py tests/test_turn_api.py tests/test_metrics_api.py -v`  
Expected: FAIL because the FastAPI app does not exist.

- [ ] **Step 3: Implement lifespan and readiness**

During startup:

1. validate ffmpeg/ffprobe availability;
2. instantiate STT/TTS adapters;
3. call their `load()` methods;
4. create the llama.cpp client and probe it;
5. start periodic audio expiry;
6. expose readiness without failing the process solely because model IDs are still empty.

Health returns `status: "ok"` only when all runtimes are ready; otherwise return `status: "degraded"` with per-runtime booleans and `"llm_base_url": "configured"`.

- [ ] **Step 4: Implement atomic admission and background execution**

Fully stream and validate upload before returning `202`. Acquire the active-turn lease atomically; if occupied, delete the rejected upload and return:

```json
{"code":"turn_in_progress","message":"Mevcut yanıt tamamlanıyor."}
```

Start `orchestrator.run()` as an owned application task. Retain completed turn event buffers until audio retention expiry so EventSource reconnects can receive the terminal event.

- [ ] **Step 5: Implement SSE and playback measurement**

Serialize with `sse-starlette`. Honor `Last-Event-ID`. The playback endpoint accepts only a produced sequence and sets the metric once:

```python
first_audio_started_ms = min(client_offset_ms, current_server_turn_age_ms)
```

Reject negative offsets and unknown sequences. This is an observed client playback time, distinct from server-side audio-ready time.

- [ ] **Step 6: Run all backend tests**

Run: `cd backend && python -m pytest -v`  
Expected: PASS; no network/model download is attempted.

- [ ] **Step 7: Commit**

```bash
git add backend/app backend/tests
git commit -m "feat: expose voice turn API and SSE"
```

---

### Task 8: React Client, Recording, and Ordered Audio Playback

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/recorder.ts`
- Create: `frontend/src/audioQueue.ts`
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/api.test.ts`
- Create: `frontend/src/recorder.test.ts`
- Create: `frontend/src/audioQueue.test.ts`

**Interfaces:**
- Consumes: API/SSE contracts from Task 7.
- Produces: `VoiceApi`, `BrowserRecorder`, and `OrderedAudioQueue`.
- `OrderedAudioQueue.enqueue({sequence, url})` tolerates out-of-order network arrival but plays only the next expected sequence.

- [ ] **Step 1: Add Vite/TypeScript/Vitest setup**

Pin React 18 and current compatible Vite/Vitest/Testing Library versions in the lockfile produced during implementation. Proxy `/api` to `http://127.0.0.1:8000` only in Vite development.

- [ ] **Step 2: Write failing browser utility tests**

Test:

- supported MediaRecorder MIME selection;
- permission denial maps to `Mikrofon izni gerekli.`;
- recording duration automatically stops at the configured limit;
- multipart field name is exactly `audio`;
- SSE event parsing is type-discriminated;
- audio chunks arriving `1, 0, 2` play `0, 1, 2`;
- rejected `audio.play()` exposes a manual-play state;
- `ended` advances the queue;
- the first successful `play` reports playback once.

- [ ] **Step 3: Run tests and verify failure**

Run: `cd frontend && npm test -- --run`  
Expected: FAIL because client modules do not exist.

- [ ] **Step 4: Implement recording**

`BrowserRecorder` uses `navigator.mediaDevices.getUserMedia({audio: true})`, selects the first supported MIME from `audio/webm;codecs=opus`, `audio/webm`, `audio/ogg;codecs=opus`, accumulates Blob chunks, exposes elapsed seconds, and always stops media tracks after stop/error.

- [ ] **Step 5: Implement API and reconnect-safe SSE**

Create the turn with `fetch`. Open `EventSource(events_url)`. Maintain the highest processed `event_id` in component state and ignore repeats after reconnect. Close EventSource on terminal event or component unmount. Convert HTTP errors into the server's Turkish `message`, falling back to a generic Turkish network error.

- [ ] **Step 6: Implement strict playback ordering**

Hold a map by sequence and one `HTMLAudioElement`. Start sequence `0` as soon as available. After `play()` resolves, report `performance.now() - turnStartedAt`; after `ended`, advance. On autoplay rejection, keep the same chunk loaded and expose `resume()` for the manual button.

- [ ] **Step 7: Run utility tests**

Run: `cd frontend && npm test -- --run src/api.test.ts src/recorder.test.ts src/audioQueue.test.ts`  
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend
git commit -m "feat: add browser recording and audio queue"
```

---

### Task 9: Demo UI and State Machine

**Files:**
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/components/HealthBanner.tsx`
- Create: `frontend/src/components/RecorderControls.tsx`
- Create: `frontend/src/components/PipelineStatus.tsx`
- Create: `frontend/src/components/Conversation.tsx`
- Create: `frontend/src/components/AudioResponse.tsx`
- Create: `frontend/src/components/MetricsCard.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: frontend utilities from Task 8.
- Produces: a responsive single-page demo covering every UI requirement.
- The reducer is the only owner of turn stage and disables recording for every non-terminal stage.

- [ ] **Step 1: Write failing interaction tests**

Cover:

1. health banner shows backend/STT/TTS/LLM independently;
2. start changes the control to stop and displays elapsed recording time;
3. stop uploads once and transitions through incoming SSE stages;
4. transcript and accumulating answer render independently;
5. recording stays disabled until `complete` or `failed`;
6. `409`, empty STT, LLM failure, TTS text-only fallback, and autoplay rejection show exact Turkish messages;
7. metrics render milliseconds, nullable token fields as `—`, and token/s only when numeric.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd frontend && npm test -- --run src/App.test.tsx`  
Expected: FAIL because UI components do not exist.

- [ ] **Step 3: Implement the reducer**

Use:

```typescript
type TurnStage =
  | "idle" | "recording" | "uploading" | "transcribing"
  | "generating" | "synthesizing" | "playing"
  | "complete" | "failed";
```

Reducer actions mirror wire events. Ignore any event whose `turn_id` is not the current turn. Clear the previous error and content only when a new recording begins.

- [ ] **Step 4: Implement required UI sections**

Render:

- one connection banner with four labeled indicators;
- one Start/Stop control with duration and disabled reason;
- a Turkish pipeline status label;
- separate user transcript and assistant response regions with `aria-live="polite"`;
- audio status/manual-play control;
- performance rows for upload, STT, LLM, TTS, total, first sentence ready, first playback, and optional LLM token telemetry.

Use visible focus states, semantic buttons, sufficient contrast, and a desktop-centered layout that collapses cleanly below 720 px.

- [ ] **Step 5: Run frontend tests and build**

Run: `cd frontend && npm test -- --run && npm run build`  
Expected: all tests PASS and `frontend/dist` is produced without TypeScript errors.

- [ ] **Step 6: Serve the production build from FastAPI**

Mount Vite assets and add an SPA fallback after `/api` routes. In backend tests, inject a temporary static directory rather than requiring a frontend build.

- [ ] **Step 7: Commit**

```bash
git add frontend backend/app/main.py backend/tests
git commit -m "feat: build Turkish voice assistant UI"
```

---

### Task 10: Orin Preflight, Service Definition, and End-to-End Smoke Runbook

**Files:**
- Create: `scripts/preflight.sh`
- Create: `scripts/smoke_turn.py`
- Create: `systemd/orin-voice-assistant.service`
- Create: `README.md`
- Modify: `backend/pyproject.toml`
- Modify: `frontend/package.json`

**Interfaces:**
- Consumes: completed backend/frontend and externally selected model configuration.
- Produces: repeatable Orin deployment and one documented cold-start smoke procedure.

- [ ] **Step 1: Write the preflight script**

Use `set -euo pipefail` and report, without secrets:

```bash
df -h /
df -h "${HF_HOME:-/var/lib/orin-voice-assistant/huggingface}"
free -h
command -v ffmpeg
command -v ffprobe
command -v nvidia-smi
curl --fail --max-time 2 http://127.0.0.1:8080/health
```

Also run a Python import check for `torch`, print `torch.cuda.is_available()`, CUDA version, and device name. Fail when any model ID is empty in the actual deployment environment, while keeping `.env.example` empty.

- [ ] **Step 2: Implement the smoke client**

`smoke_turn.py AUDIO_FILE BASE_URL` must:

1. create a multipart turn;
2. follow SSE until terminal;
3. print event stages and anonymous metrics;
4. download every announced audio chunk;
5. assert transcript and answer are non-empty;
6. assert at least one audio chunk exists;
7. assert `first_sentence_ready_ms < llm_ms`;
8. exit non-zero on API failure, missing metrics, sequence gaps, or timeout.

Do not print transcript, answer, raw SSE payloads, or configured model IDs by default. Add `--show-content` as an explicit local diagnostic switch.

- [ ] **Step 3: Add the systemd unit**

Run as a dedicated non-root user, read `/etc/orin-voice-assistant.env`, bind Uvicorn to the VPN/LAN interface or `0.0.0.0` only behind the host firewall, use `Restart=on-failure`, set a writable runtime audio directory, and never place credentials in the unit. Declare `After=network-online.target` and document that `llama-server` must already be healthy.

- [ ] **Step 4: Write the runbook**

Document exact commands for:

```bash
cd frontend && npm ci && npm run build
cd backend && python -m venv .venv
cd backend && .venv/bin/pip install -e '.[test]'
cd backend && .venv/bin/python -m pytest -v
cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Include model pre-download as a separate operator step after IDs are selected, VPN-only access, browser microphone secure-context requirements, firewall/CORS setup, log/privacy behavior, model-license verification, and recovery checks for disk pressure, CUDA OOM, timeout, llama.cpp unavailability, and autoplay blocking.

- [ ] **Step 5: Run static and automated verification**

Run:

```bash
bash -n scripts/preflight.sh
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
```

Expected: shell syntax valid; all backend/frontend tests PASS; production build succeeds.

- [ ] **Step 6: Run the Orin acceptance smoke after model selection**

On a cold backend start, run preflight, start/verify `llama-server`, start the app, open the UI from the Mac over VPN, grant microphone access, and complete one Turkish turn. Then run `smoke_turn.py` with a consented non-sensitive Turkish fixture and save only:

```text
date/time, git commit, outcome, safe device/dtype configuration,
upload/STT/LLM/TTS/total latency,
first-sentence-ready and first-playback latency,
chunk count and nullable token telemetry
```

Acceptance requires all nine criteria in the approved spec, including first audio before LLM completion, second-turn `409`, recovery after a forced stage failure, and no RAG/Qdrant/embedding dependency.

- [ ] **Step 7: Commit**

```bash
git add scripts systemd README.md backend/pyproject.toml frontend/package.json
git commit -m "docs: add Orin deployment and smoke runbook"
```

---

## Final Verification Gate

- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Run `cd frontend && npm test -- --run`.
- [ ] Run `cd frontend && npm run build`.
- [ ] Run `bash -n scripts/preflight.sh`.
- [ ] Search for forbidden dependencies: `rg -n -i 'qdrant|rag|embedding' backend frontend scripts systemd` and verify there are no runtime references.
- [ ] Search for committed model values: `rg -n '^(STT_MODEL_ID|TTS_MODEL_ID|LLAMA_CPP_MODEL)=.+' .env.example` and verify there is no output.
- [ ] Search for unsafe content logging: `rg -n 'log.*(transcript|answer|prompt|audio)' backend/app` and review every result.
- [ ] Verify a second turn receives `409` while the first is active and a new turn succeeds after completion/failure.
- [ ] Verify SSE reconnection with `Last-Event-ID` does not replay an already played sequence.
- [ ] Verify a real Orin cold-start smoke and record anonymous metrics in the README run log without speech content or secrets.
