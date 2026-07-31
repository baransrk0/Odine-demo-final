# I2S RF Audio Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional RF/I2S PTT capture and USB-headset TTS playback to the Orin voice assistant while preserving its browser-upload flow and one-active-turn contract.

**Architecture:** Extract direct normalized-WAV submission from the HTTP route, then add an application-owned RF capture service that produces the same normalized-WAV contract. Add an optional local playback sink to the existing ordered TTS worker so physical headset playback and SSE/browser audio share one sequence.

**Tech Stack:** Python 3.10+, FastAPI lifespan, asyncio subprocesses, ALSA (`amixer`, `arecord`, `aplay`), ffmpeg, pytest/pytest-asyncio, existing Transformers and llama.cpp runtimes.

## Global Constraints

- Preserve `AUDIO_INPUT_MODE=browser` as the default and keep the existing multipart `/api/turns` contract unchanged.
- Permit one active turn only; all RF and browser entries must use `TurnManager.try_create`.
- RF capture defaults are `hw:APE,0`, 8 kHz, two channels, signed 16-bit little-endian PCM, `APE` card, and `I2S2` port.
- Do not use a shell for any mixer, capture, conversion, or playback command.
- `LOCAL_AUDIO_PLAYBACK=false` by default; its target headset is `plughw:2,0` when enabled.
- Keep `demoorin/atbk-ecg` unmodified and uncommitted.

---

## File Structure

- Create `backend/app/audio/i2s.py`: APE routing and bounded PTT raw-PCM capture primitives.
- Create `backend/app/audio/playback.py`: async local WAV playback boundary.
- Create `backend/app/turns/submitter.py`: shared normalized-WAV turn admission/start boundary.
- Create `backend/app/audio/rf_service.py`: lifespan-managed RF capture loop.
- Modify `backend/app/config.py`: validated RF, local playback, and mode settings.
- Modify `backend/app/main.py`: construct shared submitter and optional RF lifecycle service; reuse submitter from HTTP route.
- Modify `backend/app/turns/orchestrator.py`: invoke optional local player in ordered TTS flow.
- Modify `frontend/src/App.tsx` and `frontend/src/components/RecorderControls.tsx`: hide/disable browser recording when health reports RF mode.
- Modify `backend/app/main.py` health response and `frontend/src/types.ts`: expose safe input mode and local-playback state.
- Create focused tests under `backend/tests/` and update existing frontend tests.
- Modify `README.md` and `systemd/orin-voice-assistant.service`: document and provision hardware mode.

### Task 1: Add validated hardware-mode settings

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces `Settings.audio_input_mode: Literal["browser", "rf_i2s"]`, `Settings.local_audio_playback: bool`, and validated RF/APE/headset fields used by later tasks.

- [ ] **Step 1: Write failing settings tests**

```python
def test_rf_mode_defaults_and_validation():
    settings = Settings(_env_file=None, AUDIO_INPUT_MODE="rf_i2s")
    assert settings.rf_mic_device == "hw:APE,0"
    assert settings.rf_mic_sample_rate == 8000
    assert settings.rf_mic_channels == 2

    with pytest.raises(ValidationError):
        Settings(_env_file=None, AUDIO_INPUT_MODE="serial")
```

- [ ] **Step 2: Run the settings test to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_config.py -q`

Expected: FAIL because the RF settings do not exist.

- [ ] **Step 3: Implement fields and cross-field validation**

```python
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
```

Add a model validator that rejects blank device strings, blank APE fields, a minimum duration above the maximum duration, and enabled local playback with a blank speaker device.

- [ ] **Step 4: Run the settings test to verify success**

Run: `cd backend && .venv/bin/python -m pytest tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the settings contract**

```bash
git add backend/app/config.py backend/tests/test_config.py
git commit -m "feat: add RF I2S audio settings"
```

### Task 2: Implement testable APE routing and PTT capture primitives

**Files:**
- Create: `backend/app/audio/i2s.py`
- Test: `backend/tests/test_i2s.py`

**Interfaces:**
- Produces `configure_ape(settings: Settings) -> None`.
- Produces `capture_ptt_pcm(settings: Settings, destination: Path) -> CaptureResult | None`, where `None` means no valid PTT utterance and `CaptureResult` contains `path: Path`, `duration_seconds: float`, and `byte_count: int`.
- Consumes only `Settings` and injected command/process factories so tests are hardware-free.

- [ ] **Step 1: Write failing capture tests**

```python
def test_configure_ape_uses_expected_mixer_controls(fake_run):
    configure_ape(_rf_settings(), run=fake_run)
    assert fake_run.calls == [
        ["amixer", "-c", "APE", "cset", "name=I2S2 codec master mode", "cbm-cfm"],
        ["amixer", "-c", "APE", "cset", "name=ADMAIF1 Mux", "I2S2"],
    ]

@pytest.mark.asyncio
async def test_capture_ends_after_started_stream_times_out(tmp_path, fake_process):
    result = await capture_ptt_pcm(_rf_settings(), tmp_path / "rf.raw", process_factory=fake_process)
    assert result is not None
    assert result.byte_count == 6400
```

- [ ] **Step 2: Run the capture test to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_i2s.py -q`

Expected: FAIL because `app.audio.i2s` is absent.

- [ ] **Step 3: Implement the bounded subprocess boundary**

Implement mixer calls with `subprocess.run(..., shell=False, capture_output=True, text=True)` and raise `RuntimeError` on non-zero results. Start `arecord` as:

```python
[
    "arecord", "-q", "-D", settings.rf_mic_device,
    "-r", str(settings.rf_mic_sample_rate), "-f", "S16_LE",
    "-c", str(settings.rf_mic_channels), "-t", "raw",
]
```

Wait indefinitely for the first frame, then apply `asyncio.wait_for` using `rf_ptt_frame_timeout_ms` to subsequent frames. Stop at timeout, EOF, or `rf_capture_max_seconds`; terminate the process in `finally`; delete the destination for no-frame and too-short captures.

- [ ] **Step 4: Add failure and cleanup tests**

```python
@pytest.mark.asyncio
async def test_short_capture_is_discarded_and_process_is_terminated(tmp_path, fake_process):
    result = await capture_ptt_pcm(_rf_settings(rf_ptt_min_seconds=1.0), tmp_path / "rf.raw", process_factory=fake_process)
    assert result is None
    assert fake_process.terminated is True
    assert not (tmp_path / "rf.raw").exists()
```

- [ ] **Step 5: Run the primitive test suite**

Run: `cd backend && .venv/bin/python -m pytest tests/test_i2s.py -q`

Expected: PASS without ALSA hardware.

- [ ] **Step 6: Commit capture primitives**

```bash
git add backend/app/audio/i2s.py backend/tests/test_i2s.py
git commit -m "feat: add I2S PTT capture primitives"
```

### Task 3: Extract direct normalized-WAV turn submission

**Files:**
- Create: `backend/app/turns/submitter.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_turn_submitter.py`

**Interfaces:**
- Produces `TurnSubmitter.submit_normalized_wav(input_path: Path, *, source_path: Path | None, duration_seconds: float, byte_count: int, content_type: str | None, started_ns: int | None = None) -> TurnCreated`.
- Consumes `AudioStorage`, `TurnManager`, and `TurnOrchestrator`.
- Produces exactly the existing `TurnCreated` response and raises the existing `ApiError("turn_in_progress", ...)` on lease contention.

- [ ] **Step 1: Write a failing direct-submission test**

```python
@pytest.mark.asyncio
async def test_submitter_admits_and_starts_the_existing_orchestrator(tmp_path):
    submitter, manager = _submitter(tmp_path)
    created = await submitter.submit_normalized_wav(
        tmp_path / "input.wav", source_path=None, duration_seconds=1.0,
        byte_count=32000, content_type="audio/wav",
    )
    assert manager.get(UUID(created.turn_id)) is not None
```

- [ ] **Step 2: Run the submitter test to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_turn_submitter.py -q`

Expected: FAIL because `TurnSubmitter` is absent.

- [ ] **Step 3: Implement submission and refactor the upload route**

Create the `TurnContext`, call `TurnManager.try_create`, start `TurnOrchestrator.run`, and return `TurnCreated` in `TurnSubmitter`. In `create_turn`, keep upload validation, duration probing, and conversion, then call `submit_normalized_wav` instead of constructing the context inline.

- [ ] **Step 4: Verify API regression coverage and direct submission**

Run: `cd backend && .venv/bin/python -m pytest tests/test_turn_submitter.py tests/test_turn_api.py -q`

Expected: PASS; browser upload still returns 202 and a second active turn still returns 409.

- [ ] **Step 5: Commit submission extraction**

```bash
git add backend/app/turns/submitter.py backend/app/main.py backend/tests/test_turn_submitter.py backend/tests/test_turn_api.py
git commit -m "refactor: share normalized audio turn submission"
```

### Task 4: Add optional local headset playback to ordered TTS

**Files:**
- Create: `backend/app/audio/playback.py`
- Modify: `backend/app/turns/orchestrator.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_playback.py`
- Test: `backend/tests/test_orchestrator.py`

**Interfaces:**
- Produces `LocalAudioPlayer.play_wav(path: Path) -> None`.
- `TurnOrchestrator` accepts `local_player: LocalAudioPlayer | None`.
- Consumes each verified TTS chunk path once, in its existing `sequence` order.

- [ ] **Step 1: Write failing playback and ordering tests**

```python
@pytest.mark.asyncio
async def test_player_calls_aplay_with_configured_device(tmp_path, fake_create_process):
    path = tmp_path / "chunk.wav"
    path.write_bytes(b"RIFF")
    player = LocalAudioPlayer("plughw:2,0", create_process=fake_create_process)
    await player.play_wav(path)
    assert fake_create_process.calls == [["aplay", "-q", "-D", "plughw:2,0", str(path)]]

@pytest.mark.asyncio
async def test_orchestrator_plays_chunks_in_sequence(tmp_path, fake_stt, fake_tts):
    player = RecordingPlayer()
    await _orchestrator(tmp_path, fake_stt, fake_tts, player).run(_turn(tmp_path))
    assert player.paths == ["0.wav", "1.wav"]
```

- [ ] **Step 2: Run the playback tests to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_playback.py tests/test_orchestrator.py -q`

Expected: FAIL because `LocalAudioPlayer` and the orchestration hook are absent.

- [ ] **Step 3: Implement the player and ordered hook**

Use `asyncio.create_subprocess_exec("aplay", "-q", "-D", self._device, str(path))`, wait for completion, and raise `RuntimeError("local playback failed")` on non-zero exit. In `_tts_worker`, call `await self._local_player.play_wav(output_path)` after `output_path.is_file()` succeeds and before `audio_ready` is published. Construct the player only when `local_audio_playback` is true.

- [ ] **Step 4: Add failure behavior test**

```python
@pytest.mark.asyncio
async def test_local_playback_failure_emits_safe_turn_failure(tmp_path, fake_stt, fake_tts):
    result = await _orchestrator(tmp_path, fake_stt, fake_tts, FailingPlayer()).run(_turn(tmp_path))
    assert result.metrics.status == "failed"
    assert result.metrics.error_code == "local_playback_failed"
```

- [ ] **Step 5: Run playback and orchestration tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_playback.py tests/test_orchestrator.py -q`

Expected: PASS; no local player is constructed in the default browser mode.

- [ ] **Step 6: Commit local playback**

```bash
git add backend/app/audio/playback.py backend/app/turns/orchestrator.py backend/app/main.py backend/tests/test_playback.py backend/tests/test_orchestrator.py
git commit -m "feat: play TTS chunks on local headset"
```

### Task 5: Add the lifespan-managed RF capture service

**Files:**
- Create: `backend/app/audio/rf_service.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_rf_service.py`

**Interfaces:**
- Produces `RFInputService(settings: Settings, storage: AudioStorage, submitter: TurnSubmitter, capture: CaptureCallable, normalize: AudioConverter)`.
- Produces `start() -> None` and `stop() -> None` for lifespan ownership.
- Consumes a raw `CaptureResult`; creates `storage.turn_dir(turn_id) / "rf-input.raw"` and normalized `stt-input.wav`; then calls `submit_normalized_wav`.

- [ ] **Step 1: Write failing RF service tests**

```python
@pytest.mark.asyncio
async def test_rf_service_submits_one_valid_capture(tmp_path):
    submitter = RecordingSubmitter()
    service = RFInputService(_rf_settings(), _storage(tmp_path), submitter, capture=one_capture, normalize=copy_to_wav)
    await service.run_once()
    assert submitter.calls == ["stt-input.wav"]

@pytest.mark.asyncio
async def test_rf_service_does_not_submit_a_none_capture(tmp_path):
    service = RFInputService(_rf_settings(), _storage(tmp_path), RecordingSubmitter(), capture=no_capture, normalize=copy_to_wav)
    await service.run_once()
    assert service.submitter.calls == []
```

- [ ] **Step 2: Run the RF service tests to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_rf_service.py -q`

Expected: FAIL because `RFInputService` is absent.

- [ ] **Step 3: Implement service loop and graceful shutdown**

Create a task only in `rf_i2s` mode. Apply APE routing before the loop. For every valid capture, allocate a UUID-owned storage directory, normalize raw input using the existing ffmpeg conversion boundary with explicit raw-input parameters, and call the submitter. Catch capture/conversion exceptions, remove only the current unsubmitted directory, and continue. `stop()` cancels and awaits the task; capture primitive cleanup terminates `arecord`.

- [ ] **Step 4: Wire service into FastAPI lifespan**

Start the service after setting `application.state` dependencies and before `yield`; stop it before `turn_manager.shutdown()` in `finally`. Extend startup tool validation in `rf_i2s` mode to require `amixer` and `arecord`; require `aplay` when local playback is enabled.

- [ ] **Step 5: Run lifecycle and RF tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_rf_service.py tests/test_turn_api.py -q`

Expected: PASS; RF mode is fully mocked and browser mode regressions remain green.

- [ ] **Step 6: Commit RF service**

```bash
git add backend/app/audio/rf_service.py backend/app/main.py backend/tests/test_rf_service.py
git commit -m "feat: add RF I2S input service"
```

### Task 6: Expose hardware mode safely in the UI and documentation

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/schemas.py`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/RecorderControls.tsx`
- Test: `backend/tests/test_health.py`
- Test: `frontend/src/App.test.tsx`
- Modify: `README.md`
- Modify: `systemd/orin-voice-assistant.service`

**Interfaces:**
- Health response adds `audio_input_mode: "browser" | "rf_i2s"` and `local_audio_playback: boolean`; it never returns ALSA device identifiers.
- UI receives those fields and hides browser record controls in `rf_i2s` mode while retaining pipeline status, transcript, response, audio queue, and metrics.

- [ ] **Step 1: Write failing health and UI tests**

```python
def test_health_exposes_safe_audio_mode(client):
    payload = client.get("/api/health").json()
    assert payload["audio_input_mode"] == "browser"
    assert "rf_mic_device" not in payload
```

```tsx
it("hides browser recording controls in RF I2S mode", async () => {
  mockHealth({ audio_input_mode: "rf_i2s", local_audio_playback: true });
  render(<App />);
  expect(await screen.findByText("RF/I²S input is active.")).toBeVisible();
  expect(screen.queryByRole("button", { name: /record/i })).toBeNull();
});
```

- [ ] **Step 2: Run the health and UI tests to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_health.py -q && cd ../frontend && npm test -- --run src/App.test.tsx`

Expected: FAIL because the safe audio-mode response and UI state are absent.

- [ ] **Step 3: Implement safe status and UI behavior**

Return only mode and boolean in `/api/health`. Add Turkish UI copy `RF/I²S input is active. Hold PTT to speak.`. Do not remove the existing recorder implementation; conditionally omit its controls in RF mode.

- [ ] **Step 4: Document operator setup and systemd permissions**

Add a README section with exact environment variables, `arecord -l`, `aplay -l`, `amixer -c APE controls`, and a non-destructive capture verification command. Update the service unit with the required `SupplementaryGroups=audio` and document that the deployment user must have access to the ALSA devices.

- [ ] **Step 5: Run final automated validation**

Run: `cd backend && .venv/bin/python -m pytest -q && cd ../frontend && npm test -- --run && npm run build`

Expected: all backend tests pass; frontend tests and production build pass.

- [ ] **Step 6: Commit UI and operator documentation**

```bash
git add backend/app/main.py backend/app/schemas.py frontend/src/types.ts frontend/src/App.tsx frontend/src/components/RecorderControls.tsx backend/tests/test_health.py frontend/src/App.test.tsx README.md systemd/orin-voice-assistant.service
git commit -m "feat: expose RF audio mode in demo UI"
```

### Task 7: Perform target-Orin hardware acceptance

**Files:**
- Modify: `README.md` only if commands differ from verified hardware behavior.

**Interfaces:**
- Consumes the deployed service from Tasks 1–6 and actual RF receiver plus USB headset.
- Produces a date-stamped, non-sensitive acceptance record outside Git containing only pass/fail, safe device/dtype summary, and latency metrics.

- [ ] **Step 1: Verify ALSA devices without changing state**

Run on Orin: `arecord -l && aplay -l && amixer -c APE controls`

Expected: an APE capture card and the target USB playback device are listed; the required I2S2/ADMAIF controls exist.

- [ ] **Step 2: Start in RF mode**

Run on Orin: `AUDIO_INPUT_MODE=rf_i2s LOCAL_AUDIO_PLAYBACK=true SPEAKER_DEVICE=plughw:2,0 systemctl restart orin-voice-assistant`

Expected: service starts with no APE routing or ALSA access error.

- [ ] **Step 3: Verify one PTT interaction**

Hold PTT, speak Turkish for at least one second, release PTT, and inspect the browser UI.

Expected: exactly one transcript/answer sequence appears and synthesized response chunks are audible in the USB headset in order.

- [ ] **Step 4: Verify single-turn and shutdown behavior**

Press PTT again while the first response is active, then stop the service.

Expected: no overlapping second turn starts; service stops without an orphan `arecord` process.

- [ ] **Step 5: Record acceptance without sensitive content**

Record date/time, Git commit, RF mode enabled, pass/fail, upload/STT/LLM/TTS/total metrics, first-audio metric, and chunk count. Do not record audio, transcript, response text, addresses, or device serials.
