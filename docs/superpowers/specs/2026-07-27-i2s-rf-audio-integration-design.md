# I2S RF Audio Integration Design

**Status:** Approved for implementation planning  
**Date:** 2026-07-27  
**Scope:** Add an optional Jetson-local RF/I2S microphone input and USB-headset TTS playback mode to the existing Orin voice-assistant demo.

## Goal

Allow the demo to receive push-to-talk RF audio through the Jetson Orin I2S/APE capture path, run the existing STT → LLM → TTS turn, and play each synthesized response chunk on a USB headset. The browser remains an observability UI for turn state, transcript, response text, generated audio URLs, and metrics.

## Non-goals

- Do not transmit synthesized speech back over RF.
- Do not remove browser microphone upload or browser playback.
- Do not add RAG, device discovery heuristics, GPIO/PTT transmission control, or a second concurrent turn.
- Do not assume the reference hardware values are valid until they are verified on the target Orin.

## Reference Hardware Contract

The added reference demo establishes the intended input path:

```text
RF receiver → Jetson I2S2 → APE ADMAIF1 → ALSA hw:APE,0
```

Its expected capture format is signed 16-bit little-endian PCM, 8 kHz, two channels. The receiver starts the I2S clock while PTT is held and stops it when PTT is released. Therefore, after the first audio frame, a bounded frame-read timeout means that the utterance has ended.

TTS is not sent to RF in the reference demo. It is played on `plughw:2,0`, a USB headset/speaker device. This integration preserves that behavior.

## Runtime Modes

`AUDIO_INPUT_MODE` selects exactly one input producer at process startup:

| Value | Input producer | Browser recording |
|---|---|---|
| `browser` (default) | Existing `POST /api/turns` multipart upload | Enabled |
| `rf_i2s` | Application-owned RF capture service | Disabled in the UI |

The HTTP `POST /api/turns` endpoint remains available in both modes for diagnostics and automated smoke tests, but the RF service is the normal producer in `rf_i2s` mode. The existing `TurnManager` remains the sole admission authority, so browser and RF attempts cannot execute simultaneously.

`LOCAL_AUDIO_PLAYBACK` controls a second, local consumer of TTS chunks. It defaults to `false`; when it is `true`, every successfully synthesized chunk is played sequentially with `aplay -D SPEAKER_DEVICE`. Browser audio URLs and SSE events continue to be generated in all modes.

## Configuration

Add these `Settings` fields, all read from the environment:

```dotenv
AUDIO_INPUT_MODE=browser
LOCAL_AUDIO_PLAYBACK=false
SPEAKER_DEVICE=plughw:2,0
RF_MIC_DEVICE=hw:APE,0
RF_MIC_SAMPLE_RATE=8000
RF_MIC_CHANNELS=2
RF_CAPTURE_MAX_SECONDS=30
RF_PTT_FRAME_TIMEOUT_MS=300
RF_PTT_MIN_SECONDS=0.30
APE_CARD=APE
APE_I2S_PORT=I2S2
```

Validation rules:

- `AUDIO_INPUT_MODE` is either `browser` or `rf_i2s`.
- Device identifiers are non-empty strings.
- Sample rate, channel count, maximum seconds, timeout, and minimum seconds are positive.
- `RF_PTT_MIN_SECONDS <= RF_CAPTURE_MAX_SECONDS`.
- `LOCAL_AUDIO_PLAYBACK=true` requires `SPEAKER_DEVICE`.

## Components and Boundaries

### `app.audio.i2s`

Owns all platform-specific shell boundaries. It applies the APE mixer routing, starts `arecord`, reads raw PCM frames, applies the PTT stream-start/end rule, writes a per-turn raw capture file, and converts it to the existing 16 kHz mono WAV contract. It does not know about STT, LLM, HTTP, SSE, or UI state.

### `app.audio.playback`

Owns local WAV playback. It invokes `aplay -q -D SPEAKER_DEVICE <chunk.wav>` without a shell, waits for completion, and maps a non-zero exit code to a typed playback failure. It never deletes the chunk; storage ownership remains with `AudioStorage`.

### `app.turns.submitter`

Creates and admits a `TurnContext` from a normalized WAV path. Both the HTTP upload endpoint and the RF capture service use this shared boundary. This avoids an HTTP loopback and ensures that the one-active-turn rule, event buffer, metrics, and task ownership are identical for browser and RF turns.

### `app.audio.rf_service`

Runs only in `rf_i2s` mode. It performs capture attempts in a task created during FastAPI lifespan. It ignores pre-PTT/no-frame conditions and recordings shorter than the configured minimum. A valid recording is normalized and submitted through the shared turn submitter. It waits until the current turn releases its lease before capturing the next utterance. On shutdown it terminates `arecord`, cancels its task, and waits for it before the turn manager is shut down.

### `TurnOrchestrator` playback hook

After TTS writes and verifies an output chunk, the worker invokes the optional local player before publishing `audio_ready`. The worker already synthesizes chunks in sequence, so this preserves audible order. A local playback failure makes the turn fail at the synthesizing stage with a distinct safe error code; it does not silently claim that audio was played.

## End-to-End Flow

```text
1. FastAPI starts models and validates ffmpeg/ffprobe.
2. In rf_i2s mode, validate amixer, arecord, and optionally aplay; apply
   “I2S2 codec master mode = cbm-cfm” and “ADMAIF1 Mux = I2S2”.
3. RF capture waits for the first raw PCM frame from hw:APE,0.
4. Once frames begin, a 300 ms frame timeout ends the utterance.
5. Convert 8 kHz stereo S16_LE capture to 16 kHz mono PCM WAV.
6. Submit the WAV directly to TurnManager/TurnOrchestrator.
7. STT, streamed LLM, and ordered TTS behave exactly as in browser mode.
8. For each TTS chunk, write the WAV, play it locally if enabled, then publish
   the existing SSE audio_ready event and URL.
9. The UI displays the turn. In rf_i2s mode it does not offer a browser record
   button, but it can still play the published browser audio URL if desired.
```

## Error Handling

| Failure | Behavior |
|---|---|
| `amixer`, `arecord`, or `aplay` unavailable in the enabled mode | Fail application startup with a clear operator-facing error. |
| APE mixer command fails | Fail application startup; do not capture from an unverified route. |
| No first I2S frame / pre-PTT I/O condition | Wait and retry; do not create a turn. |
| Stream ends before `RF_PTT_MIN_SECONDS` | Discard capture and return to listening. |
| Capture duration exceeds `RF_CAPTURE_MAX_SECONDS` | End the capture, normalize, and submit the bounded recording. |
| Capture or conversion fails after a stream starts | Log safe diagnostics, remove temporary capture files, and resume listening. |
| Active web/RF turn exists | Wait for lease release; never start a competing turn. |
| Local playback exits non-zero | Emit a safe `local_playback_failed` turn failure and preserve standard cleanup. |

## Testing and Validation

Unit tests must use injectable command runners and process doubles; CI must not require Jetson hardware. Cover configuration validation, APE command construction, PTT boundary behavior, too-short capture discard, shutdown, direct turn submission, ordered playback, and playback failure.

The target-Orin acceptance test must run with a connected RF receiver and USB headset:

1. Start with `AUDIO_INPUT_MODE=rf_i2s` and `LOCAL_AUDIO_PLAYBACK=true`.
2. Confirm the application starts only after APE routing succeeds.
3. Hold PTT, speak Turkish for at least one second, and release PTT.
4. Confirm the browser displays transcript and response events.
5. Confirm the response is audible in the USB headset, in sentence order.
6. Confirm a second PTT press while a response is active does not create a second turn.
7. Confirm Ctrl-C/systemd stop terminates capture without orphaning `arecord`.

## Security and Operational Constraints

- Use `asyncio.create_subprocess_exec` or `subprocess.run(..., shell=False)` only.
- Do not log raw PCM, transcript text, device serial numbers, or environment secrets.
- Keep raw capture and normalized input files under the existing per-turn `AudioStorage` root and use the existing retention cleanup.
- Keep the reference folder `demoorin/atbk-ecg` unmodified; it is untracked user-provided reference material.
