# Audio Input Source Cards Design

## Goal

Show both supported audio-input paths without implying that unavailable RF/I²S hardware is ready.

## User experience

The right-hand control column contains one `Ses girişleri` panel with two source cards:

- `Bilgisayar mikrofonu` remains the active browser recording surface and keeps the existing timer, start/stop button, disabled-state explanation, and error message.
- `Cihaz mikrofonu` is labeled `RF/I²S`. In browser mode it shows `Donanım bekleniyor` and explains that physical PTT input will become available when the receiver is connected and the backend is configured.

When the backend is configured with `AUDIO_INPUT_MODE=rf_i2s`, the device card shows `PTT bekleniyor` and the existing PTT instructions. The browser card remains visible but clearly shows that it is disabled for that deployment.

## Scope

- Frontend presentation only.
- No backend health-schema changes.
- No runtime source switching.
- No attempt to infer physical RF hardware presence from the ALSA controller.
- Existing recording, turn submission, pipeline, and playback behavior remain unchanged.

## Testing

Component tests verify the browser-mode and RF-mode source labels, status copy, recording action availability, and RF-mode browser action absence. The existing frontend suite and production build must remain green.
