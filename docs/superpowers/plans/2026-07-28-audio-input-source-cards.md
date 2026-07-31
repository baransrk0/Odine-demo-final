# Audio Input Source Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present browser and RF/I²S microphone sources together while keeping only the configured source actionable.

**Architecture:** Add a focused `AudioInputSources` presentation component that composes the existing `RecorderControls`. `App` supplies the backend-reported input mode and existing recorder callbacks; no API or backend behavior changes.

**Tech Stack:** React, TypeScript, Tailwind CSS, Vitest, Testing Library

## Global Constraints

- Preserve the existing visual language and responsive right-column layout.
- Do not claim that RF/I²S hardware is connected when only the APE controller is present.
- Do not add runtime source switching or backend health fields.
- Preserve existing browser recording behavior.

---

### Task 1: Specify source-card behavior

**Files:**
- Modify: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: `HealthStatus.audio_input_mode: "browser" | "rf_i2s"`
- Produces: behavioral expectations for both input modes

- [ ] **Step 1: Write failing browser-mode test**

Assert that `Bilgisayar mikrofonu`, `Tarayıcı`, `Hazır`, `Cihaz mikrofonu`, `RF/I²S`, and `Donanım bekleniyor` are visible while `Kaydı başlat` remains enabled.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: FAIL because the dual-source labels and status do not exist.

- [ ] **Step 3: Write failing RF-mode test**

Assert that RF mode shows both cards, `PTT bekleniyor`, the physical PTT instruction, and no browser recording button.

- [ ] **Step 4: Run the focused test and verify RED**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: FAIL because the dual-source panel does not exist.

### Task 2: Implement the dual-source panel

**Files:**
- Create: `frontend/src/components/AudioInputSources.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `mode`, `stage`, `elapsedSeconds`, `error`, `onStart`, and `onStop`
- Produces: `AudioInputSources` React component

- [ ] **Step 1: Add `AudioInputSources`**

Compose `RecorderControls` for browser mode and render a matching RF/I²S status card. In RF mode, replace the browser action with a disabled deployment notice and mark the device card as waiting for PTT.

- [ ] **Step 2: Integrate it into `App`**

Replace the conditional recorder/RF notice with one `AudioInputSources` instance.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: all `App` tests pass.

### Task 3: Verify the frontend

**Files:**
- No production changes expected

**Interfaces:**
- Consumes: completed frontend implementation
- Produces: fresh test and build evidence

- [ ] **Step 1: Run the full frontend suite**

Run: `cd frontend && npm test -- --run`

Expected: all tests pass.

- [ ] **Step 2: Build the production bundle**

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite build exit successfully.
