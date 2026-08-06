import { StrictMode } from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import App, {
  type AppDependencies,
  type AudioQueueFactoryOptions,
  type HealthStatus,
} from "./App";
import { VoiceApiError } from "./api";
import { BrowserRecorder } from "./recorder";
import type {
  RFDiscoveredTurn,
  TurnEvent,
  TurnMetrics,
} from "./types";

const originalMediaDevices = Object.getOwnPropertyDescriptor(
  navigator,
  "mediaDevices",
);

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function event(
  turnId: string,
  eventId: number,
  value:
    | {
        type: "state";
        stage: "transcribing" | "classifying" | "generating" | "synthesizing";
      }
    | { type: "transcript"; text: string }
    | {
        type: "intent";
        label: string;
        agent: string;
        source: string;
        confidence: number | null;
        function_call: boolean;
      }
    | { type: "answer_delta"; text: string; answer: string }
    | { type: "audio_ready"; sequence: number; text: string; audio_url: string }
    | { type: "metrics"; metrics: TurnMetrics }
    | {
        type: "complete";
        transcript: string;
        answer: string;
        audio_url: string | null;
        metrics: TurnMetrics;
      }
    | {
        type: "failed";
        code: string;
        message: string;
        metrics?: TurnMetrics | null;
      },
): TurnEvent {
  const common = { turn_id: turnId, event_id: eventId };
  switch (value.type) {
    case "state":
      return { type: "state", payload: { ...common, stage: value.stage } };
    case "transcript":
      return { type: "transcript", payload: { ...common, text: value.text } };
    case "intent":
      return {
        type: "intent",
        payload: {
          ...common,
          label: value.label,
          agent: value.agent,
          source: value.source,
          confidence: value.confidence,
          function_call: value.function_call,
        },
      };
    case "answer_delta":
      return {
        type: "answer_delta",
        payload: {
          ...common,
          text: value.text,
          answer: value.answer,
        },
      };
    case "audio_ready":
      return {
        type: "audio_ready",
        payload: {
          ...common,
          sequence: value.sequence,
          text: value.text,
          audio_url: value.audio_url,
        },
      };
    case "metrics":
      return {
        type: "metrics",
        payload: { ...common, metrics: value.metrics },
      };
    case "complete":
      return {
        type: "complete",
        payload: {
          ...common,
          transcript: value.transcript,
          answer: value.answer,
          audio_url: value.audio_url,
          metrics: value.metrics,
        },
      };
    case "failed":
      return {
        type: "failed",
        payload: {
          ...common,
          error: { code: value.code, message: value.message },
          metrics: value.metrics ?? null,
        },
      };
  }
}

function readyHealth(overrides: Partial<HealthStatus> = {}): HealthStatus {
  return {
    backend_ready: true,
    stt_ready: true,
    tts_ready: true,
    llm_ready: true,
    intent_ready: true,
    intent_enabled: true,
    audio_input_mode: "browser",
    local_audio_playback: false,
    ...overrides,
  };
}

class PendingPermissionMediaRecorder extends EventTarget {
  static instances: PendingPermissionMediaRecorder[] = [];

  static isTypeSupported(): boolean {
    return true;
  }

  readonly mimeType = "audio/webm";
  state: RecordingState = "inactive";

  constructor(readonly stream: MediaStream) {
    super();
    PendingPermissionMediaRecorder.instances.push(this);
  }

  start(): void {
    this.state = "recording";
  }

  stop(): void {
    this.state = "inactive";
    this.dispatchEvent(new Event("stop"));
  }
}

function createPendingPermissionHarness() {
  vi.useFakeTimers();
  PendingPermissionMediaRecorder.instances = [];
  vi.stubGlobal("MediaRecorder", PendingPermissionMediaRecorder);
  const permission = deferred<MediaStream>();
  const track = { stop: vi.fn() };
  const getUserMedia = vi.fn(() => permission.promise);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  const recorder = new BrowserRecorder();
  const api = {
    createTurn: vi.fn(async () => ({
      turn_id: "turn-pending",
      events_url: "/api/turns/turn-pending/events",
    })),
    watchTurn: vi.fn(() => vi.fn()),
    watchRfTurns: vi.fn(() => vi.fn()),
    watchListening: vi.fn(() => vi.fn()),
    reportPlayback: vi.fn(async () => {}),
  };
  const rendered = render(
    <App
      dependencies={{
        api,
        recorderFactory: () => recorder,
        loadHealth: async () => readyHealth(),
      }}
    />,
  );

  return {
    ...rendered,
    api,
    permission,
    track,
    resolvePermission: async () => {
      await act(async () => {
        permission.resolve(
          ({ getTracks: () => [track] }) as unknown as MediaStream,
        );
        await Promise.resolve();
        await Promise.resolve();
      });
    },
  };
}

function createHarness(options: {
  health?: HealthStatus;
  turnCreation?: Deferred<{ turn_id: string; events_url: string }>;
  createTurnError?: Error;
  strict?: boolean;
} = {}) {
  let elapsedSeconds = 0;
  let recording = false;
  let watchOptions:
    | Parameters<NonNullable<AppDependencies["api"]>["watchTurn"]>[1]
    | undefined;
  let rfWatchOptions:
    | {
        lastEventId: number;
        onTurn: (turn: RFDiscoveredTurn) => void;
        onEventId?: (eventId: number) => void;
        onProtocolError?: (error: Error) => void;
      }
    | undefined;
  let listeningWatchOptions:
    | {
        lastEventId: number;
        onListening: (listening: boolean) => void;
        onEventId?: (eventId: number) => void;
        onProtocolError?: (error: Error) => void;
      }
    | undefined;
  let queueOptions: AudioQueueFactoryOptions | undefined;
  const turnCreation =
    options.turnCreation ??
    deferred<{ turn_id: string; events_url: string }>();
  if (options.turnCreation === undefined && options.createTurnError === undefined) {
    turnCreation.resolve({
      turn_id: "turn-current",
      events_url: "/api/turns/turn-current/events",
    });
  }

  const recorder = {
    get isRecording() {
      return recording;
    },
    get isStarting() {
      return false;
    },
    get elapsedSeconds() {
      return elapsedSeconds;
    },
    start: vi.fn(async () => {
      recording = true;
    }),
    stop: vi.fn(async () => {
      recording = false;
      return new Blob(["voice"], { type: "audio/webm" });
    }),
    dispose: vi.fn(() => {
      recording = false;
    }),
  };
  const queue = {
    enqueue: vi.fn(),
    resume: vi.fn(async () => {}),
    dispose: vi.fn(),
  };
  const unsubscribe = vi.fn();
  const rfUnsubscribe = vi.fn();
  const listeningUnsubscribe = vi.fn();
  const api = {
    createTurn: options.createTurnError
      ? vi.fn(async () => {
          throw options.createTurnError;
        })
      : vi.fn(() => turnCreation.promise),
    watchTurn: vi.fn(
      (
        _eventsUrl: string,
        incomingOptions: Parameters<
          NonNullable<AppDependencies["api"]>["watchTurn"]
        >[1],
      ) => {
        watchOptions = incomingOptions;
        return unsubscribe;
      },
    ),
    watchRfTurns: vi.fn(
      (
        incomingOptions: NonNullable<typeof rfWatchOptions>,
      ) => {
        rfWatchOptions = incomingOptions;
        return rfUnsubscribe;
      },
    ),
    watchListening: vi.fn(
      (
        incomingOptions: NonNullable<typeof listeningWatchOptions>,
      ) => {
        listeningWatchOptions = incomingOptions;
        return listeningUnsubscribe;
      },
    ),
    reportPlayback: vi.fn(async () => {}),
  };
  const audioQueueFactory = vi.fn((incomingOptions: AudioQueueFactoryOptions) => {
    queueOptions = incomingOptions;
    return queue;
  });
  const dependencies: AppDependencies = {
    api,
    recorderFactory: () => recorder,
    audioQueueFactory,
    loadHealth: vi.fn(async () => options.health ?? readyHealth()),
    now: () => 1_000,
  };
  const app = <App dependencies={dependencies} />;
  const rendered = render(
    options.strict ? <StrictMode>{app}</StrictMode> : app,
  );

  return {
    ...rendered,
    api,
    audioQueueFactory,
    queue,
    recorder,
    turnCreation,
    unsubscribe,
    rfUnsubscribe,
    listeningUnsubscribe,
    emitListening(active: boolean) {
      if (listeningWatchOptions === undefined) {
        throw new Error("Listening watcher has not started");
      }
      act(() => {
        listeningWatchOptions?.onListening(active);
      });
    },
    emit(incomingEvent: TurnEvent) {
      if (watchOptions === undefined) {
        throw new Error("SSE watcher has not started");
      }
      act(() => watchOptions?.onEvent(incomingEvent));
    },
    emitRf(incomingTurn: RFDiscoveredTurn) {
      if (rfWatchOptions === undefined) {
        throw new Error("RF discovery watcher has not started");
      }
      act(() => {
        rfWatchOptions?.onEventId?.(incomingTurn.event_id);
        rfWatchOptions?.onTurn(incomingTurn);
      });
    },
    failRf(error: Error) {
      if (rfWatchOptions === undefined) {
        throw new Error("RF discovery watcher has not started");
      }
      act(() => rfWatchOptions?.onProtocolError?.(error));
    },
    currentTurnWatcher() {
      if (watchOptions === undefined) {
        throw new Error("SSE watcher has not started");
      }
      return watchOptions;
    },
    requireQueueOptions() {
      if (queueOptions === undefined) {
        throw new Error("Audio queue has not been created");
      }
      return queueOptions;
    },
    setElapsed(seconds: number) {
      elapsedSeconds = seconds;
    },
  };
}

async function beginTurn(
  harness: ReturnType<typeof createHarness>,
): Promise<void> {
  fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
  await screen.findByRole("button", { name: "Kaydı durdur" });
  fireEvent.click(screen.getByRole("button", { name: "Kaydı durdur" }));
  await waitFor(() => expect(harness.api.watchTurn).toHaveBeenCalledOnce());
}

beforeEach(() => {
  vi.useRealTimers();
});

afterEach(() => {
  if (originalMediaDevices === undefined) {
    Reflect.deleteProperty(navigator, "mediaDevices");
  } else {
    Object.defineProperty(navigator, "mediaDevices", originalMediaDevices);
  }
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Orin Turkish voice demo", () => {
  it("shows browser recording and an honest unavailable RF/I²S source together", async () => {
    createHarness();

    const sources = await screen.findByRole("region", {
      name: "Ses girişleri",
    });
    expect(within(sources).getByText("Bilgisayar mikrofonu")).toBeTruthy();
    expect(within(sources).getByText("Tarayıcı")).toBeTruthy();
    expect(
      within(sources).getByRole("status", {
        name: "Bilgisayar mikrofonu durumu",
      }).textContent,
    ).toContain("Hazır");
    expect(within(sources).getByText("Cihaz mikrofonu")).toBeTruthy();
    expect(within(sources).getByText("RF/I²S")).toBeTruthy();
    expect(
      within(sources).getByRole("status", {
        name: "Cihaz mikrofonu durumu",
      }).textContent,
    ).toContain("Donanım bekleniyor");
    expect(
      within(sources).getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", false);
  });

  it("shows RF/I²S as PTT-driven and disables browser recording in device mode", async () => {
    createHarness({
      health: readyHealth({
        audio_input_mode: "rf_i2s",
        local_audio_playback: true,
      }),
    });

    const sources = await screen.findByRole("region", {
      name: "Ses girişleri",
    });
    expect(within(sources).getByText("Bilgisayar mikrofonu")).toBeTruthy();
    expect(
      within(sources).getByText("Bu kurulumda devre dışı"),
    ).toBeTruthy();
    expect(within(sources).getByText("Cihaz mikrofonu")).toBeTruthy();
    expect(
      within(sources).getByRole("status", {
        name: "Cihaz mikrofonu durumu",
      }).textContent,
    ).toContain("PTT bekleniyor");
    expect(
      within(sources).getByText(
        "PTT düğmesine basılı tutup konuşun. Yanıt USB kulaklıkta çalınır.",
      ),
    ).toBeTruthy();
    expect(
      within(sources).queryByRole("button", { name: "Kaydı başlat" }),
    ).toBeNull();
  });

  it("reflects the backend listening signal on the device status badge", async () => {
    const harness = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");
    expect(harness.api.watchListening).toHaveBeenCalledOnce();

    harness.emitListening(true);
    const status = () =>
      within(
        screen.getByRole("region", { name: "Ses girişleri" }),
      ).getByRole("status", { name: "Cihaz mikrofonu durumu" }).textContent;
    expect(status()).toContain("Dinleniyor");
    expect(screen.getByTestId("listening-pin").textContent).toContain("HIGH");

    harness.emitListening(false);
    expect(status()).toContain("PTT bekleniyor");
    expect(screen.getByTestId("listening-pin").textContent).toContain("LOW");
  });

  it("opens RF discovery only when the backend uses device input", async () => {
    const browser = createHarness();
    await screen.findByRole("button", { name: "Kaydı başlat" });
    expect(browser.api.watchRfTurns).not.toHaveBeenCalled();
    browser.unmount();

    const rf = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");

    expect(rf.api.watchRfTurns).toHaveBeenCalledOnce();
    expect(rf.api.watchRfTurns).toHaveBeenCalledWith(
      expect.objectContaining({ lastEventId: 0 }),
    );
  });

  it("attaches a discovered RF turn to the existing replayable event stream", async () => {
    const harness = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");

    harness.emitRf({
      event_id: 1,
      turn_id: "rf-turn-1",
      events_url: "/api/turns/rf-turn-1/events",
    });
    expect(harness.api.watchTurn).toHaveBeenCalledWith(
      "/api/turns/rf-turn-1/events",
      expect.objectContaining({ lastEventId: 0 }),
    );

    harness.emit(event("rf-turn-1", 1, {
      type: "state",
      stage: "transcribing",
    }));
    harness.emit(event("rf-turn-1", 2, {
      type: "transcript",
      text: "RF üzerinden merhaba",
    }));

    expect(screen.getByText("Konuşma çözümleniyor")).toBeTruthy();
    expect(screen.getByText("RF üzerinden merhaba")).toBeTruthy();
  });

  it("observes RF audio without browser playback or playback telemetry", async () => {
    const harness = createHarness({
      health: readyHealth({
        audio_input_mode: "rf_i2s",
        local_audio_playback: true,
      }),
    });
    await screen.findByText("PTT bekleniyor");
    harness.emitRf({
      event_id: 1,
      turn_id: "rf-turn-1",
      events_url: "/api/turns/rf-turn-1/events",
    });

    harness.emit(event("rf-turn-1", 1, {
      type: "audio_ready",
      sequence: 0,
      text: "Yerel ses.",
      audio_url: "/api/audio/rf-turn-1/0.wav",
    }));

    expect(screen.getByText("Ses oynatılıyor")).toBeTruthy();
    expect(harness.audioQueueFactory).not.toHaveBeenCalled();
    expect(harness.queue.enqueue).not.toHaveBeenCalled();
    expect(harness.api.reportPlayback).not.toHaveBeenCalled();
  });

  it("atomically replaces an RF turn and rejects stale turn events", async () => {
    const harness = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");
    harness.emitRf({
      event_id: 1,
      turn_id: "rf-turn-old",
      events_url: "/api/turns/rf-turn-old/events",
    });
    const staleWatcher = harness.currentTurnWatcher();
    harness.emitRf({
      event_id: 2,
      turn_id: "rf-turn-new",
      events_url: "/api/turns/rf-turn-new/events",
    });

    act(() => staleWatcher.onEvent(event("rf-turn-old", 1, {
      type: "transcript",
      text: "Eski RF turu",
    })));
    harness.emit(event("rf-turn-new", 1, {
      type: "transcript",
      text: "Yeni RF turu",
    }));

    expect(harness.unsubscribe).toHaveBeenCalledOnce();
    expect(screen.queryByText("Eski RF turu")).toBeNull();
    expect(screen.getByText("Yeni RF turu")).toBeTruthy();
  });

  it("closes RF discovery and turn streams on unmount", async () => {
    const harness = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");
    harness.emitRf({
      event_id: 1,
      turn_id: "rf-turn-1",
      events_url: "/api/turns/rf-turn-1/events",
    });

    harness.unmount();

    expect(harness.rfUnsubscribe).toHaveBeenCalledOnce();
    expect(harness.unsubscribe).toHaveBeenCalledOnce();
  });

  it("shows a safe RF discovery protocol error until the next turn", async () => {
    const harness = createHarness({
      health: readyHealth({ audio_input_mode: "rf_i2s" }),
    });
    await screen.findByText("PTT bekleniyor");

    harness.failRf(new Error("Bağlantı hatası, tekrar deneyin."));

    expect(screen.getByRole("alert").textContent).toContain(
      "Bağlantı hatası, tekrar deneyin.",
    );

    harness.emitRf({
      event_id: 1,
      turn_id: "rf-turn-recovered",
      events_url: "/api/turns/rf-turn-recovered/events",
    });
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows backend, STT, intent, TTS, and LLM readiness independently", async () => {
    createHarness({
      health: readyHealth({ tts_ready: false, llm_ready: false }),
    });

    const healthList = await screen.findByRole("list", {
      name: "Bağlantı durumları",
    });
    const health = within(healthList);
    expect(health.getByText("Backend")).toBeTruthy();
    expect(health.getByText("STT")).toBeTruthy();
    expect(health.getByText("Niyet")).toBeTruthy();
    expect(health.getByText("TTS")).toBeTruthy();
    expect(health.getByText("LLM")).toBeTruthy();
    expect(health.getAllByText("Hazır")).toHaveLength(3);
    expect(health.getAllByText("Hazır değil")).toHaveLength(2);
  });

  it("switches start to stop and reports elapsed recording time", async () => {
    vi.useFakeTimers();
    const harness = createHarness();

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    expect(
      screen.getByRole("button", { name: "Kaydı durdur" }),
    ).toBeTruthy();
    harness.setElapsed(7);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });

    expect(screen.getByText("00:07")).toBeTruthy();
    expect(screen.getByText("Kayıt sürüyor")).toBeTruthy();
    expect(harness.recorder.start).toHaveBeenCalledOnce();
  });

  it("uploads once, follows SSE stages, and keeps transcript separate from the answer", async () => {
    const creation = deferred<{ turn_id: string; events_url: string }>();
    const harness = createHarness({ turnCreation: creation });

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    await screen.findByRole("button", { name: "Kaydı durdur" });
    fireEvent.click(screen.getByRole("button", { name: "Kaydı durdur" }));

    expect(await screen.findByText("Kayıt gönderiliyor")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", true);
    creation.resolve({
      turn_id: "turn-current",
      events_url: "/api/turns/turn-current/events",
    });
    await waitFor(() => expect(harness.api.watchTurn).toHaveBeenCalledOnce());

    harness.emit(event("turn-current", 1, { type: "state", stage: "transcribing" }));
    expect(screen.getByText("Konuşma çözümleniyor")).toBeTruthy();
    harness.emit(event("turn-current", 2, {
      type: "transcript",
      text: "Bugün sistem nasıl?",
    }));
    harness.emit(event("turn-current", 3, { type: "state", stage: "generating" }));
    harness.emit(event("turn-current", 4, {
      type: "answer_delta",
      text: "Sistem ",
      answer: "Sistem ",
    }));
    harness.emit(event("turn-current", 5, {
      type: "answer_delta",
      text: "hazır.",
      answer: "Sistem hazır.",
    }));

    expect(screen.getByText("Yanıt üretiliyor")).toBeTruthy();
    expect(
      screen.getByLabelText("Kullanıcı transkripti").textContent,
    ).toContain("Bugün sistem nasıl?");
    expect(
      screen.getByLabelText("Asistan yanıtı").textContent,
    ).toContain("Sistem hazır.");
    expect(harness.api.createTurn).toHaveBeenCalledOnce();
    expect(harness.api.watchTurn).toHaveBeenCalledOnce();
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", true);
  });

  it("rejects stale-turn events and re-enables recording only at terminal stages", async () => {
    const harness = createHarness();
    await beginTurn(harness);

    harness.emit(event("turn-stale", 1, {
      type: "transcript",
      text: "Bu eski turdan geldi.",
    }));
    harness.emit(event("turn-stale", 2, {
      type: "audio_ready",
      sequence: 0,
      text: "Eski ses.",
      audio_url: "/api/audio/turn-stale/0.wav",
    }));
    expect(screen.queryByText("Bu eski turdan geldi.")).toBeNull();
    expect(harness.queue.enqueue).not.toHaveBeenCalled();

    for (const [eventId, stage] of [
      [3, "transcribing"],
      [4, "generating"],
      [5, "synthesizing"],
    ] as const) {
      harness.emit(event("turn-current", eventId, { type: "state", stage }));
      expect(
        screen.getByRole("button", { name: "Kaydı başlat" }),
      ).toHaveProperty("disabled", true);
    }
    harness.emit(event("turn-current", 6, {
      type: "audio_ready",
      sequence: 0,
      text: "Hazır.",
      audio_url: "/api/audio/turn-current/0.wav",
    }));
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", true);
    expect(harness.queue.enqueue).toHaveBeenCalledWith({
      sequence: 0,
      url: "/api/audio/turn-current/0.wav",
    });

    harness.emit(event("turn-current", 7, {
      type: "complete",
      transcript: "Merhaba",
      answer: "Merhaba!",
      audio_url: null,
      metrics: {},
    }));
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", false);
  });

  it.each([
    ["stt_empty", "Konuşma anlaşılamadı, tekrar deneyin."],
    ["llm_failed", "Yanıt üretilemedi, tekrar deneyin."],
    ["tts_failed", "Metin yanıtı hazır, ses üretilemedi."],
  ])("shows the exact Turkish %s pipeline failure", async (code, message) => {
    const harness = createHarness();
    await beginTurn(harness);
    if (code === "tts_failed") {
      harness.emit(event("turn-current", 1, {
        type: "answer_delta",
        text: "Metin korunur.",
        answer: "Metin korunur.",
      }));
    }

    harness.emit(event("turn-current", 2, {
      type: "failed",
      code,
      message,
    }));

    expect(screen.getByRole("alert").textContent).toContain(message);
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", false);
    if (code === "tts_failed") {
      expect(screen.getByLabelText("Asistan yanıtı").textContent).toContain(
        "Metin korunur.",
      );
    }
  });

  it("shows the exact Turkish 409 message beside the recording action", async () => {
    createHarness({
      createTurnError: new VoiceApiError(
        "Mevcut yanıt tamamlanıyor.",
        "turn_in_progress",
        409,
      ),
    });

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    await screen.findByRole("button", { name: "Kaydı durdur" });
    fireEvent.click(screen.getByRole("button", { name: "Kaydı durdur" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "Mevcut yanıt tamamlanıyor.",
    );
    expect(
      screen.getByRole("button", { name: "Kaydı başlat" }),
    ).toHaveProperty("disabled", false);
  });

  it("exposes the exact manual-play prompt when autoplay is rejected", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "audio_ready",
      sequence: 0,
      text: "Dinleyin.",
      audio_url: "/api/audio/turn-current/0.wav",
    }));

    act(() => harness.requireQueueOptions().onManualPlayRequired(true));

    expect(screen.getByRole("alert").textContent).toContain(
      "Oynat düğmesine basın.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Sesi oynat" }));
    expect(harness.queue.resume).toHaveBeenCalledOnce();
  });

  it("shows first playback locally when completion arrives before play succeeds", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "audio_ready",
      sequence: 0,
      text: "Dinleyin.",
      audio_url: "/api/audio/turn-current/0.wav",
    }));
    harness.emit(event("turn-current", 2, {
      type: "complete",
      transcript: "Ölç",
      answer: "Ölçüldü.",
      audio_url: null,
      metrics: { first_audio_started_ms: null },
    }));
    const reportPlayback = harness.requireQueueOptions().reportPlayback;

    expect(screen.queryByText("432 ms")).toBeNull();
    await act(async () => {
      await reportPlayback(0, 432.4);
    });

    expect(screen.getByText("432 ms")).toBeTruthy();
    expect(harness.api.reportPlayback).toHaveBeenCalledOnce();
    expect(harness.api.reportPlayback).toHaveBeenCalledWith(
      "turn-current",
      0,
      432.4,
    );
  });

  it("ignores a playback callback from a stale turn", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    const stalePlayback = harness.requireQueueOptions().reportPlayback;
    harness.emit(event("turn-current", 1, {
      type: "complete",
      transcript: "Eski tur",
      answer: "Eski yanıt",
      audio_url: null,
      metrics: { first_audio_started_ms: null },
    }));

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    await screen.findByRole("button", { name: "Kaydı durdur" });
    await act(async () => {
      await stalePlayback(0, 777);
    });

    expect(harness.api.reportPlayback).not.toHaveBeenCalled();
    expect(screen.queryByText("777 ms")).toBeNull();
  });

  it("renders millisecond metrics, nullable token counts, and numeric token throughput", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "complete",
      transcript: "Ölç",
      answer: "Ölçüldü.",
      audio_url: null,
      metrics: {
        upload_ms: 12.4,
        stt_ms: 345.6,
        llm_ms: 789,
        tts_ms: 234.4,
        total_ms: 1_381.4,
        first_sentence_ready_ms: 932.2,
        first_audio_started_ms: null,
        llm_prompt_tokens: null,
        llm_completion_tokens: null,
        llm_tokens_per_second: 18.75,
      },
    }));

    expect(screen.getByText("12 ms")).toBeTruthy();
    expect(screen.getByText("346 ms")).toBeTruthy();
    expect(screen.getByText("1.381 ms")).toBeTruthy();
    expect(screen.getAllByText("—")).toHaveLength(4);
    expect(screen.getByText("18,75 token/sn")).toBeTruthy();
  });

  it("omits token throughput when it is unavailable", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "metrics",
      metrics: {
        llm_prompt_tokens: null,
        llm_completion_tokens: null,
        llm_tokens_per_second: null,
      },
    }));

    expect(screen.queryByText("Token hızı")).toBeNull();
    expect(screen.getAllByText("—")).toHaveLength(10);
  });

  it("names the routed agent and its confidence", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "intent",
      label: "matematik",
      agent: "matematik",
      source: "classifier",
      confidence: 0.83,
      function_call: false,
    }));

    expect(screen.getByTestId("intent-summary").textContent).toContain(
      "Matematik · %83",
    );
  });

  it("names the classified label when it differs from the agent it routes to", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "intent",
      label: "ilk yardım",
      agent: "medikal",
      source: "classifier",
      confidence: 0.83,
      function_call: false,
    }));

    expect(screen.getByTestId("intent-summary").textContent).toContain(
      "İlk yardım → Medikal · %83",
    );
  });

  it("names a label the agent map never listed", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "intent",
      label: "kbrn korunma",
      agent: "savaş yönergeleri",
      source: "classifier",
      confidence: 0.61,
      function_call: false,
    }));

    expect(screen.getByTestId("intent-summary").textContent).toContain(
      "KBRN korunma → Savaş yönergeleri · %61",
    );
  });

  it("marks a locally answered clock turn instead of implying the model ran", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "intent",
      label: "saat",
      agent: "saat",
      source: "rule",
      confidence: null,
      function_call: true,
    }));

    expect(screen.getByTestId("intent-summary").textContent).toContain(
      "Saat · yerel yanıt",
    );
  });

  it("says when the default agent answered because the classifier was down", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "intent",
      label: "sohbet",
      agent: "sohbet",
      source: "unavailable",
      confidence: null,
      function_call: false,
    }));

    expect(screen.getByTestId("intent-summary").textContent).toContain(
      "Sohbet · sınıflandırıcı yok",
    );
  });

  it("shows the intent stage on the pipeline", async () => {
    const harness = createHarness();
    await beginTurn(harness);
    harness.emit(event("turn-current", 1, {
      type: "state",
      stage: "classifying",
    }));

    expect(screen.getByText("Niyet belirleniyor")).toBeTruthy();
  });

  it("cleans up SSE and audio on unmount", async () => {
    const turnHarness = createHarness();
    await beginTurn(turnHarness);
    turnHarness.unmount();

    expect(turnHarness.unsubscribe).toHaveBeenCalledOnce();
    expect(turnHarness.queue.dispose).toHaveBeenCalledOnce();
  });

  it("cancels a pending microphone start when Stop is pressed", async () => {
    const harness = createPendingPermissionHarness();

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    const stop = screen.getByRole("button", { name: "Kaydı durdur" });
    fireEvent.click(stop);
    await harness.resolvePermission();

    expect({
      error: screen.queryByRole("alert")?.textContent ?? null,
      startDisabled: screen.getByRole("button", {
        name: "Kaydı başlat",
      }).hasAttribute("disabled"),
      stoppedTracks: harness.track.stop.mock.calls.length,
      recorderCount: PendingPermissionMediaRecorder.instances.length,
      timerCount: vi.getTimerCount(),
      uploads: harness.api.createTurn.mock.calls.length,
    }).toEqual({
      error: null,
      startDisabled: false,
      stoppedTracks: 1,
      recorderCount: 0,
      timerCount: 0,
      uploads: 0,
    });
  });

  it("disposes a pending microphone start when unmounted", async () => {
    const harness = createPendingPermissionHarness();

    fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
    harness.unmount();
    await harness.resolvePermission();

    expect({
      stoppedTracks: harness.track.stop.mock.calls.length,
      recorderCount: PendingPermissionMediaRecorder.instances.length,
      timerCount: vi.getTimerCount(),
      uploads: harness.api.createTurn.mock.calls.length,
    }).toEqual({
      stoppedTracks: 1,
      recorderCount: 0,
      timerCount: 0,
      uploads: 0,
    });
  });

  it("remains interactive after React StrictMode replays mount effects", async () => {
    const harness = createHarness({ strict: true });

    await beginTurn(harness);

    expect(harness.api.createTurn).toHaveBeenCalledOnce();
    expect(harness.api.watchTurn).toHaveBeenCalledOnce();
  });
});
