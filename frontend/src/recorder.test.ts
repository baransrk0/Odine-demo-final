import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BrowserRecorder } from "./recorder";

class FakeMediaRecorder extends EventTarget {
  static supported = new Set<string>();
  static last: FakeMediaRecorder | undefined;
  static instances: FakeMediaRecorder[] = [];

  static isTypeSupported(mimeType: string): boolean {
    return FakeMediaRecorder.supported.has(mimeType);
  }

  state: RecordingState = "inactive";

  constructor(
    readonly stream: MediaStream,
    readonly options?: MediaRecorderOptions,
  ) {
    super();
    FakeMediaRecorder.last = this;
    FakeMediaRecorder.instances.push(this);
  }

  start(): void {
    this.state = "recording";
  }

  stop(): void {
    if (this.state === "inactive") {
      throw new DOMException("Already inactive", "InvalidStateError");
    }
    this.state = "inactive";
    const dataEvent = new Event("dataavailable");
    Object.defineProperty(dataEvent, "data", {
      value: new Blob(["recorded"], {
        type: this.options?.mimeType ?? "",
      }),
    });
    this.dispatchEvent(dataEvent);
    this.dispatchEvent(new Event("stop"));
  }

  fail(error: Error): void {
    this.state = "inactive";
    const errorEvent = new Event("error");
    Object.defineProperty(errorEvent, "error", { value: error });
    this.dispatchEvent(errorEvent);
  }
}

function installMediaDevices(
  getUserMedia: () => Promise<MediaStream>,
): ReturnType<typeof vi.fn<() => Promise<MediaStream>>> {
  const getUserMediaMock = vi.fn(getUserMedia);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: getUserMediaMock },
  });
  return getUserMediaMock;
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

beforeEach(() => {
  FakeMediaRecorder.last = undefined;
  FakeMediaRecorder.instances = [];
  FakeMediaRecorder.supported = new Set();
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("BrowserRecorder", () => {
  it("selects the first supported recorder MIME type in preference order", async () => {
    FakeMediaRecorder.supported = new Set([
      "audio/webm",
      "audio/ogg;codecs=opus",
    ]);
    const track = { stop: vi.fn() };
    installMediaDevices(
      async () =>
        ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    const recorder = new BrowserRecorder();

    await recorder.start();

    expect(FakeMediaRecorder.last?.options?.mimeType).toBe("audio/webm");
    await recorder.stop();
  });

  it("maps microphone permission denial to the required Turkish message", async () => {
    installMediaDevices(async () => {
      throw new DOMException("Denied", "NotAllowedError");
    });
    const recorder = new BrowserRecorder();

    await expect(recorder.start()).rejects.toThrow(
      "Mikrofon izni gerekli.",
    );
  });

  it("automatically stops at the configured duration and releases tracks", async () => {
    vi.useFakeTimers();
    FakeMediaRecorder.supported.add("audio/webm;codecs=opus");
    const track = { stop: vi.fn() };
    installMediaDevices(
      async () =>
        ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    const recorder = new BrowserRecorder({ maxDurationSeconds: 3 });

    await recorder.start();
    await vi.advanceTimersByTimeAsync(2_999);
    expect(recorder.isRecording).toBe(true);

    await vi.advanceTimersByTimeAsync(1);

    expect(recorder.isRecording).toBe(false);
    expect(track.stop).toHaveBeenCalledOnce();
    await expect(recorder.stop()).resolves.toBeInstanceOf(Blob);
  });

  it("exposes elapsed whole seconds while recording", async () => {
    let now = 1_000;
    FakeMediaRecorder.supported.add("audio/webm");
    const track = { stop: vi.fn() };
    installMediaDevices(
      async () =>
        ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    const recorder = new BrowserRecorder({
      maxDurationSeconds: 10,
      now: () => now,
    });

    await recorder.start();
    now = 3_400;

    expect(recorder.elapsedSeconds).toBe(2);
    await recorder.stop();
  });

  it("releases media tracks when the recorder emits an error", async () => {
    FakeMediaRecorder.supported.add("audio/webm");
    const track = { stop: vi.fn() };
    installMediaDevices(
      async () =>
        ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    const recorder = new BrowserRecorder();

    await recorder.start();
    FakeMediaRecorder.last?.fail(new Error("encoder failed"));

    expect(track.stop).toHaveBeenCalledOnce();
    await expect(recorder.stop()).rejects.toThrow("encoder failed");
  });

  it("can start a new recording after the previous recording stops", async () => {
    FakeMediaRecorder.supported.add("audio/webm");
    const firstTrack = { stop: vi.fn() };
    const secondTrack = { stop: vi.fn() };
    const streams = [firstTrack, secondTrack];
    installMediaDevices(
      async () =>
        ({
          getTracks: () => [streams.shift()!],
        }) as unknown as MediaStream,
    );
    const recorder = new BrowserRecorder();

    await recorder.start();
    await recorder.stop();
    await recorder.start();
    await recorder.stop();

    expect(firstTrack.stop).toHaveBeenCalledOnce();
    expect(secondTrack.stop).toHaveBeenCalledOnce();
  });

  it("releases acquired tracks when recorder setup fails", async () => {
    const track = { stop: vi.fn() };
    installMediaDevices(
      async () =>
        ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    vi.stubGlobal(
      "MediaRecorder",
      class {
        static isTypeSupported(): boolean {
          throw new Error("codec probe failed");
        }
      },
    );
    const recorder = new BrowserRecorder();

    await expect(recorder.start()).rejects.toThrow("codec probe failed");
    expect(track.stop).toHaveBeenCalledOnce();
  });

  it("reserves start before microphone permission resolves", async () => {
    FakeMediaRecorder.supported.add("audio/webm");
    const track = { stop: vi.fn() };
    const permission = deferred<MediaStream>();
    const getUserMedia = installMediaDevices(() => permission.promise);
    const recorder = new BrowserRecorder();

    const firstStart = recorder.start();
    const concurrentStart = recorder.start();

    expect(getUserMedia).toHaveBeenCalledOnce();
    permission.resolve(
      ({ getTracks: () => [track] }) as unknown as MediaStream,
    );
    await firstStart;
    await expect(concurrentStart).rejects.toThrow("Kayıt zaten başlatıldı.");

    expect(FakeMediaRecorder.instances).toHaveLength(1);
    await recorder.stop();
    expect(track.stop).toHaveBeenCalledOnce();
  });

  it("cancels a pending permission request when stopped before it resolves", async () => {
    vi.useFakeTimers();
    FakeMediaRecorder.supported.add("audio/webm");
    const track = { stop: vi.fn() };
    const permission = deferred<MediaStream>();
    installMediaDevices(() => permission.promise);
    const recorder = new BrowserRecorder();

    const started = recorder.start().then(
      () => "resolved",
      (error: Error) => error.message,
    );
    const stopped = recorder.stop().then(
      () => "resolved",
      (error: Error) => error.message,
    );
    permission.resolve(
      ({ getTracks: () => [track] }) as unknown as MediaStream,
    );

    expect({
      started: await started,
      stopped: await stopped,
      stoppedTracks: track.stop.mock.calls.length,
      recorderCount: FakeMediaRecorder.instances.length,
      timerCount: vi.getTimerCount(),
    }).toEqual({
      started: "Kayıt iptal edildi.",
      stopped: "Kayıt iptal edildi.",
      stoppedTracks: 1,
      recorderCount: 0,
      timerCount: 0,
    });
  });

  it("disposes a pending permission request before a late stream resolves", async () => {
    vi.useFakeTimers();
    FakeMediaRecorder.supported.add("audio/webm");
    const track = { stop: vi.fn() };
    const permission = deferred<MediaStream>();
    installMediaDevices(() => permission.promise);
    const recorder = new BrowserRecorder();

    const started = recorder.start().then(
      () => "resolved",
      (error: Error) => error.message,
    );
    const dispose = (
      recorder as BrowserRecorder & { dispose?: () => void }
    ).dispose;
    dispose?.call(recorder);
    permission.resolve(
      ({ getTracks: () => [track] }) as unknown as MediaStream,
    );

    expect({
      started: await started,
      stoppedTracks: track.stop.mock.calls.length,
      recorderCount: FakeMediaRecorder.instances.length,
      timerCount: vi.getTimerCount(),
    }).toEqual({
      started: "Kayıt iptal edildi.",
      stoppedTracks: 1,
      recorderCount: 0,
      timerCount: 0,
    });
  });
});
