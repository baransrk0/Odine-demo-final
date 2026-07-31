const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
] as const;

interface BrowserRecorderOptions {
  maxDurationSeconds?: number;
  now?: () => number;
}

export class RecordingCancelledError extends Error {
  constructor() {
    super("Kayıt iptal edildi.");
    this.name = "RecordingCancelledError";
  }
}

export class BrowserRecorder {
  private readonly maxDurationSeconds: number;
  private readonly now: () => number;
  private starting = false;
  private cancelled = false;
  private recorder: MediaRecorder | undefined;
  private stream: MediaStream | undefined;
  private chunks: Blob[] = [];
  private startedAt: number | undefined;
  private elapsedAtStop = 0;
  private stopTimer: number | undefined;
  private result: Promise<Blob> | undefined;
  private resolveResult: ((blob: Blob) => void) | undefined;
  private rejectResult: ((error: Error) => void) | undefined;
  private mimeType = "";

  constructor(options: BrowserRecorderOptions = {}) {
    this.maxDurationSeconds = options.maxDurationSeconds ?? 30;
    if (
      !Number.isFinite(this.maxDurationSeconds) ||
      this.maxDurationSeconds <= 0
    ) {
      throw new RangeError("maxDurationSeconds must be positive");
    }
    this.now = options.now ?? (() => performance.now());
  }

  get isRecording(): boolean {
    return this.recorder?.state === "recording";
  }

  get isStarting(): boolean {
    return this.starting;
  }

  get elapsedSeconds(): number {
    if (this.startedAt === undefined || !this.isRecording) {
      return this.elapsedAtStop;
    }
    return Math.min(
      Math.floor(Math.max(0, this.now() - this.startedAt) / 1_000),
      Math.floor(this.maxDurationSeconds),
    );
  }

  async start(): Promise<void> {
    if (this.cancelled) {
      throw new RecordingCancelledError();
    }
    if (this.starting || this.recorder !== undefined) {
      throw new Error("Kayıt zaten başlatıldı.");
    }
    this.starting = true;

    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (error) {
        if (this.cancelled) {
          throw new RecordingCancelledError();
        }
        if (isPermissionDenied(error)) {
          throw new Error("Mikrofon izni gerekli.");
        }
        throw new Error("Mikrofon başlatılamadı.");
      }
      if (this.cancelled) {
        stopTracks(stream);
        throw new RecordingCancelledError();
      }

      this.stream = stream;
      this.chunks = [];
      this.elapsedAtStop = 0;

      try {
        this.mimeType = selectSupportedMimeType();
        this.recorder = this.mimeType
          ? new MediaRecorder(stream, { mimeType: this.mimeType })
          : new MediaRecorder(stream);
        this.result = new Promise<Blob>((resolve, reject) => {
          this.resolveResult = resolve;
          this.rejectResult = reject;
        });
        void this.result.catch(() => undefined);
        this.recorder.addEventListener("dataavailable", this.handleData);
        this.recorder.addEventListener("stop", this.handleStop, {
          once: true,
        });
        this.recorder.addEventListener("error", this.handleError, {
          once: true,
        });
        this.startedAt = this.now();
        this.recorder.start();
        this.stopTimer = window.setTimeout(
          () => void this.stop(),
          this.maxDurationSeconds * 1_000,
        );
      } catch (error) {
        this.releaseTracks();
        this.resetFailedStart();
        throw error instanceof Error
          ? error
          : new Error("Mikrofon başlatılamadı.");
      }
    } finally {
      this.starting = false;
    }
  }

  stop(): Promise<Blob> {
    if (this.starting && this.result === undefined) {
      this.cancelled = true;
      return Promise.reject(new RecordingCancelledError());
    }
    if (this.result === undefined) {
      return Promise.reject(new Error("Kayıt başlatılmadı."));
    }

    this.clearStopTimer();
    if (this.recorder !== undefined && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch (error) {
        this.finishWithError(error);
      }
    }
    return this.result;
  }

  dispose(): void {
    if (this.cancelled) {
      return;
    }
    this.cancelled = true;
    this.clearStopTimer();

    const activeRecorder = this.recorder;
    this.detachRecorder();
    if (
      activeRecorder !== undefined &&
      activeRecorder.state !== "inactive"
    ) {
      try {
        activeRecorder.stop();
      } catch {
        // Disposal owns cleanup and intentionally ignores recorder shutdown errors.
      }
    }
    this.releaseTracks();
    this.rejectResult?.(new RecordingCancelledError());
    this.result = undefined;
    this.resolveResult = undefined;
    this.rejectResult = undefined;
    this.startedAt = undefined;
  }

  private readonly handleData = (event: Event): void => {
    const data = (event as BlobEvent).data;
    if (data.size > 0) {
      this.chunks.push(data);
    }
  };

  private readonly handleStop = (): void => {
    this.captureElapsed();
    const blob = new Blob(this.chunks, {
      type: this.mimeType || this.recorder?.mimeType || "audio/webm",
    });
    this.clearStopTimer();
    this.releaseTracks();
    this.detachRecorder();
    this.resolveResult?.(blob);
  };

  private readonly handleError = (event: Event): void => {
    const recorderError = (event as ErrorEvent).error;
    this.finishWithError(
      recorderError instanceof Error
        ? recorderError
        : new Error("Ses kaydı tamamlanamadı."),
    );
  };

  private finishWithError(error: unknown): void {
    this.captureElapsed();
    this.clearStopTimer();
    this.releaseTracks();
    this.detachRecorder();
    this.rejectResult?.(
      error instanceof Error ? error : new Error("Ses kaydı tamamlanamadı."),
    );
  }

  private captureElapsed(): void {
    if (this.startedAt !== undefined) {
      this.elapsedAtStop = Math.min(
        Math.floor(Math.max(0, this.now() - this.startedAt) / 1_000),
        Math.floor(this.maxDurationSeconds),
      );
    }
  }

  private clearStopTimer(): void {
    if (this.stopTimer !== undefined) {
      window.clearTimeout(this.stopTimer);
      this.stopTimer = undefined;
    }
  }

  private releaseTracks(): void {
    for (const track of this.stream?.getTracks() ?? []) {
      track.stop();
    }
    this.stream = undefined;
  }

  private detachRecorder(): void {
    this.recorder?.removeEventListener("dataavailable", this.handleData);
    this.recorder?.removeEventListener("stop", this.handleStop);
    this.recorder?.removeEventListener("error", this.handleError);
    this.recorder = undefined;
  }

  private resetFailedStart(): void {
    this.detachRecorder();
    this.result = undefined;
    this.resolveResult = undefined;
    this.rejectResult = undefined;
    this.startedAt = undefined;
  }
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

export function selectSupportedMimeType(): string {
  return (
    MIME_CANDIDATES.find((mimeType) =>
      MediaRecorder.isTypeSupported(mimeType),
    ) ?? ""
  );
}

function isPermissionDenied(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === "NotAllowedError" || error.name === "SecurityError")
  );
}
