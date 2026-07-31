import type { AudioChunk } from "./types";

interface OrderedAudioQueueOptions {
  turnStartedAt: number;
  reportPlayback: (
    sequence: number,
    clientOffsetMs: number,
  ) => void | Promise<void>;
  createAudio?: () => HTMLAudioElement;
  now?: () => number;
  onManualPlayRequired?: (required: boolean) => void;
}

export class OrderedAudioQueue {
  private readonly pending = new Map<number, AudioChunk>();
  private readonly audio: HTMLAudioElement;
  private readonly turnStartedAt: number;
  private readonly reportPlayback: OrderedAudioQueueOptions["reportPlayback"];
  private readonly now: () => number;
  private readonly onManualPlayRequired:
    | ((required: boolean) => void)
    | undefined;
  private nextSequence = 0;
  private current: AudioChunk | undefined;
  private playAttemptInFlight = false;
  private playbackReported = false;
  private disposed = false;
  private manualRequired = false;

  constructor(options: OrderedAudioQueueOptions) {
    this.turnStartedAt = options.turnStartedAt;
    this.reportPlayback = options.reportPlayback;
    this.audio = options.createAudio?.() ?? new Audio();
    this.now = options.now ?? (() => performance.now());
    this.onManualPlayRequired = options.onManualPlayRequired;
    this.audio.addEventListener("ended", this.handleEnded);
  }

  get manualPlayRequired(): boolean {
    return this.manualRequired;
  }

  enqueue(chunk: AudioChunk): void {
    if (!Number.isInteger(chunk.sequence) || chunk.sequence < 0) {
      throw new RangeError("Audio sequence must be a non-negative integer");
    }
    if (
      this.disposed ||
      chunk.sequence < this.nextSequence ||
      this.current?.sequence === chunk.sequence ||
      this.pending.has(chunk.sequence)
    ) {
      return;
    }
    this.pending.set(chunk.sequence, chunk);
    this.playNextIfReady();
  }

  async resume(): Promise<void> {
    if (
      this.disposed ||
      !this.manualRequired ||
      this.current === undefined
    ) {
      return;
    }
    await this.attemptPlay(this.current);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.audio.removeEventListener("ended", this.handleEnded);
    this.audio.pause();
    this.audio.removeAttribute("src");
    this.audio.load();
    this.pending.clear();
    this.current = undefined;
    this.setManualRequired(false);
  }

  private playNextIfReady(): void {
    if (
      this.disposed ||
      this.current !== undefined ||
      this.playAttemptInFlight
    ) {
      return;
    }
    const next = this.pending.get(this.nextSequence);
    if (next === undefined) {
      return;
    }
    this.current = next;
    this.audio.src = next.url;
    void this.attemptPlay(next);
  }

  private async attemptPlay(chunk: AudioChunk): Promise<void> {
    if (this.playAttemptInFlight || this.disposed) {
      return;
    }
    this.playAttemptInFlight = true;
    try {
      await this.audio.play();
      if (this.disposed || this.current?.sequence !== chunk.sequence) {
        return;
      }
      this.setManualRequired(false);
      this.reportFirstPlayback(chunk.sequence);
    } catch {
      if (!this.disposed && this.current?.sequence === chunk.sequence) {
        this.setManualRequired(true);
      }
    } finally {
      this.playAttemptInFlight = false;
      if (this.current === undefined) {
        this.playNextIfReady();
      }
    }
  }

  private reportFirstPlayback(sequence: number): void {
    if (this.playbackReported) {
      return;
    }
    this.playbackReported = true;
    const clientOffsetMs = Math.max(0, this.now() - this.turnStartedAt);
    try {
      void Promise.resolve(
        this.reportPlayback(sequence, clientOffsetMs),
      ).catch(() => undefined);
    } catch {
      // Playback must continue even when optional telemetry cannot be sent.
    }
  }

  private readonly handleEnded = (): void => {
    if (this.disposed || this.current === undefined) {
      return;
    }
    this.pending.delete(this.current.sequence);
    this.nextSequence = this.current.sequence + 1;
    this.current = undefined;
    this.setManualRequired(false);
    this.playNextIfReady();
  };

  private setManualRequired(required: boolean): void {
    if (this.manualRequired === required) {
      return;
    }
    this.manualRequired = required;
    this.onManualPlayRequired?.(required);
  }
}
