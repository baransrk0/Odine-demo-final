import { describe, expect, it, vi } from "vitest";

import { OrderedAudioQueue } from "./audioQueue";

class FakeAudio extends EventTarget {
  src = "";
  playedUrls: string[] = [];
  pause = vi.fn();
  removeAttribute = vi.fn();
  load = vi.fn();
  playImplementation: () => Promise<void> = async () => {};

  play(): Promise<void> {
    this.playedUrls.push(this.src);
    return this.playImplementation();
  }

  end(): void {
    this.dispatchEvent(new Event("ended"));
  }
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

describe("OrderedAudioQueue", () => {
  it("plays chunks arriving 1, 0, 2 only in sequence 0, 1, 2", async () => {
    const audio = new FakeAudio();
    const queue = new OrderedAudioQueue({
      turnStartedAt: 0,
      reportPlayback: vi.fn(),
      createAudio: () => audio as unknown as HTMLAudioElement,
    });

    queue.enqueue({ sequence: 1, url: "/1.wav" });
    expect(audio.playedUrls).toEqual([]);
    queue.enqueue({ sequence: 0, url: "/0.wav" });
    queue.enqueue({ sequence: 2, url: "/2.wav" });
    await flushPromises();
    expect(audio.playedUrls).toEqual(["/0.wav"]);

    audio.end();
    await flushPromises();
    expect(audio.playedUrls).toEqual(["/0.wav", "/1.wav"]);

    audio.end();
    await flushPromises();
    expect(audio.playedUrls).toEqual(["/0.wav", "/1.wav", "/2.wav"]);
  });

  it("keeps a rejected autoplay chunk loaded and exposes manual resume", async () => {
    const audio = new FakeAudio();
    const manualStates: boolean[] = [];
    audio.playImplementation = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new DOMException("Blocked", "NotAllowedError"))
      .mockResolvedValueOnce();
    const queue = new OrderedAudioQueue({
      turnStartedAt: 0,
      reportPlayback: vi.fn(),
      createAudio: () => audio as unknown as HTMLAudioElement,
      onManualPlayRequired: (required) => manualStates.push(required),
    });

    queue.enqueue({ sequence: 0, url: "/0.wav" });
    await flushPromises();

    expect(queue.manualPlayRequired).toBe(true);
    expect(audio.src).toBe("/0.wav");
    expect(manualStates).toEqual([true]);

    await queue.resume();

    expect(queue.manualPlayRequired).toBe(false);
    expect(audio.src).toBe("/0.wav");
    expect(audio.playedUrls).toEqual(["/0.wav", "/0.wav"]);
    expect(manualStates).toEqual([true, false]);
  });

  it("advances only when the current audio emits ended", async () => {
    const audio = new FakeAudio();
    const queue = new OrderedAudioQueue({
      turnStartedAt: 0,
      reportPlayback: vi.fn(),
      createAudio: () => audio as unknown as HTMLAudioElement,
    });
    queue.enqueue({ sequence: 0, url: "/0.wav" });
    queue.enqueue({ sequence: 1, url: "/1.wav" });
    await flushPromises();

    expect(audio.playedUrls).toEqual(["/0.wav"]);
    audio.end();
    await flushPromises();

    expect(audio.playedUrls).toEqual(["/0.wav", "/1.wav"]);
  });

  it("reports the first successful playback once with client turn offset", async () => {
    const audio = new FakeAudio();
    const reportPlayback = vi.fn(async () => {});
    let now = 1_250;
    const queue = new OrderedAudioQueue({
      turnStartedAt: 1_000,
      reportPlayback,
      createAudio: () => audio as unknown as HTMLAudioElement,
      now: () => now,
    });

    queue.enqueue({ sequence: 0, url: "/0.wav" });
    await flushPromises();
    now = 2_000;
    queue.enqueue({ sequence: 1, url: "/1.wav" });
    audio.end();
    await flushPromises();

    expect(reportPlayback).toHaveBeenCalledOnce();
    expect(reportPlayback).toHaveBeenCalledWith(0, 250);
  });

  it("removes its ended listener when disposed", async () => {
    const audio = new FakeAudio();
    const queue = new OrderedAudioQueue({
      turnStartedAt: 0,
      reportPlayback: vi.fn(),
      createAudio: () => audio as unknown as HTMLAudioElement,
    });
    queue.enqueue({ sequence: 0, url: "/0.wav" });
    queue.enqueue({ sequence: 1, url: "/1.wav" });
    await flushPromises();

    queue.dispose();
    audio.end();
    await flushPromises();

    expect(audio.playedUrls).toEqual(["/0.wav"]);
    expect(audio.pause).toHaveBeenCalledOnce();
  });
});
