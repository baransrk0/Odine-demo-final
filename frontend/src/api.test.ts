import { afterEach, describe, expect, it, vi } from "vitest";

import { VoiceApi } from "./api";
import type { TurnEvent } from "./types";

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly listeners = new Map<string, Set<EventListener>>();
  closed = false;

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListener): void {
    const listeners = this.listeners.get(type) ?? new Set<EventListener>();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: EventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.closed = true;
  }

  emit(type: string, payload: object): void {
    const event = new MessageEvent(type, {
      data: JSON.stringify(payload),
    });
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }
}

afterEach(() => {
  FakeEventSource.instances = [];
  vi.restoreAllMocks();
});

describe("VoiceApi", () => {
  it("uploads a recording under the exact multipart field name audio", async () => {
    let submittedBody: BodyInit | null | undefined;
    const fetcher = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) => {
        submittedBody = init?.body;
        return new Response(
          JSON.stringify({
            turn_id: "43bd09f0-e780-4f78-9377-3b80ca8a533a",
            events_url:
              "/api/turns/43bd09f0-e780-4f78-9377-3b80ca8a533a/events",
          }),
          {
            status: 202,
            headers: { "content-type": "application/json" },
          },
        );
      },
    );
    const api = new VoiceApi({ fetcher });

    await api.createTurn(new Blob(["voice"], { type: "audio/webm" }));

    expect(submittedBody).toBeInstanceOf(FormData);
    expect(Array.from((submittedBody as FormData).keys())).toEqual(["audio"]);
  });

  it("turns named SSE messages into a type-discriminated event union", () => {
    const events: TurnEvent[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchTurn("/api/turns/turn-1/events", {
      lastEventId: 0,
      onEvent: (event) => events.push(event),
    });
    const source = FakeEventSource.instances[0];
    source.emit("state", {
      turn_id: "turn-1",
      event_id: 1,
      stage: "transcribing",
    });
    source.emit("audio_ready", {
      turn_id: "turn-1",
      event_id: 2,
      sequence: 0,
      text: "Merhaba.",
      audio_url: "/api/audio/turn-1/0.wav",
    });

    expect(events).toEqual([
      {
        type: "state",
        payload: {
          turn_id: "turn-1",
          event_id: 1,
          stage: "transcribing",
        },
      },
      {
        type: "audio_ready",
        payload: {
          turn_id: "turn-1",
          event_id: 2,
          sequence: 0,
          text: "Merhaba.",
          audio_url: "/api/audio/turn-1/0.wav",
        },
      },
    ]);
  });

  it("discovers RF turns once in monotonically increasing event order", () => {
    const turns: Array<{
      event_id: number;
      turn_id: string;
      events_url: string;
    }> = [];
    const storedIds: number[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchRfTurns({
      lastEventId: 4,
      onEventId: (eventId) => storedIds.push(eventId),
      onTurn: (turn) => turns.push(turn),
    });
    const source = FakeEventSource.instances[0];
    source.emit("turn_created", {
      event_id: 4,
      turn_id: "turn-old",
      events_url: "/api/turns/turn-old/events",
    });
    source.emit("turn_created", {
      event_id: 5,
      turn_id: "turn-current",
      events_url: "/api/turns/turn-current/events",
    });
    source.emit("turn_created", {
      event_id: 3,
      turn_id: "turn-stale",
      events_url: "/api/turns/turn-stale/events",
    });

    expect(source.url).toBe("/api/rf/turns/events");
    expect(turns).toEqual([
      {
        event_id: 5,
        turn_id: "turn-current",
        events_url: "/api/turns/turn-current/events",
      },
    ]);
    expect(storedIds).toEqual([5]);
  });

  it("reports malformed RF discovery without closing later valid events", () => {
    const turns: string[] = [];
    const protocolErrors: string[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchRfTurns({
      lastEventId: 0,
      onTurn: (turn) => turns.push(turn.turn_id),
      onProtocolError: (error) => protocolErrors.push(error.message),
    });
    const source = FakeEventSource.instances[0];
    source.emit("turn_created", {
      event_id: 1,
      turn_id: null,
      events_url: "/api/turns/broken/events",
    });
    source.emit("turn_created", {
      event_id: 2,
      turn_id: "turn-valid",
      events_url: "/api/turns/turn-valid/events",
    });

    expect(protocolErrors).toEqual(["Bağlantı hatası, tekrar deneyin."]);
    expect(turns).toEqual(["turn-valid"]);
    expect(source.closed).toBe(false);
  });

  it("ignores replayed event IDs and closes the stream at a terminal event", () => {
    const handledIds: number[] = [];
    const storedIds: number[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    const unsubscribe = api.watchTurn("/events", {
      lastEventId: 4,
      onEvent: (event) => handledIds.push(event.payload.event_id),
      onEventId: (eventId) => storedIds.push(eventId),
    });
    const source = FakeEventSource.instances[0];
    source.emit("state", {
      turn_id: "turn-1",
      event_id: 4,
      stage: "generating",
    });
    source.emit("state", {
      turn_id: "turn-1",
      event_id: 5,
      stage: "synthesizing",
    });
    source.emit("state", {
      turn_id: "turn-1",
      event_id: 3,
      stage: "transcribing",
    });
    source.emit("complete", {
      turn_id: "turn-1",
      event_id: 6,
      transcript: "Merhaba",
      answer: "Merhaba!",
      audio_url: null,
      metrics: {},
    });

    expect(handledIds).toEqual([5, 6]);
    expect(storedIds).toEqual([5, 6]);
    expect(source.closed).toBe(true);

    unsubscribe();
    expect(
      Array.from(source.listeners.values()).every(
        (listeners) => listeners.size === 0,
      ),
    ).toBe(true);
  });

  it("closes the stream when a replayed terminal event is already processed", () => {
    const handledEvents: TurnEvent[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchTurn("/events", {
      lastEventId: 6,
      onEvent: (event) => handledEvents.push(event),
    });
    const source = FakeEventSource.instances[0];
    const replayedTerminal = {
      turn_id: "turn-1",
      event_id: 6,
      transcript: "Merhaba",
      answer: "Merhaba!",
      audio_url: null,
      metrics: {},
    };

    source.emit("complete", replayedTerminal);
    source.emit("complete", replayedTerminal);

    expect(source.closed).toBe(true);
    expect(handledEvents).toEqual([]);
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(
      Array.from(source.listeners.values()).every(
        (listeners) => listeners.size === 0,
      ),
    ).toBe(true);
  });

  it("rejects an unknown SSE stage with a Turkish-safe protocol error", () => {
    const handledEvents: TurnEvent[] = [];
    const protocolErrors: Error[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchTurn("/events", {
      lastEventId: 0,
      onEvent: (event) => handledEvents.push(event),
      onProtocolError: (error) => protocolErrors.push(error),
    });
    FakeEventSource.instances[0].emit("state", {
      turn_id: "turn-1",
      event_id: 1,
      stage: "thinking",
    });

    expect(handledEvents).toEqual([]);
    expect(protocolErrors.map((error) => error.message)).toEqual([
      "Bağlantı hatası, tekrar deneyin.",
    ]);
  });

  it.each([
    {
      invalidField: "code",
      error: { code: 500, message: "İşlem başarısız." },
    },
    {
      invalidField: "message",
      error: { code: "runtime_error", message: null },
    },
  ])(
    "rejects a failed SSE event with a non-string $invalidField",
    ({ error }) => {
      const handledEvents: TurnEvent[] = [];
      const protocolErrors: Error[] = [];
      const api = new VoiceApi({
        eventSourceFactory: (url) =>
          new FakeEventSource(url) as unknown as EventSource,
      });

      api.watchTurn("/events", {
        lastEventId: 0,
        onEvent: (event) => handledEvents.push(event),
        onProtocolError: (protocolError) =>
          protocolErrors.push(protocolError),
      });
      const source = FakeEventSource.instances.at(-1)!;
      source.emit("failed", {
        turn_id: "turn-1",
        event_id: 1,
        error,
        metrics: null,
      });

      expect(handledEvents).toEqual([]);
      expect(protocolErrors.map((protocolError) => protocolError.message))
        .toEqual(["Bağlantı hatası, tekrar deneyin."]);
      expect(source.closed).toBe(true);
    },
  );

  it("keeps only safe failed-event error fields and nullable metrics", () => {
    const handledEvents: TurnEvent[] = [];
    const api = new VoiceApi({
      eventSourceFactory: (url) =>
        new FakeEventSource(url) as unknown as EventSource,
    });

    api.watchTurn("/events", {
      lastEventId: 0,
      onEvent: (event) => handledEvents.push(event),
    });
    FakeEventSource.instances.at(-1)!.emit("failed", {
      turn_id: "turn-1",
      event_id: 1,
      error: {
        code: "runtime_error",
        message: "İşlem tamamlanamadı.",
        internal_detail: "do not expose",
      },
    });

    expect(handledEvents).toEqual([
      {
        type: "failed",
        payload: {
          turn_id: "turn-1",
          event_id: 1,
          error: {
            code: "runtime_error",
            message: "İşlem tamamlanamadı.",
          },
          metrics: null,
        },
      },
    ]);
  });

  it("uses a Turkish server error and falls back for unreadable failures", async () => {
    const serverApi = new VoiceApi({
      fetcher: vi.fn(async () =>
        new Response(
          JSON.stringify({
            code: "turn_in_progress",
            message: "Mevcut yanıt tamamlanıyor.",
          }),
          { status: 409, headers: { "content-type": "application/json" } },
        ),
      ),
    });
    await expect(
      serverApi.createTurn(new Blob(["voice"])),
    ).rejects.toThrow("Mevcut yanıt tamamlanıyor.");

    const brokenApi = new VoiceApi({
      fetcher: vi.fn(async () => {
        throw new TypeError("connection reset");
      }),
    });
    await expect(
      brokenApi.createTurn(new Blob(["voice"])),
    ).rejects.toThrow("Bağlantı hatası, tekrar deneyin.");
  });

  it("reports browser playback using the backend playback contract", async () => {
    const fetcher = vi.fn(async () => new Response(null, { status: 204 }));
    const api = new VoiceApi({ fetcher });

    await api.reportPlayback("turn/1", 0, 321.5);

    expect(fetcher).toHaveBeenCalledWith(
      "/api/turns/turn%2F1/playback",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          sequence: 0,
          client_offset_ms: 321.5,
        }),
      }),
    );
  });
});
