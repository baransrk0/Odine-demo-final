import type {
  ApiErrorBody,
  RFDiscoveredTurn,
  Stage,
  TurnCreated,
  TurnEvent,
  TurnEventType,
  TurnMetrics,
} from "./types";

const NETWORK_ERROR_MESSAGE = "Bağlantı hatası, tekrar deneyin.";
const TURN_STAGES = new Set<Stage>([
  "uploading",
  "transcribing",
  "classifying",
  "generating",
  "synthesizing",
  "playing",
  "complete",
  "failed",
]);
const EVENT_TYPES: readonly TurnEventType[] = [
  "state",
  "transcript",
  "intent",
  "answer_delta",
  "audio_ready",
  "metrics",
  "complete",
  "failed",
];

type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

interface VoiceApiOptions {
  fetcher?: Fetcher;
  eventSourceFactory?: (url: string) => EventSource;
}

interface WatchTurnOptions {
  lastEventId: number;
  onEvent: (event: TurnEvent) => void;
  onEventId?: (eventId: number) => void;
  onProtocolError?: (error: Error) => void;
}

interface WatchRfTurnsOptions {
  lastEventId: number;
  onTurn: (turn: RFDiscoveredTurn) => void;
  onEventId?: (eventId: number) => void;
  onProtocolError?: (error: Error) => void;
}

export class VoiceApiError extends Error {
  constructor(
    message: string,
    readonly code?: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "VoiceApiError";
  }
}

export class VoiceApi {
  private readonly fetcher: Fetcher;
  private readonly eventSourceFactory: (url: string) => EventSource;

  constructor(options: VoiceApiOptions = {}) {
    this.fetcher = options.fetcher ?? fetch.bind(globalThis);
    this.eventSourceFactory =
      options.eventSourceFactory ?? ((url) => new EventSource(url));
  }

  async createTurn(audio: Blob): Promise<TurnCreated> {
    const body = new FormData();
    body.append("audio", audio, recordingFilename(audio.type));

    return this.requestJson<TurnCreated>("/api/turns", {
      method: "POST",
      body,
    });
  }

  async reportPlayback(
    turnId: string,
    sequence: number,
    clientOffsetMs: number,
  ): Promise<void> {
    await this.request(`/api/turns/${encodeURIComponent(turnId)}/playback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sequence,
        client_offset_ms: clientOffsetMs,
      }),
    });
  }

  watchTurn(eventsUrl: string, options: WatchTurnOptions): () => void {
    const source = this.eventSourceFactory(eventsUrl);
    const listeners: Array<{
      type: TurnEventType;
      listener: EventListener;
    }> = [];
    let highestEventId = Math.max(0, options.lastEventId);
    let closed = false;

    const close = () => {
      if (closed) {
        return;
      }
      closed = true;
      for (const { type, listener } of listeners) {
        source.removeEventListener(type, listener);
      }
      source.close();
    };

    for (const type of EVENT_TYPES) {
      const listener: EventListener = (rawEvent) => {
        if (closed || !(rawEvent instanceof MessageEvent)) {
          return;
        }
        const isTerminal = type === "complete" || type === "failed";

        let event: TurnEvent;
        try {
          event = parseTurnEvent(type, rawEvent.data);
        } catch {
          if (isTerminal) {
            close();
          }
          options.onProtocolError?.(
            new VoiceApiError(NETWORK_ERROR_MESSAGE),
          );
          return;
        }

        if (isTerminal) {
          close();
        }
        if (event.payload.event_id <= highestEventId) {
          return;
        }
        highestEventId = event.payload.event_id;
        options.onEventId?.(highestEventId);
        options.onEvent(event);
      };
      listeners.push({ type, listener });
      source.addEventListener(type, listener);
    }

    return close;
  }

  watchRfTurns(options: WatchRfTurnsOptions): () => void {
    const source = this.eventSourceFactory("/api/rf/turns/events");
    let highestEventId = Math.max(0, options.lastEventId);
    let closed = false;

    const listener: EventListener = (rawEvent) => {
      if (closed || !(rawEvent instanceof MessageEvent)) {
        return;
      }

      let turn: RFDiscoveredTurn;
      try {
        turn = parseRfDiscoveredTurn(rawEvent.data);
      } catch {
        options.onProtocolError?.(
          new VoiceApiError(NETWORK_ERROR_MESSAGE),
        );
        return;
      }
      if (turn.event_id <= highestEventId) {
        return;
      }
      highestEventId = turn.event_id;
      options.onEventId?.(highestEventId);
      options.onTurn(turn);
    };

    source.addEventListener("turn_created", listener);
    return () => {
      if (closed) {
        return;
      }
      closed = true;
      source.removeEventListener("turn_created", listener);
      source.close();
    };
  }

  private async requestJson<T>(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<T> {
    const response = await this.request(input, init);
    try {
      return (await response.json()) as T;
    } catch {
      throw new VoiceApiError(NETWORK_ERROR_MESSAGE);
    }
  }

  private async request(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    let response: Response;
    try {
      response = await this.fetcher(input, init);
    } catch (error) {
      if (error instanceof VoiceApiError) {
        throw error;
      }
      throw new VoiceApiError(NETWORK_ERROR_MESSAGE);
    }

    if (response.ok) {
      return response;
    }

    let body: Partial<ApiErrorBody> | undefined;
    try {
      body = (await response.json()) as Partial<ApiErrorBody>;
    } catch {
      body = undefined;
    }
    throw new VoiceApiError(
      typeof body?.message === "string" && body.message.length > 0
        ? body.message
        : NETWORK_ERROR_MESSAGE,
      typeof body?.code === "string" ? body.code : undefined,
      response.status,
    );
  }
}

function parseRfDiscoveredTurn(
  serializedPayload: unknown,
): RFDiscoveredTurn {
  const parsed =
    typeof serializedPayload === "string"
      ? (JSON.parse(serializedPayload) as unknown)
      : serializedPayload;
  const payload = requireRecord(parsed, "RF discovery payload");
  return {
    event_id: requirePositiveInteger(payload.event_id, "event_id"),
    turn_id: requireString(payload.turn_id, "turn_id"),
    events_url: requireString(payload.events_url, "events_url"),
  };
}

export function parseTurnEvent(
  type: TurnEventType,
  serializedPayload: unknown,
): TurnEvent {
  const parsed =
    typeof serializedPayload === "string"
      ? (JSON.parse(serializedPayload) as unknown)
      : serializedPayload;
  const payload = requireRecord(parsed, "SSE payload");
  const common = {
    turn_id: requireString(payload.turn_id, "turn_id"),
    event_id: requirePositiveInteger(payload.event_id, "event_id"),
  };

  switch (type) {
    case "state":
      return {
        type,
        payload: {
          ...common,
          stage: requireStage(payload.stage),
        },
      };
    case "transcript":
      return {
        type,
        payload: {
          ...common,
          text: requireString(payload.text, "text"),
        },
      };
    case "intent":
      return {
        type,
        payload: {
          ...common,
          label: requireString(payload.label, "label"),
          agent: requireString(payload.agent, "agent"),
          source: requireString(payload.source, "source"),
          confidence:
            typeof payload.confidence === "number" ? payload.confidence : null,
          function_call: payload.function_call === true,
        },
      };
    case "answer_delta":
      return {
        type,
        payload: {
          ...common,
          text: requireString(payload.text, "text"),
          answer: requireString(payload.answer, "answer"),
        },
      };
    case "audio_ready":
      return {
        type,
        payload: {
          ...common,
          sequence: requireNonNegativeInteger(
            payload.sequence,
            "sequence",
          ),
          text: requireString(payload.text, "text"),
          audio_url: requireString(payload.audio_url, "audio_url"),
        },
      };
    case "metrics":
      return {
        type,
        payload: {
          ...common,
          metrics: requireMetrics(payload.metrics),
        },
      };
    case "complete":
      return {
        type,
        payload: {
          ...common,
          transcript: requireString(payload.transcript, "transcript"),
          answer: requireString(payload.answer, "answer"),
          audio_url: requireNullableString(
            payload.audio_url,
            "audio_url",
          ),
          metrics: requireMetrics(payload.metrics),
        },
      };
    case "failed": {
      const error = requireRecord(payload.error, "error");
      return {
        type,
        payload: {
          ...common,
          error: {
            code: requireString(error.code, "error.code"),
            message: requireString(error.message, "error.message"),
          },
          metrics: requireOptionalMetrics(payload.metrics),
        },
      };
    }
  }
}

function recordingFilename(mimeType: string): string {
  if (mimeType.includes("ogg")) {
    return "recording.ogg";
  }
  return "recording.webm";
}

function requireRecord(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${field} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, field: string): string {
  if (typeof value !== "string") {
    throw new TypeError(`${field} must be a string`);
  }
  return value;
}

function requireNullableString(
  value: unknown,
  field: string,
): string | null {
  if (value !== null && typeof value !== "string") {
    throw new TypeError(`${field} must be a string or null`);
  }
  return value;
}

function requirePositiveInteger(
  value: unknown,
  field: string,
): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new TypeError(`${field} must be a positive integer`);
  }
  return value as number;
}

function requireNonNegativeInteger(
  value: unknown,
  field: string,
): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new TypeError(`${field} must be a non-negative integer`);
  }
  return value as number;
}

function requireStage(value: unknown): Stage {
  if (typeof value !== "string" || !TURN_STAGES.has(value as Stage)) {
    throw new TypeError("stage must be a supported turn stage");
  }
  return value as Stage;
}

function requireMetrics(value: unknown): TurnMetrics {
  return requireRecord(value, "metrics") as TurnMetrics;
}

function requireOptionalMetrics(value: unknown): TurnMetrics | null {
  return value == null ? null : requireMetrics(value);
}
