import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";

import { VoiceApi } from "./api";
import { OrderedAudioQueue } from "./audioQueue";
import { AudioInputSources } from "./components/AudioInputSources";
import { AudioResponse } from "./components/AudioResponse";
import { Conversation } from "./components/Conversation";
import { HealthBanner } from "./components/HealthBanner";
import { MetricsCard } from "./components/MetricsCard";
import { PipelineStatus } from "./components/PipelineStatus";
import { BrowserRecorder } from "./recorder";
import type {
  AudioChunk,
  IntentPayload,
  RFDiscoveredTurn,
  TurnCreated,
  TurnEvent,
  TurnMetrics,
} from "./types";

export type TurnStage =
  | "idle"
  | "recording"
  | "uploading"
  | "transcribing"
  | "classifying"
  | "generating"
  | "synthesizing"
  | "playing"
  | "complete"
  | "failed";

export interface HealthStatus {
  backend_ready: boolean;
  stt_ready: boolean;
  tts_ready: boolean;
  llm_ready: boolean;
  intent_ready: boolean;
  intent_enabled: boolean;
  audio_input_mode: "browser" | "rf_i2s";
  local_audio_playback: boolean;
}

interface RecorderLike {
  readonly isRecording: boolean;
  readonly isStarting: boolean;
  readonly elapsedSeconds: number;
  start(): Promise<void>;
  stop(): Promise<Blob>;
  dispose(): void;
}

interface VoiceApiLike {
  createTurn(audio: Blob): Promise<TurnCreated>;
  reportPlayback(
    turnId: string,
    sequence: number,
    clientOffsetMs: number,
  ): Promise<void>;
  watchTurn(
    eventsUrl: string,
    options: {
      lastEventId: number;
      onEvent: (event: TurnEvent) => void;
      onEventId?: (eventId: number) => void;
      onProtocolError?: (error: Error) => void;
    },
  ): () => void;
  watchRfTurns(options: {
    lastEventId: number;
    onTurn: (turn: RFDiscoveredTurn) => void;
    onEventId?: (eventId: number) => void;
    onProtocolError?: (error: Error) => void;
  }): () => void;
}

interface AudioQueueLike {
  enqueue(chunk: AudioChunk): void;
  resume(): Promise<void>;
  dispose(): void;
}

export interface AudioQueueFactoryOptions {
  turnStartedAt: number;
  reportPlayback: (
    sequence: number,
    clientOffsetMs: number,
  ) => void | Promise<void>;
  onManualPlayRequired: (required: boolean) => void;
}

export interface AppDependencies {
  api?: VoiceApiLike;
  recorderFactory?: () => RecorderLike;
  audioQueueFactory?: (
    options: AudioQueueFactoryOptions,
  ) => AudioQueueLike;
  loadHealth?: () => Promise<HealthStatus>;
  now?: () => number;
}

interface AppProps {
  dependencies?: AppDependencies;
}

interface TurnState {
  stage: TurnStage;
  lastOperationalStage: TurnStage;
  currentTurnId: string | null;
  transcript: string;
  intent: IntentPayload | null;
  answer: string;
  error: string | null;
  metrics: TurnMetrics | null;
  manualPlayRequired: boolean;
}

type TurnAction =
  | { type: "recording_started" }
  | { type: "recording_cancelled" }
  | { type: "upload_started" }
  | { type: "turn_created"; turnId: string }
  | { type: "rf_turn_created"; turnId: string }
  | { type: "turn_event"; event: TurnEvent }
  | {
      type: "playback_started";
      turnId: string;
      clientOffsetMs: number;
    }
  | { type: "manual_play_required"; required: boolean }
  | { type: "local_failure"; message: string };

const INITIAL_HEALTH: HealthStatus = {
  backend_ready: false,
  stt_ready: false,
  tts_ready: false,
  llm_ready: false,
  intent_ready: false,
  intent_enabled: false,
  audio_input_mode: "browser",
  local_audio_playback: false,
};

const INITIAL_TURN: TurnState = {
  stage: "idle",
  lastOperationalStage: "idle",
  currentTurnId: null,
  transcript: "",
  intent: null,
  answer: "",
  error: null,
  metrics: null,
  manualPlayRequired: false,
};

function turnReducer(state: TurnState, action: TurnAction): TurnState {
  switch (action.type) {
    case "recording_started":
      return {
        ...INITIAL_TURN,
        stage: "recording",
        lastOperationalStage: "recording",
      };
    case "recording_cancelled":
      return INITIAL_TURN;
    case "upload_started":
      return {
        ...state,
        stage: "uploading",
        lastOperationalStage: "uploading",
      };
    case "turn_created":
      return { ...state, currentTurnId: action.turnId };
    case "rf_turn_created":
      return {
        ...INITIAL_TURN,
        stage: "uploading",
        lastOperationalStage: "uploading",
        currentTurnId: action.turnId,
      };
    case "playback_started":
      if (
        action.turnId !== state.currentTurnId ||
        typeof state.metrics?.first_audio_started_ms === "number"
      ) {
        return state;
      }
      return {
        ...state,
        metrics: {
          ...state.metrics,
          first_audio_started_ms: action.clientOffsetMs,
        },
      };
    case "manual_play_required":
      return { ...state, manualPlayRequired: action.required };
    case "local_failure":
      return {
        ...state,
        stage: "failed",
        error: action.message,
      };
    case "turn_event": {
      if (action.event.payload.turn_id !== state.currentTurnId) {
        return state;
      }
      switch (action.event.type) {
        case "state":
          return {
            ...state,
            stage: action.event.payload.stage,
            lastOperationalStage: action.event.payload.stage,
          };
        case "transcript":
          return { ...state, transcript: action.event.payload.text };
        case "intent":
          return { ...state, intent: action.event.payload };
        case "answer_delta":
          return { ...state, answer: action.event.payload.answer };
        case "audio_ready":
          return {
            ...state,
            stage: "playing",
            lastOperationalStage: "playing",
          };
        case "metrics":
          return {
            ...state,
            metrics: mergePlaybackMetric(
              state.metrics,
              action.event.payload.metrics,
            ),
          };
        case "complete":
          return {
            ...state,
            stage: "complete",
            transcript: action.event.payload.transcript,
            answer: action.event.payload.answer,
            metrics: mergePlaybackMetric(
              state.metrics,
              action.event.payload.metrics,
            ),
            error: null,
          };
        case "failed":
          return {
            ...state,
            stage: "failed",
            error: action.event.payload.error.message,
            metrics: mergePlaybackMetric(
              state.metrics,
              action.event.payload.metrics,
            ),
          };
      }
    }
  }
}

function mergePlaybackMetric(
  current: TurnMetrics | null,
  incoming: TurnMetrics | null,
): TurnMetrics | null {
  if (incoming === null) {
    return current;
  }
  return {
    ...incoming,
    first_audio_started_ms:
      incoming.first_audio_started_ms ??
      current?.first_audio_started_ms ??
      null,
  };
}

async function loadHealthFromApi(): Promise<HealthStatus> {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error("health request failed");
  }
  const payload = (await response.json()) as {
    stt_ready?: unknown;
    tts_ready?: unknown;
    llm_ready?: unknown;
    intent_ready?: unknown;
    intent_enabled?: unknown;
    audio_input_mode?: unknown;
    local_audio_playback?: unknown;
  };
  return {
    backend_ready: true,
    stt_ready: payload.stt_ready === true,
    tts_ready: payload.tts_ready === true,
    llm_ready: payload.llm_ready === true,
    intent_ready: payload.intent_ready === true,
    intent_enabled: payload.intent_enabled === true,
    audio_input_mode: payload.audio_input_mode === "rf_i2s" ? "rf_i2s" : "browser",
    local_audio_playback: payload.local_audio_playback === true,
  };
}

function safeErrorMessage(error: unknown): string {
  return error instanceof Error && error.message.length > 0
    ? error.message
    : "Bağlantı hatası, tekrar deneyin.";
}

export default function App({
  dependencies,
}: AppProps): React.JSX.Element {
  const [resolved] = useState(() => ({
    api: dependencies?.api ?? new VoiceApi(),
    recorderFactory:
      dependencies?.recorderFactory ?? (() => new BrowserRecorder()),
    audioQueueFactory:
      dependencies?.audioQueueFactory ??
      ((options: AudioQueueFactoryOptions) =>
        new OrderedAudioQueue(options)),
    loadHealth: dependencies?.loadHealth ?? loadHealthFromApi,
    now: dependencies?.now ?? (() => performance.now()),
  }));
  const [state, dispatch] = useReducer(turnReducer, INITIAL_TURN);
  const [health, setHealth] = useState(INITIAL_HEALTH);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const recorderRef = useRef<RecorderLike>();
  const audioQueueRef = useRef<AudioQueueLike>();
  const unsubscribeRef = useRef<(() => void) | undefined>();
  const rfUnsubscribeRef = useRef<(() => void) | undefined>();
  const lastEventIdRef = useRef(0);
  const rfDiscoveryEventIdRef = useRef(0);
  const stoppingRef = useRef(false);
  const mountedRef = useRef(true);
  const recordingAttemptRef = useRef(0);
  const activeTurnIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void resolved
      .loadHealth()
      .then((nextHealth) => {
        if (!cancelled) {
          setHealth(nextHealth);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealth(INITIAL_HEALTH);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCheckingHealth(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [resolved]);

  useEffect(() => {
    if (state.stage !== "recording") {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setElapsedSeconds(recorderRef.current?.elapsedSeconds ?? 0);
    }, 250);
    return () => window.clearInterval(timer);
  }, [state.stage]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      recordingAttemptRef.current += 1;
      activeTurnIdRef.current = null;
      const stopRfDiscovery = rfUnsubscribeRef.current;
      rfUnsubscribeRef.current = undefined;
      stopRfDiscovery?.();
      const stopTurnEvents = unsubscribeRef.current;
      unsubscribeRef.current = undefined;
      stopTurnEvents?.();
      audioQueueRef.current?.dispose();
      recorderRef.current?.dispose();
      recorderRef.current = undefined;
    };
  }, []);

  const attachTurn = useCallback(
    (
      created: TurnCreated,
      options: {
        playInBrowser: boolean;
        resetState: boolean;
        turnStartedAt: number;
      },
    ): void => {
      unsubscribeRef.current?.();
      unsubscribeRef.current = undefined;
      audioQueueRef.current?.dispose();
      audioQueueRef.current = undefined;
      lastEventIdRef.current = 0;
      activeTurnIdRef.current = created.turn_id;
      dispatch({
        type: options.resetState ? "rf_turn_created" : "turn_created",
        turnId: created.turn_id,
      });

      if (options.playInBrowser) {
        audioQueueRef.current = resolved.audioQueueFactory({
          turnStartedAt: options.turnStartedAt,
          reportPlayback: (sequence, clientOffsetMs) => {
            if (
              !mountedRef.current ||
              activeTurnIdRef.current !== created.turn_id
            ) {
              return;
            }
            dispatch({
              type: "playback_started",
              turnId: created.turn_id,
              clientOffsetMs,
            });
            return resolved.api.reportPlayback(
              created.turn_id,
              sequence,
              clientOffsetMs,
            );
          },
          onManualPlayRequired: (required) => {
            if (mountedRef.current) {
              dispatch({ type: "manual_play_required", required });
            }
          },
        });
      }

      unsubscribeRef.current = resolved.api.watchTurn(
        created.events_url,
        {
          lastEventId: lastEventIdRef.current,
          onEventId: (eventId) => {
            lastEventIdRef.current = eventId;
          },
          onEvent: (incomingEvent) => {
            if (
              !mountedRef.current ||
              incomingEvent.payload.turn_id !== created.turn_id
            ) {
              return;
            }
            dispatch({ type: "turn_event", event: incomingEvent });
            if (
              options.playInBrowser &&
              incomingEvent.type === "audio_ready"
            ) {
              audioQueueRef.current?.enqueue({
                sequence: incomingEvent.payload.sequence,
                url: incomingEvent.payload.audio_url,
              });
            }
          },
          onProtocolError: (error) => {
            unsubscribeRef.current?.();
            if (mountedRef.current) {
              dispatch({
                type: "local_failure",
                message: safeErrorMessage(error),
              });
            }
          },
        },
      );
    },
    [resolved],
  );

  useEffect(() => {
    if (health.audio_input_mode !== "rf_i2s") {
      return undefined;
    }

    rfUnsubscribeRef.current?.();
    rfUnsubscribeRef.current = resolved.api.watchRfTurns({
      lastEventId: rfDiscoveryEventIdRef.current,
      onEventId: (eventId) => {
        rfDiscoveryEventIdRef.current = eventId;
      },
      onTurn: (created) => {
        if (!mountedRef.current) {
          return;
        }
        attachTurn(created, {
          playInBrowser: false,
          resetState: true,
          turnStartedAt: resolved.now(),
        });
      },
      onProtocolError: (error) => {
        if (mountedRef.current) {
          dispatch({
            type: "local_failure",
            message: safeErrorMessage(error),
          });
        }
      },
    });

    return () => {
      const stopRfDiscovery = rfUnsubscribeRef.current;
      rfUnsubscribeRef.current = undefined;
      stopRfDiscovery?.();
      const stopTurnEvents = unsubscribeRef.current;
      unsubscribeRef.current = undefined;
      stopTurnEvents?.();
      activeTurnIdRef.current = null;
    };
  }, [attachTurn, health.audio_input_mode, resolved]);

  const handleStart = (): void => {
    unsubscribeRef.current?.();
    unsubscribeRef.current = undefined;
    audioQueueRef.current?.dispose();
    audioQueueRef.current = undefined;
    lastEventIdRef.current = 0;
    activeTurnIdRef.current = null;
    stoppingRef.current = false;
    setElapsedSeconds(0);

    const recorder = resolved.recorderFactory();
    const attempt = recordingAttemptRef.current + 1;
    recordingAttemptRef.current = attempt;
    recorderRef.current = recorder;
    dispatch({ type: "recording_started" });
    void recorder.start().catch((error: unknown) => {
      if (
        mountedRef.current &&
        recordingAttemptRef.current === attempt
      ) {
        dispatch({ type: "local_failure", message: safeErrorMessage(error) });
      }
    });
  };

  const handleStop = (): void => {
    if (stoppingRef.current || recorderRef.current === undefined) {
      return;
    }
    if (recorderRef.current.isStarting) {
      stoppingRef.current = true;
      recordingAttemptRef.current += 1;
      recorderRef.current.dispose();
      recorderRef.current = undefined;
      dispatch({ type: "recording_cancelled" });
      stoppingRef.current = false;
      return;
    }
    stoppingRef.current = true;
    dispatch({ type: "upload_started" });
    const turnStartedAt = resolved.now();

    void recorderRef.current
      .stop()
      .then((audio) => resolved.api.createTurn(audio))
      .then((created) => {
        if (!mountedRef.current) {
          return;
        }
        attachTurn(created, {
          playInBrowser: true,
          resetState: false,
          turnStartedAt,
        });
      })
      .catch((error: unknown) => {
        if (mountedRef.current) {
          dispatch({
            type: "local_failure",
            message: safeErrorMessage(error),
          });
        }
      })
      .finally(() => {
        stoppingRef.current = false;
      });
  };

  const handleManualPlay = (): void => {
    void audioQueueRef.current?.resume();
  };

  return (
    <main className="min-h-dvh bg-slate-100 px-4 py-6 text-slate-950 sm:px-6 lg:py-10">
      <div className="mx-auto max-w-6xl">
        <header className="mb-6 border-b border-slate-300 pb-6">
          <p className="font-mono text-xs font-semibold text-cyan-800">
            ORIN EDGE / CANLI TUR
          </p>
          <h1 className="font-display mt-2 max-w-3xl text-balance text-3xl font-bold text-slate-950 sm:text-4xl">
            Türkçe ses hattı
          </h1>
          <p className="mt-3 max-w-2xl text-pretty text-base leading-7 text-slate-600">
            Tek bir ses turunu kayıttan oynatmaya kadar izleyin; her aşamanın
            gecikmesini aynı panelde okuyun.
          </p>
        </header>

        <HealthBanner health={health} checking={checkingHealth} />

        <div className="mt-4 grid gap-4 md:grid-cols-12">
          <div className="grid gap-4 md:col-span-8">
            <PipelineStatus
              intent={state.intent}
              lastOperationalStage={state.lastOperationalStage}
              stage={state.stage}
            />
            <Conversation
              answer={state.answer}
              transcript={state.transcript}
            />
          </div>

          <div className="grid content-start gap-4 md:col-span-4">
            <AudioInputSources
              elapsedSeconds={elapsedSeconds}
              error={state.error}
              mode={health.audio_input_mode}
              onStart={handleStart}
              onStop={handleStop}
              stage={state.stage}
            />
            <AudioResponse
              manualPlayRequired={state.manualPlayRequired}
              onManualPlay={handleManualPlay}
              stage={state.stage}
            />
            <MetricsCard metrics={state.metrics} />
          </div>
        </div>
      </div>
    </main>
  );
}
