export type Stage =
  | "uploading"
  | "transcribing"
  | "classifying"
  | "generating"
  | "synthesizing"
  | "playing"
  | "complete"
  | "failed";

export type TurnStatus = "active" | "complete" | "failed";

export interface ApiErrorBody {
  code: string;
  message: string;
}

export interface SafeConfigurationSummary {
  stt_device: string;
  tts_device: string;
  stt_dtype: string;
  tts_dtype: string;
  llm_max_tokens: number;
  stt_timeout_seconds: number;
  llm_timeout_seconds: number;
  tts_timeout_seconds: number;
  tts_queue_capacity: number;
}

export interface TurnMetrics {
  timestamp?: string | null;
  outcome?: TurnStatus | null;
  recording_duration_seconds?: number | null;
  recording_bytes?: number | null;
  recording_content_type?: string | null;
  upload_ms?: number | null;
  stt_ms?: number | null;
  llm_ms?: number | null;
  tts_ms?: number | null;
  total_ms?: number | null;
  first_sentence_ready_ms?: number | null;
  first_audio_started_ms?: number | null;
  sentence_count?: number | null;
  audio_chunk_count?: number | null;
  transcript_chars?: number | null;
  answer_chars?: number | null;
  llm_prompt_tokens?: number | null;
  llm_completion_tokens?: number | null;
  llm_tokens_per_second?: number | null;
  intent_label?: string | null;
  intent_agent?: string | null;
  intent_source?: string | null;
  intent_confidence?: number | null;
  intent_ms?: number | null;
  error_stage?: Stage | null;
  error_code?: string | null;
  timed_out?: boolean;
  device?: string | null;
  dtype?: string | null;
  configuration?: SafeConfigurationSummary | null;
}

export interface TurnCreated {
  turn_id: string;
  events_url: string;
}

export interface RFDiscoveredTurn extends TurnCreated {
  event_id: number;
}

interface EventPayload {
  turn_id: string;
  event_id: number;
}

export interface StatePayload extends EventPayload {
  stage: Stage;
}

export interface TranscriptPayload extends EventPayload {
  text: string;
}

export interface IntentPayload extends EventPayload {
  label: string;
  agent: string;
  source: string;
  confidence: number | null;
  function_call: boolean;
}

export interface AnswerDeltaPayload extends EventPayload {
  text: string;
  answer: string;
}

export interface AudioReadyPayload extends EventPayload {
  sequence: number;
  text: string;
  audio_url: string;
}

export interface MetricsPayload extends EventPayload {
  metrics: TurnMetrics;
}

export interface CompletePayload extends EventPayload {
  transcript: string;
  answer: string;
  audio_url: string | null;
  metrics: TurnMetrics;
}

export interface FailedPayload extends EventPayload {
  error: ApiErrorBody;
  metrics: TurnMetrics | null;
}

export type TurnEvent =
  | { type: "state"; payload: StatePayload }
  | { type: "transcript"; payload: TranscriptPayload }
  | { type: "intent"; payload: IntentPayload }
  | { type: "answer_delta"; payload: AnswerDeltaPayload }
  | { type: "audio_ready"; payload: AudioReadyPayload }
  | { type: "metrics"; payload: MetricsPayload }
  | { type: "complete"; payload: CompletePayload }
  | { type: "failed"; payload: FailedPayload };

export type TurnEventType = TurnEvent["type"];

export interface AudioChunk {
  sequence: number;
  url: string;
}
