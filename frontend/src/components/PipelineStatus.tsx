import { cn } from "../lib/cn";
import type { TurnStage } from "../App";
import type { IntentPayload } from "../types";

interface PipelineStatusProps {
  stage: TurnStage;
  lastOperationalStage: TurnStage;
  intent?: IntentPayload | null;
}

const STAGE_LABELS: Record<TurnStage, string> = {
  idle: "Kayda hazır",
  recording: "Kayıt sürüyor",
  uploading: "Kayıt gönderiliyor",
  transcribing: "Konuşma çözümleniyor",
  classifying: "Niyet belirleniyor",
  generating: "Yanıt üretiliyor",
  synthesizing: "Ses hazırlanıyor",
  playing: "Ses oynatılıyor",
  complete: "Tur tamamlandı",
  failed: "Tur tamamlanamadı",
};

// Display overrides for names whose casing plain capitalization gets wrong.
// Deliberately not a complete list of the classes: `display` falls back to
// capitalizing whatever the backend sent, so a new label added to the
// evaluation set shows up here without a frontend change. The map this
// replaced listed every agent by hand and silently hid the six labels the
// `savaş yönergeleri` class was split into.
const DISPLAY_OVERRIDES: Record<string, string> = {
  "kbrn korunma": "KBRN korunma",
};

function display(name: string): string {
  const override = DISPLAY_OVERRIDES[name];
  if (override) {
    return override;
  }
  // Turkish casing: "ilk yardım" must capitalize to "İlk yardım", not "Ilk".
  return name.slice(0, 1).toLocaleUpperCase("tr") + name.slice(1);
}

const STEPS = [
  { label: "Kayıt", stages: ["recording", "uploading"] },
  { label: "STT", stages: ["transcribing"] },
  { label: "Niyet", stages: ["classifying"] },
  { label: "LLM", stages: ["generating"] },
  { label: "TTS", stages: ["synthesizing"] },
  { label: "Oynatma", stages: ["playing"] },
] as const;

// The classified label and the agent it routes to are different things: six
// labels share the `savaş yönergeleri` agent and `ilk yardım` routes to
// `medikal`, so showing the agent alone reports the same five names no matter
// which of the ten classes won. Both are shown, collapsed to one name when the
// label routes to an agent of its own.
function intentRoute(intent: IntentPayload): string {
  const agent = display(intent.agent);
  if (!intent.label || intent.label === intent.agent) {
    return agent;
  }
  return `${display(intent.label)} → ${agent}`;
}

function intentSummary(intent: IntentPayload): string {
  const route = intentRoute(intent);
  if (intent.function_call) {
    return `${route} · yerel yanıt`;
  }
  if (intent.source === "rule") {
    return `${route} · kural`;
  }
  if (intent.source === "classifier") {
    const confidence =
      typeof intent.confidence === "number"
        ? ` · %${Math.round(intent.confidence * 100)}`
        : "";
    return `${route}${confidence}`;
  }
  // low_confidence and unavailable both land on the default agent; saying which
  // one keeps a quiet classifier outage from reading as a confident route.
  return intent.source === "unavailable"
    ? `${route} · sınıflandırıcı yok`
    : `${route} · varsayılan`;
}

function stepIndex(stage: TurnStage): number {
  return STEPS.findIndex((step) =>
    (step.stages as readonly TurnStage[]).includes(stage),
  );
}

export function PipelineStatus({
  stage,
  lastOperationalStage,
  intent = null,
}: PipelineStatusProps): React.JSX.Element {
  const currentIndex =
    stage === "complete" ? STEPS.length : stepIndex(lastOperationalStage);

  return (
    <section
      aria-labelledby="pipeline-heading"
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2
          id="pipeline-heading"
          className="text-balance text-sm font-semibold text-slate-900"
        >
          Canlı sinyal yolu
        </h2>
        <p
          aria-live="polite"
          className={cn(
            "text-right text-sm font-semibold",
            stage === "failed"
              ? "text-red-700"
              : stage === "complete"
                ? "text-emerald-700"
                : "text-cyan-800",
          )}
        >
          {STAGE_LABELS[stage]}
        </p>
      </div>

      {intent ? (
        <p className="mt-2 text-xs text-slate-600" data-testid="intent-summary">
          Niyet: <span className="font-medium">{intentSummary(intent)}</span>
        </p>
      ) : null}

      <ol className="mt-6 grid grid-cols-6" aria-label="Ses işleme aşamaları">
        {STEPS.map((step, index) => {
          const active =
            stage !== "complete" &&
            stage !== "idle" &&
            index === currentIndex;
          const failed = stage === "failed" && index === currentIndex;
          const complete =
            stage === "complete" ||
            (currentIndex >= 0 && index < currentIndex);
          return (
            <li
              className="relative flex min-w-0 flex-col items-center gap-2"
              key={step.label}
            >
              {index < STEPS.length - 1 ? (
                <span
                  aria-hidden="true"
                  className={cn(
                    "absolute left-1/2 top-2 h-px w-full",
                    complete ? "bg-emerald-500" : "bg-slate-200",
                  )}
                />
              ) : null}
              <span
                aria-hidden="true"
                className={cn(
                  "relative z-10 size-4 rounded-full border-2 bg-white",
                  failed
                    ? "border-red-600 bg-red-50"
                    : active
                      ? "border-cyan-700 bg-cyan-100"
                      : complete
                        ? "border-emerald-600 bg-emerald-100"
                        : "border-slate-300",
                )}
              />
              <span
                className={cn(
                  "text-center text-xs font-medium",
                  active
                    ? "text-cyan-800"
                    : failed
                      ? "text-red-700"
                      : complete
                        ? "text-emerald-700"
                        : "text-slate-500",
                )}
              >
                {step.label}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
