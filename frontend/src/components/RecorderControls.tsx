import { cn } from "../lib/cn";
import type { TurnStage } from "../App";

interface RecorderControlsProps {
  stage: TurnStage;
  elapsedSeconds: number;
  error: string | null;
  onStart: () => void;
  onStop: () => void;
}

const PROCESSING_STAGES = new Set<TurnStage>([
  "uploading",
  "transcribing",
  "generating",
  "synthesizing",
  "playing",
]);

function formatElapsed(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder
    .toString()
    .padStart(2, "0")}`;
}

export function RecorderControls({
  stage,
  elapsedSeconds,
  error,
  onStart,
  onStop,
}: RecorderControlsProps): React.JSX.Element {
  const recording = stage === "recording";
  const disabled = PROCESSING_STAGES.has(stage);
  const describedBy = [
    disabled ? "recording-disabled-reason" : null,
    error ? "recording-error" : null,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section
      aria-labelledby="recorder-heading"
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase text-cyan-800">
            Tarayıcı
          </p>
          <h2
            id="recorder-heading"
            className="mt-1 text-balance text-lg font-semibold text-slate-950"
          >
            Bilgisayar mikrofonu
          </h2>
        </div>
        <div className="flex flex-col items-end gap-2">
          <output
            aria-label="Kayıt süresi"
            className="font-mono text-sm font-semibold tabular-nums text-slate-700"
          >
            {formatElapsed(elapsedSeconds)}
          </output>
          <span
            aria-label="Bilgisayar mikrofonu durumu"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-800"
            role="status"
          >
            <span
              aria-hidden="true"
              className="size-2 rounded-full bg-emerald-600"
            />
            Hazır
          </span>
        </div>
      </div>

      <button
        aria-describedby={describedBy || undefined}
        className={cn(
          "mt-5 w-full rounded-lg px-5 py-3 text-sm font-semibold text-white shadow-sm",
          "focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-600 disabled:shadow-none",
          recording
            ? "bg-slate-900 hover:bg-slate-800 focus-visible:outline-slate-900"
            : "bg-cyan-700 hover:bg-cyan-800 focus-visible:outline-cyan-700",
        )}
        disabled={disabled}
        onClick={recording ? onStop : onStart}
        type="button"
      >
        {recording ? "Kaydı durdur" : "Kaydı başlat"}
      </button>

      {disabled ? (
        <p
          className="mt-3 text-pretty text-sm text-slate-600"
          id="recording-disabled-reason"
        >
          Tur tamamlanana kadar yeni kayıt başlatılamaz.
        </p>
      ) : null}
      {error ? (
        <p
          className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-pretty text-sm text-red-800"
          id="recording-error"
          role="alert"
        >
          {error}
        </p>
      ) : null}
    </section>
  );
}
