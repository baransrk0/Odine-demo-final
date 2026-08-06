import type { TurnStage } from "../App";
import { cn } from "../lib/cn";
import { RecorderControls } from "./RecorderControls";

interface AudioInputSourcesProps {
  mode: "browser" | "rf_i2s";
  stage: TurnStage;
  elapsedSeconds: number;
  error: string | null;
  listening?: boolean;
  onStart: () => void;
  onStop: () => void;
}

interface SourceStatusProps {
  label: string;
  tone: "active" | "waiting" | "disabled";
}

function SourceStatus({
  label,
  tone,
}: SourceStatusProps): React.JSX.Element {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs font-semibold",
        tone === "active" && "bg-emerald-50 text-emerald-800",
        tone === "waiting" && "bg-amber-50 text-amber-800",
        tone === "disabled" && "bg-slate-100 text-slate-600",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "size-2 rounded-full",
          tone === "active" && "bg-emerald-600",
          tone === "waiting" && "bg-amber-500",
          tone === "disabled" && "bg-slate-400",
        )}
      />
      {label}
    </span>
  );
}

export function AudioInputSources({
  mode,
  stage,
  elapsedSeconds,
  error,
  listening = false,
  onStart,
  onStop,
}: AudioInputSourcesProps): React.JSX.Element {
  const rfActive = mode === "rf_i2s";
  const deviceLabel = rfActive
    ? listening
      ? "Dinleniyor"
      : "PTT bekleniyor"
    : "Donanım bekleniyor";

  return (
    <div
      aria-label="Ses girişleri"
      className="grid gap-3"
      role="region"
    >
      {rfActive ? (
        <section
          aria-labelledby="browser-source-heading"
          className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase text-cyan-800">
                Tarayıcı
              </p>
              <h2
                className="mt-1 text-balance text-lg font-semibold text-slate-950"
                id="browser-source-heading"
              >
                Bilgisayar mikrofonu
              </h2>
            </div>
            <div
              aria-label="Bilgisayar mikrofonu durumu"
              role="status"
            >
              <SourceStatus label="Bu kurulumda devre dışı" tone="disabled" />
            </div>
          </div>
          <p className="mt-4 text-pretty text-sm text-slate-600">
            Ses girişi Orin üzerindeki RF/I²S cihazından otomatik alınır.
          </p>
        </section>
      ) : (
        <RecorderControls
          elapsedSeconds={elapsedSeconds}
          error={error}
          onStart={onStart}
          onStop={onStop}
          stage={stage}
        />
      )}

      <section
        aria-labelledby="device-source-heading"
        className={cn(
          "rounded-xl border p-5 shadow-sm",
          rfActive
            ? "border-cyan-200 bg-cyan-50"
            : "border-slate-200 bg-white",
        )}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase text-cyan-800">
              RF/I²S
            </p>
            <h2
              className="mt-1 text-balance text-lg font-semibold text-slate-950"
              id="device-source-heading"
            >
              Cihaz mikrofonu
            </h2>
          </div>
          <div aria-label="Cihaz mikrofonu durumu" role="status">
            <SourceStatus
              label={deviceLabel}
              tone={rfActive ? (listening ? "active" : "waiting") : "waiting"}
            />
          </div>
        </div>
        <p className="mt-4 text-pretty text-sm text-slate-600">
          {rfActive
            ? "PTT düğmesine basılı tutup konuşun. Yanıt USB kulaklıkta çalınır."
            : "RF alıcı bağlanıp backend cihaz modunda başlatıldığında PTT ile otomatik kayıt yapılır."}
        </p>
        {rfActive ? (
          <div
            aria-label="Dinleme pini durumu"
            className="mt-3 flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2"
            role="status"
          >
            <span className="text-sm font-medium text-slate-700">
              Dinleme pini (GPIO)
            </span>
            <span
              className={cn(
                "inline-flex items-center gap-2 font-mono text-sm font-bold",
                listening ? "text-emerald-700" : "text-slate-500",
              )}
              data-testid="listening-pin"
            >
              <span
                aria-hidden="true"
                className={cn(
                  "size-2 rounded-full",
                  listening ? "bg-emerald-600" : "bg-slate-400",
                )}
              />
              {listening ? "HIGH" : "LOW"}
            </span>
          </div>
        ) : null}
        {rfActive && error ? (
          <p
            className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-pretty text-sm text-red-800"
            role="alert"
          >
            {error}
          </p>
        ) : null}
      </section>
    </div>
  );
}
