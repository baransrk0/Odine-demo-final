import { cn } from "../lib/cn";
import type { HealthStatus } from "../App";

interface HealthBannerProps {
  health: HealthStatus;
  checking: boolean;
}

const INDICATORS = [
  ["Backend", "backend_ready"],
  ["STT", "stt_ready"],
  ["Niyet", "intent_ready"],
  ["TTS", "tts_ready"],
  ["LLM", "llm_ready"],
] as const;

export function HealthBanner({
  health,
  checking,
}: HealthBannerProps): React.JSX.Element {
  return (
    <section
      aria-labelledby="connection-heading"
      className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2
            id="connection-heading"
            className="text-balance text-sm font-semibold text-slate-900"
          >
            Sistem bağlantısı
          </h2>
          <p className="text-pretty text-xs text-slate-500">
            Yerel inference hattının canlı durumu
          </p>
        </div>
        <ul
          aria-label="Bağlantı durumları"
          className="grid grid-cols-2 gap-x-5 gap-y-2 sm:flex sm:items-center sm:gap-5"
        >
          {INDICATORS.map(([label, key]) => {
            const ready = health[key];
            const status = checking
              ? "Kontrol ediliyor"
              : ready
                ? "Hazır"
                : "Hazır değil";
            return (
              <li
                aria-label={`${label}: ${status}`}
                className="flex items-center gap-2"
                key={key}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "size-2 rounded-full",
                    checking
                      ? "bg-slate-300"
                      : ready
                        ? "bg-emerald-600"
                        : "bg-red-600",
                  )}
                />
                <span className="text-xs font-medium text-slate-700">
                  {label}
                </span>
                <span className="text-xs text-slate-500">
                  {status}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
