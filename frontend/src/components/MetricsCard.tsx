import type { TurnMetrics } from "../types";

interface MetricsCardProps {
  metrics: TurnMetrics | null;
}

interface MetricRow {
  label: string;
  value: number | null | undefined;
}

const TURKISH_NUMBER = new Intl.NumberFormat("tr-TR", {
  maximumFractionDigits: 2,
});

function formatMilliseconds(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value).toLocaleString("tr-TR")} ms`
    : "—";
}

function formatTokenCount(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value).toLocaleString("tr-TR")
    : "—";
}

export function MetricsCard({
  metrics,
}: MetricsCardProps): React.JSX.Element {
  const timingRows: MetricRow[] = [
    { label: "Yükleme", value: metrics?.upload_ms },
    { label: "STT", value: metrics?.stt_ms },
    { label: "Niyet", value: metrics?.intent_ms },
    { label: "LLM", value: metrics?.llm_ms },
    { label: "TTS", value: metrics?.tts_ms },
    { label: "Toplam", value: metrics?.total_ms },
    {
      label: "İlk cümle hazır",
      value: metrics?.first_sentence_ready_ms,
    },
    {
      label: "İlk oynatma",
      value: metrics?.first_audio_started_ms,
    },
  ];
  const tokenRate = metrics?.llm_tokens_per_second;
  const hasTokenRate =
    typeof tokenRate === "number" && Number.isFinite(tokenRate);

  return (
    <section
      aria-labelledby="metrics-heading"
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2
          id="metrics-heading"
          className="text-balance text-sm font-semibold text-slate-900"
        >
          Tur gecikmesi
        </h2>
        <span className="font-mono text-xs text-slate-500">ms</span>
      </div>

      <dl className="mt-4 divide-y divide-slate-100">
        {timingRows.map((row) => (
          <div
            className="flex items-center justify-between gap-4 py-2 text-sm"
            key={row.label}
          >
            <dt className="text-slate-600">{row.label}</dt>
            <dd className="font-mono font-semibold tabular-nums text-slate-900">
              {formatMilliseconds(row.value)}
            </dd>
          </div>
        ))}
      </dl>

      <div className="mt-5 border-t border-slate-200 pt-4">
        <h3 className="text-balance text-xs font-semibold uppercase text-slate-500">
          Token telemetrisi
        </h3>
        <dl className="mt-2 divide-y divide-slate-100">
          <div className="flex items-center justify-between gap-4 py-2 text-sm">
            <dt className="text-slate-600">İstem tokenı</dt>
            <dd className="font-mono font-semibold tabular-nums text-slate-900">
              {formatTokenCount(metrics?.llm_prompt_tokens)}
            </dd>
          </div>
          <div className="flex items-center justify-between gap-4 py-2 text-sm">
            <dt className="text-slate-600">Yanıt tokenı</dt>
            <dd className="font-mono font-semibold tabular-nums text-slate-900">
              {formatTokenCount(metrics?.llm_completion_tokens)}
            </dd>
          </div>
          {hasTokenRate ? (
            <div className="flex items-center justify-between gap-4 py-2 text-sm">
              <dt className="text-slate-600">Token hızı</dt>
              <dd className="font-mono font-semibold tabular-nums text-slate-900">
                {TURKISH_NUMBER.format(tokenRate)} token/sn
              </dd>
            </div>
          ) : null}
        </dl>
      </div>
    </section>
  );
}
