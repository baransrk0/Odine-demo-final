import type { TurnStage } from "../App";

interface AudioResponseProps {
  stage: TurnStage;
  manualPlayRequired: boolean;
  onManualPlay: () => void;
}

function audioStatus(stage: TurnStage): string {
  if (stage === "playing") {
    return "Ses otomatik oynatılıyor.";
  }
  if (stage === "complete") {
    return "Ses yanıtı tamamlandı.";
  }
  if (stage === "failed") {
    return "Ses yanıtı kullanılamıyor.";
  }
  if (stage === "synthesizing") {
    return "İlk ses parçası hazırlanıyor.";
  }
  return "Yanıt üretildiğinde ses otomatik oynatılır.";
}

export function AudioResponse({
  stage,
  manualPlayRequired,
  onManualPlay,
}: AudioResponseProps): React.JSX.Element {
  return (
    <section
      aria-labelledby="audio-heading"
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <h2
        id="audio-heading"
        className="text-balance text-sm font-semibold text-slate-900"
      >
        Ses yanıtı
      </h2>
      {manualPlayRequired ? (
        <div
          className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3"
          role="alert"
        >
          <p className="text-pretty text-sm font-medium text-amber-900">
            Oynat düğmesine basın.
          </p>
          <button
            className="mt-3 rounded-md bg-amber-800 px-3 py-2 text-sm font-semibold text-white hover:bg-amber-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-800"
            onClick={onManualPlay}
            type="button"
          >
            Sesi oynat
          </button>
        </div>
      ) : (
        <p
          aria-live="polite"
          className="mt-3 text-pretty text-sm text-slate-600"
        >
          {audioStatus(stage)}
        </p>
      )}
    </section>
  );
}
