interface ConversationProps {
  transcript: string;
  answer: string;
}

export function Conversation({
  transcript,
  answer,
}: ConversationProps): React.JSX.Element {
  return (
    <section
      aria-labelledby="conversation-heading"
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2
          id="conversation-heading"
          className="text-balance text-lg font-semibold text-slate-950"
        >
          Konuşma
        </h2>
        <span className="font-mono text-xs text-slate-500">CANLI METİN</span>
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <section
          aria-label="Kullanıcı transkripti"
          aria-live="polite"
          className="min-h-36 rounded-lg border border-slate-200 bg-slate-50 p-4"
        >
          <h3 className="text-balance text-sm font-semibold text-slate-700">
            Siz
          </h3>
          <p className="mt-3 text-pretty text-base leading-7 text-slate-800">
            {transcript || "Kaydı başlatın; çözümlenen konuşma burada görünecek."}
          </p>
        </section>
        <section
          aria-label="Asistan yanıtı"
          aria-live="polite"
          className="min-h-36 rounded-lg border border-cyan-100 bg-cyan-50/50 p-4"
        >
          <h3 className="text-balance text-sm font-semibold text-cyan-900">
            Asistan
          </h3>
          <p className="mt-3 text-pretty text-base leading-7 text-slate-900">
            {answer || "Bir tur başlatın; üretilen yanıt burada birikecek."}
          </p>
        </section>
      </div>
    </section>
  );
}
