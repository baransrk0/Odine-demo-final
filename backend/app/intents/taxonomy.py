"""Intent classes derived from the shipped ATBK evaluation set.

The label set, confidence threshold, and default agent are read from
`atbk_knowledge_base.json` rather than hardcoded, so revising the evaluation set
revises the routing table with it. Each label's routing path is inferred from its
own records: a label whose records are marked `fonksiyon_cagrisi` is answered
locally and never reaches the language model.

`VERBALIZATIONS` and `HYPOTHESIS_TEMPLATE` are the tuning surface for zero-shot
accuracy and deliberately do not live in the evaluation set -- they describe how
the labels are phrased to the NLI model, not what the labels are. Measure changes
to them with `scripts/eval_intent.py` before keeping them.
"""

from collections import Counter
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from app.knowledge import DEFAULT_KNOWLEDGE_BASE


logger = logging.getLogger(__name__)

HYPOTHESIS_TEMPLATE = "Bu metin {} ile ilgilidir."

# Bare labels like "saat" or "sohbet" are weak NLI hypotheses on their own.
# Each entry expands its label into a phrase the model can actually entail.
#
# Every phrase below is drawn from the questions its own label carries in the
# evaluation set, not from what the label name suggests. The label names are
# misleading on their own: "savaş yönergeleri" holds no combat questions, and a
# hypothesis built around "muharebe" described none of its 30 records -- so
# equipment and procedure questions lost to the broader medical phrasing.
VERBALIZATIONS: dict[str, str] = {
    # 30 records, all treating a casualty: turnike, kanama, şok, yanık, kırık,
    # hipotermi, triyaj. Anchored on the wounded person, because an unanchored
    # "tıbbi müdahale" also entails any question about checking equipment.
    "medikal": "yaralı veya hastaya uygulanan ilk yardım ve tıbbi müdahale",
    # 30 records of field procedure: telsiz disiplini, fonetik alfabe, nöbet
    # devir teslimi, kamuflaj, harita ve pusula, mevzi, KBRN alarmı, angajman
    # kuralları, konvoy, ikmal, durum raporu.
    "savaş yönergeleri": "telsiz, nöbet, keşif, mevzi ve KBRN gibi askeri saha talimatları",
    # 15 records: yüzde, dört işlem, karekök, denklem, ortalama, ve birim
    # çevirme (derece-mil, metre-kilometre, mililitre-litre).
    "matematik": "sayı hesabı, birim çevirme ve matematik işlemi",
    # 20 records: selamlama, veda, hal hatır, teşekkür, moral ve motivasyon,
    # şaka, ve questions about the assistant itself (adın ne, sen insan mısın).
    "sohbet": "selamlaşma, hâl hatır sorma ve asistanla günlük sohbet",
    # 5 records, all asking the wall clock and nothing else.
    "saat": "şu anki saatin ne olduğunun sorulması",
}

_FUNCTION_CALL_PATH = "fonksiyon_cagrisi"
_FALLBACK_LABELS = ("sohbet", "matematik", "savaş yönergeleri", "medikal", "saat")
_FALLBACK_THRESHOLD = 0.25
_FALLBACK_AGENT = "sohbet"


@dataclass(frozen=True, slots=True)
class IntentLabel:
    """One routable intent class and everything the pipeline needs to act on it."""

    name: str
    agent: str
    verbalization: str
    function_call: bool
    rag_collection: str

    @property
    def hypothesis(self) -> str:
        """Return the NLI hypothesis that tests this label against a transcript."""
        return HYPOTHESIS_TEMPLATE.format(self.verbalization)


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """The full routing table: labels, the accept threshold, and the fallback."""

    labels: tuple[IntentLabel, ...]
    threshold: float
    default_agent: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return label names in evaluation-set order, as sent to the classifier."""
        return tuple(label.name for label in self.labels)

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """Return one NLI hypothesis per label, aligned with `names`."""
        return tuple(label.hypothesis for label in self.labels)

    @property
    def agents(self) -> tuple[str, ...]:
        """Return the distinct agents any label can route to, in label order."""
        return tuple(dict.fromkeys(label.agent for label in self.labels))

    def get(self, name: str) -> IntentLabel | None:
        """Return the label with this exact name, or None when unknown."""
        for label in self.labels:
            if label.name == name:
                return label
        return None

    @property
    def default_label(self) -> IntentLabel:
        """Return the label answering for the default agent, synthesized if absent."""
        for label in self.labels:
            if label.agent == self.default_agent:
                return label
        return IntentLabel(
            name=self.default_agent,
            agent=self.default_agent,
            verbalization=VERBALIZATIONS.get(self.default_agent, self.default_agent),
            function_call=False,
            rag_collection="none",
        )


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Read the routing table, falling back to the shipped classes when unusable."""
    source = path or DEFAULT_KNOWLEDGE_BASE
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("Intent taxonomy could not be read; using shipped classes.")
        return _fallback_taxonomy()

    if not isinstance(payload, dict):
        logger.warning("Intent taxonomy is malformed; using shipped classes.")
        return _fallback_taxonomy()

    names = _label_names(payload)
    if not names:
        logger.warning("Intent taxonomy has no labels; using shipped classes.")
        return _fallback_taxonomy()

    records = payload.get("kayitlar")
    records = records if isinstance(records, list) else []
    labels = tuple(_build_label(name, records) for name in names)

    return Taxonomy(
        labels=labels,
        threshold=_threshold(payload),
        default_agent=_default_agent(payload, labels),
    )


def _label_names(payload: dict) -> tuple[str, ...]:
    raw = payload.get("etiketler")
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for entry in raw:
        name = str(entry).strip() if isinstance(entry, str) else ""
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _build_label(name: str, records: list) -> IntentLabel:
    matching = [
        record
        for record in records
        if isinstance(record, dict) and str(record.get("beklenen_etiket", "")).strip() == name
    ]
    return IntentLabel(
        name=name,
        agent=_majority(matching, "beklenen_ajan", default=name),
        verbalization=VERBALIZATIONS.get(name, name),
        # A label counts as a function call only when its records agree on it;
        # one stray record must not divert a whole class away from the model.
        function_call=bool(matching)
        and all(
            str(record.get("yol", "")).strip() == _FUNCTION_CALL_PATH
            for record in matching
        ),
        rag_collection=_majority(matching, "rag_koleksiyonu", default="none"),
    )


def _majority(records: list[dict], field: str, *, default: str) -> str:
    values = [
        value
        for record in records
        if (value := str(record.get(field, "")).strip())
    ]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _threshold(payload: dict) -> float:
    raw = payload.get("guven_esigi")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return _FALLBACK_THRESHOLD
    threshold = float(raw)
    if not 0.0 <= threshold <= 1.0:
        logger.warning("Intent confidence threshold is out of range; using default.")
        return _FALLBACK_THRESHOLD
    return threshold


def _default_agent(payload: dict, labels: tuple[IntentLabel, ...]) -> str:
    raw = payload.get("varsayilan_ajan")
    candidate = str(raw).strip() if isinstance(raw, str) else ""
    known = {label.agent for label in labels}
    if candidate and candidate in known:
        return candidate
    if candidate:
        logger.warning("Default agent is not a routable agent; using the first label.")
    return labels[0].agent if labels else _FALLBACK_AGENT


def _fallback_taxonomy() -> Taxonomy:
    return Taxonomy(
        labels=tuple(
            IntentLabel(
                name=name,
                agent=name,
                verbalization=VERBALIZATIONS.get(name, name),
                function_call=name == "saat",
                rag_collection="small" if name in {"medikal", "savaş yönergeleri"} else "none",
            )
            for name in _FALLBACK_LABELS
        ),
        threshold=_FALLBACK_THRESHOLD,
        default_agent=_FALLBACK_AGENT,
    )
