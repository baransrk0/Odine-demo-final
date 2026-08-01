"""Static reference answers folded into the llama.cpp system prompt.

The block this module builds is deliberately constant for the process lifetime.
llama-server reuses the cached prefix of an unchanged prompt, so the reference
answers are prefilled once and every later turn pays only for the transcript
and the generated tokens.

With intent routing enabled the process builds one such constant prompt per
agent, each carrying only its own label's answers. Every agent prompt stays
byte-identical across turns, so each keeps its own prefill cache slot, and the
prompts are a fraction of the size of the combined block.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple


logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE = Path(__file__).parent / "data" / "atbk_knowledge_base.json"

_HEADER = (
    "Aşağıda onaylanmış soru ve cevaplar var. Gelen soru bunlardan biriyle aynı "
    "ya da anlamca çok yakınsa, ilgili cevabı olduğu gibi ver; yeniden yazma, "
    "genişletme veya madde işareti ekleme. Hiçbiriyle eşleşmiyorsa kendi "
    "bilginle kısa ve yalnızca Türkçe yanıtla. Saat sorularında gerçek saati "
    "bilemezsin; saat bilgisine erişimin olmadığını kısaca söyle."
)

# Spoken through Piper, so every agent is told to stay plain-text; markdown and
# symbols are read aloud literally and ruin the response.
_VOICE_CONSTRAINT = (
    "Yanıtın sesli okunacak: madde işareti, başlık veya simge kullanma; "
    "kısa, açık ve yalnızca Türkçe yanıtla."
)

# Per-agent instructions used when intent routing selects an agent. Tune the
# wording here; the reference answers each prompt carries are filtered by label.
AGENT_INSTRUCTIONS: dict[str, str] = {
    "medikal": (
        "Sen kask içi taktik sağlık asistanısın. Yaralı bakımı, ilk yardım ve "
        "tıbbi müdahale sorularını yanıtla. Adımları uygulanma sırasına göre ver. "
        f"{_VOICE_CONSTRAINT}"
    ),
    "savaş yönergeleri": (
        "Sen kask içi harekât asistanısın. Muharebe, görev ve nöbet yönergelerini "
        f"yanıtla. Yönergeye bağlı kal, kendi kuralını uydurma. {_VOICE_CONSTRAINT}"
    ),
    "matematik": (
        "Sen kask içi hesap asistanısın. İstenen işlemi yap ve sonucu tek cümlede, "
        f"sayısıyla birlikte söyle. Ara işlemleri sıralama. {_VOICE_CONSTRAINT}"
    ),
    "sohbet": f"Sen kask içi sesli asistansın. {_VOICE_CONSTRAINT}",
}


class ReferenceAnswer(NamedTuple):
    """One approved question/answer pair from the PoC evaluation set."""

    question: str
    answer: str
    label: str = ""


def load_reference_answers(path: Path | None = None) -> list[ReferenceAnswer]:
    """Read approved pairs, returning an empty list when the file is unusable."""
    source = path or DEFAULT_KNOWLEDGE_BASE
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("Reference answer set could not be read.")
        return []

    records = payload.get("kayitlar") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        logger.warning("Reference answer set has no usable records.")
        return []

    answers: list[ReferenceAnswer] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        question = str(record.get("soru", "")).strip()
        answer = str(record.get("referans_cevap", "")).strip()
        if question and answer:
            answers.append(
                ReferenceAnswer(
                    question=question,
                    answer=answer,
                    label=str(record.get("beklenen_etiket", "")).strip(),
                )
            )
    return answers


def answers_for_label(
    label: str,
    answers: list[ReferenceAnswer] | None = None,
) -> list[ReferenceAnswer]:
    """Return only the approved pairs belonging to one intent label."""
    pairs = answers if answers is not None else load_reference_answers()
    return [pair for pair in pairs if pair.label == label]


def agent_instruction(agent: str, base_prompt: str) -> str:
    """Return the routed agent's instruction, or the base one when unconfigured."""
    return AGENT_INSTRUCTIONS.get(agent, base_prompt)


def build_system_prompt(
    base_prompt: str,
    answers: list[ReferenceAnswer] | None = None,
) -> str:
    """Prepend the reference block to the base instruction, base-only when empty.

    The reference header leads every prompt so llama-server's cache can reuse
    it as a shared prefix on an agent switch, even with a single KV-cache slot
    (see docs/superpowers/specs/2026-08-01-shared-prompt-prefix-cache-design.md).
    """
    pairs = answers if answers is not None else load_reference_answers()
    if not pairs:
        return base_prompt

    lines = [_HEADER, "", base_prompt, ""]
    lines.extend(f"S: {pair.question}\nC: {pair.answer}" for pair in pairs)
    return "\n".join(lines)
