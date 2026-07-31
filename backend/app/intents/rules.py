"""Deterministic shortcuts that answer before any model is consulted.

The evaluation set marks every `saat` record as `fonksiyon_cagrisi`: the clock is
answered locally, so neither the classifier nor the language model can be asked
for a time they do not have. These patterns cover the phrasings in that set and
the obvious variations around them.

They are deliberately anchored to the whole utterance. A loose `saat` substring
would swallow "nöbet saati kaçta başlıyor" and "kaç saatte alırım", which belong
to other agents. Anything phrased outside these patterns is left to the
classifier, which routes it to `saat` on its own and still reaches the clock
without a language-model call.
"""

from collections.abc import Callable
from datetime import datetime
import re


Clock = Callable[[], datetime]

# Optional politeness that may wrap any of the clock phrasings below.
_PREFIX = r"(?:(?:lütfen|bana|acaba|peki|bi|bir)\s+)*"
_SUFFIX = r"(?:\s+(?:oldu|acaba|lütfen|biliyor\s+musun(?:uz)?))*"
_CLOCK_BODIES = (
    # "saat"
    r"saat",
    # "saat kaç", "saat kaçtır", "saatin kaç", "şu anda saat kaç"
    r"(?:şu\s+an(?:da)?\s+)?saat(?:in|i)?\s+kaç(?:tır|ta|te)?",
    # "saati söyler misin", "saat söyle", "saati söyleyebilir misiniz"
    r"saat(?:in|i)?\s+söyle(?:r|yebilir)?(?:\s+mi(?:sin|siniz))?",
    # "saat ne", "şu anda saat nedir"
    r"(?:şu\s+an(?:da)?\s+)?saat\s+ne(?:dir)?",
)
_CLOCK_PATTERN = re.compile(
    rf"^{_PREFIX}(?:{'|'.join(_CLOCK_BODIES)}){_SUFFIX}$"
)


def matches_clock(normalized_text: str) -> bool:
    """Whether this canonical utterance is unambiguously a request for the time."""
    return bool(normalized_text) and _CLOCK_PATTERN.fullmatch(normalized_text) is not None


def clock_answer(now: datetime | None = None) -> str:
    """Return the evaluation set's approved clock reply for the current time.

    The reference answers spell this as "Şu an saat HH:MM"; the wording and the
    24-hour format are what the evaluation checks, so neither is configurable.
    """
    moment = now or default_clock()
    return f"Şu an saat {moment:%H:%M}"


def default_clock() -> datetime:
    """Return device-local wall-clock time."""
    return datetime.now().astimezone()
