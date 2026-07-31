"""Turkish-aware canonicalization shared by the rule layer and the evaluation.

Only the deterministic rule layer consumes this. The classifier is given the raw
transcript on purpose: NLI models are trained on natural, cased, punctuated text,
and stripping that signal costs entailment accuracy for no gain.

The rule layer and `scripts/eval_intent.py` must canonicalize identically, so the
single `process` entrypoint here is the only place the rules are allowed to fold
case or drop punctuation.
"""

import re
import unicodedata


# str.lower() maps "I" to "i" and "İ" to "i" plus a combining dot -- both wrong
# for Turkish, and the second one silently breaks equality against typed text.
# Fold the dotted/dotless pair explicitly, before lowering the rest.
_UPPERCASE_FOLD = str.maketrans({"İ": "i", "I": "ı"})
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def process(text: str) -> str:
    """Return the canonical form the intent rules match against."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(_UPPERCASE_FOLD).lower()
    normalized = _PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()
