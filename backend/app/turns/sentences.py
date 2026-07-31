"""Incremental sentence segmentation for Turkish language-model streams."""

ABBREVIATIONS = {
    "Dr.",
    "Doç.",
    "Prof.",
    "Sn.",
    "Say.",
    "No.",
    "örn.",
    "vb.",
    "vs.",
    "bkz.",
    "T.C.",
}
CLOSERS = '"”’»)]}'
OPENERS = '"“‘«([{'


class TurkishSentenceBuffer:
    """Accumulate token deltas and return each completed sentence once."""

    def __init__(self) -> None:
        self._pending = ""

    def push(self, delta: str) -> list[str]:
        """Append a stream delta and emit any now-complete sentences."""
        self._pending += delta
        return self._drain_boundaries()

    def finish(self) -> list[str]:
        """Flush the remaining stream tail exactly once."""
        tail = self._pending
        self._pending = ""
        return [tail] if tail else []

    def _drain_boundaries(self) -> list[str]:
        emitted: list[str] = []
        start = 0

        while start < len(self._pending):
            boundary = self._next_boundary(start)
            if boundary is None:
                break
            emitted.append(self._pending[:boundary])
            self._pending = self._pending[boundary:]
            start = 0

        return emitted

    def _next_boundary(self, start: int) -> int | None:
        for index in range(start, len(self._pending)):
            if self._pending[index] not in ".!?":
                continue

            end = index + 1
            while end < len(self._pending) and self._pending[end] in CLOSERS:
                end += 1

            if end < len(self._pending) and not self._pending[end].isspace():
                continue
            if self._pending[index] == "." and self._is_abbreviation(index):
                continue
            if self._is_unfinished_numeric_boundary(index, end):
                continue

            return end
        return None

    def _is_abbreviation(self, punctuation_index: int) -> bool:
        token_start = punctuation_index
        while token_start > 0 and not self._pending[token_start - 1].isspace():
            token_start -= 1
        token = self._pending[token_start : punctuation_index + 1].lstrip(OPENERS)
        return token in ABBREVIATIONS

    def _is_unfinished_numeric_boundary(self, punctuation_index: int, end: int) -> bool:
        previous = self._pending[punctuation_index - 1] if punctuation_index else ""
        return previous.isdigit() and end == len(self._pending)
