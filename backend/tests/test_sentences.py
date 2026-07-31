import pytest

from app.turns.sentences import TurkishSentenceBuffer


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Merhaba! Nasılsın?"], ["Merhaba!", " Nasılsın?"]),
        (["Değer 3.", "14 oldu. Sonuç iyi"], ["Değer 3.14 oldu."]),
        (["Dr. Ayşe geldi. Devam ediyor."], ["Dr. Ayşe geldi.", " Devam ediyor."]),
        (['"Hazır mısın?" Evet.'], ['"Hazır mısın?"', " Evet."]),
        (["İlk cümle.", " İkinci"], ["İlk cümle."]),
    ],
)
def test_incremental_boundaries(chunks, expected):
    """Known Turkish stream boundaries emit exactly once and in order."""
    buffer = TurkishSentenceBuffer()

    assert [sentence for chunk in chunks for sentence in buffer.push(chunk)] == expected


def test_finish_flushes_unpunctuated_tail_once():
    """A final incomplete tail must become audible once at stream completion."""
    buffer = TurkishSentenceBuffer()

    buffer.push("Son parça")

    assert buffer.finish() == ["Son parça"]
    assert buffer.finish() == []


def test_retains_digit_adjacent_punctuation_until_the_next_chunk_resolves_it():
    """Punctuation may not split text when the following chunk begins with a digit."""
    buffer = TurkishSentenceBuffer()

    assert buffer.push("Skor 3!") == []
    assert buffer.push("4 değişti.") == ["Skor 3!4 değişti."]


def test_does_not_split_an_abbreviation_after_an_opening_bracket():
    """Opening punctuation does not make a known abbreviation a sentence end."""
    buffer = TurkishSentenceBuffer()

    assert buffer.push("(Dr. Ayşe geldi.) Son.") == ["(Dr. Ayşe geldi.)", " Son."]
