import json

from app.config import Settings
from app.knowledge import (
    DEFAULT_KNOWLEDGE_BASE,
    ReferenceAnswer,
    build_system_prompt,
    load_reference_answers,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, llama_cpp_model="gemma", **overrides)


def test_shipped_knowledge_base_loads_every_record():
    """Losing records from the shipped set must not go unnoticed."""
    answers = load_reference_answers()

    assert len(answers) == 100
    assert all(pair.question and pair.answer and pair.label for pair in answers)
    assert (
        ReferenceAnswer(
            "Merhaba",
            "Merhaba, buradayım. Nasıl yardımcı olabilirim?",
            "sohbet",
        )
        in answers
    )


def test_system_prompt_contains_base_instruction_and_every_pair():
    """Truncating the reference block must fail rather than silently shrink."""
    answers = load_reference_answers()
    prompt = build_system_prompt("TEMEL", answers)

    assert prompt.startswith("TEMEL")
    for pair in answers:
        assert f"S: {pair.question}" in prompt
        assert f"C: {pair.answer}" in prompt


def test_disabled_knowledge_base_sends_base_prompt_only():
    """The escape hatch must actually remove the block from the prompt."""
    settings = _settings(knowledge_base_enabled=False)

    assert settings.system_prompt == settings.turkish_system_prompt


def test_system_prompt_is_built_once_and_stays_identical():
    """A prompt that changes between turns would defeat llama.cpp prefix caching."""
    settings = _settings()

    assert settings.system_prompt is settings.system_prompt


def test_unreadable_knowledge_base_degrades_to_base_prompt(tmp_path, caplog):
    """A bad path must not take the assistant down mid-demo."""
    settings = _settings(knowledge_base_path=str(tmp_path / "missing.json"))

    assert settings.system_prompt == settings.turkish_system_prompt
    assert "missing.json" not in caplog.text


def test_malformed_records_are_skipped_without_failing(tmp_path):
    """One broken record must not discard the usable ones."""
    source = tmp_path / "kb.json"
    source.write_text(
        json.dumps(
            {
                "kayitlar": [
                    {"soru": "Soru bir", "referans_cevap": "Cevap bir"},
                    {"soru": "", "referans_cevap": "Cevapsiz soru"},
                    "bozuk kayit",
                    {"soru": "Soru iki", "referans_cevap": "Cevap iki"},
                ]
            }
        ),
        encoding="utf-8",
    )

    answers = load_reference_answers(source)

    assert answers == [
        ReferenceAnswer("Soru bir", "Cevap bir"),
        ReferenceAnswer("Soru iki", "Cevap iki"),
    ]


def test_knowledge_base_ships_inside_the_package():
    """Moving the data file without updating the default path must fail."""
    assert DEFAULT_KNOWLEDGE_BASE.is_file()
    payload = json.loads(DEFAULT_KNOWLEDGE_BASE.read_text(encoding="utf-8"))
    assert payload["kayit_sayisi"] == 100
