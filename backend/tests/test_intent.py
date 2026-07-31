"""Contracts for normalization, clock rules, the routing table, and the engine."""

from datetime import datetime
import json

import pytest
import respx
from httpx import Response

from app.config import Settings
from app.intents import normalize, rules
from app.intents.engine import IntentEngine
from app.intents.taxonomy import Taxonomy, load_taxonomy
from app.runtimes.intent import ZeroShotIntentClassifier
from app.runtimes.protocols import IntentPrediction

from conftest import FakeIntentClassifier


FIXED_NOON = datetime(2026, 7, 31, 14, 5)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, llama_cpp_model="gemma", **overrides)


def _engine(taxonomy: Taxonomy | None = None, **overrides) -> IntentEngine:
    values = {
        "taxonomy": taxonomy or load_taxonomy(),
        "clock": lambda: FIXED_NOON,
    }
    values.update(overrides)
    return IntentEngine(**values)


# --- normalization -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # str.lower() alone turns "İ" into "i" plus a combining dot, which never
        # compares equal to a typed "i" and would silently kill every rule.
        ("SAAT KAÇ?", "saat kaç"),
        ("İLK YARDIM", "ilk yardım"),
        ("IŞIK", "ışık"),
        ("Saati söyler misin?", "saati söyler misin"),
        ("  çok   boşluk  ", "çok boşluk"),
        ("ŞOK, BELİRTİLERİ!", "şok belirtileri"),
    ],
)
def test_normalizer_folds_turkish_case_and_strips_noise(raw: str, expected: str):
    """Turkish dotted/dotless casing must not be left to str.lower()."""
    assert normalize.process(raw) == expected


def test_normalizer_preserves_turkish_diacritics():
    """Losing ç/ğ/ı/ö/ş/ü would break every rule and every reference comparison."""
    assert normalize.process("Çğıöşü ÇĞIÖŞÜ") == "çğıöşü çğıöşü"


# --- clock rules -------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "Saat kaç?",
        "Saati söyler misin?",
        "Saatin kaç?",
        "Saat kaçtır?",
        "Saat",
        "Şu anda saat kaç",
        "Lütfen saati söyler misiniz",
        "saat kaç oldu",
    ],
)
def test_clock_rules_cover_the_evaluation_set_phrasings(utterance: str):
    """These are the fonksiyon_cagrisi records; missing one sends a clock question to the model."""
    assert rules.matches_clock(normalize.process(utterance))


@pytest.mark.parametrize(
    "utterance",
    [
        # Duration arithmetic, not a clock reading.
        "Altmış kilometreyi kırk kilometre hızla kaç saatte alırım",
        # A duty-roster question that belongs to the yönergeler agent.
        "Nöbet saati kaçta başlıyor",
        "Yirmi dört saat sonra ne yapmalıyım",
        "Turnike nasıl uygulanır",
        "",
    ],
)
def test_clock_rules_leave_other_agents_alone(utterance: str):
    """A loose 'saat' substring would steal these from their own agents."""
    assert not rules.matches_clock(normalize.process(utterance))


def test_clock_answer_uses_the_approved_wording_and_24_hour_format():
    """The evaluation checks the 'Şu an saat' pattern and HH:MM exactly."""
    assert rules.clock_answer(FIXED_NOON) == "Şu an saat 14:05"


# --- taxonomy ----------------------------------------------------------------


def test_shipped_taxonomy_matches_the_evaluation_set():
    """Routing silently disagreeing with the evaluation set would invalidate every score."""
    taxonomy = load_taxonomy()

    assert set(taxonomy.names) == {
        "sohbet",
        "matematik",
        "savaş yönergeleri",
        "medikal",
        "saat",
    }
    assert taxonomy.threshold == 0.25
    assert taxonomy.default_agent == "sohbet"
    assert all(label.agent == label.name for label in taxonomy.labels)


def test_only_the_clock_label_is_a_function_call():
    """Routing another class away from the model would leave it unanswered."""
    taxonomy = load_taxonomy()

    assert [label.name for label in taxonomy.labels if label.function_call] == ["saat"]
    assert taxonomy.get("medikal").rag_collection == "small"
    assert taxonomy.get("savaş yönergeleri").rag_collection == "small"
    assert taxonomy.get("matematik").rag_collection == "none"


def test_every_label_gets_a_distinct_hypothesis():
    """Two labels sharing a hypothesis would make their scores meaningless."""
    taxonomy = load_taxonomy()
    hypotheses = taxonomy.hypotheses

    assert len(set(hypotheses)) == len(hypotheses)
    assert all(label.name != label.verbalization for label in taxonomy.labels)


def test_unreadable_evaluation_set_falls_back_to_the_shipped_classes(tmp_path):
    """A bad path must not leave the pipeline with no routing table at all."""
    taxonomy = load_taxonomy(tmp_path / "missing.json")

    assert set(taxonomy.names) == set(load_taxonomy().names)
    assert taxonomy.default_agent == "sohbet"


def test_a_single_stray_record_cannot_divert_a_whole_class(tmp_path):
    """One mislabeled path must not send a model-answered class to a function call."""
    source = tmp_path / "kb.json"
    source.write_text(
        json.dumps(
            {
                "etiketler": ["medikal", "saat"],
                "guven_esigi": 0.4,
                "varsayilan_ajan": "medikal",
                "kayitlar": [
                    {
                        "beklenen_etiket": "medikal",
                        "beklenen_ajan": "medikal",
                        "yol": "fonksiyon_cagrisi",
                        "rag_koleksiyonu": "small",
                    },
                    {
                        "beklenen_etiket": "medikal",
                        "beklenen_ajan": "medikal",
                        "yol": "siniflandirici",
                        "rag_koleksiyonu": "small",
                    },
                    {
                        "beklenen_etiket": "saat",
                        "beklenen_ajan": "saat",
                        "yol": "fonksiyon_cagrisi",
                        "rag_koleksiyonu": "none",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    taxonomy = load_taxonomy(source)

    assert taxonomy.get("medikal").function_call is False
    assert taxonomy.get("saat").function_call is True
    assert taxonomy.threshold == 0.4
    assert taxonomy.default_agent == "medikal"


# --- engine ------------------------------------------------------------------


async def test_clock_rule_answers_without_consulting_the_classifier():
    """Spending a model call on a phrasing the rules already cover is pure latency."""
    classifier = FakeIntentClassifier(label="sohbet", confidence=0.99)
    result = await _engine(classifier=classifier).classify("Saat kaç?")

    assert result.label == "saat"
    assert result.source == "rule"
    assert result.function_answer == "Şu an saat 14:05"
    assert classifier.calls == []


async def test_classifier_routes_an_unrecognized_clock_phrasing_to_the_function():
    """The whole point of the classifier fallback: reach the clock without the model."""
    classifier = FakeIntentClassifier(label="saat", confidence=0.71)
    result = await _engine(classifier=classifier).classify(
        "Vaktin ne olduğunu öğrenebilir miyim"
    )

    assert result.label == "saat"
    assert result.source == "classifier"
    assert result.function_answer == "Şu an saat 14:05"


async def test_confident_prediction_routes_to_its_agent():
    classifier = FakeIntentClassifier(label="medikal", confidence=0.83)
    result = await _engine(classifier=classifier).classify("Turnike nasıl uygulanır?")

    assert (result.label, result.agent, result.source) == (
        "medikal",
        "medikal",
        "classifier",
    )
    assert result.function_answer is None
    assert result.rag_collection == "small"
    assert classifier.calls[0][0] == "Turnike nasıl uygulanır?"


async def test_the_classifier_receives_the_raw_transcript_and_every_label():
    """Normalizing away case and punctuation costs an NLI model entailment accuracy."""
    classifier = FakeIntentClassifier(label="sohbet", confidence=0.5)
    await _engine(classifier=classifier).classify("Şok BELİRTİLERİ nelerdir?")

    text, labels = classifier.calls[0]
    assert text == "Şok BELİRTİLERİ nelerdir?"
    assert set(labels) == set(load_taxonomy().names)


async def test_a_prediction_below_the_threshold_falls_back_to_the_default_agent():
    """A confidently wrong agent is worse for the operator than a handoff."""
    classifier = FakeIntentClassifier(label="medikal", confidence=0.24)
    result = await _engine(classifier=classifier).classify("Bir şey soracaktım")

    assert (result.label, result.agent) == ("sohbet", "sohbet")
    assert result.source == "low_confidence"
    assert result.confidence == 0.24
    assert result.routed is False


async def test_the_threshold_can_be_overridden_without_touching_the_evaluation_set():
    classifier = FakeIntentClassifier(label="medikal", confidence=0.24)
    result = await _engine(classifier=classifier, threshold=0.2).classify("Soru")

    assert result.source == "classifier"


@pytest.mark.parametrize(
    "classifier",
    [
        FakeIntentClassifier(error=RuntimeError("service down")),
        FakeIntentClassifier(label="tanımsız", confidence=0.99),
        None,
    ],
)
async def test_a_broken_classifier_degrades_instead_of_failing_the_turn(classifier):
    """Losing routing must cost answer quality, never the answer itself."""
    result = await _engine(classifier=classifier).classify("Turnike nasıl uygulanır?")

    assert (result.label, result.agent) == ("sohbet", "sohbet")
    assert result.source == "unavailable"
    assert result.function_answer is None


async def test_a_classifier_that_was_down_at_startup_is_still_tried():
    """Gating on the startup probe would keep routing dead until a restart."""
    classifier = FakeIntentClassifier(label="medikal", confidence=0.9, ready=False)
    result = await _engine(classifier=classifier).classify("Turnike nasıl uygulanır?")

    assert result.agent == "medikal"
    assert result.source == "classifier"


async def test_a_slow_classifier_is_abandoned_at_the_timeout():
    """An unbounded classifier call would add its latency to every single turn."""

    class HangingClassifier:
        ready = True

        async def classify(self, text, labels):
            import asyncio

            await asyncio.sleep(5)
            return IntentPrediction(label="medikal", confidence=1.0)

    result = await _engine(
        classifier=HangingClassifier(),
        timeout_seconds=0.05,
    ).classify("Turnike nasıl uygulanır?")

    assert result.source == "unavailable"
    assert result.agent == "sohbet"


async def test_the_clock_rule_still_fires_with_no_classifier_at_all():
    """The one answer that must survive a dead intent service."""
    result = await _engine(classifier=None).classify("Saat kaç?")

    assert result.function_answer == "Şu an saat 14:05"
    assert result.source == "rule"


# --- routed agent prompts ----------------------------------------------------


def test_each_agent_prompt_carries_only_its_own_reference_answers():
    """A prompt leaking other agents' answers wastes the context it was split to save."""
    prompts = _settings().agent_system_prompts

    assert set(prompts) == {"medikal", "savaş yönergeleri", "matematik", "sohbet"}
    assert "Turnike nasıl uygulanır?" in prompts["medikal"]
    assert "Turnike nasıl uygulanır?" not in prompts["sohbet"]
    assert "Merhaba" in prompts["sohbet"]


def test_no_agent_prompt_teaches_the_model_the_clock_placeholder():
    """Folding "Şu an saat HH:MM" into a prompt makes the model answer literally that."""
    prompts = _settings().agent_system_prompts

    assert "saat" not in prompts
    assert not any("HH:MM" in prompt for prompt in prompts.values())


def test_agent_prompts_are_built_once_and_stay_identical():
    """A prompt that changes between turns defeats llama.cpp prefix caching."""
    settings = _settings()

    assert settings.agent_system_prompts is settings.agent_system_prompts


def test_disabling_the_knowledge_base_leaves_instruction_only_agent_prompts():
    prompts = _settings(knowledge_base_enabled=False).agent_system_prompts

    assert "Turnike nasıl uygulanır?" not in prompts["medikal"]
    assert "sağlık asistanısın" in prompts["medikal"]


# --- zero-shot HTTP adapter --------------------------------------------------


async def test_adapter_sends_verbalized_labels_and_maps_the_winner_back():
    """Bare labels are weak hypotheses; the answer must still arrive under its label name."""
    runtime = ZeroShotIntentClassifier(
        "http://intent.test",
        verbalizations={"medikal": "tıbbi ilk yardım", "sohbet": "genel sohbet"},
        timeout_seconds=3,
    )

    with respx.mock(assert_all_called=True) as router:
        route = router.post("http://intent.test/classify").mock(
            return_value=Response(
                200,
                json={
                    "labels": ["tıbbi ilk yardım", "genel sohbet"],
                    "scores": [0.88, 0.12],
                },
            )
        )
        prediction = await runtime.classify("Turnike nasıl?", ["medikal", "sohbet"])

    assert prediction.label == "medikal"
    assert prediction.confidence == 0.88
    assert prediction.scores == {"medikal": 0.88, "sohbet": 0.12}

    body = json.loads(route.calls[0].request.content)
    assert body["sequence"] == "Turnike nasıl?"
    assert body["candidate_labels"] == ["tıbbi ilk yardım", "genel sohbet"]
    assert body["multi_label"] is False
    assert "{}" in body["hypothesis_template"]


async def test_adapter_also_reads_a_list_of_label_score_objects():
    """The second common zero-shot response shape must not read as a service outage."""
    runtime = ZeroShotIntentClassifier("http://intent.test", timeout_seconds=3)

    with respx.mock(assert_all_called=True) as router:
        router.post("http://intent.test/classify").mock(
            return_value=Response(
                200,
                json=[{"label": "matematik", "score": 0.6}, {"label": "sohbet", "score": 0.4}],
            )
        )
        prediction = await runtime.classify("Kaç eder", ["matematik", "sohbet"])

    assert prediction.label == "matematik"


@pytest.mark.parametrize(
    "payload",
    [
        {"labels": ["medikal"], "scores": [0.9, 0.1]},
        {"labels": ["bilinmeyen"], "scores": [0.9]},
        {"unexpected": "payload"},
        [{"label": "medikal", "score": "yüksek"}],
    ],
)
async def test_adapter_rejects_responses_it_cannot_trust(payload):
    """A malformed reply must raise, not resolve to a confident arbitrary label."""
    runtime = ZeroShotIntentClassifier("http://intent.test", timeout_seconds=3)

    with respx.mock(assert_all_called=True) as router:
        router.post("http://intent.test/classify").mock(
            return_value=Response(200, json=payload)
        )
        with pytest.raises(RuntimeError):
            await runtime.classify("Soru", ["medikal", "sohbet"])


async def test_adapter_reports_an_unreachable_service_as_not_ready():
    runtime = ZeroShotIntentClassifier("http://intent.test", timeout_seconds=3)
    runtime.load()

    with respx.mock(assert_all_called=True) as router:
        router.get("http://intent.test/health").mock(return_value=Response(503))
        healthy = await runtime.health()

    assert healthy is False
    assert runtime.ready is False
