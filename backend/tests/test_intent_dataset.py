"""Tests for the recipe-driven intent dataset generator."""

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util

import pytest

from app.intent_dataset import (
    Candidate,
    ROUTING_LABELS,
    build_openai_request,
    filter_candidates,
    load_recipes,
    parse_generated_question,
    plan_candidates,
    split_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = REPO_ROOT / "data" / "intent_dataset" / "recipes.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_intent_dataset.py"


def test_planning_is_seeded_balanced_and_keeps_recipe_families() -> None:
    """A changed randomization path must not make review plans non-reproducible."""
    book = load_recipes(RECIPES_PATH)

    first = plan_candidates(book, per_label=3, seed=17)
    second = plan_candidates(book, per_label=3, seed=17)

    assert first == second
    assert Counter(item.label for item in first) == {label: 3 for label in book.labels}
    assert all(item.family_id.startswith(f"{item.label}:") for item in first)
    assert all("JSON" in item.prompt for item in first)


def test_planning_can_start_after_a_previous_generation_round() -> None:
    """A follow-up plan must not reuse candidate IDs from the first round."""
    book = load_recipes(RECIPES_PATH)

    plans = plan_candidates(book, per_label=3, seed=17, start_index=140)

    assert plans[0].id.endswith("-141")
    assert plans[-1].id.endswith("-143")
    assert all(":141" in item.family_id or ":142" in item.family_id or ":143" in item.family_id for item in plans)


def test_planning_accepts_per_label_followup_counts() -> None:
    """A deficit round must generate only the labels that still need candidates."""
    book = load_recipes(RECIPES_PATH)

    plans = plan_candidates(
        book,
        per_label={"mevzi ve gizlenme": 4, "saat": 2},
        seed=17,
        start_index=140,
    )

    assert Counter(item.label for item in plans) == {"mevzi ve gizlenme": 4, "saat": 2}
    assert {item.id.rsplit("-", 1)[1] for item in plans} <= {"141", "142", "143", "144"}


def test_generation_prompt_requires_natural_semantic_rephrasing() -> None:
    """Removing the natural-language rule would restore literal slot concatenation."""
    book = load_recipes(RECIPES_PATH)
    first_aid_plan = next(item for item in plan_candidates(book, per_label=1, seed=17) if item.label == "ilk yardım")

    prompt = first_aid_plan.prompt.lower()
    assert "kelimelerin anlamına bağlı kal" in prompt
    assert "tek cümle" in prompt
    assert "savaş sahasında" in prompt
    assert "anlamsız" in prompt


def test_recipe_document_rejects_a_missing_routing_label(tmp_path: Path) -> None:
    """A recipe edit must not silently remove a deployed intent label."""
    recipes = tmp_path / "recipes.json"
    recipes.write_text('{"version": 1, "labels": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing labels"):
        load_recipes(recipes)


def test_filter_rejects_a_normalized_duplicate_without_losing_the_audit() -> None:
    """Removing Turkish-character normalization would retain duplicate training rows."""
    first = Candidate(
        id="one",
        label="telsiz ve raporlama",
        family_id="telsiz:cagri:001",
        recipe_id="cagri",
        slots={},
        question="Telsiz çağrı işareti nasıl verilir",
    )
    duplicate = Candidate(
        id="two",
        label="telsiz ve raporlama",
        family_id="telsiz:cagri:002",
        recipe_id="cagri",
        slots={},
        question="telsiz cagri isareti nasil verilir",
    )

    result = filter_candidates([first, duplicate])

    assert result.accepted == [first]
    assert result.rejected[0].candidate == duplicate
    assert result.rejected[0].reason == "duplicate"


def test_split_keeps_each_recipe_family_in_exactly_one_partition() -> None:
    """Splitting one paraphrase family across partitions would leak evaluation text."""
    candidates = [
        Candidate(
            id=f"{label}-{family}-{member}",
            label=label,
            family_id=f"{label}:family-{family}",
            recipe_id="fixture",
            slots={},
            question=f"{label} soru {family} {member}",
        )
        for label in ROUTING_LABELS
        for family in range(20)
        for member in range(5)
    ]

    splits = split_candidates(candidates, seed=9)

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 700,
        "validation": 150,
        "test": 150,
    }
    family_partitions: dict[str, set[str]] = {}
    for partition, items in splits.items():
        for item in items:
            family_partitions.setdefault(item.family_id, set()).add(partition)
    assert all(partitions == {next(iter(partitions))} for partitions in family_partitions.values())


def test_plan_command_writes_reviewable_requests_without_an_api_key(tmp_path: Path) -> None:
    """Accidentally requiring a secret for dry-run would block local review."""
    result = _run_cli("plan", "--out-dir", str(tmp_path), "--per-label", "2", "--seed", "17")

    assert result.returncode == 0, result.stderr
    requests = tmp_path / "requests.jsonl"
    rows = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert {row["label"] for row in rows} == set(ROUTING_LABELS)
    assert all("OPENAI_API_KEY" not in row["prompt"] for row in rows)


def test_generate_command_requires_an_explicit_execute_flag(tmp_path: Path) -> None:
    """Removing the execution guard could spend API credits during review."""
    plan = tmp_path / "requests.jsonl"
    plan.write_text(json.dumps({"id": "fixture"}) + "\n", encoding="utf-8")

    result = _run_cli("generate", "--plan", str(plan), "--out-dir", str(tmp_path))

    assert result.returncode != 0
    assert "--execute" in result.stderr


def test_openai_request_uses_structured_output_without_storing_the_turn() -> None:
    """A provider payload regression must not re-enable storage or free-form output."""
    plan = plan_candidates(load_recipes(RECIPES_PATH), per_label=1, seed=17)[0]

    request = build_openai_request(plan, model="gpt-test")

    assert request["model"] == "gpt-test"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["input"] == plan.prompt
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["required"] == ["soru"]


def test_parse_generated_question_rejects_explanation_wrapped_output() -> None:
    """Accepting prose around JSON would make audit records ambiguous."""
    assert parse_generated_question('{"soru":"saat kaç"}') == "saat kaç"

    with pytest.raises(ValueError, match="JSON object"):
        parse_generated_question('İşte sonuç: {"soru":"saat kaç"}')


def test_terra_cost_uses_the_configured_input_and_output_rates() -> None:
    """A missing Terra price entry must not turn a real API run into an unknown cost."""
    module = _load_generator_module()

    cost = module.estimate_cost_usd(
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000},
        model="gpt-5.6-terra",
    )

    assert cost == 17.5


def test_generation_records_a_structured_provider_answer_without_the_api_key() -> None:
    """A provider adapter regression must not write the key into an audit row."""
    module = _load_generator_module()
    plan = module._plan_row(plan_candidates(load_recipes(RECIPES_PATH), per_label=1, seed=17)[0])
    captured: dict[str, object] = {}

    def post(url: str, *, headers: dict[str, str], json: dict[str, object]) -> dict[str, object]:
        captured.update({"url": url, "headers": headers, "json": json})
        return {
            "output_text": '{"soru":"saat kaç"}',
            "usage": {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        }

    rows = list(module.generate_rows([plan], api_key="secret-value", model="gpt-5-mini", post=post))

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"] == {"Authorization": "Bearer secret-value"}
    assert captured["json"]["store"] is False
    assert rows[0]["question"] == "saat kaç"
    assert rows[0]["usage"] == {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160}
    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.00011)
    assert "secret-value" not in json.dumps(rows[0])


def test_live_generation_output_resumes_successes_retries_errors_and_keeps_csv(tmp_path: Path) -> None:
    """An interrupted run must retain completed questions and retry only unfinished work."""
    module = _load_generator_module()
    audit_path = tmp_path / "candidates.jsonl"
    csv_path = tmp_path / "candidates.csv"
    successful = {
        "id": "done",
        "label": "saat",
        "question": "Saat kaç?",
        "status": "accepted",
    }
    failed = {
        "id": "retry",
        "label": "ilk yardım",
        "question": None,
        "status": "error",
        "error": "TimeoutError: timed out",
    }

    module.append_generation_record(audit_path, successful)
    module.append_generation_record(audit_path, failed)
    module.sync_live_csv(audit_path, csv_path)

    plans = [{"id": "done"}, {"id": "retry"}, {"id": "new"}]
    assert module.successful_candidate_ids(audit_path) == {"done"}
    assert module.pending_plans(plans, completed_ids={"done"}) == [{"id": "retry"}, {"id": "new"}]
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["soru,sinif", "Saat kaç?,saat"]


def test_generation_progress_starts_at_the_resumed_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed run must expose completed work instead of restarting the bar at zero."""
    module = _load_generator_module()
    captured: dict[str, object] = {}

    class Progress:
        pass

    def fake_tqdm(**kwargs: object) -> Progress:
        captured.update(kwargs)
        return Progress()

    monkeypatch.setattr(module, "tqdm", fake_tqdm)

    assert isinstance(module.generation_progress(total=1400, completed=427), Progress)
    assert captured == {
        "total": 1400,
        "initial": 427,
        "desc": "Intent dataset",
        "unit": "soru",
        "dynamic_ncols": True,
    }


def test_review_request_requires_one_non_editing_decision_per_candidate() -> None:
    """A review response must cover every candidate without silently rewriting any text."""
    module = _load_generator_module()
    candidates = [
        {"id": "saat-001", "question": "Saat kaç oldu?"},
        {"id": "saat-002", "question": "Toplanmaya ne kadar var?"},
    ]

    request = module.build_review_request("saat", candidates, model="gpt-5.6-terra")
    reviews = module.parse_review_results(
        json.dumps(
            {
                "reviews": [
                    {
                        "id": "saat-001",
                        "decision": "keep",
                        "reason_code": "good",
                        "duplicate_of": None,
                    },
                    {
                        "id": "saat-002",
                        "decision": "revise",
                        "reason_code": "unclear_meaning",
                        "duplicate_of": None,
                    },
                ]
            }
        ),
        expected_ids={"saat-001", "saat-002"},
    )

    assert request["model"] == "gpt-5.6-terra"
    assert request["reasoning"] == {"effort": "none"}
    assert "Yeniden yazma" in request["input"]
    assert request["text"]["format"]["schema"]["properties"]["reviews"]["items"]["properties"]["decision"]["enum"] == [
        "keep",
        "revise",
        "reject",
    ]
    assert reviews[1]["decision"] == "revise"

    with pytest.raises(ValueError, match="exactly once"):
        module.parse_review_results(
            '{"reviews":[{"id":"saat-001","decision":"keep","reason_code":"good","duplicate_of":null}]}',
            expected_ids={"saat-001", "saat-002"},
        )


def test_export_writes_only_the_requested_csv_columns_and_keeps_audit_files(tmp_path: Path) -> None:
    """Changing export columns or dropping audit records would break training reproducibility."""
    module = _load_generator_module()
    rows = [
        asdict(
            Candidate(
                id=f"{label}-{family}-{member}",
                label=label,
                family_id=f"{label}:family-{family}",
                recipe_id="fixture",
                slots={},
                question=f"{label} benzersiz soru {family} {member}",
            )
        )
        for label in ROUTING_LABELS
        for family in range(20)
        for member in range(5)
    ]

    module.export_rows(rows, out_dir=tmp_path, seed=9)

    train = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()
    assert train[0] == "soru,sinif"
    assert len(train) == 701
    assert (tmp_path / "accepted.jsonl").exists()
    assert (tmp_path / "rejections.jsonl").read_text(encoding="utf-8") == ""


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONPATH"] = str(REPO_ROOT / "backend")
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("intent_dataset_generator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
