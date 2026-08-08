"""Tests for the recipe-driven intent dataset generator."""

from collections import Counter
from pathlib import Path

import pytest

from app.intent_dataset import Candidate, ROUTING_LABELS, filter_candidates, load_recipes, plan_candidates, split_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = REPO_ROOT / "data" / "intent_dataset" / "recipes.json"


def test_planning_is_seeded_balanced_and_keeps_recipe_families() -> None:
    """A changed randomization path must not make review plans non-reproducible."""
    book = load_recipes(RECIPES_PATH)

    first = plan_candidates(book, per_label=3, seed=17)
    second = plan_candidates(book, per_label=3, seed=17)

    assert first == second
    assert Counter(item.label for item in first) == {label: 3 for label in book.labels}
    assert all(item.family_id.startswith(f"{item.label}:") for item in first)
    assert all("JSON" in item.prompt for item in first)


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
