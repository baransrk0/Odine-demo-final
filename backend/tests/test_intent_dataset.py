"""Tests for the recipe-driven intent dataset generator."""

from collections import Counter
from pathlib import Path

import pytest

from app.intent_dataset import load_recipes, plan_candidates


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
