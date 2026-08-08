"""Pure helpers for planning a recipe-driven intent dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Mapping


ROUTING_LABELS = (
    "ilk yardım",
    "telsiz ve raporlama",
    "nöbet ve emniyet",
    "harita ve intikal",
    "mevzi ve gizlenme",
    "kbrn korunma",
    "angajman ve esir hukuku",
    "matematik",
    "sohbet",
    "saat",
)


@dataclass(frozen=True)
class Recipe:
    id: str
    scenario: str
    slots: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class RecipeBook:
    labels: tuple[str, ...]
    recipes_by_label: Mapping[str, tuple[Recipe, ...]]


@dataclass(frozen=True)
class CandidatePlan:
    id: str
    label: str
    family_id: str
    recipe_id: str
    slots: Mapping[str, str]
    prompt: str


def load_recipes(path: Path) -> RecipeBook:
    """Load and validate recipes against the deployed routing taxonomy."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read recipes: {error}") from error

    labels_payload = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels_payload, dict):
        raise ValueError("recipes must define a labels object")

    present = set(labels_payload)
    expected = set(ROUTING_LABELS)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        raise ValueError(f"missing labels: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown labels: {', '.join(extra)}")

    recipes_by_label: dict[str, tuple[Recipe, ...]] = {}
    for label in ROUTING_LABELS:
        group = labels_payload[label]
        recipes_payload = group.get("recipes") if isinstance(group, dict) else None
        if not isinstance(recipes_payload, list) or not recipes_payload:
            raise ValueError(f"{label} must define at least one recipe")
        recipes_by_label[label] = tuple(_parse_recipe(label, item) for item in recipes_payload)

    return RecipeBook(labels=ROUTING_LABELS, recipes_by_label=recipes_by_label)


def _parse_recipe(label: str, payload: object) -> Recipe:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} recipe must be an object")
    recipe_id = payload.get("id")
    scenario = payload.get("scenario")
    slots_payload = payload.get("slots")
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError(f"{label} recipe id is required")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError(f"{label}:{recipe_id} scenario is required")
    if not isinstance(slots_payload, dict) or not slots_payload:
        raise ValueError(f"{label}:{recipe_id} slots are required")

    slots: dict[str, tuple[str, ...]] = {}
    for name, choices in slots_payload.items():
        if not isinstance(name, str) or not isinstance(choices, list) or not choices:
            raise ValueError(f"{label}:{recipe_id} has an invalid slot")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError(f"{label}:{recipe_id} has an invalid slot choice")
        slots[name] = tuple(choices)
    return Recipe(id=recipe_id, scenario=scenario, slots=slots)


def plan_candidates(book: RecipeBook, *, per_label: int, seed: int) -> list[CandidatePlan]:
    """Plan reproducible, recipe-compatible provider requests without calling a model."""
    if per_label <= 0:
        raise ValueError("per_label must be positive")

    randomizer = random.Random(seed)
    plans: list[CandidatePlan] = []
    for label in book.labels:
        recipes = book.recipes_by_label[label]
        for index in range(per_label):
            recipe = recipes[(index + randomizer.randrange(len(recipes))) % len(recipes)]
            selected_slots = {name: randomizer.choice(choices) for name, choices in recipe.slots.items()}
            family_id = f"{label}:{recipe.id}:{index + 1:03d}"
            plans.append(
                CandidatePlan(
                    id=f"{_slug(label)}-{index + 1:03d}",
                    label=label,
                    family_id=family_id,
                    recipe_id=recipe.id,
                    slots=selected_slots,
                    prompt=_build_prompt(label, recipe.scenario, selected_slots),
                )
            )
    return plans


def _build_prompt(label: str, scenario: str, slots: Mapping[str, str]) -> str:
    rendered_slots = "\\n".join(f"- {name}: {value}" for name, value in slots.items())
    return (
        "Yalnızca geçerli JSON döndür: {\\\"soru\\\": \\\"...\\\"}.\\n"
        "Tek, kısa, Türkçe ve STT-benzeri kullanıcı sorusu yaz. Cevap, açıklama, "
        "etiket adı veya birden fazla soru yazma.\\n"
        f"Hedef intent: {label}\\nSenaryo: {scenario}\\nDeğişkenler:\\n{rendered_slots}"
    )


def _slug(value: str) -> str:
    replacements = str.maketrans("çğıöşü ", "cgiosu-")
    return value.lower().translate(replacements).replace(" ", "-")

