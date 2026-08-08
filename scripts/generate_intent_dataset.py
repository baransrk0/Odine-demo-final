#!/usr/bin/env python3
"""Prepare and generate an auditable Turkish intent fine-tuning dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.intent_dataset import (
    CandidatePlan,
    Candidate,
    build_openai_request,
    filter_candidates,
    load_recipes,
    parse_generated_question,
    plan_candidates,
    split_candidates,
)


DEFAULT_RECIPES = REPO_ROOT / "data" / "intent_dataset" / "recipes.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write deterministic provider requests without a network call")
    plan.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    plan.add_argument("--out-dir", type=Path, required=True)
    plan.add_argument("--per-label", type=int, default=140)
    plan.add_argument("--seed", type=int, default=17)

    generate = subparsers.add_parser("generate", help="call OpenAI only after explicit approval")
    generate.add_argument("--plan", type=Path, required=True)
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--execute", action="store_true")
    generate.add_argument("--model", default="gpt-5-mini")

    export = subparsers.add_parser("export", help="filter generated candidates and write train/validation/test CSV files")
    export.add_argument("--candidates", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--seed", type=int, default=17)

    return parser


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def plan_command(args: argparse.Namespace) -> int:
    book = load_recipes(args.recipes)
    plans = plan_candidates(book, per_label=args.per_label, seed=args.seed)
    write_jsonl(args.out_dir / "requests.jsonl", (_plan_row(item) for item in plans))
    return 0


def _plan_row(plan: CandidatePlan) -> dict[str, object]:
    return asdict(plan)


def generate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.execute:
        parser.error("generate requires --execute; use plan for a network-free review")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("generate requires OPENAI_API_KEY in the environment")
    try:
        import httpx
    except ImportError as error:
        parser.error(f"generate requires httpx: {error}")

    plans = list(_read_jsonl(args.plan))
    with httpx.Client(timeout=60.0) as client:
        def post(url: str, *, headers: dict[str, str], json: dict[str, object]) -> dict[str, object]:
            response = client.post(url, headers=headers, json=json)
            response.raise_for_status()
            return response.json()

        rows = list(
            generate_rows(
                plans,
                api_key=os.environ["OPENAI_API_KEY"],
                model=args.model,
                post=post,
            )
        )
    write_jsonl(args.out_dir / "candidates.jsonl", rows)
    return 0 if all(row["status"] == "accepted" for row in rows) else 1


def export_command(args: argparse.Namespace) -> int:
    export_rows(list(_read_jsonl(args.candidates)), out_dir=args.out_dir, seed=args.seed)
    return 0


def export_rows(rows: Iterable[Mapping[str, object]], *, out_dir: Path, seed: int) -> None:
    """Filter provider rows, preserve rejections, and export family-safe CSV splits."""
    candidates: list[Candidate] = []
    rejected_rows: list[dict[str, object]] = []
    for row in rows:
        try:
            candidates.append(_candidate_from_row(row))
        except ValueError as error:
            rejected_rows.append({**dict(row), "reason": f"invalid_response: {error}"})

    filtered = filter_candidates(candidates)
    accepted = filtered.accepted
    rejected_rows.extend(
        {**asdict(rejection.candidate), "reason": rejection.reason}
        for rejection in filtered.rejected
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "accepted.jsonl", (asdict(candidate) for candidate in accepted))
    write_jsonl(out_dir / "rejections.jsonl", rejected_rows)

    splits = split_candidates(accepted, seed=seed)
    for name, split in splits.items():
        _write_csv(out_dir / f"{name}.csv", split)


def _candidate_from_row(row: Mapping[str, object]) -> Candidate:
    required_strings = ("id", "label", "family_id", "recipe_id")
    values: dict[str, str] = {}
    for name in required_strings:
        value = row.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"candidate row requires a nonempty {name}")
        values[name] = value
    slots = row.get("slots")
    if not isinstance(slots, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in slots.items()
    ):
        raise ValueError("candidate row requires string slots")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("candidate row requires a nonempty question")
    return Candidate(question=question.strip(), slots=slots, **values)


def _write_csv(path: Path, candidates: Iterable[Candidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["soru", "sinif"])
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({"soru": candidate.question, "sinif": candidate.label})


def generate_rows(
    plans: Iterable[Mapping[str, object]],
    *,
    api_key: str,
    model: str,
    post: Callable[..., dict[str, object]],
) -> Iterable[dict[str, object]]:
    """Generate one auditable candidate per plan through an injected HTTP boundary."""
    for row in plans:
        plan = _plan_from_row(row)
        audit = dict(row)
        try:
            payload = post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=build_openai_request(plan, model=model),
            )
            output_text = _response_output_text(payload)
            audit.update(
                {
                    "question": parse_generated_question(output_text),
                    "provider_response": output_text,
                    "provider_model": model,
                    "status": "accepted",
                }
            )
        except Exception as error:  # Provider failures are audit artifacts, not lost plans.
            audit.update(
                {
                    "question": None,
                    "provider_model": model,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        yield audit


def _plan_from_row(row: Mapping[str, object]) -> CandidatePlan:
    required_strings = ("id", "label", "family_id", "recipe_id", "prompt")
    values: dict[str, str] = {}
    for name in required_strings:
        value = row.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"plan row requires a nonempty {name}")
        values[name] = value
    slots = row.get("slots")
    if not isinstance(slots, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in slots.items()
    ):
        raise ValueError("plan row requires string slots")
    return CandidatePlan(slots=slots, **values)


def _response_output_text(payload: Mapping[str, object]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    raise ValueError("provider response has no output text")


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            yield row


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return plan_command(args)
    if args.command == "generate":
        return generate_command(args, parser)
    if args.command == "export":
        return export_command(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
