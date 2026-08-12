#!/usr/bin/env python3
"""Prepare and generate an auditable Turkish intent fine-tuning dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping

from tqdm.auto import tqdm

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
PRICE_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-5.6-terra": (2.50, 15.00),
}


def generation_progress(*, total: int, completed: int):
    """Show progress over the full deterministic plan, including resumed work."""
    return tqdm(
        total=total,
        initial=completed,
        desc="Intent dataset",
        unit="soru",
        dynamic_ncols=True,
    )


def build_review_request(
    label: str, candidates: list[dict[str, str]], *, model: str
) -> dict[str, object]:
    """Build a review-only request for all candidates in one intent label."""
    if not label.strip() or not model.strip() or not candidates:
        raise ValueError("label, model, and candidates are required")
    items = json.dumps(candidates, ensure_ascii=False)
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "none"},
        "input": (
            "Yalnızca geçerli JSON döndür. Aşağıdaki adayları aynı intent sınıfı içinde "
            "değerlendir. Yeniden yazma, soru üretme, cevap verme veya metni değiştirme. "
            "Her aday için tam bir karar döndür: keep doğal, açık ve yeterince farklıysa; "
            "revise anlam korunarak daha sonra düzeltilebilecekse; reject anlamsız, etikete "
            "uyumsuz veya aynı sınıftaki daha iyi bir adayın gereksiz tekrarıysa. "
            "too_similar için duplicate_of alanına tercih edilen adayın id'sini yaz; diğer "
            "durumlarda null yaz.\n"
            f"Intent sınıfı: {label}\nAdaylar: {items}"
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "intent_review",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reviews": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "decision": {"type": "string", "enum": ["keep", "revise", "reject"]},
                                    "reason_code": {
                                        "type": "string",
                                        "enum": [
                                            "good",
                                            "too_similar",
                                            "unclear_meaning",
                                            "unnatural_language",
                                            "label_mismatch",
                                            "not_a_question",
                                        ],
                                    },
                                    "duplicate_of": {"type": ["string", "null"]},
                                },
                                "required": ["id", "decision", "reason_code", "duplicate_of"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["reviews"],
                    "additionalProperties": False,
                },
            }
        },
    }


def parse_review_results(response_text: str, *, expected_ids: set[str]) -> list[dict[str, str | None]]:
    """Validate that the reviewer decided exactly once for each supplied candidate."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("review response must be a JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {"reviews"} or not isinstance(payload["reviews"], list):
        raise ValueError("review response must contain only a reviews array")

    allowed_decisions = {"keep", "revise", "reject"}
    allowed_reasons = {
        "good",
        "too_similar",
        "unclear_meaning",
        "unnatural_language",
        "label_mismatch",
        "not_a_question",
    }
    parsed: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for review in payload["reviews"]:
        if not isinstance(review, dict) or set(review) != {"id", "decision", "reason_code", "duplicate_of"}:
            raise ValueError("each review must contain id, decision, reason_code, and duplicate_of")
        candidate_id = review["id"]
        decision = review["decision"]
        reason_code = review["reason_code"]
        duplicate_of = review["duplicate_of"]
        if not isinstance(candidate_id, str) or candidate_id not in expected_ids or candidate_id in seen:
            raise ValueError("review ids must identify each supplied candidate exactly once")
        if not isinstance(decision, str) or decision not in allowed_decisions:
            raise ValueError("review decision is invalid")
        if not isinstance(reason_code, str) or reason_code not in allowed_reasons:
            raise ValueError("review reason_code is invalid")
        if duplicate_of is not None and (
            not isinstance(duplicate_of, str)
            or duplicate_of not in expected_ids
            or duplicate_of == candidate_id
        ):
            raise ValueError("review duplicate_of is invalid")
        if reason_code == "too_similar" and duplicate_of is None:
            raise ValueError("too_similar reviews require duplicate_of")
        seen.add(candidate_id)
        parsed.append(
            {
                "id": candidate_id,
                "decision": decision,
                "reason_code": reason_code,
                "duplicate_of": duplicate_of,
            }
        )
    if seen != expected_ids:
        raise ValueError("review must decide each supplied candidate exactly once")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write deterministic provider requests without a network call")
    plan.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    plan.add_argument("--out-dir", type=Path, required=True)
    plan.add_argument("--per-label", type=int, default=140)
    plan.add_argument("--seed", type=int, default=17)
    plan.add_argument("--start-index", type=int, default=0)

    followup = subparsers.add_parser("plan-followup", help="plan only the post-review candidate deficits")
    followup.add_argument("--reviews", type=Path, required=True)
    followup.add_argument("--out-dir", type=Path, required=True)
    followup.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    followup.add_argument("--target-per-label", type=int, default=100)
    followup.add_argument("--oversample", type=float, default=3.0)
    followup.add_argument("--seed", type=int, default=23)
    followup.add_argument("--start-index", type=int, default=140)

    generate = subparsers.add_parser("generate", help="call OpenAI only after explicit approval")
    generate.add_argument("--plan", type=Path, required=True)
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--execute", action="store_true")
    generate.add_argument("--model", default="gpt-5-mini")

    review = subparsers.add_parser("review", help="review generated candidates with an LLM without rewriting them")
    review.add_argument("--candidates", type=Path, required=True)
    review.add_argument("--out-dir", type=Path, required=True)
    review.add_argument("--execute", action="store_true")
    review.add_argument("--model", default="gpt-5.6-terra")

    fill = subparsers.add_parser("fill", help="generate exactly the post-review deficits per label")
    fill.add_argument("--candidates", type=Path, required=True)
    fill.add_argument("--reviews", type=Path, required=True)
    fill.add_argument("--out-dir", type=Path, required=True)
    fill.add_argument("--target-per-label", type=int, default=100)
    fill.add_argument("--execute", action="store_true")
    fill.add_argument("--model", default="gpt-5.6-terra")

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


def append_generation_record(path: Path, row: Mapping[str, object]) -> None:
    """Durably append one provider outcome before the next network request starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def successful_candidate_ids(path: Path) -> set[str]:
    """Return only successful IDs; error rows are deliberately eligible for retry."""
    if not path.exists():
        return set()
    return {
        row["id"]
        for row in _read_jsonl(path)
        if row.get("status") == "accepted" and isinstance(row.get("id"), str)
    }


def pending_plans(
    plans: Iterable[Mapping[str, object]], *, completed_ids: set[str]
) -> list[dict[str, object]]:
    """Skip completed plans while retaining provider failures for the next attempt."""
    return [
        dict(row)
        for row in plans
        if not isinstance(row.get("id"), str) or row["id"] not in completed_ids
    ]


def sync_live_csv(audit_path: Path, csv_path: Path) -> None:
    """Rebuild the visible CSV from durable accepted audit rows at process start."""
    rows = _read_jsonl(audit_path) if audit_path.exists() else ()
    accepted = [
        row
        for row in rows
        if row.get("status") == "accepted"
        and isinstance(row.get("question"), str)
        and isinstance(row.get("label"), str)
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["soru", "sinif"])
        writer.writeheader()
        for row in accepted:
            writer.writerow({"soru": row["question"], "sinif": row["label"]})
        handle.flush()
        os.fsync(handle.fileno())


def append_live_csv(csv_path: Path, row: Mapping[str, object]) -> None:
    """Durably mirror one accepted audit row into the user-facing CSV."""
    question = row.get("question")
    label = row.get("label")
    if row.get("status") != "accepted" or not isinstance(question, str) or not isinstance(label, str):
        return
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["soru", "sinif"])
        writer.writerow({"soru": question, "sinif": label})
        handle.flush()
        os.fsync(handle.fileno())


def plan_command(args: argparse.Namespace) -> int:
    book = load_recipes(args.recipes)
    plans = plan_candidates(
        book,
        per_label=args.per_label,
        seed=args.seed,
        start_index=args.start_index,
    )
    write_jsonl(args.out_dir / "requests.jsonl", (_plan_row(item) for item in plans))
    return 0


def plan_followup_command(args: argparse.Namespace) -> int:
    if args.target_per_label <= 0 or args.oversample <= 0:
        raise ValueError("target-per-label and oversample must be positive")
    counts = followup_counts(
        list(_read_jsonl(args.reviews)),
        target_per_label=args.target_per_label,
        oversample=args.oversample,
    )
    book = load_recipes(args.recipes)
    plans = plan_candidates(
        book,
        per_label=counts,
        seed=args.seed,
        start_index=args.start_index,
    )
    write_jsonl(args.out_dir / "requests.jsonl", (_plan_row(item) for item in plans))
    (args.out_dir / "followup_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"followup planned={len(plans)} counts={json.dumps(counts, ensure_ascii=False, sort_keys=True)}")
    return 0


def followup_counts(
    review_rows: Iterable[Mapping[str, object]], *, target_per_label: int, oversample: float
) -> dict[str, int]:
    """Calculate a conservative follow-up generation count from keep/revise decisions."""
    survivors: dict[str, int] = {}
    for row in review_rows:
        if row.get("status") != "accepted":
            continue
        label = row.get("label")
        reviews = row.get("reviews")
        if not isinstance(label, str) or not isinstance(reviews, list):
            raise ValueError("accepted review rows require label and reviews")
        survivors[label] = sum(
            1
            for review in reviews
            if isinstance(review, dict) and review.get("decision") in {"keep", "revise"}
        )
    return {
        label: math.ceil(max(0, target_per_label - count) * oversample)
        for label, count in survivors.items()
        if count < target_per_label
    }


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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "candidates.jsonl"
    live_csv_path = args.out_dir / "candidates.csv"
    plans = list(_read_jsonl(args.plan))
    sync_live_csv(audit_path, live_csv_path)
    completed_ids = successful_candidate_ids(audit_path)
    pending = pending_plans(plans, completed_ids=completed_ids)
    print(f"resume completed={len(completed_ids)} pending={len(pending)}")

    progress = generation_progress(total=len(plans), completed=len(completed_ids))
    try:
        with httpx.Client(timeout=60.0) as client:
            def post(url: str, *, headers: dict[str, str], json: dict[str, object]) -> dict[str, object]:
                response = client.post(url, headers=headers, json=json)
                response.raise_for_status()
                return response.json()

            for row in generate_rows(
                pending,
                api_key=os.environ["OPENAI_API_KEY"],
                model=args.model,
                post=post,
            ):
                append_generation_record(audit_path, row)
                append_live_csv(live_csv_path, row)
                progress.update(1)
    finally:
        progress.close()

    rows = list(_read_jsonl(audit_path))
    summary = summarize_usage(rows)
    print(
        "usage "
        f"input_tokens={summary['input_tokens']} "
        f"output_tokens={summary['output_tokens']} "
        f"total_tokens={summary['total_tokens']} "
        f"estimated_cost_usd={summary['estimated_cost_usd']:.8f}"
        if summary["estimated_cost_usd"] is not None
        else "usage cost=unknown (supply a priced model)"
    )
    plan_ids = {row["id"] for row in plans if isinstance(row.get("id"), str)}
    return 0 if plan_ids <= successful_candidate_ids(audit_path) else 1


def review_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.execute:
        parser.error("review requires --execute; review requests can spend API credits")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("review requires OPENAI_API_KEY in the environment")
    try:
        import httpx
    except ImportError as error:
        parser.error(f"review requires httpx: {error}")

    grouped = reviewable_candidates(list(_read_jsonl(args.candidates)))
    if not grouped:
        parser.error("review requires at least one accepted generated candidate")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "reviews.jsonl"
    completed_labels = successful_review_labels(audit_path)
    pending = {label: candidates for label, candidates in grouped.items() if label not in completed_labels}
    print(f"resume reviewed={len(completed_labels)} pending={len(pending)}")

    progress = tqdm(
        total=len(grouped),
        initial=len(completed_labels & set(grouped)),
        desc="Intent review",
        unit="sınıf",
        dynamic_ncols=True,
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            def post(url: str, *, headers: dict[str, str], json: dict[str, object]) -> dict[str, object]:
                response = client.post(url, headers=headers, json=json)
                response.raise_for_status()
                return response.json()

            for row in review_rows(
                pending,
                api_key=os.environ["OPENAI_API_KEY"],
                model=args.model,
                post=post,
            ):
                append_generation_record(audit_path, row)
                progress.update(1)
    finally:
        progress.close()

    rows = list(_read_jsonl(audit_path))
    summary = summarize_usage(rows)
    print(
        "usage "
        f"input_tokens={summary['input_tokens']} "
        f"output_tokens={summary['output_tokens']} "
        f"total_tokens={summary['total_tokens']} "
        f"estimated_cost_usd={summary['estimated_cost_usd']:.8f}"
        if summary["estimated_cost_usd"] is not None
        else "usage cost=unknown (supply a priced model)"
    )
    return 0 if set(grouped) <= successful_review_labels(audit_path) else 1


def fill_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.execute:
        parser.error("fill requires --execute")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("fill requires OPENAI_API_KEY in the environment")
    candidates = {
        row["id"]: row
        for row in _read_jsonl(args.candidates)
        if row.get("status") == "accepted" and isinstance(row.get("id"), str)
    }
    canonical: dict[str, list[str]] = {}
    for batch in _read_jsonl(args.reviews):
        if batch.get("status") != "accepted":
            continue
        for decision in batch.get("reviews", []):
            if not isinstance(decision, dict) or decision.get("decision") not in {"keep", "revise"}:
                continue
            source = candidates.get(decision.get("id"))
            if source and isinstance(source.get("label"), str) and isinstance(source.get("question"), str):
                canonical.setdefault(source["label"], []).append(source["question"])
    gaps = {label: args.target_per_label - len(items) for label, items in canonical.items() if len(items) < args.target_per_label}
    if not gaps:
        return 0
    import httpx
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "fills.jsonl"
    completed = successful_review_labels(audit_path)
    with httpx.Client(timeout=60.0) as client:
        for label, amount in gaps.items():
            if label in completed:
                continue
            prompt = (
                "Yalnızca geçerli JSON döndür. Verilen sınıf için mevcut soruların olayını, "
                "anlamını ve kalıbını tekrar etmeden tam istenen sayıda doğal askerî/saha "
                "konuşması sorusu üret. Tek cümle yaz; cevap veya açıklama verme.\n"
                f"Sınıf: {label}\nİstenen yeni soru sayısı: {amount}\nMevcut sorular: "
                + json.dumps(canonical[label], ensure_ascii=False)
            )
            request = {
                "model": args.model, "store": False, "reasoning": {"effort": "none"}, "input": prompt,
                "text": {"format": {"type": "json_schema", "name": "intent_fill", "strict": True,
                    "schema": {"type": "object", "properties": {"questions": {"type": "array",
                    "minItems": amount, "maxItems": amount, "items": {"type": "string"}}},
                    "required": ["questions"], "additionalProperties": False}}}
            }
            audit = {"label": label, "requested": amount, "provider_model": args.model}
            try:
                response = client.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, json=request)
                response.raise_for_status()
                text = _response_output_text(response.json())
                payload = json.loads(text)
                questions = payload.get("questions") if isinstance(payload, dict) else None
                if not isinstance(questions, list) or len(questions) != amount or not all(isinstance(q, str) and q.strip() for q in questions):
                    raise ValueError("fill response must contain exactly the requested nonempty questions")
                audit.update({"questions": [q.strip() for q in questions], "status": "accepted"})
            except Exception as error:
                audit.update({"questions": None, "status": "error", "error": f"{type(error).__name__}: {error}"})
            append_generation_record(audit_path, audit)
    return 0 if set(gaps) <= successful_review_labels(audit_path) else 1


def reviewable_candidates(rows: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, str]]]:
    """Group successful generator results by label and reject duplicate candidate IDs."""
    grouped: dict[str, list[dict[str, str]]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        if row.get("status") != "accepted":
            continue
        candidate_id = row.get("id")
        label = row.get("label")
        question = row.get("question")
        if not all(isinstance(value, str) and value.strip() for value in (candidate_id, label, question)):
            raise ValueError("accepted generated candidates require id, label, and question")
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        grouped.setdefault(label, []).append({"id": candidate_id, "question": question})
    return grouped


def successful_review_labels(path: Path) -> set[str]:
    """Return labels whose complete review result was durably recorded."""
    if not path.exists():
        return set()
    return {
        row["label"]
        for row in _read_jsonl(path)
        if row.get("status") == "accepted" and isinstance(row.get("label"), str)
    }


def review_rows(
    grouped: Mapping[str, list[dict[str, str]]],
    *,
    api_key: str,
    model: str,
    post: Callable[..., dict[str, object]],
) -> Iterable[dict[str, object]]:
    """Review each label as one durable LLM task without changing candidate wording."""
    for label, candidates in grouped.items():
        audit: dict[str, object] = {
            "label": label,
            "candidate_ids": [candidate["id"] for candidate in candidates],
            "provider_model": model,
        }
        try:
            payload = post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=build_review_request(label, candidates, model=model),
            )
            output_text = _response_output_text(payload)
            usage = _usage_from_response(payload)
            audit.update(
                {
                    "reviews": parse_review_results(
                        output_text,
                        expected_ids={candidate["id"] for candidate in candidates},
                    ),
                    "provider_response": output_text,
                    "usage": usage,
                    "estimated_cost_usd": estimate_cost_usd(usage, model=model),
                    "status": "accepted",
                }
            )
        except Exception as error:
            audit.update(
                {
                    "reviews": None,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        yield audit


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
            usage = _usage_from_response(payload)
            audit.update(
                {
                    "question": parse_generated_question(output_text),
                    "provider_response": output_text,
                    "provider_model": model,
                    "usage": usage,
                    "estimated_cost_usd": estimate_cost_usd(usage, model=model),
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


def estimate_cost_usd(usage: Mapping[str, int], *, model: str) -> float | None:
    """Estimate standard, non-cached text-token cost from the response usage object."""
    prices = PRICE_PER_MILLION.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return round(
        usage["input_tokens"] * input_price / 1_000_000
        + usage["output_tokens"] * output_price / 1_000_000,
        10,
    )


def summarize_usage(rows: Iterable[Mapping[str, object]]) -> dict[str, int | float | None]:
    """Sum successful response usage and its per-response standard-rate estimate."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cost = 0.0
    has_unknown_cost = False
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
        row_cost = row.get("estimated_cost_usd")
        if isinstance(row_cost, (int, float)):
            cost += float(row_cost)
        else:
            has_unknown_cost = True
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": None if has_unknown_cost else round(cost, 10),
    }


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


def _usage_from_response(payload: Mapping[str, object]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    values: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"provider usage {name} must be a non-negative integer")
        values[name] = value
    return values


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
    if args.command == "plan-followup":
        return plan_followup_command(args)
    if args.command == "generate":
        return generate_command(args, parser)
    if args.command == "review":
        return review_command(args, parser)
    if args.command == "fill":
        return fill_command(args, parser)
    if args.command == "export":
        return export_command(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
