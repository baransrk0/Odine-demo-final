#!/usr/bin/env python3
"""Score the intent layer against the 100-record ATBK evaluation set.

Follows the set's own `kullanim_notu`: each record's `stt_varyanti` -- the
lowercase, unpunctuated variant that imitates Whisper output -- is fed to the
pipeline and the returned label is compared against `beklenen_etiket`.

Two modes, because they answer different questions:

  engine (default)  What the deployed pipeline actually routes, clock rules and
                    confidence threshold included. This is the number that
                    predicts demo behaviour.
  --classifier-only Raw zero-shot accuracy with the rules and the threshold
                    switched off. Use this when tuning hypotheses or comparing
                    models; it is the only mode that shows what the classifier
                    does with the clock records on its own.

Run from the repository root with the backend virtualenv:

    backend/.venv/bin/python scripts/eval_intent.py --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.intents.engine import IntentEngine  # noqa: E402
from app.intents.taxonomy import HYPOTHESIS_TEMPLATE, load_taxonomy  # noqa: E402
from app.runtimes.intent import ZeroShotIntentClassifier  # noqa: E402


@dataclass
class Prediction:
    """One scored record."""

    record_id: str
    text: str
    expected: str
    predicted: str
    correct: bool
    expected_agent: str
    predicted_agent: str
    agent_correct: bool
    source: str
    confidence: float | None
    difficulty: str
    latency_ms: float


@dataclass
class LabelScore:
    """Per-label precision and recall, so a dominant class cannot hide a dead one."""

    label: str
    support: int
    correct: int
    predicted: int
    recall: float
    precision: float


@dataclass
class Report:
    """The full run, dumped verbatim to results.json."""

    mode: str
    model: str
    base_url: str
    threshold: float
    hypothesis_template: str
    total: int
    correct: int
    accuracy: float
    # Several labels share one agent, so a label-level miss between two military
    # classes routes to the same prompt and costs the operator nothing. Agent
    # accuracy is what the pipeline actually delivers; label accuracy is the
    # diagnostic underneath it.
    agent_correct: int
    agent_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    by_label: list[LabelScore]
    by_source: dict[str, int]
    by_difficulty: dict[str, str]
    confusions: list[str]
    predictions: list[Prediction] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure intent accuracy and latency against the ATBK set."
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=REPO_ROOT / "backend" / "app" / "data" / "atbk_knowledge_base.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--classify-path", default="/classify")
    parser.add_argument(
        "--model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the evaluation set's guven_esigi.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Bypass the clock rules and the threshold to score the model alone.",
    )
    parser.add_argument(
        "--field",
        default="stt_varyanti",
        choices=("stt_varyanti", "soru"),
        help="Which text to feed; the set specifies stt_varyanti.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print every misrouted record instead of just the summary.",
    )
    return parser.parse_args()


def load_records(path: Path, field_name: str, limit: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload.get("kayitlar", [])
        if isinstance(record, dict)
        and str(record.get(field_name, "")).strip()
        and str(record.get("beklenen_etiket", "")).strip()
    ]
    return records[:limit] if limit > 0 else records


async def run(args: argparse.Namespace) -> Report:
    taxonomy = load_taxonomy(args.knowledge_base)
    records = load_records(args.knowledge_base, args.field, args.limit)
    if not records:
        raise SystemExit("Evaluation set has no usable records.")

    classifier = ZeroShotIntentClassifier(
        args.base_url,
        classify_path=args.classify_path,
        model=args.model,
        verbalizations={label.name: label.verbalization for label in taxonomy.labels},
        timeout_seconds=args.timeout,
    )
    classifier.load()

    threshold = taxonomy.threshold if args.threshold is None else args.threshold
    engine = IntentEngine(
        taxonomy=taxonomy,
        classifier=classifier,
        timeout_seconds=args.timeout,
        threshold=threshold,
    )

    predictions: list[Prediction] = []
    for index, record in enumerate(records, start=1):
        text = str(record[args.field]).strip()
        expected = str(record["beklenen_etiket"]).strip()
        started = time.perf_counter()
        try:
            if args.classifier_only:
                # No rules, no threshold: the argmax the model actually produced.
                verdict = await classifier.classify(text, taxonomy.names)
                predicted, source, confidence = verdict.label, "classifier", verdict.confidence
            else:
                result = await engine.classify(text)
                predicted, source, confidence = result.label, result.source, result.confidence
        except Exception as error:  # noqa: BLE001 - a dead service must not abort the run
            predicted, source, confidence = "", f"error:{type(error).__name__}", None
        latency_ms = (time.perf_counter() - started) * 1000

        expected_agent = str(record.get("beklenen_ajan", "")).strip()
        predicted_label = taxonomy.get(predicted)
        predicted_agent = predicted_label.agent if predicted_label else ""

        predictions.append(
            Prediction(
                record_id=str(record.get("id", f"#{index}")),
                text=text,
                expected=expected,
                predicted=predicted,
                correct=predicted == expected,
                expected_agent=expected_agent,
                predicted_agent=predicted_agent,
                agent_correct=bool(expected_agent) and predicted_agent == expected_agent,
                source=source,
                confidence=confidence,
                difficulty=str(record.get("zorluk", "")),
                latency_ms=latency_ms,
            )
        )
        print(f"\r{index}/{len(records)}", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)

    return build_report(args, taxonomy, threshold, predictions)


def build_report(
    args: argparse.Namespace,
    taxonomy,
    threshold: float,
    predictions: list[Prediction],
) -> Report:
    latencies = sorted(item.latency_ms for item in predictions)
    correct = sum(1 for item in predictions if item.correct)
    agent_correct = sum(1 for item in predictions if item.agent_correct)

    by_label: list[LabelScore] = []
    for name in taxonomy.names:
        support = sum(1 for item in predictions if item.expected == name)
        hit = sum(1 for item in predictions if item.expected == name and item.correct)
        predicted = sum(1 for item in predictions if item.predicted == name)
        by_label.append(
            LabelScore(
                label=name,
                support=support,
                correct=hit,
                predicted=predicted,
                recall=round(hit / support, 4) if support else 0.0,
                precision=round(hit / predicted, 4) if predicted else 0.0,
            )
        )

    difficulty = Counter(item.difficulty for item in predictions if item.difficulty)
    difficulty_hits = Counter(
        item.difficulty for item in predictions if item.difficulty and item.correct
    )
    confusions = Counter(
        f"{item.expected} -> {item.predicted or '(none)'}"
        for item in predictions
        if not item.correct
    )

    return Report(
        mode="classifier-only" if args.classifier_only else "engine",
        model=args.model,
        base_url="configured",
        threshold=threshold,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        total=len(predictions),
        correct=correct,
        accuracy=round(correct / len(predictions), 4),
        agent_correct=agent_correct,
        agent_accuracy=round(agent_correct / len(predictions), 4),
        latency_p50_ms=round(percentile(latencies, 0.50), 2),
        latency_p95_ms=round(percentile(latencies, 0.95), 2),
        latency_max_ms=round(latencies[-1], 2) if latencies else 0.0,
        by_label=by_label,
        by_source=dict(Counter(item.source for item in predictions)),
        by_difficulty={
            level: f"{difficulty_hits[level]}/{count}"
            for level, count in sorted(difficulty.items())
        },
        confusions=[f"{pair} ({count})" for pair, count in confusions.most_common()],
        predictions=predictions,
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: always an observed latency, never above the max.

    Interpolating quantiles extrapolate past the sample on runs this small, which
    prints a p95 higher than the slowest measured call.
    """
    if not sorted_values:
        return 0.0
    rank = math.ceil(fraction * len(sorted_values))
    return sorted_values[min(len(sorted_values) - 1, max(0, rank - 1))]


def print_summary(report: Report, predictions: list[Prediction], show_errors: bool) -> None:
    print(f"\nmode={report.mode} threshold={report.threshold}")
    print(f"label     {report.correct}/{report.total} = {report.accuracy:.1%}")
    print(
        f"agent     {report.agent_correct}/{report.total} = {report.agent_accuracy:.1%}"
        "   <- what the operator actually gets"
    )
    print(
        f"latency   p50 {report.latency_p50_ms} ms  "
        f"p95 {report.latency_p95_ms} ms  max {report.latency_max_ms} ms"
    )

    print("\nlabel                   support  recall  precision")
    for score in report.by_label:
        print(
            f"{score.label:<22}{score.support:>9}"
            f"{score.recall:>8.1%}{score.precision:>11.1%}"
        )

    print("\nroute source:", ", ".join(f"{k}={v}" for k, v in report.by_source.items()))
    if report.by_difficulty:
        print("by difficulty:", ", ".join(f"{k}={v}" for k, v in report.by_difficulty.items()))
    if report.confusions:
        print("\nconfusions:")
        for line in report.confusions:
            print(f"  {line}")
    if show_errors:
        print("\nmisrouted:")
        for item in predictions:
            if not item.correct:
                print(
                    f"  {item.record_id} [{item.expected} -> {item.predicted or '(none)'}] "
                    f"{item.text}"
                )


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    print_summary(report, report.predictions, args.show_errors)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0 if report.accuracy > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
