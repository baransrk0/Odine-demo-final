#!/usr/bin/env python3
"""Measure llama-server prefill reuse across the routed agent prompts.

Each agent carries its own constant system prompt (`Settings.agent_system_prompts`),
and llama-server only skips prefill when a slot already holds that exact prefix.
With fewer slots than agents the prompts evict each other, so every agent switch
pays a full prefill -- invisible in accuracy numbers, plainly visible in the time
to the first token.

This script asks the same questions the demo asks, in the order the demo asks
them, and reports time-to-first-token split two ways:

  cold   the first turn an agent is used in this run. Nothing is cached yet, so
         this is the full prefill cost and it is expected to be slow.
  warm   every later turn for that same agent. If the slot survived, the prefix
         is already resident and this should be far below cold. When warm is
         close to cold, the prompt cache is not being reused.

Two orderings, because they bracket the real behaviour:

  interleave (default)  Round-robin across agents, as intent routing produces.
                        The case that thrashes a single slot.
  grouped               Every question for one agent, then the next. The
                        best case; warm should be fast here even with one slot,
                        which confirms the measurement itself is sound.

Run from the repository root with the backend virtualenv:

    backend/.venv/bin/python scripts/bench_prefill.py --output before.json
    backend/.venv/bin/python scripts/bench_prefill.py --compare before.json after.json

Only prefill is measured, so generation is capped at a handful of tokens. The
script never prints an answer; questions come from the approved evaluation set.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import sys
import time

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import Settings  # noqa: E402


@dataclass
class Turn:
    """One measured request."""

    index: int
    agent: str
    label: str
    record_id: str
    phase: str  # "cold" on an agent's first turn, "warm" afterwards
    prompt_chars: int
    ttft_ms: float
    total_ms: float
    # Populated only when llama-server reports them; left null, never zeroed.
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    prompt_ms: float | None = None
    tokens_per_second: float | None = None


@dataclass
class AgentSummary:
    """Cold-versus-warm time to first token for one agent."""

    agent: str
    turns: int
    cold_ttft_ms: float | None
    warm_median_ttft_ms: float | None
    warm_max_ttft_ms: float | None
    # warm / cold. Near 1.0 means the prefix was re-prefilled every time.
    warm_cold_ratio: float | None


@dataclass
class Report:
    """The full run, dumped verbatim to the output file."""

    order: str
    base_url: str
    model: str
    cache_prompt: bool
    agents: list[str]
    total_turns: int
    cold_median_ttft_ms: float | None
    warm_median_ttft_ms: float | None
    warm_cold_ratio: float | None
    verdict: str
    by_agent: list[AgentSummary]
    slots_before: object | None = None
    slots_after: object | None = None
    turns: list[Turn] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure agent-prompt prefill reuse against llama-server."
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        type=Path,
        default=None,
        help="Print a delta between two previous runs and exit without measuring.",
    )
    parser.add_argument("--base-url", default=None, help="Defaults to llama_cpp_base_url.")
    parser.add_argument("--model", default=None, help="Defaults to llama_cpp_model.")
    parser.add_argument(
        "--order",
        default="interleave",
        choices=("interleave", "grouped"),
        help="interleave imitates intent routing; grouped is the best case.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Turns per agent. Needs at least 2 for a warm measurement.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Generation is irrelevant here; keep it small.",
    )
    parser.add_argument(
        "--no-cache-prompt",
        action="store_true",
        help="Send cache_prompt=false to measure the uncached floor.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded turn per agent before measuring.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--knowledge-base", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_plan(
    settings: Settings,
    order: str,
    rounds: int,
) -> list[tuple[str, str, str, str]]:
    """Return (agent, label, record_id, question) in the order to be asked.

    Questions come from the evaluation set, restricted to labels that actually
    reach the language model; function-call labels never build a prompt.
    """
    from app.knowledge import load_reference_answers

    answers = load_reference_answers(settings.knowledge_base_file)
    prompts = settings.agent_system_prompts

    by_agent: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for label in settings.taxonomy.labels:
        if label.function_call or label.agent not in prompts:
            continue
        pool = [pair for pair in answers if pair.label == label.name]
        for offset in range(rounds):
            if not pool:
                break
            pair = pool[offset % len(pool)]
            by_agent[label.agent].append(
                (label.agent, label.name, f"{label.name}#{offset + 1}", pair.question)
            )

    if not by_agent:
        raise SystemExit("Ölçülecek ajan promptu yok; bilgi tabanını kontrol edin.")

    if order == "grouped":
        return [turn for agent in sorted(by_agent) for turn in by_agent[agent]]

    plan: list[tuple[str, str, str, str]] = []
    for offset in range(rounds):
        for agent in sorted(by_agent):
            if offset < len(by_agent[agent]):
                plan.append(by_agent[agent][offset])
    return plan


async def measure_turn(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int,
    cache_prompt: bool,
) -> tuple[float, float, dict]:
    """Return (ttft_ms, total_ms, telemetry) for one streamed completion.

    Time to the first content token is the measurement that matters: it is the
    prefill the user waits through before any audio can start.
    """
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "cache_prompt": cache_prompt,
    }

    telemetry: dict = {}
    ttft_ms: float | None = None
    started = time.perf_counter()

    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=request
    ) as response:
        if response.status_code != 200:
            await response.aread()
            raise SystemExit(
                f"llama-server {response.status_code} döndürdü; sunucu ayakta mı?"
            )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if ttft_ms is None and _has_content(payload):
                ttft_ms = (time.perf_counter() - started) * 1000
            _absorb_telemetry(payload, telemetry)

    total_ms = (time.perf_counter() - started) * 1000
    if ttft_ms is None:
        raise SystemExit("llama-server hiç içerik üretmedi; modeli kontrol edin.")
    return ttft_ms, total_ms, telemetry


def _has_content(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    return bool(isinstance(delta, dict) and delta.get("content"))


def _absorb_telemetry(payload: object, sink: dict) -> None:
    """Keep whatever llama-server volunteers; every field is optional."""
    if not isinstance(payload, dict):
        return
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "cache_n", "prompt_cache_hit_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                sink[key] = value
    timings = payload.get("timings")
    if isinstance(timings, dict):
        for key in ("prompt_n", "prompt_ms", "cache_n", "predicted_per_second"):
            value = timings.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sink[key] = value


async def fetch_slots(client: httpx.AsyncClient, base_url: str) -> object | None:
    """Best-effort slot dump; llama-server may have the endpoint disabled."""
    try:
        response = await client.get(f"{base_url}/slots", timeout=5.0)
    except httpx.RequestError:
        return None
    if response.status_code != 200:
        return None
    try:
        slots = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(slots, list):
        return None
    return [
        {
            "id": slot.get("id"),
            "n_ctx": slot.get("n_ctx"),
            "prompt_chars": len(str(slot.get("prompt", ""))),
        }
        for slot in slots
        if isinstance(slot, dict)
    ]


async def run(args: argparse.Namespace) -> Report:
    settings = (
        Settings(knowledge_base_path=str(args.knowledge_base))
        if args.knowledge_base
        else Settings()
    )
    base_url = (args.base_url or settings.llama_cpp_base_url).rstrip("/")
    model = args.model or settings.llama_cpp_model
    cache_prompt = not args.no_cache_prompt
    prompts = settings.agent_system_prompts

    if args.rounds < 2:
        raise SystemExit("--rounds en az 2 olmalı; tek turda sıcak ölçüm yapılamaz.")

    plan = build_plan(settings, args.order, args.rounds)
    timeout = httpx.Timeout(args.timeout)

    turns: list[Turn] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=timeout) as client:
        slots_before = await fetch_slots(client, base_url)

        if args.warmup:
            for agent in sorted(prompts):
                question = next(
                    (item[3] for item in plan if item[0] == agent), None
                )
                if question is not None:
                    await measure_turn(
                        client, base_url, model, prompts[agent],
                        question, args.max_tokens, cache_prompt,
                    )

        for index, (agent, label, record_id, question) in enumerate(plan, start=1):
            phase = "warm" if agent in seen else "cold"
            seen.add(agent)
            ttft_ms, total_ms, telemetry = await measure_turn(
                client, base_url, model, prompts[agent],
                question, args.max_tokens, cache_prompt,
            )
            turns.append(
                Turn(
                    index=index,
                    agent=agent,
                    label=label,
                    record_id=record_id,
                    phase=phase,
                    prompt_chars=len(prompts[agent]),
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    prompt_tokens=_first_int(telemetry, "prompt_tokens", "prompt_n"),
                    cached_tokens=_first_int(
                        telemetry, "cache_n", "prompt_cache_hit_tokens"
                    ),
                    prompt_ms=_first_float(telemetry, "prompt_ms"),
                    tokens_per_second=_first_float(telemetry, "predicted_per_second"),
                )
            )
            print(
                f"  {index:>3}. {phase:<4} {agent:<20} ttft={ttft_ms:8.1f} ms",
                flush=True,
            )

        slots_after = await fetch_slots(client, base_url)

    by_agent = summarize_agents(turns)
    cold = _median([turn.ttft_ms for turn in turns if turn.phase == "cold"])
    warm = _median([turn.ttft_ms for turn in turns if turn.phase == "warm"])
    ratio = (warm / cold) if cold and warm else None

    return Report(
        order=args.order,
        base_url=base_url,
        model=model,
        cache_prompt=cache_prompt,
        agents=sorted(prompts),
        total_turns=len(turns),
        cold_median_ttft_ms=cold,
        warm_median_ttft_ms=warm,
        warm_cold_ratio=ratio,
        verdict=verdict_for(ratio),
        by_agent=by_agent,
        slots_before=slots_before,
        slots_after=slots_after,
        turns=turns,
    )


def verdict_for(ratio: float | None) -> str:
    """Turn the warm/cold ratio into the one sentence the run exists to produce."""
    if ratio is None:
        return "Yetersiz veri."
    if ratio > 0.8:
        return "Prompt cache kullanılmıyor: her ajan değişimi yeniden prefill ediyor."
    if ratio > 0.4:
        return "Prompt cache kısmen tutuyor; slotlar hâlâ birbirini düşürüyor."
    return "Prompt cache tutuyor: sıcak turlar prefill'i atlıyor."


def summarize_agents(turns: list[Turn]) -> list[AgentSummary]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        grouped[turn.agent].append(turn)

    summaries: list[AgentSummary] = []
    for agent in sorted(grouped):
        cold = next(
            (turn.ttft_ms for turn in grouped[agent] if turn.phase == "cold"), None
        )
        warm_values = [
            turn.ttft_ms for turn in grouped[agent] if turn.phase == "warm"
        ]
        warm_median = _median(warm_values)
        summaries.append(
            AgentSummary(
                agent=agent,
                turns=len(grouped[agent]),
                cold_ttft_ms=cold,
                warm_median_ttft_ms=warm_median,
                warm_max_ttft_ms=max(warm_values) if warm_values else None,
                warm_cold_ratio=(warm_median / cold) if cold and warm_median else None,
            )
        )
    return summaries


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _first_int(source: dict, *keys: str) -> int | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _first_float(source: dict, *keys: str) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def print_report(report: Report) -> None:
    print()
    print(f"order={report.order} cache_prompt={report.cache_prompt} turns={report.total_turns}")
    print(f"cold  median ttft : {_fmt(report.cold_median_ttft_ms)}")
    print(f"warm  median ttft : {_fmt(report.warm_median_ttft_ms)}")
    print(f"warm/cold         : {_fmt_ratio(report.warm_cold_ratio)}")
    print()
    print(f"{'agent':<22}{'turns':>6}{'cold ms':>11}{'warm ms':>11}{'ratio':>9}")
    for summary in report.by_agent:
        print(
            f"{summary.agent:<22}{summary.turns:>6}"
            f"{_fmt(summary.cold_ttft_ms):>11}"
            f"{_fmt(summary.warm_median_ttft_ms):>11}"
            f"{_fmt_ratio(summary.warm_cold_ratio):>9}"
        )
    print()
    print(report.verdict)


def print_comparison(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    print(f"{before_path.name} -> {after_path.name}")
    print()
    for key in ("cold_median_ttft_ms", "warm_median_ttft_ms"):
        old, new = before.get(key), after.get(key)
        print(f"{key:<22}{_fmt(old):>11}{_fmt(new):>11}  {_delta(old, new)}")
    print(f"{'warm/cold':<22}{_fmt_ratio(before.get('warm_cold_ratio')):>11}"
          f"{_fmt_ratio(after.get('warm_cold_ratio')):>11}")
    print()

    old_agents = {item["agent"]: item for item in before.get("by_agent", [])}
    print(f"{'agent':<22}{'warm before':>13}{'warm after':>12}{'':>4}")
    for item in after.get("by_agent", []):
        old = old_agents.get(item["agent"], {}).get("warm_median_ttft_ms")
        new = item.get("warm_median_ttft_ms")
        print(
            f"{item['agent']:<22}{_fmt(old):>13}{_fmt(new):>12}  {_delta(old, new)}"
        )
    print()
    print(f"before: {before.get('verdict', '')}")
    print(f"after : {after.get('verdict', '')}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _delta(old: float | None, new: float | None) -> str:
    if old is None or new is None or not old:
        return ""
    change = (new - old) / old * 100
    return f"{change:+.0f}%"


def main() -> int:
    args = parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return 0

    report = asyncio.run(run(args))
    print_report(report)

    if args.output:
        args.output.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nrapor: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
