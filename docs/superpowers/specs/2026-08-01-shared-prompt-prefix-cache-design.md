# Shared prompt-prefix cache reuse (Approach A)

## Problem

`llama-server` runs with `--parallel 1` on the Orin device (see `RUNBOOK.md`), meaning
a single KV-cache slot. Intent routing selects one of four agent system prompts per
turn (`medikal`, `savaş yönergeleri`, `matematik`, `sohbet`, built in
`Settings.agent_system_prompts`, `backend/app/config.py`). Each prompt is built by
`build_system_prompt()` (`backend/app/knowledge.py`) as:

```
[agent_instruction, "", _HEADER, "", QA pairs...]
```

The agent-specific instruction is first, so the four prompts diverge from the very
first token. With one slot, routing to a different agent than the previous turn
evicts the cached prefix entirely and forces a full prefill — this is the behavior
`scripts/bench_prefill.py` (recovered from commit `f7adf56`, previously on the now
deleted `feat/improve-cache` branch) was written to measure, split into `cold`
(first use of an agent) and `warm` (repeat use) time-to-first-token.

## Change

Reorder `build_system_prompt()` so the constant, agent-independent block comes
first:

```
[_HEADER, "", agent_instruction, "", QA pairs...]
```

`_HEADER` is always included, even when the QA list for a label is empty, so every
agent prompt has an identical byte-for-byte prefix of the same length. This gives
`llama-server`'s longest-common-prefix prompt cache a guaranteed hit on that shared
segment on every request, regardless of which agent served the previous turn under
a single slot. The agent instruction and QA pairs remain agent-specific and are
still reprefilled on an agent switch — this change does not eliminate thrash, it
shrinks the portion of every prompt that must be reprefilled after a switch.

No change to `AGENT_INSTRUCTIONS` text, no change to what the model reads overall
(same content, same meaning), no change to the function signature.

## Scope

- `backend/app/knowledge.py`: reorder the block inside `build_system_prompt()`.
- No changes to `config.py`, `llm.py`, or the intent engine — they consume
  `build_system_prompt()`'s output unchanged.
- Out of scope (tracked as separate approaches, not part of this spec):
  increasing `--parallel` / per-agent slots (Approach B), shrinking per-agent
  knowledge base content (Approach C).

## Testing

- Existing tests (`backend/tests/test_intent.py`, `test_intent_routing.py`) assert
  prompt *identity* and *substring* membership, not internal ordering — confirmed
  by inspection, no test changes expected.
- Recover `scripts/bench_prefill.py` from commit `f7adf56` onto this branch.
- Run it in `interleave` mode before and after the reorder
  (`--output before.json` / `--output after.json`, then `--compare`) against a
  running `llama-server` instance, and confirm the shared-prefix segment is no
  longer reprefilled on every agent switch (visible as a drop in warm TTFT for
  the interleaved case, converging toward the grouped-order baseline).

## Risks

- None to correctness: output content to the model is unchanged, only its
  internal ordering.
- The gain is bounded by `_HEADER`'s size relative to the full prompt — QA pairs
  dominate prompt length per the RUNBOOK's ~17.7k character note, so expect a
  partial, not total, reduction in post-switch prefill cost. Approach B remains
  the option for eliminating cross-agent thrash entirely, pending Orin VRAM
  headroom.
