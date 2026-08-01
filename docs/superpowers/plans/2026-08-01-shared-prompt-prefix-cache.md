# Shared Prompt-Prefix Cache Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the routed-agent system prompt so its constant `_HEADER` block comes before the agent-specific instruction, giving `llama-server`'s single KV-cache slot a guaranteed cache hit on that shared prefix on every agent switch, instead of zero reuse.

**Architecture:** One function, `build_system_prompt()` in `backend/app/knowledge.py`, changes its string-concatenation order. No new files, no signature changes, no changes to callers (`backend/app/config.py`). A benchmark script (`scripts/bench_prefill.py`) is recovered from a prior commit to measure the effect against a live `llama-server`.

**Tech Stack:** Python 3, pytest, httpx (bench script only).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-shared-prompt-prefix-cache-design.md`.
- Output content shown to the model must not change in meaning — only ordering of blocks that are already present.
- The empty-QA short-circuit (`if not pairs: return base_prompt`) stays exactly as-is; do not add handling for a case that does not occur in the current dataset (YAGNI, per spec).
- No changes to `AGENT_INSTRUCTIONS` text, `_HEADER` text, `_VOICE_CONSTRAINT` text, or `config.py`.

---

### Task 1: Reorder `build_system_prompt()` and prove the shared prefix

**Files:**
- Modify: `backend/app/knowledge.py:112-123` (`build_system_prompt`)
- Test: `backend/tests/test_intent.py`

**Interfaces:**
- Consumes: nothing new — `build_system_prompt(base_prompt: str, answers: list[ReferenceAnswer] | None = None) -> str`, `_HEADER: str` (module-level constant, unchanged), `ReferenceAnswer` (existing `NamedTuple` with `.question`, `.answer`, `.label`), already defined in `backend/app/knowledge.py`.
- Produces: same `build_system_prompt` signature and return type, only internal ordering changes. `Settings.agent_system_prompts` (`backend/app/config.py:97-129`) consumes this unchanged and is what later verification (Task 2) reads.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_intent.py` in the `# --- routed agent prompts ---` section (near `test_each_agent_prompt_carries_only_its_own_reference_answers`):

```python
def test_agent_prompts_share_an_identical_header_prefix():
    """The constant header must lead every agent prompt so a single llama-server
    slot can reuse it as a cache hit even when the previous turn used a
    different agent."""
    prompts = _settings().agent_system_prompts

    header_len = len(_HEADER)
    prefixes = {prompt[:header_len] for prompt in prompts.values()}

    assert len(prefixes) == 1
    assert prefixes.pop() == _HEADER


def test_agent_instruction_still_follows_the_header():
    """The header must lead, but each agent's own instruction must still be
    present right after it, not dropped or reordered further."""
    prompts = _settings().agent_system_prompts

    assert prompts["medikal"].index("sağlık asistanısın") > prompts["medikal"].index(_HEADER)
```

Add the import at the top of `backend/tests/test_intent.py` (check the existing import block first — if `app.knowledge` is not yet imported, add):

```python
from app.knowledge import _HEADER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_intent.py -k "share_an_identical_header or instruction_still_follows" -v`
Expected: FAIL — current order puts `base_prompt` (the agent instruction) first, so `prompts[...][:header_len]` will not equal `_HEADER`.

- [ ] **Step 3: Reorder the prompt assembly**

In `backend/app/knowledge.py`, change `build_system_prompt`:

```python
def build_system_prompt(
    base_prompt: str,
    answers: list[ReferenceAnswer] | None = None,
) -> str:
    """Prepend the reference block to the base instruction, base-only when empty.

    The reference header leads every prompt so llama-server's cache can reuse
    it as a shared prefix on an agent switch, even with a single KV-cache slot
    (see docs/superpowers/specs/2026-08-01-shared-prompt-prefix-cache-design.md).
    """
    pairs = answers if answers is not None else load_reference_answers()
    if not pairs:
        return base_prompt

    lines = [_HEADER, "", base_prompt, ""]
    lines.extend(f"S: {pair.question}\nC: {pair.answer}" for pair in pairs)
    return "\n".join(lines)
```

This is the only code change: `_HEADER` moves from the third list element to the
first, `base_prompt` moves from the first to the third. The one-line docstring
explaining *why* replaces the old one (same reason, updated for the new order).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_intent.py -v`
Expected: PASS — all tests in the file, including the two new ones and the
pre-existing ones (`test_each_agent_prompt_carries_only_its_own_reference_answers`,
`test_no_agent_prompt_teaches_the_model_the_clock_placeholder`,
`test_agent_prompts_are_built_once_and_stay_identical`,
`test_disabling_the_knowledge_base_leaves_instruction_only_agent_prompts`).

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && .venv/bin/pytest -v`
Expected: PASS, no regressions in `test_intent_routing.py` or elsewhere — those
tests assert prompt *identity* (`is`) and *substring* membership, not internal
ordering, per the spec's testing section.

- [ ] **Step 6: Commit**

```bash
git add backend/app/knowledge.py backend/tests/test_intent.py
git commit -m "perf: lead agent system prompts with the shared header

Puts the constant reference-answer header before the agent-specific
instruction so llama-server's single KV-cache slot can reuse it as a
shared prefix on an agent switch, instead of reprefilling from byte one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Recover the prefill benchmark script for manual verification

**Files:**
- Create: `scripts/bench_prefill.py` (recovered verbatim from commit `f7adf56`)

**Interfaces:**
- Consumes: `app.config.Settings` (imports `backend/app` via `sys.path.insert`, unchanged by Task 1).
- Produces: nothing consumed by other tasks — this is a standalone manual-run script, not part of the test suite.

- [ ] **Step 1: Recover the file from git history**

```bash
git show f7adf56:scripts/bench_prefill.py > scripts/bench_prefill.py
```

- [ ] **Step 2: Confirm it is byte-identical to the historical version and untouched by Task 1**

```bash
git show f7adf56:scripts/bench_prefill.py | diff - scripts/bench_prefill.py
```

Expected: no output (files identical). This script only reads
`Settings.agent_system_prompts` at request time; it needs no changes to work
with the Task 1 reorder.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_prefill.py
git commit -m "chore: recover bench_prefill.py from f7adf56 for cache-prefix verification

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual verification against a live llama-server (not automatable here — run on the Orin device or any host running llama-server per RUNBOOK.md)**

```bash
# Before Task 1's change is deployed to the running server (i.e. on main, or by
# temporarily reverting knowledge.py), capture a baseline:
backend/.venv/bin/python scripts/bench_prefill.py --order interleave --output before.json

# With Task 1's change deployed:
backend/.venv/bin/python scripts/bench_prefill.py --order interleave --output after.json

# Compare:
backend/.venv/bin/python scripts/bench_prefill.py --compare before.json after.json
```

Expected: `after.json`'s warm-median TTFT for the interleaved order is lower
than `before.json`'s — the shared-header segment of the prompt should no
longer be reprefilled on every agent switch. Per the spec, this is a partial
improvement (QA pairs still dominate prompt length and still get reprefilled
on a switch), not a full elimination of thrash.

This step has no pass/fail assertion in this plan because it requires a
running `llama-server` instance, which is not available in this repo's test
environment — record the actual before/after numbers when run.

---

## Self-Review Notes

- **Spec coverage:** "Change" section → Task 1. "Testing" section → Task 1 (unit
  tests) + Task 2 (bench recovery and manual run). "Scope" section (no changes
  outside `knowledge.py`) → respected, Task 1 touches only that file plus its
  test file. "Risks" section (bounded gain) → reflected in Task 2 Step 4's
  expected-outcome wording.
- **No placeholders:** every step has literal code or literal commands.
- **Type/name consistency:** `build_system_prompt(base_prompt, answers)` signature
  matches across Task 1's steps and the spec; `_HEADER` name matches the existing
  module constant in `backend/app/knowledge.py`.
