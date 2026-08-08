# Intent Fine-Tune Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable, recipe-driven Turkish intent-dataset generator that exports `soru,sinif` CSV files and makes OpenAI generation opt-in.

**Architecture:** `data/intent_dataset/recipes.json` owns the ten-label scenario vocabulary and compatible slot recipes. `backend/app/intent_dataset.py` is a dependency-light pure domain module for planning, parsing, filtering, splitting, and artifact writing; `scripts/generate_intent_dataset.py` is the CLI adapter that adds optional OpenAI Responses API calls through `httpx`.

**Tech Stack:** Python 3.10+, standard library, existing `httpx`, existing `pytest`.

## Global Constraints

- Preserve exactly the existing ten label names from `backend/app/data/atbk_knowledge_base.json`.
- The public CSV has exactly `soru,sinif` columns in that order and UTF-8 encoding.
- Never call a network API unless the user passes `--execute`; dry-run requires no `OPENAI_API_KEY`.
- Read `OPENAI_API_KEY` only from the environment and never emit it in logs, JSONL, or CSV.
- Keep all rejected candidates with a machine-readable rejection reason.
- Keep recipe families intact when splitting 70/15/15 per label.
- Do not drop a valid candidate because an evaluated classifier misroutes it.

---

### Task 1: Define recipe data and deterministic candidate planning

**Files:**
- Create: `data/intent_dataset/recipes.json`
- Create: `backend/tests/test_intent_dataset.py`
- Create: `backend/app/intent_dataset.py`

**Interfaces:**
- Consumes: JSON recipe document with `version`, `labels`, and per-label `recipes`.
- Produces: `load_recipes(path: Path) -> RecipeBook` and `plan_candidates(book: RecipeBook, *, per_label: int, seed: int) -> list[CandidatePlan]`.
- `CandidatePlan` exposes `id`, `label`, `family_id`, `recipe_id`, `slots`, and `prompt`.

- [ ] **Step 1: Write the failing planning tests**

```python
def test_plan_candidates_is_reproducible_and_balanced() -> None:
    book = load_recipes(_recipes_path())
    first = plan_candidates(book, per_label=3, seed=17)
    second = plan_candidates(book, per_label=3, seed=17)

    assert first == second
    assert Counter(item.label for item in first) == {label: 3 for label in book.labels}
    assert all(item.family_id.startswith(f"{item.label}:") for item in first)
```

- [ ] **Step 2: Run the focused test to verify RED**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q`

Expected: FAIL because `app.intent_dataset` does not exist.

- [ ] **Step 3: Add the recipe document and minimal planning domain module**

Implement immutable recipe and plan dataclasses. The JSON must contain at least two compatible recipes per label, each with a stable recipe identifier and explicit slot values. `plan_candidates` must use `random.Random(seed)`, rotate/select recipes without global cross-label slot mixing, and build a strict Turkish-generation prompt from the label, scenario, slots, and noise mode.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q`

Expected: PASS for deterministic output, exact ten-label coverage, and invalid schema rejection.

- [ ] **Step 5: Commit the planning layer**

```bash
git add data/intent_dataset/recipes.json backend/app/intent_dataset.py backend/tests/test_intent_dataset.py
git commit -m "feat: add intent dataset recipe planner"
```

### Task 2: Add response validation, duplicate rejection, and family-safe splits

**Files:**
- Modify: `backend/app/intent_dataset.py`
- Modify: `backend/tests/test_intent_dataset.py`

**Interfaces:**
- Consumes: `CandidatePlan` plus a provider response text.
- Produces: `parse_generated_question(response_text: str) -> str`, `filter_candidates(candidates: list[Candidate]) -> FilterResult`, and `split_candidates(candidates: list[Candidate], *, seed: int) -> dict[str, list[Candidate]]`.
- `FilterResult` exposes `accepted` and `rejected`; every rejected candidate has one of `invalid_response`, `duplicate`, `near_duplicate`, or `recipe_mismatch`.

- [ ] **Step 1: Write the failing filter and split tests**

```python
def test_filter_rejects_normalized_duplicates() -> None:
    first = _candidate(question="Telsiz çağrı işareti nasıl verilir")
    duplicate = _candidate(question="telsiz cagri isareti nasil verilir")

    result = filter_candidates([first, duplicate])

    assert result.accepted == [first]
    assert result.rejected[0].reason == "duplicate"
```

- [ ] **Step 2: Run the focused test to verify RED**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q`

Expected: FAIL because filtering and splitting APIs are unavailable.

- [ ] **Step 3: Implement the smallest quality gate and split algorithm**

Normalize case and Turkish-character substitutions before exact duplicate detection. Reject malformed multi-line/provider responses and empty or overly long questions. Use family IDs to assign a whole family to one split, preserving 70/15/15 examples per label. Keep duplicate and malformed candidates in `rejected` rather than deleting them.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q`

Expected: PASS for duplicate auditability and leakage-free 70/15/15 splitting.

- [ ] **Step 5: Commit the quality layer**

```bash
git add backend/app/intent_dataset.py backend/tests/test_intent_dataset.py
git commit -m "feat: filter and split intent candidates"
```

### Task 3: Add dry-run artifacts and opt-in OpenAI generation CLI

**Files:**
- Create: `scripts/generate_intent_dataset.py`
- Modify: `backend/app/intent_dataset.py`
- Modify: `backend/tests/test_intent_dataset.py`
- Create: `docs/intent-dataset-generator.md`

**Interfaces:**
- CLI: `python3 scripts/generate_intent_dataset.py plan --recipes ... --out-dir ... --per-label 140 --seed 17`
- CLI: `python3 scripts/generate_intent_dataset.py generate --plan ... --out-dir ... --execute --model ...`
- CLI: `python3 scripts/generate_intent_dataset.py export --candidates ... --out-dir ... --seed 17`
- OpenAI request: `POST https://api.openai.com/v1/responses` with bearer authentication, `store: false`, and strict JSON schema under `text.format`.

- [ ] **Step 1: Write the failing CLI/artifact tests**

```python
def test_plan_command_writes_requests_without_api_key(tmp_path: Path) -> None:
    result = _run_cli("plan", "--out-dir", str(tmp_path), "--per-label", "2")

    assert result.returncode == 0
    assert (tmp_path / "requests.jsonl").exists()
    assert "OPENAI_API_KEY" not in (tmp_path / "requests.jsonl").read_text()


def test_generate_requires_execute_and_key(tmp_path: Path) -> None:
    result = _run_cli("generate", "--plan", str(_plan_file(tmp_path)), "--out-dir", str(tmp_path))

    assert result.returncode != 0
    assert "--execute" in result.stderr
```

- [ ] **Step 2: Run the focused test to verify RED**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q`

Expected: FAIL because the CLI and artifact writers do not exist.

- [ ] **Step 3: Implement the CLI and provider boundary**

Use `argparse` subcommands. `plan` writes deterministic `requests.jsonl` with resolved prompts and metadata but no secret. `generate` refuses to run unless both `--execute` and a nonempty `OPENAI_API_KEY` are present; it sends one request at a time through an injectable `httpx.Client`, posts `store: false`, parses only the JSON question field, and records raw response text plus status in `candidates.jsonl`. `export` filters candidates, writes `accepted.jsonl`, `rejections.jsonl`, and per-split `train.csv`, `validation.csv`, `test.csv` with `soru,sinif` headers.

- [ ] **Step 4: Run focused tests and a network-free dry-run**

Run:

```bash
PYTHONPATH=backend python3 -m pytest backend/tests/test_intent_dataset.py -q
PYTHONPATH=backend python3 scripts/generate_intent_dataset.py plan \
  --recipes data/intent_dataset/recipes.json \
  --out-dir /tmp/intent-dataset-dry-run --per-label 2 --seed 17
```

Expected: all focused tests pass; the command writes 20 deterministic request rows without a network request or API key.

- [ ] **Step 5: Document local review and explicit execution**

Document the dry-run command, artifact meanings, environment-only key setup, explicit `--execute` command template, the no-secret rule, and the rule against filtering examples merely because a benchmark model misses them.

- [ ] **Step 6: Commit the CLI and documentation**

```bash
git add scripts/generate_intent_dataset.py backend/app/intent_dataset.py backend/tests/test_intent_dataset.py docs/intent-dataset-generator.md
git commit -m "feat: add intent dataset generation CLI"
```

### Task 4: Verify the complete feature

**Files:**
- Verify: `backend/tests/test_intent_dataset.py`
- Verify: `scripts/generate_intent_dataset.py`

- [ ] **Step 1: Run complete backend tests**

Run: `cd backend && python3 -m pytest -q`

Expected: all tests pass. If the local interpreter is lower than the documented Python 3.10 minimum or required packages are unavailable, report the exact environment blocker separately from feature-test results.

- [ ] **Step 2: Run static and artifact checks**

Run:

```bash
python3 -m py_compile backend/app/intent_dataset.py scripts/generate_intent_dataset.py
git diff --check
git status --short
```

Expected: no syntax errors, no whitespace errors, and only intentional branch changes before commit.

