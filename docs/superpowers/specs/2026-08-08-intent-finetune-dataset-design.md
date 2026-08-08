# Intent Fine-Tune Dataset Design

## Goal

Create an auditable, reproducible generator for a balanced Turkish intent-routing dataset. It will prepare 100 accepted utterances for each of the existing ten intent labels and export the requested `soru,sinif` CSV format. API generation is opt-in; reviewing the request plan must not require an API key or make network calls.

## Scope

The dataset preserves the repository's existing labels:

- `ilk yardım`
- `telsiz ve raporlama`
- `nöbet ve emniyet`
- `harita ve intikal`
- `mevzi ve gizlenme`
- `kbrn korunma`
- `angajman ve esir hukuku`
- `matematik`
- `sohbet`
- `saat`

The generator produces only short Turkish user utterances for intent classification. It does not generate answers, alter the backend routing implementation, train a model, or invoke the OpenAI API by default.

## Data Contract

The public training artifact is UTF-8 CSV with exactly these columns:

```csv
soru,sinif
arazide kanama var ilk ne yapayim,ilk yardım
```

`soru` is an STT-like user transcript, not a polished assistant-facing question. It may use lowercase text, missing punctuation, Turkish-character substitutions, or a controlled dropped filler word.

The pipeline keeps a separate JSONL audit trail. Each candidate records its identifier, label, recipe, sampled slots, prompt version, raw model response, quality decisions, and rejection reason. The CSV never contains API keys, request headers, or raw provider metadata.

## Controlled Generation

Each label owns a set of compatible scenario recipes. A recipe declares the allowed values for its slots, rather than allowing a global shuffle of unrelated attributes. Generic variation dimensions include utterance style, length, urgency, and STT-noise mode; label-specific slots describe the scenario.

Examples:

- `ilk yardım`: injury or symptom, setting, available equipment, requested action.
- `telsiz ve raporlama`: communication goal, radio state, report type, task phase.
- `harita ve intikal`: navigation objective, terrain or visibility, available tool, movement problem.
- `matematik`: operation, quantities, units, operational context.
- `sohbet`: dialogue act, tone, addressee framing.
- `saat`: time-request wording and urgency.

The CLI samples a seeded recipe and slot combination, then asks the generation model to return one question in a strict JSON response. The prompt forbids answers, explanations, multiple questions, label names, and unsupported operational facts. A dry-run mode writes the fully resolved requests without sending them.

## Quality Gate

Generate more than the target count per label, then retain 100 accepted candidates per label. A candidate is rejected only for data quality, not because a baseline classifier finds it difficult.

The gate applies, in order:

1. Schema and language checks: nonempty string, one question, Turkish text, allowed label, length bounds, and no generated answer or explanation.
2. Normalized exact and near-duplicate checks across the complete candidate pool.
3. Recipe-consistency checks against the sampled slots and forbidden wording.
4. Optional independent LLM label validation. The validator model must be configured separately from the generator; disagreement or inadequate confidence sends the candidate to review/rejection rather than silently relabelling it.
5. Manual audit queue: all validator disagreements plus a reproducible random sample from each label.
6. Per-label balancing: generation resumes only for labels that have fewer than 100 accepted candidates.

Every rejection is preserved in the audit artifact with a machine-readable reason.

## Dataset Splits and Evaluation

Recipes have stable family identifiers. All variants from one family remain in one split to prevent paraphrase leakage. The accepted 1,000-example dataset is divided per label into 70 training, 15 validation, and 15 final-test examples.

The final test is frozen before model selection. It is used unchanged to compare the current zero-shot route and later fine-tuned candidates under one evaluator. Report label macro-F1, per-label precision/recall, agent-routing accuracy, confusion matrix, and p50/p95 latency. A model error alone never makes an example low quality.

## CLI and Secrets

The first implementation provides a single Python CLI with commands for request planning, generation, filtering, split creation, and CSV export. OpenAI integration is enabled only by an explicit execution flag and reads `OPENAI_API_KEY` from the environment. The API key is never placed in configuration, audit output, logs, or version control.

Model names, output directories, random seed, candidate count, and target count are CLI options. The default seed makes candidate planning and split assignment reproducible.

## Verification

Unit tests cover deterministic slot sampling, recipe compatibility, response parsing, duplicate detection, rejection recording, per-label balance, family-safe splitting, CSV column order, and dry-run behavior without credentials or network access. A small fixture replaces the API client in tests.

The implementation is accepted only after focused dataset-generator tests pass and a dry-run produces inspectable, reproducible request and audit artifacts.
