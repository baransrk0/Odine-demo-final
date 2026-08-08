# Intent dataset generator

This generator prepares a balanced Turkish intent-classification dataset for the ten labels already used by the demo. It does not train a model or change demo routing.

## Review without API access

Create deterministic, reviewable requests without an API key or network call:

```bash
PYTHONPATH=backend /opt/anaconda3/bin/python3.12 scripts/generate_intent_dataset.py plan \
  --recipes data/intent_dataset/recipes.json \
  --out-dir /tmp/intent-dataset-plan \
  --per-label 140 \
  --seed 17
```

The result is `requests.jsonl`. Each line records a candidate ID, assigned label, recipe family, sampled slots, and the fully resolved model prompt. It never includes a secret.

Recipe definitions are in `data/intent_dataset/recipes.json`. A recipe has class-compatible slots; do not shuffle variables across labels. Update recipes first, then regenerate the plan with a new documented seed.

## Explicit OpenAI generation

Generation is deliberately guarded. Export the key only in the shell that will make the request:

```bash
export OPENAI_API_KEY="..."
PYTHONPATH=backend /opt/anaconda3/bin/python3.12 scripts/generate_intent_dataset.py generate \
  --plan /tmp/intent-dataset-plan/requests.jsonl \
  --out-dir /tmp/intent-dataset-generated \
  --model gpt-5-mini \
  --execute
```

Without `--execute`, the script exits before creating an HTTP client. Without `OPENAI_API_KEY`, it also exits. The API request uses the Responses endpoint, strict `{ "soru": "..." }` JSON output, and `store: false`. The key is sent only in the bearer header and is not written to artifacts.

Generation writes `candidates.jsonl`. Each source request remains represented even if the provider response is malformed or the HTTP request fails; such rows have `status: "error"` and an error field.

After a successful run, the CLI prints aggregate `input_tokens`, `output_tokens`, `total_tokens`, and `estimated_cost_usd`. It also stores the per-response usage and estimate in `candidates.jsonl`. The current standard-rate map covers `gpt-5-mini` at `$0.25 / 1M` input and `$2.00 / 1M` output, plus `gpt-5.6-terra` at `$2.50 / 1M` input and `$15.00 / 1M` output. It does not apply a prompt-cache discount.

## Filter and export

After manual review of the generated candidates, export only once there are exactly 100 accepted candidates for each label:

```bash
PYTHONPATH=backend /opt/anaconda3/bin/python3.12 scripts/generate_intent_dataset.py export \
  --candidates /tmp/intent-dataset-generated/candidates.jsonl \
  --out-dir /tmp/intent-dataset-final \
  --seed 17
```

The export step:

- rejects empty, multiline, one-token, and normalized duplicate questions;
- retains rejected rows in `rejections.jsonl` with a reason;
- keeps accepted audit records in `accepted.jsonl`;
- keeps every recipe family in one split only;
- writes `train.csv`, `validation.csv`, and `test.csv`, each with exactly `soru,sinif` columns.

For 100 accepted candidates per label, the split is 70 training, 15 validation, and 15 frozen test examples per label.

Do not remove an example merely because a baseline classifier gets it wrong. Remove only malformed, duplicate, scenario-incompatible, or human-audited mislabelled examples. Keep model-disagreement and manual-review decisions in a separate audit file when adding the later independent validation pass.
