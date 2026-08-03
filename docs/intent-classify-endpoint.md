# `/classify` endpoint for the mDeBERTa service

The intent layer in this repo talks to the mDeBERTa service on the Orin
(port 6006). Alongside `/health` and `/embed`, that service exposes the
`/classify` endpoint specified below — **deployed and in use**. Zero-shot
routing needs it because NLI classification runs the model as a
**cross-encoder** over (transcript, hypothesis) pairs and reads its entailment
head — a score that pooled embeddings cannot reproduce.

This document is the contract `backend/app/runtimes/intent.py` is written
against; the deployed handler matches it, so neither side needs changing.

## Contract

`POST /classify`

```json
{
  "sequence": "turnike nasıl uygulanır",
  "candidate_labels": ["yaralı bakımı, ilk yardım ve tıbbi müdahale", "..."],
  "hypothesis_template": "Bu metin {} ile ilgilidir.",
  "multi_label": false,
  "model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
}
```

```json
{
  "sequence": "turnike nasıl uygulanır",
  "labels": ["yaralı bakımı, ilk yardım ve tıbbi müdahale", "..."],
  "scores": [0.87, 0.13]
}
```

Notes on the fields:

- `candidate_labels` carries **verbalizations**, not the bare class names.
  "saat" and "sohbet" are weak NLI hypotheses on their own, so the client sends
  the phrase each class stands for and maps the winner back to its label. The
  service does not need to know the ATBK classes at all.
- `labels` and `scores` must stay index-aligned and sorted descending. The
  client also accepts `[{"label": ..., "score": ...}, ...]` if that is easier.
- `multi_label` is always `false`: single-label routing needs the scores to
  compete via softmax across labels, not stand alone per label.
- `model` is informational. The service owns which model is loaded and should
  accept and ignore this field.

## Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

zero_shot = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,  # -1 for CPU
)


class ClassifyRequest(BaseModel):
    sequence: str = Field(min_length=1)
    candidate_labels: list[str] = Field(min_length=1)
    hypothesis_template: str = "Bu metin {} ile ilgilidir."
    multi_label: bool = False
    # Accepted and ignored: the service owns which model is loaded.
    model: str | None = None

    @field_validator("hypothesis_template")
    @classmethod
    def template_has_a_slot(cls, value: str) -> str:
        # The pipeline raises on a template without "{}". Rejecting it here
        # returns a 422 that names the problem instead of a 500 that doesn't.
        if "{}" not in value:
            raise ValueError("hypothesis_template must contain '{}'")
        return value


class ClassifyResponse(BaseModel):
    sequence: str
    labels: list[str]
    scores: list[float]


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Score candidate labels against one transcript.

    Declared `def`, not `async def`, on purpose: FastAPI runs a sync handler in
    its threadpool, so a GPU forward pass cannot stall the event loop and block
    /health for every other caller.
    """
    result = zero_shot(
        request.sequence,
        candidate_labels=request.candidate_labels,
        hypothesis_template=request.hypothesis_template,
        multi_label=request.multi_label,
    )
    return ClassifyResponse(
        sequence=request.sequence,
        labels=list(result["labels"]),
        scores=[float(score) for score in result["scores"]],
    )
```

## Integrating it without loading the model twice

A second `from_pretrained` costs roughly another gigabyte of resident memory on
a device already holding Whisper, Gemma, and Piper. Load the model **once** as
`AutoModelForSequenceClassification` and serve both endpoints from it:

- `/classify` uses the full model, entailment head included, as above.
- `/embed` can use the base encoder submodule of that same object — for this
  architecture, `model.deberta` — instead of a separate `AutoModel`. The encoder
  weights are then shared rather than duplicated.

If the existing service loads `AutoModel`, that path discards the classification
head at load time, which is why `/embed` alone cannot be made to do NLI.

One environment note for the Orin: set `OMP_NUM_THREADS` and
`TOKENIZERS_PARALLELISM` **before** torch and transformers are imported, or the
settings are ignored.

## Verifying it

```bash
curl -s -X POST http://127.0.0.1:6006/classify \
  -H 'content-type: application/json' \
  -d '{
        "sequence": "turnike nasıl uygulanır",
        "candidate_labels": [
          "yaralı bakımı, ilk yardım ve tıbbi müdahale",
          "askeri harekât, muharebe ve görev yönergeleri",
          "sayısal hesaplama ve matematik işlemi",
          "selamlaşma ve genel sohbet",
          "içinde bulunulan saatin sorulması"
        ],
        "hypothesis_template": "Bu metin {} ile ilgilidir.",
        "multi_label": false
      }' | python3 -m json.tool
```

The first entry of `labels` should be the medical verbalization. Then score the
whole set from the repository root:

```bash
backend/.venv/bin/python scripts/eval_intent.py --output intent_results.json
```

`route source: classifier=95, rule=5` in that output confirms the backend is
reaching the endpoint. `unavailable=95` means it is not.
