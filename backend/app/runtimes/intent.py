"""HTTP adapter for the mDeBERTa zero-shot NLI service.

UNVERIFIED CONTRACT. `_build_request` and `_parse_prediction` below assume the
service speaks the Hugging Face `zero-shot-classification` pipeline shape:

    -> {"sequence": ..., "candidate_labels": [...], "hypothesis_template": ...}
    <- {"labels": [...], "scores": [...]}      (descending, aligned)

The parser also accepts `[{"label": ..., "score": ...}, ...]`. If the deployed
service on :6006 speaks anything else, those two functions are the only places
that need to change -- the engine, the routing table, and the evaluation harness
are all written against `IntentPrediction`, not against this wire format.
"""

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.intents.taxonomy import HYPOTHESIS_TEMPLATE
from app.runtimes.protocols import IntentPrediction


_UNSUPPORTED_HEALTH_STATUSES = frozenset({404, 405, 501})


class ZeroShotIntentClassifier:
    """Score candidate intent labels through a local zero-shot NLI server."""

    def __init__(
        self,
        base_url: str,
        *,
        classify_path: str = "/classify",
        model: str = "",
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
        verbalizations: Mapping[str, str] | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/{classify_path.lstrip('/')}"
        self._host_label = urlsplit(base_url).hostname or "configured"
        self._model = model
        self._hypothesis_template = hypothesis_template
        # Labels are terse; the service is given the phrase each one stands for
        # and the answer is mapped back to the label name it was sent under.
        self._verbalizations = dict(verbalizations or {})
        self._timeout_seconds = timeout_seconds
        self.ready = False

    def load(self) -> None:
        """Mark a configured local HTTP runtime ready without loading a model."""
        self.ready = True

    async def health(self) -> bool:
        """Probe the service without revealing its configuration."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
                response = await client.get(f"{self._base_url}/health")
                if 200 <= response.status_code < 300:
                    healthy = True
                elif response.status_code in _UNSUPPORTED_HEALTH_STATUSES:
                    fallback = await client.get(f"{self._base_url}/")
                    healthy = 200 <= fallback.status_code < 300
                else:
                    healthy = False
        except httpx.HTTPError:
            healthy = False

        self.ready = healthy
        return healthy

    async def classify(
        self,
        text: str,
        labels: Sequence[str],
    ) -> IntentPrediction:
        """Return the winning label and the full score distribution."""
        if not labels:
            raise RuntimeError("intent classification needs at least one label")

        # Sent under their verbalizations, read back under their label names.
        sent = {self._verbalize(label): label for label in labels}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds)
            ) as client:
                response = await client.post(
                    self._url,
                    json=self._build_request(text, tuple(sent)),
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError("intent classification request failed") from error

        scores = _parse_scores(payload)
        named = {
            sent[candidate]: score
            for candidate, score in scores.items()
            if candidate in sent
        }
        if not named:
            raise RuntimeError("intent service returned no known label")

        top = max(named, key=named.__getitem__)
        return IntentPrediction(label=top, confidence=named[top], scores=named)

    def _verbalize(self, label: str) -> str:
        return self._verbalizations.get(label, label)

    def _build_request(self, text: str, candidates: Sequence[str]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "sequence": text,
            "candidate_labels": list(candidates),
            "hypothesis_template": self._hypothesis_template,
            # Single-label routing: scores must compete, not stand alone.
            "multi_label": False,
        }
        if self._model:
            request["model"] = self._model
        return request


def _parse_scores(payload: Any) -> dict[str, float]:
    """Read a label/score mapping out of either supported response shape."""
    if isinstance(payload, dict):
        labels = payload.get("labels")
        scores = payload.get("scores")
        if isinstance(labels, list) and isinstance(scores, list):
            if len(labels) != len(scores):
                raise RuntimeError("intent service returned misaligned scores")
            return {
                str(label): _as_score(score)
                for label, score in zip(labels, scores)
            }
        payload = payload.get("predictions", payload.get("results"))

    if isinstance(payload, list):
        parsed: dict[str, float] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                raise RuntimeError("intent service returned an invalid prediction")
            label = entry.get("label")
            if not isinstance(label, str):
                raise RuntimeError("intent service returned an unlabelled score")
            parsed[label] = _as_score(entry.get("score"))
        if parsed:
            return parsed

    raise RuntimeError("intent service returned an invalid response")


def _as_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("intent service returned a non-numeric score")
    return float(value)
