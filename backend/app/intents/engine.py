"""Rule-first intent routing with a zero-shot classifier as the fallback path.

Order matters. The deterministic rules are free and cannot be wrong about the
phrasings they cover, so they run first. The classifier is consulted only for
what the rules do not recognize, and its verdict is accepted only above the
evaluation set's confidence threshold -- a confidently wrong agent is worse for
the operator than a handoff to the default one.

The engine never fails a turn. A classifier that is down, slow, or incoherent
degrades to the default agent and records why, because losing intent routing
should cost answer quality, not the answer itself.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from typing import Literal

from app.intents import normalize, rules
from app.intents.taxonomy import IntentLabel, Taxonomy
from app.runtimes.protocols import IntentRuntime


logger = logging.getLogger(__name__)

IntentSource = Literal["rule", "classifier", "low_confidence", "unavailable"]


@dataclass(frozen=True, slots=True)
class IntentResult:
    """One routing decision and the evidence behind it."""

    label: str
    agent: str
    source: IntentSource
    confidence: float | None = None
    function_answer: str | None = None
    rag_collection: str = "none"
    elapsed_ms: float = 0.0

    @property
    def routed(self) -> bool:
        """Whether a rule or an accepted prediction chose this agent."""
        return self.source in ("rule", "classifier")


class IntentEngine:
    """Route one transcript to an agent, answering the clock without a model."""

    def __init__(
        self,
        *,
        taxonomy: Taxonomy,
        classifier: IntentRuntime | None = None,
        timeout_seconds: float = 5.0,
        threshold: float | None = None,
        clock: Callable[[], datetime] = rules.default_clock,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("intent timeout must be positive")
        self._taxonomy = taxonomy
        self._classifier = classifier
        self._timeout_seconds = timeout_seconds
        self._threshold = taxonomy.threshold if threshold is None else threshold
        self._clock = clock

    @property
    def taxonomy(self) -> Taxonomy:
        return self._taxonomy

    async def classify(self, transcript: str) -> IntentResult:
        """Return the routing decision for one transcript, never raising."""
        started_ns = time.perf_counter_ns()

        if rules.matches_clock(normalize.process(transcript)):
            return self._resolve(
                self._taxonomy.get("saat") or self._taxonomy.default_label,
                source="rule",
                confidence=None,
                started_ns=started_ns,
            )

        prediction = await self._predict(transcript)
        if prediction is None:
            return self._fallback("unavailable", None, started_ns)

        label = self._taxonomy.get(prediction.label)
        if label is None:
            logger.warning("Intent classifier returned an unconfigured label.")
            return self._fallback("unavailable", None, started_ns)
        if prediction.confidence < self._threshold:
            return self._fallback("low_confidence", prediction.confidence, started_ns)

        return self._resolve(
            label,
            source="classifier",
            confidence=prediction.confidence,
            started_ns=started_ns,
        )

    async def _predict(self, transcript: str):
        classifier = self._classifier
        if classifier is None:
            return None

        # Deliberately not gated on the startup probe's `ready` flag. That flag
        # is for health reporting; gating on it would leave routing dead for the
        # whole session if the service happened to be down at backend start.
        # Attempting every turn costs a refused connection and recovers by itself.
        try:
            return await asyncio.wait_for(
                classifier.classify(transcript, self._taxonomy.candidate_names),
                timeout=self._timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Intent classification timed out; using the default agent.")
            return None
        except Exception:
            logger.warning("Intent classification failed; using the default agent.")
            return None

    def _resolve(
        self,
        label: IntentLabel,
        *,
        source: IntentSource,
        confidence: float | None,
        started_ns: int,
    ) -> IntentResult:
        return IntentResult(
            label=label.name,
            agent=label.agent,
            source=source,
            confidence=confidence,
            function_answer=rules.clock_answer(self._clock()) if label.function_call else None,
            rag_collection=label.rag_collection,
            elapsed_ms=self._elapsed_ms(started_ns),
        )

    def _fallback(
        self,
        source: IntentSource,
        confidence: float | None,
        started_ns: int,
    ) -> IntentResult:
        default = self._taxonomy.default_label
        return IntentResult(
            label=default.name,
            agent=default.agent,
            source=source,
            confidence=confidence,
            function_answer=None,
            rag_collection=default.rag_collection,
            elapsed_ms=self._elapsed_ms(started_ns),
        )

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return max(time.perf_counter_ns() - started_ns, 0) / 1_000_000
