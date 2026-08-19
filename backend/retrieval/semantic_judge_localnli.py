"""Local NLI entailment judge adapter for the selective semantic-judge layer.

Wraps a local 3-way NLI cross-encoder (``contradiction / entailment / neutral``)
and maps it onto the :mod:`semantic_judge` contract.  ``SEMANTIC_JUDGE_DEFAULT``
stays OFF; this adapter is only ever invoked for router-triggered queries during
experiments.  It never sees ground truth labels or failure classes.

The model is injected (``CrossEncoder``-like ``predict`` callable) so the label
mapping, thresholds and UNKNOWN fallback are unit-testable without loading
weights and without network access.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

from .semantic_judge import (
    JudgeDecision,
    SemanticJudgeResult,
    validate_judge_schema,
)

# Experimental candidate identity.  The V3.25 study validated this adapter against
# the DEV oracle ceiling; it is NOT promoted to READY and remains behind
# SEMANTIC_JUDGE_DEFAULT = OFF in production.
LOCAL_NLI_CANDIDATE_VERSION = "evidence-v325-local-nli-candidate"
LOCAL_NLI_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
LOCAL_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
# Policy selected on DEV-TRAIN-V2 only (never per-case TUNE hardcoding).
LOCAL_NLI_ENTAILMENT_THRESHOLD = 0.5
LOCAL_NLI_CONTRADICTION_THRESHOLD = 0.5
LOCAL_NLI_UNKNOWN_FLOOR = 0.33
LOCAL_NLI_UNKNOWN_POLICY = "A"  # UNKNOWN -> rule fallback

# cross-encoder/nli-deberta-v3-xsmall id2label: {0: contradiction, 1: entailment, 2: neutral}
CONTRADICTION_INDEX = 0
ENTAILMENT_INDEX = 1
NEUTRAL_INDEX = 2
LABEL_NAMES = ("contradiction", "entailment", "neutral")

# Raw NLI label -> contract verdict (NLI_ENTAILMENT -> ENTAILS, etc.)
NLI_LABEL_TO_DECISION = {
    "contradiction": JudgeDecision.CONTRADICTS.value,
    "entailment": JudgeDecision.ENTAILS.value,
    "neutral": JudgeDecision.INSUFFICIENT.value,
}

_POLAR_LEAD_RE = re.compile(
    r"^\s*(?:does|is|are|do|can|will|should|must|was|were|has|have)\b",
    re.IGNORECASE,
)


def to_declarative(query: str) -> str:
    """Turn a polar verification question into a declarative relation proposition.

    ``Does A broadcast while B targets X?`` -> ``A broadcast while B targets X``.
    This is the hypothesis the NLI model must entail or contradict.
    """
    text = _POLAR_LEAD_RE.sub("", (query or "").strip(), count=1).strip()
    text = re.sub(r"[?？]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def build_premise(claims: Iterable[Any] | None) -> str:
    """Concatenate the selected supporting candidate text into a minimal premise."""
    parts: list[str] = []
    for claim in list(claims or ()):
        text = getattr(claim, "text", None)
        if text:
            parts.append(str(text).strip())
    return " ".join(parts).strip()


def build_hypothesis(query: str, requirement: Any = None) -> str:
    """Build the hypothesis (relation proposition) for an NLI pair."""
    return to_declarative(query)


def _softmax(scores: Sequence[float]) -> tuple[float, float, float]:
    values = [float(value) for value in scores]
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return tuple(round(exp / total, 6) for exp in exponents)


class LocalNliJudge:
    """Maps a local 3-way NLI model onto the SemanticEvidenceJudge contract.

    Parameters
    ----------
    model:
        A ``CrossEncoder``-like object exposing ``predict(list[(premise, hypothesis)])``.
    entailment_threshold:
        Minimum entailment probability to emit ENTAILS.  Defaults are neutral
        starting points; the study tunes them on DEV-TRAIN-V2 only.
    contradiction_threshold:
        Minimum contradiction probability to emit CONTRADICTS.
    unknown_floor:
        If the argmax probability is below this floor the judge returns UNKNOWN
        (low confidence), deferring to the explicit fallback policy.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        unknown_floor: float = 0.4,
    ):
        self.model = model
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.unknown_floor = float(unknown_floor)

    def predict_probs(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Return ``(p_contradiction, p_entailment, p_neutral)`` for one pair."""
        raw = self.model.predict([(premise, hypothesis)])
        scores = raw[0]
        if isinstance(scores, (list, tuple)) or (hasattr(scores, "shape") and len(getattr(scores, "shape", ())) >= 1):
            flat = [float(value) for value in scores]
            if len(flat) >= 3:
                return _softmax(flat[:3])
        # Single-score/regression fallback: treat score as P(entailment).
        value = float(scores)
        contra = 1.0 - value
        return (round(contra, 6), round(value, 6), 0.0)

    def decide_from_probs(self, probs: tuple[float, float, float]) -> tuple[str, float]:
        """Return ``(decision, confidence)`` given the three label probabilities."""
        p_contradiction, p_entailment, p_neutral = probs
        probabilities = (p_contradiction, p_entailment, p_neutral)
        top_label = LABEL_NAMES[max(range(3), key=lambda index: probabilities[index])]
        top_prob = probabilities[LABEL_NAMES.index(top_label)]

        if top_prob < self.unknown_floor:
            return JudgeDecision.UNKNOWN.value, top_prob
        if top_label == "entailment" and top_prob >= self.entailment_threshold:
            return JudgeDecision.ENTAILS.value, top_prob
        if top_label == "contradiction" and top_prob >= self.contradiction_threshold:
            return JudgeDecision.CONTRADICTS.value, top_prob
        if top_label == "neutral":
            return JudgeDecision.INSUFFICIENT.value, top_prob
        return JudgeDecision.UNKNOWN.value, top_prob

    def judge(
        self,
        query: str,
        requirements: Any,
        claims: Any = None,
        scope: Any = None,
    ) -> SemanticJudgeResult:
        if self.model is None:
            return SemanticJudgeResult(
                decision=JudgeDecision.UNKNOWN.value,
                confidence=0.0,
                reason_code="NLI_MODEL_NOT_LOADED",
            )
        premise = build_premise(claims)
        hypothesis = build_hypothesis(query, requirements)
        probs = self.predict_probs(premise, hypothesis)
        decision, confidence = self.decide_from_probs(probs)
        p_contradiction, p_entailment, p_neutral = probs
        result = SemanticJudgeResult(
            decision=decision,
            confidence=confidence,
            reason_code=f"probs=({p_contradiction:.3f},{p_entailment:.3f},{p_neutral:.3f})",
            subject_match=p_entailment >= self.entailment_threshold,
            predicate_match=p_entailment >= self.entailment_threshold,
        )
        violations = validate_judge_schema(result)
        if violations:
            return SemanticJudgeResult(
                decision=JudgeDecision.UNKNOWN.value,
                confidence=0.0,
                reason_code="SCHEMA_" + ";".join(violations),
            )
        return result


def load_cross_encoder(model_name: str) -> Any:
    """Load a local 3-way NLI cross-encoder (requires sentence-transformers)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)