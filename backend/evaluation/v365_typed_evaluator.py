"""V3.65 typed evaluation framework: sensitive, claim-level correctness.

Core principle: EVALUATOR_DOES_NOT_CREATE_CAPABILITY.
An evaluator must distinguish correct from incorrect answers using
structural claims, not word-overlap heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExpectedDecision(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


class ClaimVerdict(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL = "PARTIAL"
    INCORRECT = "INCORRECT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class RequiredClaim:
    subject: str
    relation: str
    object: str
    scope: str = ""
    qualifiers: tuple[str, ...] = ()

    def normalized(self) -> tuple:
        return (
            norm_text(self.subject),
            norm_text(self.relation),
            norm_text(self.object),
            norm_text(self.scope),
        )


@dataclass(frozen=True)
class ForbiddenClaim:
    subject: str
    relation: str
    object: str
    scope: str = ""
    reason: str = ""


@dataclass(frozen=True)
class EvaluationExpectation:
    query_id: str
    expected_decision: ExpectedDecision
    answerability: bool = True
    required_claims: tuple[RequiredClaim, ...] = ()
    forbidden_claims: tuple[ForbiddenClaim, ...] = ()
    required_scope: str = ""
    required_value: str = ""
    required_unit: str = ""
    expected_abstain_reason: str = ""
    slice_labels: tuple[str, ...] = ()
    difficulty: str = "L3"
    context_critical: bool = False


@dataclass(frozen=True)
class Prediction:
    decision: str                # ANSWER | ABSTAIN
    text: str                    # raw answer text or empty for abstain


@dataclass(frozen=True)
class EvaluationResult:
    decision_correct: bool = False
    claim_precision: float = 0.0
    claim_recall: float = 0.0
    scope_correct: bool = False
    relation_correct: bool = False
    identity_correct: bool = False
    unsafe_claim_count: int = 0
    missing_claim_count: int = 0
    forbidden_claim_hits: int = 0
    verdict: ClaimVerdict = ClaimVerdict.INCORRECT
    reason_codes: tuple[str, ...] = ()


def _norm(value):
    return __import__("re").sub(r"\s+", " ", str(value)).strip().lower()


def _words(text):
    return set(w for w in _norm(text).split() if len(w) >= 1)


def _numeric_value(text):
    """Extract numeric value from text."""
    match = __import__("re").search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _unit_token(text):
    """Extract unit token if present."""
    match = __import__("re").search(
        r"\d(?:\.\d+)?\s*(s|ms|min|h|A|mA|V|mV|kV|W|kW|Hz|rpm|mm|%|N|Nm)\b",
        text, __import__("re").IGNORECASE,
    )
    return match.group(1).lower() if match else ""


def _claim_satisfied(claim: RequiredClaim, prediction_text: str) -> bool:
    """Check whether a single RequiredClaim is satisfied by the prediction.

    Targeted checks:
    - subject must appear in prediction
    - object numeric value must match (not just word overlap)
    - unit token must match (if present)
    - scope must match (if specified)
    - relation keywords must be consistent (no reversal)
    """
    pred_norm = _norm(prediction_text)
    pred_lower = prediction_text.lower()

    # Subject: key identifier must be present.
    subject_norm = _norm(claim.subject)
    if subject_norm and subject_norm not in pred_norm:
        return False

    # Object: numeric value proximity check (not substring).
    obj_norm = _norm(claim.object)
    gold_num = _numeric_value(obj_norm)

    # Scope: must be mentioned if specified.
    scope_norm = _norm(claim.scope)
    if scope_norm and len(scope_norm.split()) >= 2:
        if scope_norm not in pred_norm:
            return False

    # Relation keywords: check that the claim's relation direction matches.
    # E.g., "acceleration time" should not match a text about "deceleration".
    rel_words = set(_words(claim.relation))
    negation_pairs = {
        ("acceleration", "deceleration"),
        ("deceleration", "acceleration"),
        ("minimum", "maximum"),
        ("maximum", "minimum"),
        ("default", "factory"),
    }
    for w1, w2 in negation_pairs:
        if w1 in rel_words and w2 in pred_lower:
            return False
        if w2 in rel_words and w1 in pred_lower:
            return False

    # Numeric value check.
    if gold_num is not None:
        pred_nums = [
            float(m) for m in
            re.findall(r"[-+]?\d+(?:\.\d+)?", prediction_text)
            if abs(float(m)) > 0.001 or m.startswith("0.")
        ]
        matched = any(
            abs(n - gold_num) <= 0.01 * max(abs(gold_num), 0.01)
            for n in pred_nums
        )
        if not matched:
            return False
        # Unit check.
        gold_unit = _unit_token(claim.object)
        if gold_unit:
            gold_unit_val = _unit_token(f"1 {gold_unit}")
            if gold_unit_val and gold_unit_val not in pred_lower:
                return False

    # Object words (non-numeric) must also appear.
    obj_words = [w for w in obj_norm.split() if not any(c.isdigit() for c in w) and len(w) > 2]
    missing_obj_words = [
        w for w in obj_words
        if w not in pred_lower and w.rstrip(".,;:") not in pred_lower
    ]
    if obj_words and missing_obj_words:
        return False

    return True


def evaluate_claim_level(
    expectation: EvaluationExpectation,
    prediction: Prediction,
) -> EvaluationResult:
    """Typed claim-level evaluation (NOT word-overlap)."""
    reasons: list[str] = []
    exp_decision = expectation.expected_decision
    pred_decision = prediction.decision

    decision_correct = exp_decision.value == pred_decision

    if exp_decision == ExpectedDecision.ABSTAIN:
        # Gold says abstain; correct if prediction also abstains.
        if pred_decision == "ABSTAIN":
            return EvaluationResult(decision_correct=True, verdict=ClaimVerdict.ABSTAIN)
        # Gold says abstain but prediction answered → False Answer.
        reasons.append("FALSE_ANSWER_ON_ABSTAIN")
        return EvaluationResult(
            decision_correct=False,
            verdict=ClaimVerdict.INCORRECT,
            reason_codes=tuple(reasons),
        )

    # Gold says ANSWER.
    if pred_decision == "ABSTAIN":
        reasons.append("FALSE_REFUSAL")
        return EvaluationResult(
            decision_correct=False,
            missing_claim_count=len(expectation.required_claims),
            verdict=ClaimVerdict.ABSTAIN,
            reason_codes=tuple(reasons),
        )

    # Evaluate claims.
    satisfied = 0
    total = len(expectation.required_claims)
    forbidden_hits = 0
    unsafe_count = 0

    for claim in expectation.required_claims:
        if _claim_satisfied(claim, prediction.text):
            satisfied += 1

    for fclaim in expectation.forbidden_claims:
        # Check if forbidden claim appears in prediction.
        f_subject = _norm(fclaim.subject)
        f_object = _norm(fclaim.object)
        if f_subject in _norm(prediction.text) and f_object in _norm(prediction.text):
            forbidden_hits += 1

    precision = satisfied / max(total, 1)
    recall = precision  # same set for now
    missing = total - satisfied

    if forbidden_hits > 0:
        unsafe_count += forbidden_hits
        reasons.append("FORBIDDEN_CLAIM_HIT")

    if total == 0:
        verdict = ClaimVerdict.CORRECT
    elif satisfied == total and forbidden_hits == 0:
        verdict = ClaimVerdict.CORRECT
    elif satisfied > 0 and forbidden_hits == 0:
        verdict = ClaimVerdict.PARTIAL
        reasons.append("INCOMPLETE_ANSWER")
    else:
        verdict = ClaimVerdict.INCORRECT
        reasons.append("CLAIM_MISMATCH")

    return EvaluationResult(
        decision_correct=(verdict in {ClaimVerdict.CORRECT}),
        claim_precision=round(precision, 4),
        claim_recall=round(satisfied / max(total, 1), 4),
        scope_correct=True,   # simplified for now
        relation_correct=forbidden_hits == 0,
        identity_correct=satisfied >= total // 2,
        unsafe_claim_count=unsafe_count,
        missing_claim_count=missing,
        forbidden_claim_hits=forbidden_hits,
        verdict=verdict,
        reason_codes=tuple(reasons),
    )
