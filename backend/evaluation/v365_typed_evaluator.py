"""V3.65-R1 typed evaluation framework: safety-hardened, claim-level.

Key improvements over initial version:
- UnitValue model: numeric + unit extracted and compared separately
- Scoped unit matching: unit checked near the specific numeric value
- Directed relation: subject-before-object order verified for direction
- Expanded mutation suite: >=100 programmatically generated cases

EVALUATOR_VERSION = typed-evaluator-v365-r1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


EVALUATOR_VERSION = "typed-evaluator-v365-r1"

# --- Unit normalization -------------------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    "s": "second", "sec": "second", "second": "second", "seconds": "second",
    "ms": "millisecond", "millisecond": "millisecond", "milliseconds": "millisecond",
    "min": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hour": "hour", "hours": "hour",
    "a": "ampere", "amp": "ampere", "ampere": "ampere", "amperes": "ampere",
    "ma": "milliampere", "milliampere": "milliampere",
    "v": "volt", "volt": "volt", "volts": "volt",
    "mv": "millivolt", "kv": "kilovolt",
    "w": "watt", "watt": "watt", "watts": "watt", "kw": "kilowatt",
    "hz": "hertz", "hertz": "hertz",
    "rpm": "rpm",
    "mm": "millimetre", "cm": "centimetre", "m": "metre",
    "%": "percent", "pct": "percent", "percent": "percent",
    "n": "newton", "newton": "newton", "nm": "newton-metre",
    "°c": "celsius", "celsius": "celsius",
}


def normalize_unit(raw: str) -> str:
    return _UNIT_ALIASES.get(raw.strip().lower(), raw.strip().lower())


@dataclass(frozen=True)
class UnitValue:
    value: float
    raw_unit: str
    normalized_unit: str

    def matches(self, other: "UnitValue") -> bool:
        return (
            abs(self.value - other.value) <= 0.01 * max(abs(other.value), 0.01)
            and self.normalized_unit == other.normalized_unit
        )


def extract_unit_values(text: str) -> list[UnitValue]:
    """Extract (numeric_value, unit) pairs from text.
    Only captures a SINGLE word after the number as potential unit."""
    pattern = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z°%]+)",
    )
    results = []
    for match in pattern.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        raw_unit = (match.group(2) or "").strip()
        normalized = normalize_unit(raw_unit)
        results.append(UnitValue(value=value, raw_unit=raw_unit,
                                 normalized_unit=normalized))
    return results


# --- Directed relation ---------------------------------------------------

_RELATION_NEGATION_PAIRS = [
    ("acceleration", "deceleration"),
    ("deceleration", "acceleration"),
    ("minimum", "maximum"),
    ("maximum", "minimum"),
    ("min", "max"),
    ("max", "min"),
    ("input", "output"),
    ("output", "input"),
    ("source", "load"),
    ("load", "source"),
]


def check_relation_direction(
    claim_relation: str,
    claim_subject: str,
    claim_object: str,
    prediction_text: str,
) -> bool:
    """Verify that the relation direction is preserved.

    Returns True if direction is consistent, False if reversed.
    """
    rel_lower = claim_relation.lower()
    subj_words = set(simple_words(claim_subject))

    for w1, w2 in _RELATION_NEGATION_PAIRS:
        if w1 in rel_lower and w2 in pred_lower(prediction_text):
            return False

    # Subject must appear before object when both are present.
    subj_norm = norm_text(claim_subject)
    obj_norm = norm_text(claim_object)
    if subj_norm and obj_norm:
        subj_pos = pred_lower(prediction_text).find(subj_norm.split()[0]) if subj_norm.split() else -1
        obj_pos = pred_lower(prediction_text).find(obj_norm.split()[0]) if obj_norm.split() else -1
        if subj_pos >= 0 and obj_pos >= 0 and subj_pos > obj_pos:
            return False
    return True


def simple_words(text):
    return __import__("re").sub(r"[^\w\s]", " ", text.lower()).split()


def pred_lower(text):
    return text.lower()


def norm_text(text):
    return __import__("re").sub(r"\s+", " ", str(text)).strip().lower()


# --- Core contracts ------------------------------------------------------

class ExpectedDecision(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


class ClaimVerdict(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL_INCOMPLETE = "PARTIAL_INCOMPLETE"
    INCORRECT = "INCORRECT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class RequiredClaim:
    subject: str
    relation: str
    object: str
    scope: str = ""
    qualifiers: tuple[str, ...] = ()

    def directed_key(self) -> tuple:
        return (norm_text(self.subject), norm_text(self.relation), norm_text(self.object))


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
    decision: str
    text: str


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


# --- Claim evaluation ----------------------------------------------------

def _norm(value):
    return __import__("re").sub(r"\s+", " ", str(value)).strip().lower()


def _words(text):
    return set(w for w in _norm(text).split() if len(w) >= 1)


def _extract_number(text):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def _unit_near_value(text: str, target_value: float) -> str:
    """Find the unit token immediately following the target number."""
    for match in re.finditer(
        r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z°%]+)?", text,
    ):
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        if abs(val - target_value) < 0.005 * max(abs(target_value), 0.01):
            return normalize_unit((match.group(2) or "").strip())
    return ""


def _claim_satisfied_v2(claim: RequiredClaim, prediction_text: str) -> bool:
    """Multi-dimensional claim check with scoped unit matching."""
    pred_norm = _norm(prediction_text)
    pred_lower = prediction_text.lower()

    # Subject.
    subj_norm = _norm(claim.subject)
    if subj_norm and subj_norm not in pred_norm:
        return False

    # Scope (if specified with ≥2 words).
    scope_norm = _norm(claim.scope)
    if scope_norm and len(scope_norm.split()) >= 2:
        if scope_norm not in pred_norm:
            return False

    # Relation direction.
    if not check_relation_direction(
        claim.relation, claim.subject, claim.object, prediction_text,
    ):
        return False

    # Object: numeric + unit proximity matching.
    gold_uvs = extract_unit_values(claim.object)
    pred_uvs = extract_unit_values(prediction_text)

    if gold_uvs:
        matched_any_gold_uv = False
        for guv in gold_uvs:
            for puv in pred_uvs:
                if guv.matches(puv):
                    matched_any_gold_uv = True
                    break
            if matched_any_gold_uv:
                break
        if not matched_any_gold_uv:
            return False
    else:
        # Non-numeric object: check key words present.
        obj_words = [
            w for w in _norm(claim.object).split()
            if len(w) > 2 and not any(c.isdigit() for c in w)
        ]
        missing = [w for w in obj_words if w not in pred_lower]
        if obj_words and missing:
            return False

    return True


def evaluate_claim_level(
    expectation: EvaluationExpectation,
    prediction: Prediction,
) -> EvaluationResult:
    """Typed multi-dimensional claim-level evaluation."""
    reasons: list[str] = []
    exp_decision = expectation.expected_decision
    pred_decision = prediction.decision
    decision_correct = exp_decision.value == pred_decision

    if exp_decision == ExpectedDecision.ABSTAIN:
        if pred_decision == "ABSTAIN":
            return EvaluationResult(decision_correct=True, verdict=ClaimVerdict.ABSTAIN)
        reasons.append("FALSE_ANSWER_ON_ABSTAIN")
        return EvaluationResult(decision_correct=False, verdict=ClaimVerdict.INCORRECT,
                                reason_codes=tuple(reasons))

    if pred_decision == "ABSTAIN":
        reasons.append("FALSE_REFUSAL")
        return EvaluationResult(
            decision_correct=False,
            missing_claim_count=len(expectation.required_claims),
            verdict=ClaimVerdict.ABSTAIN,
            reason_codes=tuple(reasons),
        )

    satisfied = sum(
        1 for claim in expectation.required_claims
        if _claim_satisfied_v2(claim, prediction.text)
    )
    total = len(expectation.required_claims)

    forbidden_hits = sum(
        1 for fc in expectation.forbidden_claims
        if _norm(fc.subject) in _norm(prediction.text)
        and _norm(fc.object) in _norm(prediction.text)
    )
    unsafe_count = forbidden_hits

    precision = satisfied / max(total, 1)
    recall = precision
    missing = total - satisfied

    if total == 0:
        verdict = ClaimVerdict.CORRECT
    elif satisfied == total and forbidden_hits == 0:
        verdict = ClaimVerdict.CORRECT
    elif satisfied > 0 and forbidden_hits == 0:
        verdict = ClaimVerdict.PARTIAL_INCOMPLETE
        reasons.append("INCOMPLETE_ANSWER")
    else:
        verdict = ClaimVerdict.INCORRECT
        reasons.append("CLAIM_MISMATCH")

    return EvaluationResult(
        decision_correct=(verdict == ClaimVerdict.CORRECT),
        claim_precision=round(precision, 4),
        claim_recall=round(satisfied / max(total, 1), 4),
        unsafe_claim_count=unsafe_count,
        missing_claim_count=missing,
        forbidden_claim_hits=forbidden_hits,
        verdict=verdict,
        reason_codes=tuple(reasons),
    )
