"""V3.67 Contract-Native Evidence Evaluation Authority.

Evaluates Evidence correctness using STRUCTURED FIELDS from
RetrievalEvidence, NOT natural-language answer text.

Machine properties:
  EVALUATION_TARGET = EVIDENCE_CONTRACT
  NATURAL_LANGUAGE_ANSWER_AUTHORITY = NONE
  EVALUATION_ADAPTER_MUST_NOT_CHANGE_DECISION = TRUE
  DETERMINISTIC = TRUE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


def _norm(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


CONTRACT_EVALUATOR_VERSION = "contract-native-evaluator-v367-r0"


# --- Structured Expectation ---------------------------------------------------


class ExpectedDecision(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class ExpectedClaim:
    subject: str
    relation: str
    obj_value: str = ""
    obj_unit: str = ""
    scope: str = ""

    def key(self) -> tuple:
        return (
            self.subject.strip().lower(),
            self.relation.strip().lower(),
            self.obj_value.strip().lower(),
        )


@dataclass(frozen=True)
class ForbiddenClaim:
    subject: str
    relation: str
    obj_value: str = ""
    scope: str = ""
    reason: str = ""

    def key(self) -> tuple:
        return (
            self.subject.strip().lower(),
            self.relation.strip().lower(),
            self.obj_value.strip().lower(),
        )


@dataclass(frozen=True)
class EvidenceExpectation:
    query_id: str
    expected_decision: ExpectedDecision
    expected_reason_family: str = ""
    required_claims: tuple[ExpectedClaim, ...] = ()
    forbidden_claims: tuple[ForbiddenClaim, ...] = ()
    expected_identity: dict = field(default_factory=dict)
    expected_scope: str = ""
    expected_abstain_reason: str = ""
    slice_labels: tuple[str, ...] = ()
    difficulty: str = "L3"


# --- Evaluation Record --------------------------------------------------------


@dataclass(frozen=True)
class EvidenceEvaluationRecord:
    """Serialized Evidence runtime state for evaluation."""

    decision: str                    # ANSWER | ABSTAIN
    reason: str                      # DecisionReason value
    query_identity: dict             # e.g. {"manufacturer": "...", "product_series": "..."}
    candidate_identity: dict
    identity_relation: str           # e.g. EXACT_MATCH, FAMILY_COMPATIBLE
    contract_requirements_covered: bool
    lexical_score: float | None = None
    vector_distance: float | None = None
    metadata_consistency: bool = True
    # Set of (subject, relation, obj) tuples that the runtime marked as covered.
    covered_claim_keys: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_retrieval_evidence(cls, evidence) -> "EvidenceEvaluationRecord":
        d = evidence.as_dict() if hasattr(evidence, "as_dict") else dict(evidence)
        return cls(
            decision=d.get("decision", ""),
            reason=d.get("reason", ""),
            query_identity=d.get("query_identity", {}),
            candidate_identity=d.get("candidate_identity", {}),
            identity_relation=d.get("identity_relation", ""),
            contract_requirements_covered=d.get(
                "contract", {},
            ).get("requirements_covered", False),
            lexical_score=d.get("lexical_score"),
            vector_distance=d.get("vector_distance"),
            metadata_consistency=d.get("metadata_consistency", True),
        )


# --- Evaluation Result ----------------------------------------------------------


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL_INCOMPLETE = "PARTIAL_INCOMPLETE"
    INCORRECT = "INCORRECT"
    CONTRACT_UNSUPPORTED = "CONTRACT_UNSUPPORTED"
    ABSTAIN_CORRECT = "ABSTAIN_CORRECT"


@dataclass(frozen=True)
class ContractEvaluationResult:
    verdict: Verdict
    decision_correct: bool
    reason_family_match: bool | None
    identity_correct: bool | None
    unsafe_structural_acceptance: bool
    missing_claim_count: int
    forbidden_claim_hits: int
    contract_coverage: float          # fraction of evaluable dimensions
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


# --- Contract-Native Evaluator ----------------------------------------------------


def evaluate_contract_native(
    expectation: EvidenceExpectation,
    runtime_record: EvidenceEvaluationRecord,
) -> ContractEvaluationResult:
    """Deterministic contract-native evaluation using structured fields only."""
    reasons: list[str] = []
    evaluable_dims = 0
    total_dims = 0

    # --- 1. Decision check (always evaluable) ---
    exp_dec = expectation.expected_decision.value
    run_dec = runtime_record.decision
    decision_correct = exp_dec == run_dec
    total_dims += 1
    if decision_correct:
        evaluable_dims += 1

    # --- 2. Reason family check ---
    reason_match = None
    if expectation.expected_reason_family:
        total_dims += 1
        evaluable_dims += 1
        exp_family = _reason_family(expectation.expected_reason_family)
        run_family = _reason_family(runtime_record.reason)
        reason_match = exp_family == run_family
        if not reason_match:
            reasons.append(f"REASON_FAMILY_MISMATCH:{run_family}!={exp_family}")

    # --- 3. Identity check ---
    identity_correct = None
    if expectation.expected_identity:
        total_dims += 1
        evaluable_dims += 1
        mismatches = []
        for key, expected_val in expectation.expected_identity.items():
            runtime_val = runtime_record.candidate_identity.get(key, "")
            if _norm(str(expected_val)) != _norm(str(runtime_val)):
                mismatches.append(key)
        identity_correct = len(mismatches) == 0
        if not identity_correct:
            reasons.append(f"IDENTITY_MISMATCH:{','.join(mismatches)}")

    # --- 4. Unsafe structural acceptance ---
    unsafe_acceptance = False
    if exp_dec == "ABSTAIN" and run_dec == "ANSWER":
        reasons.append("FALSE_ANSWER_ON_ABSTAIN")
        unsafe_acceptance = True

    # --- 5. Claim checks (for ANSWER decisions) ---
    forbidden_hits = 0
    missing_claims = 0
    claims_ok = True

    if exp_dec == "ANSWER":
        # Check forbidden claims against runtime covered claim set.
        for fc in expectation.forbidden_claims:
            if fc.key() in runtime_record.covered_claim_keys:
                forbidden_hits += 1
                reasons.append(f"FORBIDDEN_CLAIM_HIT:{fc.subject}:{fc.obj_value}")
                unsafe_acceptance = True
        if expectation.required_claims:
            total_dims += 1
            evaluable_dims += 1
            if not runtime_record.contract_requirements_covered:
                missing_claims = len(expectation.required_claims)
                claims_ok = False
                reasons.append("CONTRACT_REQUIREMENTS_NOT_COVERED")

    if exp_dec == "ABSTAIN" and expectation.forbidden_claims:
        total_dims += 1
        evaluable_dims += 1

    # --- Compute coverage ---
    coverage = evaluable_dims / max(total_dims, 1)

    # --- Verdict ---
    if unsafe_acceptance:
        verdict = Verdict.INCORRECT
    elif not decision_correct:
        if exp_dec == "ANSWER":
            reasons.append("FALSE_REFUSAL")
            verdict = Verdict.INCORRECT
        else:
            reasons.append("FALSE_ANSWER")
            verdict = Verdict.INCORRECT
    elif identity_correct is False:
        # Identity mismatch is a structural error → INCORRECT.
        verdict = Verdict.INCORRECT
    elif reason_match is False:
        # Decision correct but reason family differs.
        verdict = Verdict.PARTIAL_INCOMPLETE
    elif not claims_ok:
        verdict = Verdict.PARTIAL_INCOMPLETE
    elif exp_dec == "ABSTAIN":
        verdict = Verdict.ABSTAIN_CORRECT
    else:
        verdict = Verdict.CORRECT

    return ContractEvaluationResult(
        verdict=verdict,
        decision_correct=decision_correct,
        reason_family_match=reason_match,
        identity_correct=identity_correct,
        unsafe_structural_acceptance=unsafe_acceptance,
        missing_claim_count=missing_claims,
        forbidden_claim_hits=forbidden_hits,
        contract_coverage=round(coverage, 3),
        reason_codes=tuple(reasons),
    )


def _reason_family(reason: str) -> str:
    """Map specific reason codes to families."""
    reason_upper = reason.upper()
    if "IDENTIFIER" in reason_upper or "MODEL" in reason_upper or "FAMILY_COMPATIBLE" in reason_upper:
        return "IDENTIFIER_MATCH"
    if "LEXICAL" in reason_upper or "VECTOR" in reason_upper or "COMBINED" in reason_upper:
        return "RETRIEVAL_EVIDENCE"
    if "NO_CANDIDATE" in reason_upper or "UNKNOWN_IDENTIFIER" in reason_upper:
        return "NO_EVIDENCE"
    if "MISMATCH" in reason_upper or "CROSS_EQUIPMENT" in reason_upper or "PROTOCOL" in reason_upper:
        return "MISMATCH_REJECTION"
    if "MISSING" in reason_upper or "UNSUPPORTED" in reason_upper or "PARTIAL" in reason_upper or "UNKNOWN_PARAMETER" in reason_upper:
        return "INSUFFICIENT_CONTENT"
    if "CONTRACT_REQUIREMENTS_COVERED" in reason_upper:
        return "CONTRACT_COVERED"
    return "OTHER"


def _accept_as_pass(verdict: Verdict) -> bool:
    """Determine whether an evaluation verdict counts as a passing case."""
    return verdict in {Verdict.CORRECT, Verdict.ABSTAIN_CORRECT}
