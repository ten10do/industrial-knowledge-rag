"""V3.66 Selective Semantic Evaluation Framework.

Architecture (directive §7):
    Typed Deterministic Evaluator
        ↓
    SemanticEvaluationRouter
        ↓ (only ambiguous / coverage-limited cases)
    Selective Semantic Judge (local NLI)
        ↓
    Combined Final Result

Safety invariants:
- Hard deterministic safety failures CANNOT be overridden
- Semantic judge only handles relation/scope/extra-claim ambiguity
- All decisions are deterministic and replayable
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- Reuse V3.65 typed evaluator contracts ---

from backend.evaluation.v365_typed_evaluator import (
    ClaimVerdict,
    EvaluationExpectation,
    EvaluationResult,
    ExpectedDecision,
    ForbiddenClaim,
    Prediction,
    RequiredClaim,
    evaluate_claim_level,
)


# --- Semantic router ----------------------------------------------------------

class RoutingVerdict(str, Enum):
    DETERMINISTIC_FINAL = "DETERMINISTIC_FINAL"
    SEMANTIC_REQUIRED = "SEMANTIC_REQUIRED"
    HARD_SAFETY_FINAL = "HARD_SAFETY_FINAL"


# Reason codes that indicate semantic ambiguity (route to semantic judge).
SEMANTIC_AMBIGUITY_REASONS = {
    "INCOMPLETE_ANSWER",
}

# Reason codes that indicate hard safety failure (never route).
HARD_SAFETY_REASONS = {
    "FORBIDDEN_CLAIM_HIT",
    "FALSE_ANSWER_ON_ABSTAIN",
    "FALSE_REFUSAL",
}


@dataclass(frozen=True)
class RoutingDecision:
    verdict: RoutingVerdict
    reason_codes: tuple[str, ...]
    deterministic_result: EvaluationResult


def route_evaluation(
    expectation: EvaluationExpectation,
    prediction: Prediction,
    deterministic_result: EvaluationResult,
) -> RoutingDecision:
    """Decide whether a case needs semantic evaluation or can be
    resolved deterministically."""
    # Hard safety failures are FINAL — never routed.
    if any(r in HARD_SAFETY_REASONS for r in deterministic_result.reason_codes):
        return RoutingDecision(
            verdict=RoutingVerdict.HARD_SAFETY_FINAL,
            reason_codes=("DETERMINISTIC_HARD_SAFETY_FAILURE",),
            deterministic_result=deterministic_result,
        )

    # Clear INCORRECT from claim mismatch: also final (deterministic found
    # concrete evidence of mismatch).
    if (
        deterministic_result.verdict.value == "INCORRECT"
        and "CLAIM_MISMATCH" in deterministic_result.reason_codes
        and deterministic_result.forbidden_claim_hits > 0
    ):
        return RoutingDecision(
            verdict=RoutingVerdict.HARD_SAFETY_FINAL,
            reason_codes=("FORBIDDEN_CLAIM_DETERMINISTIC",),
            deterministic_result=deterministic_result,
        )

    # Cases that might benefit from semantic understanding.
    semantic_reasons = [
        r for r in deterministic_result.reason_codes
        if r in SEMANTIC_AMBIGUITY_REASONS
    ]
    if semantic_reasons:
        return RoutingDecision(
            verdict=RoutingVerdict.SEMANTIC_REQUIRED,
            reason_codes=tuple(semantic_reasons),
            deterministic_result=deterministic_result,
        )

    # Relation direction ambiguity: when claims partially satisfied but
    # not fully, and no forbidden hits.
    if (
        deterministic_result.verdict.value == "PARTIAL_INCOMPLETE"
        or (0 < deterministic_result.claim_precision < 1)
    ):
        return RoutingDecision(
            verdict=RoutingVerdict.SEMANTIC_REQUIRED,
            reason_codes=("PARTIAL_CLAIM_SEMANTIC_CHECK",),
            deterministic_result=deterministic_result,
        )

    # Fully correct or fully incorrect with clear reasons: deterministic.
    return RoutingDecision(
        verdict=RoutingVerdict.DETERMINISTIC_FINAL,
        reason_codes=tuple(deterministic_result.reason_codes),
        deterministic_result=deterministic_result,
    )


# --- Local NLI semantic judge --------------------------------------------------


@dataclass(frozen=True)
class SemanticJudgeResult:
    verdict: str                # CORRECT | PARTIAL_INCOMPLETE | INCORRECT | UNSUPPORTED
    claim_entailment: float     # 0-1
    relation_direction_ok: bool | None
    scope_consistent: bool | None
    unsupported_extra_claim: bool | None
    confidence: float
    reason_codes: tuple[str, ...]
    judge_source: str           # nli_local | llm_provider | ...


class LocalNLISemanticJudge:
    """NLI-based semantic evaluation using cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self._model = None
        self._model_name = model_name
        self._invocation_count = 0

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, max_length=256)

    def _nli_score(self, premise: str, hypothesis: str) -> str:
        """Return 'entailment', 'contradiction', or 'neutral'."""
        self._ensure_model()
        self._invocation_count += 1
        scores = self._model.predict([(premise, hypothesis)])
        labels = ["contradiction", "entailment", "neutral"]
        best_idx = max(range(len(scores[0])), key=lambda i: scores[0][i])
        return labels[best_idx] if best_idx < len(labels) else "neutral"

    def evaluate_semantic_claims(
        self,
        expectation: EvaluationExpectation,
        prediction: Prediction,
    ) -> SemanticJudgeResult:
        """Evaluate prediction against required/forbidden claims using NLI."""
        pred_text = prediction.text.strip()
        reasons: list[str] = []

        # Check required claims via entailment.
        satisfied = 0
        total = len(expectation.required_claims)
        relation_direction_ok: bool | None = None
        scope_consistent: bool | None = True

        for claim in expectation.required_claims:
            gold_statement = (
                f"{_norm(claim.subject)} {_norm(claim.relation)} "
                f"{_norm(claim.object)}"
            )
            if claim.scope:
                gold_statement += f" for {_norm(claim.scope)}"

            entailment = self._nli_score(pred_text, gold_statement)
            if entailment == "entailment":
                satisfied += 1

        # Check forbidden claims.
        forbidden_hits = 0
        for fc in expectation.forbidden_claims:
            fc_statement = (
                f"{_norm(fc.subject)} {_norm(fc.relation)} "
                f"{_norm(fc.object)}"
            )
            contradiction = self._nli_score(
                pred_text,
                f"It is NOT true that {fc_statement}",
            )
            entailment_fc = self._nli_score(pred_text, fc_statement)
            if entailment_fc == "entailment":
                forbidden_hits += 1

        # Check relation direction using NLI on reversed statement.
        if expectation.required_claims:
            rc = expectation.required_claims[0]
            forward_stmt = f"{_norm(rc.subject)} {_norm(rc.relation)} {_norm(rc.object)}"
            reverse_stmt = f"{_norm(rc.object)} {_norm(rc.relation)} {_norm(rc.subject)}"
            fwd_nli = self._nli_score(pred_text, forward_stmt)
            rev_nli = self._nli_score(pred_text, reverse_stmt)
            if fwd_nli == "entailment" and rev_nli != "entailment":
                relation_direction_ok = True
            elif rev_nli == "entailment" and fwd_nli != "entailment":
                relation_direction_ok = False
            else:
                relation_direction_ok = None

        precision = satisfied / max(total, 1)

        if total == 0:
            verdict = "CORRECT"
        elif precision >= 1.0 and forbidden_hits == 0:
            verdict = "CORRECT"
        elif precision > 0 and forbidden_hits == 0:
            verdict = "PARTIAL_INCOMPLETE"
        else:
            verdict = "INCORRECT"

        return SemanticJudgeResult(
            verdict=verdict,
            claim_entailment=round(precision, 3),
            relation_direction_ok=relation_direction_ok,
            scope_consistent=scope_consistent,
            unsupported_extra_claim=None,  # TODO: extra claim detection
            confidence=round(precision, 2),
            reason_codes=tuple(reasons),
            judge_source="nli_local",
        )


def _norm(value):
    return __import__("re").sub(r"\s+", " ", str(value)).strip().lower()


# --- Combined selective evaluator ---------------------------------------------


def selective_evaluate(
    expectation: EvaluationExpectation,
    prediction: Prediction,
    semantic_judge: LocalNLISemanticJudge | None = None,
    *,
    enable_routing: bool = True,
) -> dict:
    """Combined deterministic + selective semantic evaluation."""
    det_result = evaluate_claim_level(expectation, prediction)
    routing = route_evaluation(expectation, prediction, det_result)

    final_result = det_result
    semantic_used = False

    if routing.verdict == RoutingVerdict.HARD_SAFETY_FINAL:
        pass  # Keep deterministic result
    elif routing.verdict == RoutingVerdict.SEMANTIC_REQUIRED and semantic_judge:
        sem_result = semantic_judge.evaluate_semantic_claims(
            expectation, prediction,
        )
        semantic_used = True
        # Combine: use semantic verdict but keep deterministic safety flags.
        final_verdict = sem_result.verdict
        final_reasons = list(det_result.reason_codes) + list(sem_result.reason_codes)
        final_result = EvaluationResult(
            decision_correct=(final_verdict == "CORRECT"),
            claim_precision=det_result.claim_precision,
            claim_recall=det_result.claim_recall,
            unsafe_claim_count=det_result.unsafe_claim_count,
            missing_claim_count=max(det_result.missing_claim_count, 0),
            forbidden_claim_hits=det_result.forbidden_claim_hits,
            verdict=__import__(
                "backend.evaluation.v365_typed_evaluator", fromlist=["ClaimVerdict"],
            ).ClaimVerdict(final_verdict) if final_verdict != "UNSUPPORTED" else det_result.verdict,
            reason_codes=tuple(set(final_reasons)),
        )

    return {
        "result": final_result,
        "routing": routing,
        "semantic_used": semantic_used,
        "deterministic_verdict": det_result.verdict.value,
        "semantic_verdict": sem_result.verdict if semantic_used else None,
    }
