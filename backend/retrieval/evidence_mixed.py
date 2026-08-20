"""Explicit mixed Evidence candidate orchestrator (V3.32).

The V3.31 sealed gate froze the intended composition — V3.25 verification
(rule + V3.24 ambiguity router + V3.24 local NLI), V3.30 open-question
sufficiency, V3.29 boundary — but the official runner never wired the local NLI
judge through the verification path (it wrapped the judge in a non-forwarding
recorder that exposes neither ``model`` nor ``predict_probs``).  This module
turns the composition into an explicit, testable runtime artifact so a future
sealed gate cannot silently drop a declared capability.

Composition (all-frozen components; NO algorithm change):

* VERIFICATION  -> frozen V3.25 rule + selective NLI
  (``analyze_querytype_evidence(mode="VERIFIER_ONLY", judge=judge)``).
* OPEN (EXTRACTION route) -> frozen base Evidence + V3.30 open sufficiency
  augmentation (``analyze_open_question_evidence(..., apply_open_sufficiency)``).
* FALLBACK (UNKNOWN route) -> conservative frozen baseline (VERIFIER_ONLY).
* Boundary: V3.29 ``EvidenceDecision`` / ``OpenQuestionEvidenceDecision``.
* Grounding/normalization: optional enrichment only; decision authority NONE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence_contract import build_typed_requirement
from .evidence_openquestion import (
    analyze_open_question_evidence,
    analyze_open_sufficiency,
)
from .evidence_querytype import (
    EvidenceQueryType,
    analyze_querytype_evidence,
    route_query_type,
)
from .filters import analyze_query
from .semantic_judge import route_ambiguity

MIXED_CANDIDATE_VERSION = "evidence-v332-integrated-candidate"
MIXED_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

# Frozen component identities (declaration only; the components themselves are
# unchanged from V3.25 / V3.30 and remain behind their own version constants).
MIXED_COMPOSITION = {
    "verification": "evidence-v323.1-candidate + V3.24 ambiguity router + V3.25 local NLI",
    "open": "evidence-v330-open-sufficiency-candidate",
    "boundary": "V3.29 EvidenceDecisionV2",
    "grounding_decision_authority": "NONE",
    "normalization_decision_authority": "NONE",
}

GROUNDING_DECISION_AUTHORITY = "NONE"
NORMALIZATION_DECISION_AUTHORITY = "NONE"

# Hard safety/identity abstention reasons the open path must never relax (§21).
_HARD_GATE_REASONS = frozenset({
    "NO_CANDIDATE",
    "UNKNOWN_IDENTIFIER",
    "CROSS_EQUIPMENT",
    "UNKNOWN_PARAMETER",
    "UNSUPPORTED_PROCEDURE",
})

_NLI_REASONS = frozenset({
    "NLI_ENTAILS",
    "NLI_CONTRADICTS",
    "NLI_INSUFFICIENT",
    "NLI_UNKNOWN_FALLBACK",
})

_NLI_REASON_TO_DECISION = {
    "NLI_ENTAILS": "ENTAILS",
    "NLI_CONTRADICTS": "CONTRADICTS",
    "NLI_INSUFFICIENT": "INSUFFICIENT",
    "NLI_UNKNOWN_FALLBACK": "UNKNOWN",
}


@dataclass(frozen=True)
class MixedEvidenceDecision:
    """A mixed-candidate decision with full path instrumentation.

    ``base_rule_decision`` is the deterministic rule-only decision (judge=None);
    ``nli_router_triggered`` is whether the V3.24 router flagged the query;
    ``nli_decision`` is the judge verdict only when the judge actually ran;
    ``final_decision_source`` attributes the final ANSWER/ABSTAIN so a sealed
    gate always knows which module produced it.
    """

    query: str
    query_path: str
    decision: str
    reason: str
    base_rule_decision: str
    base_rule_reason: str
    nli_router_triggered: bool
    nli_decision: str | None
    open_sufficiency_invoked: bool
    open_sufficiency_status: str | None
    final_decision_source: str
    grounding_status: str
    query_type_route: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decision_source(reason: str, rule_reason: str) -> str:
    if reason == "OPEN_RELATION_SUPPORTED":
        return "OPEN_SUFFICIENCY"
    if reason in _NLI_REASONS:
        return "NLI"
    if rule_reason in _HARD_GATE_REASONS:
        return "HARD_GATE"
    return "RULE"


def assert_mixed_candidate_capabilities(judge: Any = None, *, require_nli: bool = True) -> list[str]:
    """Return capability-activation violations for the mixed candidate.

    A declared Local-NLI branch must have a loaded, duck-type-compatible judge
    (``model``, ``predict_probs``, ``decide_from_probs`` are what the frozen
    V3.25 verification path actually calls).  The open-sufficiency branch must
    be importable/reachable.  Any violation means the candidate's declared
    capability is NOT wired and the run must be refused before evaluation.
    """
    violations: list[str] = []
    if require_nli:
        if judge is None:
            violations.append("NLI_JUDGE_NOT_PROVIDED")
        else:
            if getattr(judge, "model", None) is None:
                violations.append("NLI_MODEL_NOT_LOADED")
            if not callable(getattr(judge, "predict_probs", None)):
                violations.append("NLI_PREDICT_PROBS_NOT_CALLABLE")
            if not callable(getattr(judge, "decide_from_probs", None)):
                violations.append("NLI_DECIDE_FROM_PROBS_NOT_CALLABLE")
    if not callable(analyze_open_sufficiency):
        violations.append("OPEN_SUFFICIENCY_NOT_REACHABLE")
    return violations


def analyze_mixed_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    policy: Any = None,
    identity_matching: bool = True,
    requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> MixedEvidenceDecision:
    """Route a query through the explicit mixed Evidence candidate.

    Dispatch is by the deterministic V3.29 query-type route (VERIFICATION /
    EXTRACTION / UNKNOWN); it is exactly-equivalent to the V3.31 frozen
    composition (which runs ``analyze_open_question_evidence`` over every query)
    while making each decision's source observable.
    """
    analysis = getattr(result, "query_analysis", None) or analyze_query(query, documents)
    typed_requirement = build_typed_requirement(query, documents, analysis)
    route = route_query_type(query, typed_requirement)
    query_type = route.query_type

    # Deterministic rule-only baseline (judge=None) for path diagnostics.
    base = analyze_querytype_evidence(
        query, result, documents, retrieval_mode,
        mode="VERIFIER_ONLY", judge=None, policy=policy, identity_matching=identity_matching,
    )
    base_rule_decision = base.decision
    router_triggered = route_ambiguity(query, typed_requirement) is not None

    if query_type == EvidenceQueryType.EXTRACTION.value:
        final = analyze_open_question_evidence(
            query, result, documents, retrieval_mode,
            judge=judge, policy=policy, identity_matching=identity_matching,
            requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
        )
        query_path = "OPEN"
    else:
        final = analyze_querytype_evidence(
            query, result, documents, retrieval_mode,
            mode="VERIFIER_ONLY", judge=judge, policy=policy, identity_matching=identity_matching,
        )
        query_path = "VERIFICATION" if query_type == EvidenceQueryType.VERIFICATION.value else "FALLBACK"

    open_payload: dict | None = getattr(final, "open_sufficiency", None)
    open_invoked = open_payload is not None
    open_status = open_payload.get("status") if isinstance(open_payload, dict) else None
    reason = final.reason
    rule_reason = (final.evidence or {}).get("reason", "")
    nli_decision = _NLI_REASON_TO_DECISION.get(reason)

    return MixedEvidenceDecision(
        query=query,
        query_path=query_path,
        decision=final.decision,
        reason=reason,
        base_rule_decision=base_rule_decision,
        base_rule_reason=rule_reason,
        nli_router_triggered=router_triggered,
        nli_decision=nli_decision,
        open_sufficiency_invoked=open_invoked,
        open_sufficiency_status=open_status,
        final_decision_source=_decision_source(reason, rule_reason),
        grounding_status=GROUNDING_DECISION_AUTHORITY,
        query_type_route=route.as_dict(),
        evidence=final.evidence,
    )