"""V3.79 Shadow claim-support validation contract (SHADOW ONLY).

Answers the V3.79 feasibility question ``is there a claim-support quality
signal independent of contract completeness, retrieval distance and identity
compatibility`` without touching any runtime decision path.

Layered candidates over existing frozen assets:

* ``S0`` - structured support replay: ``support-v316.1``
  (:func:`backend.retrieval.evidence_support.validate_evidence_support`,
  SUPPORT_RULE_VERSION pinned) mapped onto the claim-support state machine.
* ``S2`` - local NLI shadow: ``cross-encoder/nli-deberta-v3-xsmall`` through
  :mod:`backend.retrieval.semantic_judge_localnli` with its FROZEN thresholds;
  model injected, never loaded implicitly by this module.
* ``S3`` - hybrid: S0 verdict governs; a high-confidence NLI contradiction can
  only DOWNGRADE towards unsupported. Entailment never overrides S0.

None of these functions reads evaluation gold, query ids, slice labels,
support states or expected decisions; see ``GOLD_INDEPENDENCE_FIELDS`` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from backend.retrieval.evidence_support import (
    SUPPORT_RULE_VERSION,
    EvidenceSupport,
    SupportStatus,
    validate_evidence_support,
)
from backend.retrieval.semantic_judge_localnli import (
    LOCAL_NLI_CONTRADICTION_THRESHOLD,
    LOCAL_NLI_ENTAILMENT_THRESHOLD,
    LOCAL_NLI_MODEL_NAME,
    LOCAL_NLI_UNKNOWN_FLOOR,
    LocalNliJudge,
    to_declarative,
)

CLAIM_SUPPORT_SIGNAL_VERSION = "claim-support-shadow-v379-r0"
STRUCTURED_SOURCE = f"struct:{SUPPORT_RULE_VERSION}"
NLI_SOURCE = f"nli:{LOCAL_NLI_MODEL_NAME}"
HYBRID_SOURCE = "hybrid:struct+nli-veto"

# Machine-checked inputs the validator is allowed to consume. Any benchmark /
# gold field names listed here MUST NOT be passed into validators.
GOLD_INDEPENDENCE_FIELDS = (
    "expected_decision",
    "support_state",
    "support_reason",
    "slice_labels",
    "query_domain",
    "anchors",
    "claims",
)


class ClaimSupportState(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ClaimSupportResult:
    state: str                       # ClaimSupportState value
    support_source: str              # STRUCTURED_SOURCE / NLI_SOURCE / HYBRID_SOURCE
    support_reason: str
    confidence: float | None = None
    identity_compatible: bool = True
    scope_compatible: bool = True
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "support_source": self.support_source,
            "support_reason": self.support_reason,
            "confidence": self.confidence,
            "identity_compatible": self.identity_compatible,
            "scope_compatible": self.scope_compatible,
            "provenance": self.provenance,
        }


def _claim_state_from_support(support: EvidenceSupport) -> str:
    if support.status == SupportStatus.SUPPORTED.value:
        return ClaimSupportState.SUPPORTED.value
    if support.status == SupportStatus.INSUFFICIENT.value:
        return ClaimSupportState.UNSUPPORTED.value
    return ClaimSupportState.AMBIGUOUS.value


def _identity_flags(support: EvidenceSupport) -> tuple[bool, bool]:
    coverage = support.coverage or {}
    identity_ok = bool(coverage.get("identity", True)) and not any(
        item.startswith("target_identity") for item in (support.missing_requirements or ())
    )
    # Scope: identifier scoped to another product family inside evidence text is
    # reported by MISSING_IDENTIFIER_SUPPORT targets elsewhere; keep scope True
    # unless identity fails (support-v316.1 folds scope into identity/identifier).
    return identity_ok, True


def structured_claim_support(query: str, result, documents: list) -> ClaimSupportResult:
    """S0: replay the frozen structured support gate as a claim-support signal."""
    support = validate_evidence_support(query, result, documents)
    identity_ok, scope_ok = _identity_flags(support)
    return ClaimSupportResult(
        state=_claim_state_from_support(support),
        support_source=STRUCTURED_SOURCE,
        support_reason=support.reason,
        identity_compatible=identity_ok,
        scope_compatible=scope_ok,
        provenance={
            "status": support.status,
            "missing_requirements": list(support.missing_requirements),
            "candidate_count": len(list(getattr(result, "candidates", []) or [])),
        },
    )


class NliClaimSupportValidator:
    """S2: local-NLI entailment of the query proposition against evidence text.

    Uses the frozen thresholds from ``semantic_judge_localnli`` verbatim -
    deliberately NOT tuned in V3.79. Invocation counters make capability
    coverage observable (invocation=0 => INCONCLUSIVE).
    """

    def __init__(self, model=None):
        self._judge = LocalNliJudge(
            model=model,
            entailment_threshold=LOCAL_NLI_ENTAILMENT_THRESHOLD,
            contradiction_threshold=LOCAL_NLI_CONTRADICTION_THRESHOLD,
            unknown_floor=LOCAL_NLI_UNKNOWN_FLOOR,
        )
        self.invocations = 0

    def judge_case(self, query: str, chunk_texts: list[str]) -> ClaimSupportResult:
        """Any-chunk entailment: best (non-neutral-conflicting) decision wins."""
        hypothesis = to_declarative(query)
        best: ClaimSupportResult | None = None
        order = {"entailment": 0, "contradiction": 1}
        for text in chunk_texts:
            premise = " ".join(str(text).split())
            if not premise:
                continue
            self.invocations += 1
            probs = self._judge.predict_probs(premise[:4000], hypothesis)
            decision, confidence = self._judge.decide_from_probs(probs)
            p_contradiction, p_entailment, p_neutral = probs
            reason = f"probs=({p_contradiction:.3f},{p_entailment:.3f},{p_neutral:.3f})"
            state = {
                "ENTAILS": ClaimSupportState.SUPPORTED.value,
                "INSUFFICIENT": ClaimSupportState.UNSUPPORTED.value,
                "CONTRADICTS": ClaimSupportState.UNSUPPORTED.value,
                "UNKNOWN": ClaimSupportState.AMBIGUOUS.value,
            }.get(decision, ClaimSupportState.AMBIGUOUS.value)
            candidate = ClaimSupportResult(
                state=state,
                support_source=NLI_SOURCE,
                support_reason=reason,
                confidence=round(confidence, 6),
                provenance={"decision": decision, "chunk_chars": len(premise)},
            )
            rank = order.get(decision, 2)
            if best is None or rank < order.get(best.provenance.get("decision", ""), 3):
                best = candidate
        if best is None:
            self.invocations += 1
            return ClaimSupportResult(
                state=ClaimSupportState.NOT_APPLICABLE.value,
                support_source=NLI_SOURCE,
                support_reason="NO_EVIDENCE_TEXT",
            )
        return best


def hybrid_claim_support(structured: ClaimSupportResult, nli: ClaimSupportResult) -> ClaimSupportResult:
    """S3: S0 governs; NLI contradiction downgrades, never upgrades."""
    if structured.state == ClaimSupportState.SUPPORTED.value and (
        nli.support_reason.startswith("probs=") and "CONTRADICTS" in nli.provenance.get("decision", "")
    ):
        return ClaimSupportResult(
            state=ClaimSupportState.UNSUPPORTED.value,
            support_source=HYBRID_SOURCE,
            support_reason=f"{structured.support_reason}|nli_contradicts",
            confidence=nli.confidence,
            identity_compatible=structured.identity_compatible,
            scope_compatible=structured.scope_compatible,
            provenance={"structured": structured.as_dict(), "nli": nli.as_dict()},
        )
    return ClaimSupportResult(
        state=structured.state,
        support_source=HYBRID_SOURCE,
        support_reason=structured.support_reason,
        confidence=nli.confidence if nli.confidence is not None else None,
        identity_compatible=structured.identity_compatible,
        scope_compatible=structured.scope_compatible,
        provenance={"structured": structured.as_dict(), "nli": nli.as_dict()},
    )


def case_admissible(result: ClaimSupportResult) -> bool:
    """Pre-registered shadow policy: SUPPORTED-or-nothing (AMBIGUOUS fails)."""
    return result.state == ClaimSupportState.SUPPORTED.value


def counterfactual_transitions(rows: list[dict], *, policy_name: str) -> dict:
    """Shadow counterfactual over per-case rows.

    Each row needs: ``query_id``, ``runtime_answered`` (bool), ``correct_answer``
    (bool; verdict accepted-as-pass), ``admissible`` (signal says accept).
    Classifies SAFE_BLOCK / FALSE_REFUSAL_REGRESSION / NO_EFFECT / UNSAFE_CHANGE.
    """
    transitions = {
        "policy": policy_name,
        "SAFE_BLOCK": [],
        "FALSE_REFUSAL_REGRESSION": [],
        "NO_EFFECT": [],
        "UNSAFE_CHANGE": [],
    }
    for row in rows:
        answered, admissible = bool(row["runtime_answered"]), bool(row["admissible"])
        correct = bool(row["correct_answer"])
        if answered and not admissible:
            if correct:
                transitions["FALSE_REFUSAL_REGRESSION"].append(row["query_id"])
            else:
                transitions["SAFE_BLOCK"].append(row["query_id"])
        elif answered and admissible:
            transitions["NO_EFFECT"].append(row["query_id"]) if correct else (
                transitions["UNSAFE_CHANGE"].append(row["query_id"]))
        else:
            transitions["NO_EFFECT"].append(row["query_id"])
    return transitions
