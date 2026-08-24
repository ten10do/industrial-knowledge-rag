"""Bidirectional Evidence decision-scope candidate for V3.45.

The candidate composes on the frozen V3.42 decision.  It preserves V3.42
upgrades and abstentions.  A pre-existing answer may be vetoed only when the
same bounded relation checker finds an explicit unsafe conflict; insufficiency
or missing evidence never revokes an answer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import evidence_sufficiency_v342 as v342


EVIDENCE_DECISION_SCOPE_CANDIDATE_VERSION = "evidence-v345-decision-scope-candidate"
EVIDENCE_DECISION_SCOPE_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
VETO_CONFIDENCE_FLOOR = 0.96


class EvidenceDecisionScopeAction(str, Enum):
    UPGRADE = "UPGRADE"
    VETO = "VETO"
    PRESERVE = "PRESERVE"


_EXPLICIT_VETO_REASONS = frozenset({
    "ABBREVIATION_DEFINITION_CONFLICT",
    "CROSS_MODEL_LEAKAGE_BLOCKED",
    "MANUFACTURER_EXPANSION_BLOCKED",
    "NEGATED_RELAXATION_FORBIDDEN",
    "REFERENCE_TARGET_MISMATCH",
    "SECTION_SCOPE_MISMATCH",
    "SIBLING_MODEL_VALUE_BLOCKED",
})

_VALUE_TOKEN = re.compile(
    r"\bip\s*\d{2}\b|"
    r"\b\d+(?:\.\d+)?\s*(?:us|ms|ma|vdc|v|mm2|mm|m|g|bytes?|bits?|inputs?|outputs?|%)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceDecisionScopeDecision:
    query: str
    decision: str
    action: str
    reason_code: str
    confidence: float
    final_decision_source: str
    baseline_decision: str
    baseline_reason: str
    relation: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _document(candidate: Any) -> Any:
    return getattr(candidate, "document", candidate)


def _dominant_document_id(result: Any) -> str:
    values = {
        str((getattr(_document(candidate), "metadata", {}) or {}).get("document_id", ""))
        for candidate in list(getattr(result, "candidates", []) or [])
    }
    values.discard("")
    return next(iter(values)) if len(values) == 1 else ""


def _unit(value: str) -> str:
    normalized = v342._norm(value)
    if normalized.startswith("ip "):
        return "ip"
    match = re.search(r"(?:us|ms|ma|vdc|v|mm2|mm|m|g|bytes?|bits?|inputs?|outputs?|%)$", normalized)
    return match.group(0) if match else ""


def _explicit_attribute_value_conflict(query: str, candidates: list[Any], relation: dict[str, Any]) -> bool:
    """Require a competing same-unit value beside the same attribute anchor."""
    attribute = str(relation.get("attribute_anchor", ""))
    if attribute not in v342._ATTRIBUTE_GROUPS:
        return False
    query_models = v342._query_models(query)
    requested = v342._query_values(query, query_models)
    if not requested:
        return False
    chunk_id = str(relation.get("chunk_id", ""))
    candidate = next((
        item for item in candidates
        if str((getattr(_document(item), "metadata", {}) or {}).get("chunk_id", "")) == chunk_id
    ), None)
    if candidate is None:
        return False
    text = str(getattr(_document(candidate), "page_content", ""))
    windows = v342._local_windows(text, v342._ATTRIBUTE_GROUPS[attribute], radius=2)
    for window in windows:
        if all(v342._value_present(window, value) for value in requested):
            return False
        observed = tuple(dict.fromkeys(v342._norm(match.group(0)) for match in _VALUE_TOKEN.finditer(window)))
        if all(any(_unit(value) and _unit(value) == _unit(other) and value != other for other in observed)
               for value in requested):
            return True
    return False


def analyze_evidence_decision_scope(
    query: str,
    result: Any,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    policy: Any = None,
    identity_matching: bool = True,
    requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> EvidenceDecisionScopeDecision:
    """Apply a bounded veto only after the unchanged V3.42 decision."""
    baseline = v342.analyze_evidence_sufficiency(
        query, result, documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    baseline_dict = baseline.as_dict()
    if baseline.decision == "ABSTAIN":
        return EvidenceDecisionScopeDecision(
            query, "ABSTAIN", EvidenceDecisionScopeAction.PRESERVE.value,
            "V342_ABSTAIN_PRESERVED", baseline.confidence,
            baseline.final_decision_source, baseline.decision, baseline.reason,
            baseline.relation, baseline_dict,
        )
    if baseline.relaxed:
        return EvidenceDecisionScopeDecision(
            query, "ANSWER", EvidenceDecisionScopeAction.UPGRADE.value,
            "V342_SAFE_UPGRADE_PRESERVED", baseline.confidence,
            baseline.final_decision_source, baseline.decision, baseline.reason,
            baseline.relation, baseline_dict,
        )

    candidates = list(getattr(result, "candidates", []) or [])
    allowed_document_id = _dominant_document_id(result) if baseline.baseline.get("expanded") else ""
    relation = v342.classify_evidence_sufficiency_relation(
        query, candidates, allowed_document_id=allowed_document_id,
    ).as_dict()
    explicit = relation["reason_code"] in _EXPLICIT_VETO_REASONS
    if relation["reason_code"] == "ATTRIBUTE_VALUE_MISMATCH":
        explicit = _explicit_attribute_value_conflict(query, candidates, relation)
    veto = (
        relation["relation"] == v342.EvidenceSufficiencyRelation.UNSAFE.value
        and float(relation["confidence"]) >= VETO_CONFIDENCE_FLOOR
        and explicit
    )
    if veto:
        reason = (
            "EXPLICIT_ATTRIBUTE_VALUE_CONFLICT"
            if relation["reason_code"] == "ATTRIBUTE_VALUE_MISMATCH"
            else relation["reason_code"]
        )
        return EvidenceDecisionScopeDecision(
            query, "ABSTAIN", EvidenceDecisionScopeAction.VETO.value,
            reason, float(relation["confidence"]), "V345_EVIDENCE_DECISION_SCOPE",
            baseline.decision, baseline.reason, relation, baseline_dict,
        )
    return EvidenceDecisionScopeDecision(
        query, "ANSWER", EvidenceDecisionScopeAction.PRESERVE.value,
        "NO_EXPLICIT_CONFLICT_PRESERVED", 1.0,
        baseline.final_decision_source, baseline.decision, baseline.reason,
        relation, baseline_dict,
    )
