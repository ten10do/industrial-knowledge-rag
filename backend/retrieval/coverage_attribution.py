"""V3.75 observability: coverage/sufficiency predicate attribution.

Pure helpers that classify an Evidence ABSTAIN into a blocking mechanism and
identify coverage-bound false refusals. These helpers read ONLY runtime
``RetrievalEvidence`` fields (decision/reason/identity_relation/vector_distance/
lexical_score/has_candidates) and therefore never recompute a score that the
runtime did not use. No decision is changed here.
"""
from __future__ import annotations

# Reasons routed through the coverage/sufficiency gate (the region governed by
# max_vector_distance / lexical evidence), as opposed to identity / identifier /
# contract coverage / safety gates.
COVERAGE_REASONS = frozenset({"INSUFFICIENT_EVIDENCE", "WEAK_RETRIEVAL_EVIDENCE"})

CONTRACT_REASONS = frozenset({
    "MISSING_ATTRIBUTE_EVIDENCE",
    "MISSING_VALUE_EVIDENCE",
    "MISSING_REQUIREMENT_EVIDENCE",
    "MISSING_ACTION_EVIDENCE",
    "PARTIAL_EVIDENCE_ONLY",
    "IDENTIFIER_NOT_IN_EVIDENCE",
    "PROTOCOL_MISMATCH",
})

IDENTITY_REASONS = frozenset({"MODEL_MISMATCH"})

COMPATIBLE_RELATIONS = frozenset({"EXACT_MODEL", "SAME_SERIES", "SAME_FAMILY"})


def blocking_mechanism(ev) -> str:
    """Classify the primary blocking mechanism of an ABSTAIN from runtime fields."""
    reason = ev.reason
    if reason == "NO_CANDIDATE":
        return "RETRIEVAL_MISS"
    if reason in IDENTITY_REASONS:
        return "IDENTITY"
    if reason == "UNKNOWN_IDENTIFIER":
        return "IDENTIFIER"
    if reason == "CROSS_EQUIPMENT":
        return "CROSS_EQUIPMENT"
    if reason == "UNKNOWN_PARAMETER":
        return "UNKNOWN_PARAMETER"
    if reason == "UNSUPPORTED_PROCEDURE":
        return "SECURITY_BYPASS"
    if reason in CONTRACT_REASONS:
        return "CONTRACT_COVERAGE"
    if reason in COVERAGE_REASONS:
        if ev.vector_distance is None and ev.lexical_score is None:
            return "COVERAGE_SCORE_UNWIRED"
        return "VECTOR_LEXICAL_COVERAGE"
    return "OTHER"


def is_coverage_bound(ev) -> bool:
    """True when a refusal is a coverage-bound FR.

    Definition (V3.75 contract section 13): gold evidence present (has
    candidates) + identity accepted (compatible relation) + coverage/sufficiency
    direct reject (COVERAGE_REASONS).
    """
    return (
        ev.reason in COVERAGE_REASONS
        and ev.identity_relation in COMPATIBLE_RELATIONS
        and bool(ev.has_candidates)
    )
