"""Public V3.41 DEV benchmark, identity-claim taxonomy, metrics, acceptance.

Private manuals, query text, evidence excerpts, candidates, and result rows are
gitignored.  This module has no production decision authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any


BENCHMARK_VERSION = "v341-identity-claim-dev-v1"
QUERY_COUNT = 60
ANSWER_COUNT = 30
ABSTAIN_COUNT = 30

# §Phase 1 identity case types (10 queries each).
IDENTITY_CASES = (
    "DOCUMENT_TITLE_IDENTITY",
    "SECTION_INHERITED_IDENTITY",
    "PRONOUN_IDENTITY",
    "FAMILY_IMPLICIT_REFERENCE",
    "SUBMODULE_OWNERSHIP",
    "CROSS_REFERENCE_IDENTITY",
)

IDENTITY_ANSWER_QUOTA = {
    "DOCUMENT_TITLE_IDENTITY": 6,
    "SECTION_INHERITED_IDENTITY": 5,
    "PRONOUN_IDENTITY": 5,
    "FAMILY_IMPLICIT_REFERENCE": 5,
    "SUBMODULE_OWNERSHIP": 5,
    "CROSS_REFERENCE_IDENTITY": 4,
}


class IdentityEvidenceRelation(str, Enum):
    """How the identity claim between a query and a document was established."""

    EXPLICIT = "EXPLICIT"
    SECTION_INHERITED = "SECTION_INHERITED"
    DOCUMENT_INHERITED = "DOCUMENT_INHERITED"
    FAMILY_INHERITED = "FAMILY_INHERITED"
    UNSUPPORTED = "UNSUPPORTED"


CASE_TO_RELATION = {
    "DOCUMENT_TITLE_IDENTITY": IdentityEvidenceRelation.EXPLICIT.value,
    "SECTION_INHERITED_IDENTITY": IdentityEvidenceRelation.SECTION_INHERITED.value,
    "PRONOUN_IDENTITY": IdentityEvidenceRelation.DOCUMENT_INHERITED.value,
    "FAMILY_IMPLICIT_REFERENCE": IdentityEvidenceRelation.FAMILY_INHERITED.value,
    "SUBMODULE_OWNERSHIP": IdentityEvidenceRelation.DOCUMENT_INHERITED.value,
    "CROSS_REFERENCE_IDENTITY": IdentityEvidenceRelation.DOCUMENT_INHERITED.value,
}

REASON_CODES = (
    # safe expansions
    "EXPLICIT_MODEL_ALIAS_MATCH",
    "DOCUMENT_TITLE_IDENTITY_BOUND",
    "SECTION_CONTEXT_INHERITED",
    "PRONOUN_BOUND_TO_DOCUMENT_PRODUCT",
    "FAMILY_REFERENCE_BOUND",
    "SUBMODULE_OWNED_BY_DOCUMENT_PRODUCT",
    "CROSS_REFERENCE_BOUND_TO_DOCUMENT_PRODUCT",
    # refusals (safety constraints)
    "MANUFACTURER_MISMATCH_REJECTED",
    "PRODUCT_LINE_MISMATCH_REJECTED",
    "PARAMETER_OWNER_MISMATCH_REJECTED",
    "MULTI_PRODUCT_CORPUS_UNRESOLVED",
    "NO_IDENTITY_ANCHOR",
    # inherited guards
    "NON_IDENTITY_BLOCK_PRESERVED",
    "ALREADY_ANSWERED",
)

FAILURE_TAXONOMY = (
    "SAFE_CLAIM_EXPANSION",
    "UNSAFE_EXPANSION",
    "MISSING_EVIDENCE",
    "SCOPE_AMBIGUITY",
    "PARSER_LIMIT",
)


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(payload: dict) -> ValidationReport:
    rows = payload.get("queries", [])
    errors: list[str] = []
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if payload.get("uses_v335_data") is not False:
        errors.append("V335_EXCLUSION")
    if payload.get("uses_v337_data") is not False:
        errors.append("V337_EXCLUSION")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")
    answers = sum(row.get("expected") == "ANSWER" for row in rows)
    if answers != ANSWER_COUNT:
        errors.append(f"ANSWER_COUNT:{answers}")
    required = {
        "query_id", "query", "document_id", "expected", "identity_case",
        "relevant_chunk_ids", "confidence", "new_document",
        "identity_expected", "scope_ambiguous", "parser_recoverable",
    }
    distribution: Counter = Counter()
    answer_distribution: Counter = Counter()
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(missing)}")
        case = row.get("identity_case")
        if case not in IDENTITY_CASES:
            errors.append(f"IDENTITY_CASE:{index}")
            continue
        distribution[case] += 1
        if row.get("expected") == "ANSWER":
            answer_distribution[case] += 1
        if row.get("expected") not in {"ANSWER", "ABSTAIN"}:
            errors.append(f"EXPECTED:{index}")
        if row.get("confidence") != "HIGH":
            errors.append(f"CONFIDENCE:{index}")
        if row.get("new_document") is not True:
            errors.append(f"NEW_DOCUMENT:{index}")
        if row.get("identity_expected") != "COMPATIBLE":
            errors.append(f"IDENTITY_EXPECTED:{index}")
        if (row.get("expected") == "ANSWER") != bool(row.get("relevant_chunk_ids")):
            errors.append(f"RELEVANCE:{index}")
    ids = [str(row.get("query_id", "")) for row in rows]
    texts = [" ".join(str(row.get("query", "")).casefold().split()) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_TEXTS")
    for case in IDENTITY_CASES:
        if distribution[case] != 10:
            errors.append(f"CASE_COUNT:{case}:{distribution[case]}")
        if answer_distribution[case] != IDENTITY_ANSWER_QUOTA[case]:
            errors.append(
                f"CASE_ANSWER_QUOTA:{case}:{answer_distribution[case]}!={IDENTITY_ANSWER_QUOTA[case]}"
            )
    return ValidationReport(tuple(errors))


def classify_baseline(record: dict) -> tuple[str, str]:
    expected = record.get("expected")
    decision = record.get("decision")
    if expected == "ABSTAIN":
        reason = "BASELINE_CORRECT_REFUSAL" if decision == "ABSTAIN" else "BASELINE_ALREADY_UNSAFE"
        return "UNSAFE_EXPANSION", reason
    if decision == "ANSWER":
        return "UNSAFE_EXPANSION", "NO_EXPANSION_NEEDED"
    if record.get("identity_result") == "INCOMPATIBLE":
        return "SAFE_CLAIM_EXPANSION", str(record.get("identity_reason") or "IDENTITY_BLOCKED")
    if not record.get("parser_recoverable", True):
        return "PARSER_LIMIT", str(record.get("parser_reason") or "STRUCTURE_NOT_RECOVERED")
    if not record.get("relevant_evidence_retrieved", False):
        return "MISSING_EVIDENCE", "GOLD_CHUNK_NOT_SELECTED"
    if record.get("scope_ambiguous", False):
        return "SCOPE_AMBIGUITY", str(record.get("scope_reason") or "MORE_CONTEXT_REQUIRED")
    return "SAFE_CLAIM_EXPANSION", str(record.get("evidence_reason") or "EVIDENCE_REFUSED")


def metrics(records: list[dict], decision_key: str = "decision") -> dict[str, Any]:
    answerable = [row for row in records if row.get("expected") == "ANSWER"]
    abstain = [row for row in records if row.get("expected") == "ABSTAIN"]
    false_refusals = sum(row.get(decision_key) == "ABSTAIN" for row in answerable)
    false_answers = sum(row.get(decision_key) == "ANSWER" for row in abstain)
    return {
        "queries": len(records),
        "accuracy": sum(row.get(decision_key) == row.get("expected") for row in records) / len(records),
        "answerable_recall": 1.0 - false_refusals / len(answerable) if answerable else None,
        "abstain_recall": 1.0 - false_answers / len(abstain) if abstain else None,
        "false_refusals": false_refusals,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else None,
        "false_answers": false_answers,
        "false_answer_rate": false_answers / len(abstain) if abstain else None,
    }


def case_slices(records: list[dict], decision_key: str = "decision") -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for case in IDENTITY_CASES:
        subset = [row for row in records if row.get("identity_case") == case]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "modeled_relation": CASE_TO_RELATION[case],
            **metrics(subset, decision_key),
        }
        if decision_key == "decision":
            entry["expanded_from_baseline"] = sum(
                row.get("baseline_decision") == "ABSTAIN" and row.get("decision") == "ANSWER"
                for row in subset
            )
            entry["unsafe_expansion"] = sum(
                row.get("expected") == "ABSTAIN"
                and row.get("baseline_decision") == "ABSTAIN"
                and row.get("decision") == "ANSWER"
                for row in subset
            )
        slices[case] = entry
    return slices


def acceptance(
    baseline: dict,
    candidate: dict,
    *,
    unsafe_expansion: int,
    baseline_hard_negative_fa: int,
    candidate_hard_negative_fa: int,
    runtime_integrity: bool,
) -> dict[str, Any]:
    baseline_fr = int(baseline["false_refusals"])
    fr_reduction = (baseline_fr - int(candidate["false_refusals"])) / baseline_fr if baseline_fr else 0.0
    fa_increase = float(candidate["false_answer_rate"]) - float(baseline["false_answer_rate"])
    checks = {
        "fr_reduction_at_least_50pct": fr_reduction >= 0.5,
        "fa_increase_at_most_5pp": fa_increase <= 0.05,
        "identity_hard_negative_fa_unchanged": candidate_hard_negative_fa <= baseline_hard_negative_fa,
        "no_unsafe_identity_expansion": unsafe_expansion == 0,
        "runtime_integrity": runtime_integrity,
    }
    return {
        "status": "DEV_READY" if all(checks.values()) else "PARTIAL",
        "fr_reduction": fr_reduction,
        "fa_increase_pp": fa_increase,
        "checks": checks,
    }