"""Public V3.39 DEV benchmark, taxonomy, metrics, and acceptance policy.

Private manuals, query text, evidence excerpts, candidates, and result rows are
gitignored.  This module has no production decision authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v339-evidence-sufficiency-dev-v1"
QUERY_COUNT = 80
ANSWER_COUNT = 40
ABSTAIN_COUNT = 40

COVERAGE = (
    "SAME_PRODUCT_SUPPORTED_FACT",
    "SAME_FAMILY_INHERITED_SPECIFICATION",
    "PARAMETER_BLOCK_ASSOCIATION",
    "TABLE_VALUE_ASSOCIATION",
    "CROSS_SECTION_REFERENCE",
    "PROCEDURE_PREREQUISITE",
    "CONFIGURATION_DEPENDENCY",
    "NEGATIVE_NEAR_MISS",
)

FAILURE_TAXONOMY = (
    "SAFE_RELAX_CANDIDATE",
    "UNSAFE_RELAX",
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
    abstains = sum(row.get("expected") == "ABSTAIN" for row in rows)
    if answers != ANSWER_COUNT:
        errors.append(f"ANSWER_COUNT:{answers}")
    if abstains != ABSTAIN_COUNT:
        errors.append(f"ABSTAIN_COUNT:{abstains}")
    required = {
        "query_id", "query", "document_id", "expected", "coverage",
        "relevant_chunk_ids", "confidence", "new_document",
        "identity_expected", "scope_ambiguous", "parser_recoverable",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(missing)}")
        if row.get("coverage") not in COVERAGE:
            errors.append(f"COVERAGE:{index}")
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
    distribution = Counter(row.get("coverage") for row in rows)
    for coverage in COVERAGE:
        if distribution[coverage] != 10:
            errors.append(f"COVERAGE_COUNT:{coverage}:{distribution[coverage]}")
    return ValidationReport(tuple(errors))


def classify_baseline(record: dict) -> tuple[str, str]:
    """Classify a frozen baseline observation without candidate information."""
    expected = record.get("expected")
    decision = record.get("decision")
    if expected == "ABSTAIN":
        reason = "BASELINE_CORRECT_REFUSAL" if decision == "ABSTAIN" else "BASELINE_ALREADY_UNSAFE"
        return "UNSAFE_RELAX", reason
    if decision == "ANSWER":
        return "UNSAFE_RELAX", "NO_RELAXATION_NEEDED"
    if record.get("identity_result") == "INCOMPATIBLE":
        return "UNSAFE_RELAX", "IDENTITY_BOUNDARY_NOT_RELAXABLE"
    if not record.get("parser_recoverable", True):
        return "PARSER_LIMIT", str(record.get("parser_reason") or "STRUCTURE_NOT_RECOVERED")
    if not record.get("relevant_evidence_retrieved", False):
        return "MISSING_EVIDENCE", "GOLD_CHUNK_NOT_SELECTED"
    if record.get("scope_ambiguous", False):
        return "SCOPE_AMBIGUITY", str(record.get("scope_reason") or "MORE_CONTEXT_REQUIRED")
    return "SAFE_RELAX_CANDIDATE", str(record.get("evidence_reason") or "SUFFICIENT_EVIDENCE_REFUSED")


def metrics(records: list[dict], decision_key: str = "decision") -> dict[str, Any]:
    answerable = [row for row in records if row.get("expected") == "ANSWER"]
    abstain = [row for row in records if row.get("expected") == "ABSTAIN"]
    false_refusals = sum(row.get(decision_key) == "ABSTAIN" for row in answerable)
    false_answers = sum(row.get(decision_key) == "ANSWER" for row in abstain)
    return {
        "queries": len(records),
        "accuracy": sum(row.get(decision_key) == row.get("expected") for row in records) / len(records),
        "answerable_recall": 1.0 - false_refusals / len(answerable),
        "abstain_recall": 1.0 - false_answers / len(abstain),
        "false_refusals": false_refusals,
        "false_refusal_rate": false_refusals / len(answerable),
        "false_answers": false_answers,
        "false_answer_rate": false_answers / len(abstain),
    }


def acceptance(
    baseline: dict,
    candidate: dict,
    *,
    unsafe_relax: int,
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
        "unsafe_relax_zero": unsafe_relax == 0,
        "identity_hard_negative_fa_unchanged": candidate_hard_negative_fa <= baseline_hard_negative_fa,
        "runtime_integrity": runtime_integrity,
    }
    return {
        "status": "DEV_READY" if all(checks.values()) else "PARTIAL",
        "fr_reduction": fr_reduction,
        "fa_increase_pp": fa_increase,
        "checks": checks,
    }
