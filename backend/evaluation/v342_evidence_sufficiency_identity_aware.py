"""Public V3.42 DEV benchmark contract, metrics, and acceptance policy.

The official manuals, query text, evidence excerpts, retrieval candidates, and
result rows remain under the gitignored private benchmark directory.  This
module has no production decision authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any


BENCHMARK_VERSION = "v342-evidence-sufficiency-identity-aware-dev-v1"
QUERY_COUNT = 60
ANSWER_COUNT = 30
ABSTAIN_COUNT = 30

EVIDENCE_CASES = (
    "IDENTITY_COMPATIBLE_RECOVERY",
    "TECHNICAL_SYNONYM",
    "ABBREVIATION",
    "PARAMETER_TABLE_RELATION",
    "SECTION_INHERITED_EVIDENCE",
    "CROSS_REFERENCE_EVIDENCE",
)

HARD_NEGATIVE_TYPES = (
    "SAME_FAMILY_WRONG_PARAMETER",
    "SAME_MANUFACTURER_WRONG_MODEL",
    "SAME_VALUE_WRONG_ATTRIBUTE",
    "IDENTITY_CORRECT_EVIDENCE_INSUFFICIENT",
)


class EvidenceSufficiencyRelation(str, Enum):
    DIRECT_SUPPORTED = "DIRECT_SUPPORTED"
    SEMANTIC_SUPPORTED = "SEMANTIC_SUPPORTED"
    REFERENCE_SUPPORTED = "REFERENCE_SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNSAFE = "UNSAFE"


FAILURE_TAXONOMY = (
    "COMPATIBLE_EVIDENCE_FALSE_REFUSAL",
    "IDENTITY_BOUNDARY_REFUSAL",
    "MISSING_RETRIEVAL",
    "PARSER_LIMIT",
    "CORRECT_REFUSAL",
    "FALSE_ANSWER",
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
    for key in (
        "uses_a_to_h_data", "uses_j_data", "uses_k_data",
        "uses_historical_sealed_data",
    ):
        if payload.get(key) is not False:
            errors.append(f"FORBIDDEN_CORPUS:{key}")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")
    answers = sum(row.get("expected") == "ANSWER" for row in rows)
    abstains = sum(row.get("expected") == "ABSTAIN" for row in rows)
    if answers != ANSWER_COUNT:
        errors.append(f"ANSWER_COUNT:{answers}")
    if abstains != ABSTAIN_COUNT:
        errors.append(f"ABSTAIN_COUNT:{abstains}")

    required = {
        "query_id", "query", "document_id", "expected", "evidence_case",
        "evidence_relation_expected", "target", "relation", "attribute",
        "value_or_action", "relevant_chunk_ids", "confidence", "new_document",
        "identity_expected", "identity_compatible", "hard_negative_type",
        "parser_recoverable",
    }
    distribution: Counter = Counter()
    answer_distribution: Counter = Counter()
    negative_distribution: Counter = Counter()
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(missing)}")
        case = row.get("evidence_case")
        if case not in EVIDENCE_CASES:
            errors.append(f"EVIDENCE_CASE:{index}")
            continue
        distribution[case] += 1
        expected = row.get("expected")
        if expected == "ANSWER":
            answer_distribution[case] += 1
            if row.get("evidence_relation_expected") not in {
                EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value,
                EvidenceSufficiencyRelation.SEMANTIC_SUPPORTED.value,
                EvidenceSufficiencyRelation.REFERENCE_SUPPORTED.value,
            }:
                errors.append(f"ANSWER_RELATION:{index}")
            if not all(str(row.get(key, "")).strip() for key in (
                "target", "relation", "attribute", "value_or_action",
            )):
                errors.append(f"ANSWER_COMPONENTS:{index}")
            if row.get("hard_negative_type") is not None:
                errors.append(f"ANSWER_HARD_NEGATIVE:{index}")
        elif expected == "ABSTAIN":
            hard_type = row.get("hard_negative_type")
            if hard_type not in HARD_NEGATIVE_TYPES:
                errors.append(f"HARD_NEGATIVE:{index}")
            else:
                negative_distribution[hard_type] += 1
            if row.get("evidence_relation_expected") not in {
                EvidenceSufficiencyRelation.INSUFFICIENT.value,
                EvidenceSufficiencyRelation.UNSAFE.value,
            }:
                errors.append(f"ABSTAIN_RELATION:{index}")
        else:
            errors.append(f"EXPECTED:{index}")
        if row.get("confidence") != "HIGH":
            errors.append(f"CONFIDENCE:{index}")
        if row.get("new_document") is not True:
            errors.append(f"NEW_DOCUMENT:{index}")
        if row.get("identity_expected") not in {"COMPATIBLE", "INCOMPATIBLE"}:
            errors.append(f"IDENTITY_EXPECTED:{index}")
        if row.get("identity_compatible") != (row.get("identity_expected") == "COMPATIBLE"):
            errors.append(f"IDENTITY_COMPATIBILITY:{index}")
        if (expected == "ANSWER") != bool(row.get("relevant_chunk_ids")):
            errors.append(f"RELEVANCE:{index}")

    ids = [str(row.get("query_id", "")) for row in rows]
    texts = [" ".join(str(row.get("query", "")).casefold().split()) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_TEXTS")
    for case in EVIDENCE_CASES:
        if distribution[case] != 10:
            errors.append(f"CASE_COUNT:{case}:{distribution[case]}")
        if answer_distribution[case] != 5:
            errors.append(f"CASE_ANSWER_COUNT:{case}:{answer_distribution[case]}")
    if set(negative_distribution) != set(HARD_NEGATIVE_TYPES):
        errors.append("HARD_NEGATIVE_COVERAGE")
    return ValidationReport(tuple(errors))


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


def case_slices(records: list[dict], decision_key: str = "decision") -> dict[str, Any]:
    return {
        case: metrics([row for row in records if row.get("evidence_case") == case], decision_key)
        for case in EVIDENCE_CASES
    }


def compatible_identity_false_refusals(records: list[dict], decision_key: str = "decision") -> int:
    return sum(
        row.get("expected") == "ANSWER"
        and row.get("identity_compatible") is True
        and row.get(decision_key) == "ABSTAIN"
        for row in records
    )


def acceptance(
    baseline: dict,
    candidate: dict,
    *,
    unsafe_relax: int,
    baseline_hard_negative_fa: int,
    candidate_hard_negative_fa: int,
    v341_regressions: int,
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
        "v341_regression_zero": v341_regressions == 0,
        "runtime_integrity": runtime_integrity,
    }
    return {
        "status": "DEV_READY" if all(checks.values()) else "PARTIAL",
        "fr_reduction": fr_reduction,
        "fa_increase_pp": fa_increase,
        "checks": checks,
    }
