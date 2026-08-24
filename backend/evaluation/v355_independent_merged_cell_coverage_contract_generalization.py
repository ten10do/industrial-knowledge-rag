"""Public V3.55 independent merged-cell coverage generalization protocol."""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v354_candidate import (
    MERGED_CELL_COVERAGE_VERSION,
)


BENCHMARK_VERSION = "v355-independent-merged-cell-coverage-contract-generalization-v1"
CANDIDATE_VERSION = MERGED_CELL_COVERAGE_VERSION
CANDIDATE_SHA256 = "b02a2e73323ab05c63f6679f226c8840a17841104c650087b1d37b1fee263aa7"
V351_CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"
V354_DEFINITION_RESULT_SHA256 = "371a2b04acc4b0740f15c40112aec635118393a7bc167b8996c5fef54c463263"
FIXTURE_COUNT = 64
VALID_COUNT = 32
INVALID_COUNT = 32
CASE_COUNTS = {
    "DEEP_VARIABLE_VERTICAL": (8, 4),
    "NESTED_SAME_OWNER": (8, 4),
    "OVERLAPPING_OWNER_CONFLICT": (8, 4),
    "MULTIPAGE_REPLICATION": (8, 4),
    "MEMBERSHIP_BOUNDARY": (8, 4),
    "SCOPE_BOUNDARY": (8, 4),
    "ADVISORY_SPAN_VARIATION": (6, 3),
    "BASE_RELATION_REGRESSION": (10, 5),
}
ALL_BASE_RELATIONS = frozenset({
    TableOwnershipRelation.DIRECT_ROW.value,
    TableOwnershipRelation.COLUMN_BOUND.value,
    TableOwnershipRelation.HEADER_INHERITED.value,
    TableOwnershipRelation.SECTION_INHERITED.value,
    TableOwnershipRelation.CROSS_REFERENCE.value,
})


def validate_fixture_dataset(payload: dict) -> tuple[str, ...]:
    rows = payload.get("fixtures", [])
    errors: list[str] = []
    checks = {
        "BENCHMARK_VERSION": payload.get("benchmark_version") == BENCHMARK_VERSION,
        "CANDIDATE_VERSION": payload.get("candidate_version") == CANDIDATE_VERSION,
        "CANDIDATE_SHA256": payload.get("candidate_sha256_at_freeze") == CANDIDATE_SHA256,
        "V351_SHA256": payload.get("v351_contract_sha256") == V351_CONTRACT_SHA256,
        "V354_RESULT_SHA256": payload.get("v354_definition_result_sha256") == V354_DEFINITION_RESULT_SHA256,
        "INDEPENDENT_AUTHORING": payload.get("independent_authoring") is True,
        "V354_FIXTURE_REUSE_FORBIDDEN": payload.get("uses_v354_definition_fixtures") is False,
        "V352_FIXTURE_REUSE_FORBIDDEN": payload.get("uses_v352_fixtures") is False,
        "PARSER_FROZEN": payload.get("modified_parser") is False,
        "RUNTIME_INTEGRATION_FORBIDDEN": payload.get("runtime_integration") is False,
        "SPAN_ARITHMETIC_FORBIDDEN": payload.get("uses_span_arithmetic_for_membership") is False,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    if len(rows) != FIXTURE_COUNT:
        errors.append(f"FIXTURE_COUNT:{len(rows)}")
    ids = [str(row.get("fixture_id", "")) for row in rows]
    if not all(value.startswith("V355-") for value in ids) or len(ids) != len(set(ids)):
        errors.append("FIXTURE_IDS")
    if Counter(row.get("expected") for row in rows) != Counter({"VALID": VALID_COUNT, "INVALID": INVALID_COUNT}):
        errors.append("EXPECTED_DISTRIBUTION")
    case_counts = Counter(row.get("case") for row in rows)
    case_valid = Counter(row.get("case") for row in rows if row.get("expected") == "VALID")
    for case, (count, valid_count) in CASE_COUNTS.items():
        if case_counts[case] != count:
            errors.append(f"CASE_COUNT:{case}:{case_counts[case]}")
        if case_valid[case] != valid_count:
            errors.append(f"CASE_VALID_COUNT:{case}:{case_valid[case]}")
    if set(case_counts) != set(CASE_COUNTS):
        errors.append("CASE_COVERAGE")
    required = {
        "fixture_id", "origin", "case", "expected", "expected_reason_code",
        "claim", "proof", "coverage", "proof_lineage_peer",
        "coverage_lineage_peer", "source_query_id",
    }
    for index, row in enumerate(rows):
        if required - row.keys():
            errors.append(f"FIXTURE_FIELDS:{index}")
        if row.get("origin") != "INDEPENDENT_SYNTHETIC":
            errors.append(f"FIXTURE_ORIGIN:{index}")
        if row.get("source_query_id") is not None:
            errors.append(f"SOURCE_QUERY_FORBIDDEN:{index}")
        expected_reason = row.get("expected_reason_code")
        if row.get("expected") == "INVALID" and not str(expected_reason or ""):
            errors.append(f"INVALID_REASON:{index}")
        if row.get("expected") == "VALID" and expected_reason not in {None, ""}:
            errors.append(f"VALID_REASON:{index}")
        if row.get("case") == "BASE_RELATION_REGRESSION" and row.get("coverage") is not None:
            errors.append(f"BASE_REGRESSION_COVERAGE_FORBIDDEN:{index}")
    regression_relations = {
        row.get("proof", {}).get("relation") for row in rows
        if row.get("case") == "BASE_RELATION_REGRESSION" and row.get("expected") == "VALID"
    }
    if regression_relations != ALL_BASE_RELATIONS:
        errors.append("BASE_RELATION_COVERAGE")
    document_ids = {str(row.get("claim", {}).get("document_id", "")) for row in rows}
    if not document_ids or any("v354" in value.casefold() or "v352" in value.casefold() for value in document_ids):
        errors.append("INDEPENDENT_DOCUMENT_IDS")
    return tuple(errors)


def _slice_metrics(rows: list[dict]) -> dict[str, Any]:
    valid = [row for row in rows if row["expected"] == "VALID"]
    invalid = [row for row in rows if row["expected"] == "INVALID"]
    return {
        "fixtures": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "valid_recall": sum(row["correct"] for row in valid) / len(valid),
        "invalid_rejection": sum(row["correct"] for row in invalid) / len(invalid),
        "unsafe_acceptance_count": sum(row["observed"] == "VALID" for row in invalid),
        "false_rejection_count": sum(row["observed"] == "INVALID" for row in valid),
    }


def generalization_metrics(records: list[dict]) -> dict[str, Any]:
    metrics = _slice_metrics(records)
    invalid = [row for row in records if row["expected"] == "INVALID"]
    reason_hits = sum(
        row.get("expected_reason_code") in row.get("reason_codes", [])
        for row in invalid
    )
    metrics.update({
        "reason_target_accuracy": reason_hits / len(invalid),
        "coverage_used_count": sum(row.get("coverage_used", False) for row in records),
        "base_relation_regression_count": sum(
            not row["correct"] for row in records
            if row["case"] == "BASE_RELATION_REGRESSION"
        ),
        "case_metrics": {
            case: _slice_metrics([row for row in records if row["case"] == case])
            for case in CASE_COUNTS
        },
        "reason_counts": dict(sorted(Counter(
            reason for row in records for reason in row.get("reason_codes", [])
        ).items())),
    })
    return metrics


def decide(
    metrics: dict[str, Any], *, dataset_valid: bool, source_integrity: bool,
    candidate_frozen: bool, one_shot: bool,
) -> dict[str, Any]:
    cases = metrics.get("case_metrics", {})
    checks = {
        "dataset_valid": dataset_valid,
        "source_integrity": source_integrity,
        "candidate_frozen": candidate_frozen,
        "one_shot": one_shot,
        "accuracy_at_least_95pct": metrics.get("accuracy", 0.0) >= 0.95,
        "valid_recall_at_least_90pct": metrics.get("valid_recall", 0.0) >= 0.90,
        "invalid_rejection_100pct": metrics.get("invalid_rejection") == 1.0,
        "unsafe_acceptance_zero": metrics.get("unsafe_acceptance_count") == 0,
        "reason_target_accuracy_at_least_95pct": metrics.get("reason_target_accuracy", 0.0) >= 0.95,
        "base_relation_regression_zero": metrics.get("base_relation_regression_count") == 0,
        "each_case_valid_recall_at_least_75pct": all(
            row.get("valid_recall", 0.0) >= 0.75 for row in cases.values()
        ) and set(cases) == set(CASE_COUNTS),
        "each_case_unsafe_acceptance_zero": all(
            row.get("unsafe_acceptance_count") == 0 for row in cases.values()
        ) and set(cases) == set(CASE_COUNTS),
    }
    ready = all(checks.values())
    return {
        "status": "GENERALIZATION_READY" if ready else "PARTIAL",
        "recommendation": (
            "ENTER_V356_MERGED_CELL_COVERAGE_PRODUCER_FEASIBILITY"
            if ready else "REASSESS_MERGED_CELL_COVERAGE_CONTRACT"
        ),
        "checks": checks,
    }
