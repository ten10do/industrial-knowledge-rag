"""Public V3.54 merged-cell coverage contract-candidate protocol."""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend.retrieval.table_structure_proof_contract_v354_candidate import (
    MERGED_CELL_COVERAGE_VERSION,
)


BENCHMARK_VERSION = "v354-merged-cell-coverage-contract-candidate-definition-v1"
CANDIDATE_VERSION = MERGED_CELL_COVERAGE_VERSION
CANDIDATE_SHA256 = "b02a2e73323ab05c63f6679f226c8840a17841104c650087b1d37b1fee263aa7"
V351_CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"
V352_FIXTURE_FILE_SHA256 = "95e9d21a469541bbfd9ca1359b9e754449c6abfff55151de43e01c1f90bee380"
FIXTURE_COUNT = 72
VALID_COUNT = 36
INVALID_COUNT = 36
CASE_COUNTS = {
    "FROZEN_V352_VERTICAL": (8, 4),
    "DEPTH_VARIATION": (8, 4),
    "NESTED_OVERLAPPING": (8, 4),
    "MULTIPAGE_COVERAGE_LINEAGE": (8, 4),
    "OUTSIDE_INCOMPLETE_DUPLICATE": (8, 4),
    "SCOPE_TABLE_MODEL_CONFLICT": (8, 4),
    "REGRESSION_V351_RELATIONS": (12, 6),
    "REGRESSION_V352_NONVERTICAL": (12, 6),
}
FROZEN_V352_VERTICAL_IDS = frozenset(
    [f"V352-VMC-V{index}" for index in range(1, 5)]
    + [f"V352-VMC-N{index}" for index in range(1, 5)]
)
FROZEN_V352_NONVERTICAL_IDS = frozenset({
    "V352-MRH-V1", "V352-MRH-N1", "V352-HMH-V1", "V352-HMH-N1",
    "V352-MPL-V1", "V352-MPL-N1", "V352-MSV-V1", "V352-MSV-N1",
    "V352-NDC-V1", "V352-NDC-N1", "V352-SDC-V1", "V352-SDC-N1",
})


def validate_fixture_dataset(payload: dict) -> tuple[str, ...]:
    rows = payload.get("fixtures", [])
    errors: list[str] = []
    checks = {
        "BENCHMARK_VERSION": payload.get("benchmark_version") == BENCHMARK_VERSION,
        "CANDIDATE_VERSION": payload.get("candidate_version") == CANDIDATE_VERSION,
        "CANDIDATE_SHA256": payload.get("candidate_sha256_at_freeze") == CANDIDATE_SHA256,
        "V351_SHA256": payload.get("v351_contract_sha256") == V351_CONTRACT_SHA256,
        "V352_SHA256": payload.get("v352_fixture_file_sha256") == V352_FIXTURE_FILE_SHA256,
        "PRE_REGISTERED": payload.get("pre_registered") is True,
        "RUNTIME_INTEGRATION_FORBIDDEN": payload.get("runtime_integration") is False,
        "PARSER_CHANGE_FORBIDDEN": payload.get("modified_parser") is False,
        "SPAN_ARITHMETIC_FORBIDDEN": payload.get("uses_span_arithmetic_for_membership") is False,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    if len(rows) != FIXTURE_COUNT:
        errors.append(f"FIXTURE_COUNT:{len(rows)}")
    ids = [str(row.get("fixture_id", "")) for row in rows]
    if not all(value.startswith("V354-") for value in ids) or len(ids) != len(set(ids)):
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
        "coverage_lineage_peer", "source_fixture_id",
    }
    for index, row in enumerate(rows):
        if required - row.keys():
            errors.append(f"FIXTURE_FIELDS:{index}")
        expected_reason = row.get("expected_reason_code")
        if row.get("expected") == "INVALID" and not str(expected_reason or ""):
            errors.append(f"INVALID_REASON:{index}")
        if row.get("expected") == "VALID" and expected_reason not in {None, ""}:
            errors.append(f"VALID_REASON:{index}")
        if row.get("case") in {
            "REGRESSION_V351_RELATIONS", "REGRESSION_V352_NONVERTICAL",
        } and row.get("coverage") is not None:
            errors.append(f"REGRESSION_COVERAGE_FORBIDDEN:{index}")
    vertical_sources = {
        row.get("source_fixture_id") for row in rows
        if row.get("case") == "FROZEN_V352_VERTICAL"
    }
    nonvertical_sources = {
        row.get("source_fixture_id") for row in rows
        if row.get("case") == "REGRESSION_V352_NONVERTICAL"
    }
    if vertical_sources != FROZEN_V352_VERTICAL_IDS:
        errors.append("FROZEN_VERTICAL_SOURCE_COVERAGE")
    if nonvertical_sources != FROZEN_V352_NONVERTICAL_IDS:
        errors.append("FROZEN_NONVERTICAL_SOURCE_COVERAGE")
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


def candidate_metrics(records: list[dict]) -> dict[str, Any]:
    metrics = _slice_metrics(records)
    invalid = [row for row in records if row["expected"] == "INVALID"]
    reason_hits = sum(
        row.get("expected_reason_code") in row.get("reason_codes", [])
        for row in invalid
    )
    cases = {
        case: _slice_metrics([row for row in records if row["case"] == case])
        for case in CASE_COUNTS
    }
    vertical = [row for row in records if row["case"] == "FROZEN_V352_VERTICAL"]
    metrics.update({
        "reason_target_accuracy": reason_hits / len(invalid),
        "case_metrics": cases,
        "coverage_used_count": sum(row.get("coverage_used", False) for row in records),
        "frozen_vertical_valid_recovered": sum(
            row["expected"] == "VALID" and row["observed"] == "VALID" for row in vertical
        ),
        "frozen_vertical_invalid_preserved": sum(
            row["expected"] == "INVALID" and row["observed"] == "INVALID" for row in vertical
        ),
        "v351_regression_count": sum(
            not row["correct"] for row in records
            if row["case"] == "REGRESSION_V351_RELATIONS"
        ),
        "v352_nonvertical_regression_count": sum(
            not row["correct"] for row in records
            if row["case"] == "REGRESSION_V352_NONVERTICAL"
        ),
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
        "accuracy_at_least_98pct": metrics.get("accuracy", 0.0) >= 0.98,
        "valid_recall_at_least_95pct": metrics.get("valid_recall", 0.0) >= 0.95,
        "invalid_rejection_100pct": metrics.get("invalid_rejection") == 1.0,
        "unsafe_acceptance_zero": metrics.get("unsafe_acceptance_count") == 0,
        "reason_target_accuracy_at_least_95pct": metrics.get("reason_target_accuracy", 0.0) >= 0.95,
        "frozen_vertical_valid_recovered_4_of_4": metrics.get("frozen_vertical_valid_recovered") == 4,
        "frozen_vertical_invalid_preserved_4_of_4": metrics.get("frozen_vertical_invalid_preserved") == 4,
        "v351_regression_zero": metrics.get("v351_regression_count") == 0,
        "v352_nonvertical_regression_zero": metrics.get("v352_nonvertical_regression_count") == 0,
        "each_case_unsafe_acceptance_zero": all(
            row.get("unsafe_acceptance_count") == 0 for row in cases.values()
        ) and set(cases) == set(CASE_COUNTS),
    }
    ready = all(checks.values())
    return {
        "status": "CONTRACT_CANDIDATE_READY" if ready else "PARTIAL",
        "recommendation": (
            "ENTER_V355_INDEPENDENT_COVERAGE_CONTRACT_GENERALIZATION"
            if ready else "REASSESS_MERGED_CELL_COVERAGE_CONTRACT"
        ),
        "checks": checks,
    }
