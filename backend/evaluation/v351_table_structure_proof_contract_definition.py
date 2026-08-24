"""Public V3.51 table-structure proof contract evaluation protocol."""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v351 import (
    SUPPORTED_RELATIONS,
    TABLE_STRUCTURE_PROOF_VERSION,
)


BENCHMARK_VERSION = "v351-table-structure-proof-contract-definition-v1"
CONTRACT_VERSION = TABLE_STRUCTURE_PROOF_VERSION
CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"
FIXTURE_COUNT = 36
VALID_COUNT = 18
INVALID_COUNT = 18
SYNTHETIC_COUNT = 30
FROZEN_V349_COUNT = 6
FROZEN_V349_IDS = frozenset({
    "V349-004", "V349-007", "V349-035", "V349-036", "V349-040", "V349-041",
})
REQUIRED_CASES = frozenset({
    "SIMPLE_PARAMETER_TABLE",
    "MULTI_MODEL_TABLE",
    "MERGED_HEADER",
    "DEFAULT_CONFIGURATION_COLUMN",
    "SAFETY_LIMIT_TABLE",
    "SECTION_INHERITANCE",
    "CROSS_REFERENCE",
    "LINEAGE_STABILITY",
})
ALL_RELATIONS = frozenset({item.value for item in TableOwnershipRelation})
SOURCE_V349_SHA256 = {
    "dev_benchmark.json": "e99e3ad1134dbe6ee997e09d16a0ef044d7d7c78a65a6c298698d17c9511516b",
    "baseline_results.json": "735751e4577124656a9e77b6284168ca01c33ff490365e6fc4ed3bfd6a782c40",
    "candidate_results.json": "77a5a0d6476baca90bac7d2cde9165001126845e1a38c49a2f0b021aef45dee6",
}


def validate_fixture_dataset(payload: dict) -> tuple[str, ...]:
    errors: list[str] = []
    rows = payload.get("fixtures", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if payload.get("contract_version") != CONTRACT_VERSION:
        errors.append("CONTRACT_VERSION")
    if payload.get("contract_sha256_at_freeze") != CONTRACT_SHA256:
        errors.append("CONTRACT_SHA256")
    if payload.get("source_v349_sha256") != SOURCE_V349_SHA256:
        errors.append("SOURCE_V349_SHA256")
    if payload.get("modified_parser") is not False:
        errors.append("PARSER_MUST_REMAIN_FROZEN")
    if payload.get("runtime_integration") is not False:
        errors.append("RUNTIME_INTEGRATION_FORBIDDEN")
    if len(rows) != FIXTURE_COUNT:
        errors.append(f"FIXTURE_COUNT:{len(rows)}")
    ids = [str(row.get("fixture_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("FIXTURE_IDS")
    expected_counts = Counter(row.get("expected") for row in rows)
    if expected_counts != Counter({"VALID": VALID_COUNT, "INVALID": INVALID_COUNT}):
        errors.append("EXPECTED_DISTRIBUTION")
    origin_counts = Counter(row.get("origin") for row in rows)
    if origin_counts != Counter({"SYNTHETIC": SYNTHETIC_COUNT, "V349_FROZEN": FROZEN_V349_COUNT}):
        errors.append("ORIGIN_DISTRIBUTION")
    cases = {str(row.get("case", "")) for row in rows}
    if not REQUIRED_CASES.issubset(cases):
        errors.append("CASE_COVERAGE")
    relations = {str(row.get("proof", {}).get("relation", "")) for row in rows}
    if relations != ALL_RELATIONS:
        errors.append("RELATION_COVERAGE")
    frozen_ids = {
        str(row.get("source_query_id")) for row in rows
        if row.get("origin") == "V349_FROZEN"
    }
    if frozen_ids != FROZEN_V349_IDS:
        errors.append("FROZEN_V349_COVERAGE")
    required = {
        "fixture_id", "origin", "case", "expected", "claim", "proof",
        "expected_reason_code", "source_query_id",
    }
    for index, row in enumerate(rows):
        if required - row.keys():
            errors.append(f"FIXTURE_FIELDS:{index}")
        if row.get("expected") == "INVALID" and not str(row.get("expected_reason_code", "")):
            errors.append(f"INVALID_REASON:{index}")
        if row.get("expected") == "VALID" and row.get("expected_reason_code") not in {None, ""}:
            errors.append(f"VALID_REASON:{index}")
        if row.get("origin") == "SYNTHETIC" and row.get("source_query_id") is not None:
            errors.append(f"SYNTHETIC_SOURCE_ID:{index}")
        if row.get("origin") == "V349_FROZEN" and not row.get("source_query_id"):
            errors.append(f"FROZEN_SOURCE_ID:{index}")
        if not isinstance(row.get("lineage_peer"), (dict, type(None))):
            errors.append(f"LINEAGE_PEER:{index}")
    return tuple(errors)


def contract_metrics(records: list[dict]) -> dict[str, Any]:
    valid = [row for row in records if row["expected"] == "VALID"]
    invalid = [row for row in records if row["expected"] == "INVALID"]
    frozen = [row for row in records if row["origin"] == "V349_FROZEN"]
    correct = lambda row: row["observed"] == row["expected"]
    reason_target_misses = sum(
        row["expected"] == "INVALID"
        and row.get("expected_reason_code") not in row.get("reason_codes", [])
        for row in records
    )
    return {
        "fixtures": len(records),
        "accuracy": sum(correct(row) for row in records) / len(records),
        "valid_recall": sum(correct(row) for row in valid) / len(valid),
        "invalid_rejection": sum(correct(row) for row in invalid) / len(invalid),
        "unsafe_acceptance_count": sum(row["observed"] == "VALID" for row in invalid),
        "false_rejection_count": sum(row["observed"] == "INVALID" for row in valid),
        "reason_target_miss_count": reason_target_misses,
        "reason_target_accuracy": 1.0 - reason_target_misses / len(invalid),
        "frozen_v349_correct": sum(correct(row) for row in frozen),
        "frozen_v349_total": len(frozen),
        "reason_counts": dict(sorted(Counter(
            reason for row in records for reason in row.get("reason_codes", [])
        ).items())),
    }


def decide(
    metrics: dict[str, Any], *, dataset_valid: bool, source_integrity: bool,
    contract_frozen: bool, one_shot: bool,
) -> dict[str, Any]:
    checks = {
        "dataset_valid": dataset_valid,
        "source_integrity": source_integrity,
        "contract_frozen": contract_frozen,
        "one_shot": one_shot,
        "accuracy_100pct": metrics.get("accuracy") == 1.0,
        "valid_recall_100pct": metrics.get("valid_recall") == 1.0,
        "invalid_rejection_100pct": metrics.get("invalid_rejection") == 1.0,
        "unsafe_acceptance_zero": metrics.get("unsafe_acceptance_count") == 0,
        "false_rejection_zero": metrics.get("false_rejection_count") == 0,
        "reason_targets_100pct": metrics.get("reason_target_miss_count") == 0,
        "frozen_v349_all_correct": (
            metrics.get("frozen_v349_correct") == FROZEN_V349_COUNT
            and metrics.get("frozen_v349_total") == FROZEN_V349_COUNT
        ),
    }
    return {
        "status": "CONTRACT_READY" if all(checks.values()) else "PARTIAL",
        "recommendation": (
            "ENTER_INDEPENDENT_CONTRACT_GENERALIZATION"
            if all(checks.values()) else "REASSESS_TABLE_STRUCTURE_PROOF_SCHEMA"
        ),
        "checks": checks,
    }
