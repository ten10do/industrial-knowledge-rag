"""Public V3.52 independent proof-contract generalization protocol."""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v351 import TABLE_STRUCTURE_PROOF_VERSION


BENCHMARK_VERSION = "v352-independent-table-structure-proof-contract-generalization-v1"
CONTRACT_VERSION = TABLE_STRUCTURE_PROOF_VERSION
CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"
FIXTURE_COUNT = 64
VALID_COUNT = 32
INVALID_COUNT = 32
CASE_COUNTS = {
    "MULTILEVEL_REPEATED_HEADER": (8, 4),
    "VERTICAL_MERGED_MODEL_CELL": (8, 4),
    "HORIZONTAL_MERGED_PARAMETER_HEADER": (8, 4),
    "MULTIPAGE_CHUNK_LINEAGE": (8, 4),
    "MULTI_MODEL_SAME_VALUE": (8, 4),
    "NESTED_DEFAULT_CONFIGURATION": (8, 4),
    "SECTION_DEVICE_CONFLICT": (8, 4),
    "EXTERNAL_CROSS_REFERENCE": (8, 4),
}
ALL_RELATIONS = frozenset(item.value for item in TableOwnershipRelation)
FORBIDDEN_DEFINITION_DOCUMENT_IDS = frozenset({
    "synthetic-industrial-table",
    "siemens-et200sp-ai-4xtc-hs-2019",
})


def validate_fixture_dataset(payload: dict) -> tuple[str, ...]:
    errors: list[str] = []
    rows = payload.get("fixtures", [])
    checks = {
        "BENCHMARK_VERSION": payload.get("benchmark_version") == BENCHMARK_VERSION,
        "CONTRACT_VERSION": payload.get("contract_version") == CONTRACT_VERSION,
        "CONTRACT_SHA256": payload.get("contract_sha256_at_freeze") == CONTRACT_SHA256,
        "INDEPENDENT_AUTHORING": payload.get("independent_authoring") is True,
        "V351_FIXTURE_REUSE_FORBIDDEN": payload.get("uses_v351_definition_fixtures") is False,
        "V349_FIXTURE_REUSE_FORBIDDEN": payload.get("uses_v349_frozen_fixtures") is False,
        "PARSER_FROZEN": payload.get("modified_parser") is False,
        "RUNTIME_INTEGRATION_FORBIDDEN": payload.get("runtime_integration") is False,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    if len(rows) != FIXTURE_COUNT:
        errors.append(f"FIXTURE_COUNT:{len(rows)}")
    ids = [str(row.get("fixture_id", "")) for row in rows]
    if not all(value.startswith("V352-") for value in ids) or len(ids) != len(set(ids)):
        errors.append("FIXTURE_IDS")
    expected_counts = Counter(row.get("expected") for row in rows)
    if expected_counts != Counter({"VALID": VALID_COUNT, "INVALID": INVALID_COUNT}):
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
    relations = {str(row.get("proof", {}).get("relation", "")) for row in rows}
    if relations != ALL_RELATIONS:
        errors.append("RELATION_COVERAGE")
    document_ids = {str(row.get("claim", {}).get("document_id", "")) for row in rows}
    if not document_ids or document_ids & FORBIDDEN_DEFINITION_DOCUMENT_IDS:
        errors.append("INDEPENDENT_DOCUMENT_IDS")
    required = {
        "fixture_id", "origin", "case", "expected", "claim", "proof",
        "lineage_peer", "expected_reason_code", "source_query_id",
    }
    for index, row in enumerate(rows):
        if required - row.keys():
            errors.append(f"FIXTURE_FIELDS:{index}")
        if row.get("origin") != "INDEPENDENT_SYNTHETIC":
            errors.append(f"FIXTURE_ORIGIN:{index}")
        if row.get("source_query_id") is not None:
            errors.append(f"SOURCE_QUERY_FORBIDDEN:{index}")
        if row.get("expected") == "INVALID" and not str(row.get("expected_reason_code", "")):
            errors.append(f"INVALID_REASON:{index}")
        if row.get("expected") == "VALID" and row.get("expected_reason_code") not in {None, ""}:
            errors.append(f"VALID_REASON:{index}")
        if not isinstance(row.get("lineage_peer"), (dict, type(None))):
            errors.append(f"LINEAGE_PEER:{index}")
    return tuple(errors)


def _slice_metrics(rows: list[dict]) -> dict[str, Any]:
    valid = [row for row in rows if row["expected"] == "VALID"]
    invalid = [row for row in rows if row["expected"] == "INVALID"]
    correct = lambda row: row["expected"] == row["observed"]
    return {
        "fixtures": len(rows),
        "accuracy": sum(correct(row) for row in rows) / len(rows),
        "valid_recall": sum(correct(row) for row in valid) / len(valid),
        "invalid_rejection": sum(correct(row) for row in invalid) / len(invalid),
        "unsafe_acceptance_count": sum(row["observed"] == "VALID" for row in invalid),
        "false_rejection_count": sum(row["observed"] == "INVALID" for row in valid),
    }


def generalization_metrics(records: list[dict]) -> dict[str, Any]:
    metrics = _slice_metrics(records)
    invalid = [row for row in records if row["expected"] == "INVALID"]
    reason_misses = sum(
        row.get("expected_reason_code") not in row.get("reason_codes", [])
        for row in invalid
    )
    metrics.update({
        "reason_target_miss_count": reason_misses,
        "reason_target_accuracy": 1.0 - reason_misses / len(invalid),
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
    contract_frozen: bool, one_shot: bool,
) -> dict[str, Any]:
    cases = metrics.get("case_metrics", {})
    checks = {
        "dataset_valid": dataset_valid,
        "source_integrity": source_integrity,
        "contract_frozen": contract_frozen,
        "one_shot": one_shot,
        "accuracy_at_least_95pct": metrics.get("accuracy", 0.0) >= 0.95,
        "valid_recall_at_least_90pct": metrics.get("valid_recall", 0.0) >= 0.90,
        "invalid_rejection_100pct": metrics.get("invalid_rejection") == 1.0,
        "unsafe_acceptance_zero": metrics.get("unsafe_acceptance_count") == 0,
        "reason_target_accuracy_at_least_95pct": metrics.get("reason_target_accuracy", 0.0) >= 0.95,
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
            "ENTER_TABLE_STRUCTURE_PROOF_PRODUCER_FEASIBILITY"
            if ready else "REASSESS_TABLE_STRUCTURE_PROOF_CONTRACT"
        ),
        "checks": checks,
    }
