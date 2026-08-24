"""Public V3.56 merged-cell coverage producer feasibility protocol.

This module records the preregistered benchmark shape and acceptance gates
for the V3.56 feasibility phase. The producer itself lives in
backend.retrieval.table_structure_producer_v356 and is intentionally NOT
wired into any runtime path. Fixture PDFs, their ground truth, the runner,
and raw results stay in the gitignored private benchmark area.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


BENCHMARK_VERSION = "v356-producer-feasibility-fixtures-v1"
PRODUCER_VERSION = "table-structure-producer-v356-feasibility"
PRODUCER_SHA256 = "4dc23edafb34d4e4a22fb1ff354a719b149d93f2901f82c8db88b6080e8d54ad"
V351_CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"
V354_CANDIDATE_SHA256 = "b02a2e73323ab05c63f6679f226c8840a17841104c650087b1d37b1fee263aa7"

FIXTURE_COUNT = 72
CATEGORIES = (
    "SIMPLE_TABLE",
    "MULTI_LEVEL_HEADER",
    "MERGED_HEADER_COLSPAN",
    "MERGED_MODEL_COLUMN_ROWSPAN",
    "DEEP_ROWSPAN",
    "COLSPAN_DATA_QUALIFIER",
    "MIN_DEFAULT_MAX",
    "MODEL_MATRIX",
    "UNIT_INHERITANCE",
    "CONFIG_MATRIX",
    "REFERENCE_NOTE_CELLS",
    "MULTIPAGE_REPLICATION",
)
CLAIM_TYPES = ("VALID", "INVALID", "AMBIGUOUS")

# Preregistered feasibility gate (protocol 12.7).
GATE = {
    "min_valid_acceptance": 0.95,
    "min_invalid_rejection": 0.95,
    "max_unsafe_structure_acceptance": 0,
    "min_cell_ownership_precision": 0.90,
    "min_merged_region_precision": 0.90,
}

REQUIRED_RESULT_KEYS = {
    "benchmark_version",
    "metrics",
    "gates",
    "fixtures",
}


def validate_benchmark_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    """Structural checks for the private benchmark result payload."""
    errors: list[str] = []
    missing = REQUIRED_RESULT_KEYS - payload.keys()
    if missing:
        errors.append("MISSING_KEYS:" + ",".join(sorted(missing)))
        return tuple(errors)
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION_MISMATCH")
    metrics = payload.get("metrics", {})
    producer = metrics.get("condition_B_producer", {})
    if not producer:
        errors.append("CONDITION_B_MISSING")
    structure = metrics.get("structure_means", {})
    flat = metrics.get("condition_C_flat_control", {})
    ceiling = metrics.get("condition_A_ceiling", {})
    for key in ("valid_acceptance", "invalid_rejection", "unsafe_structure_acceptance", "ownership_precision"):
        if key not in producer:
            errors.append(f"METRIC_MISSING:{key}")
    if "merge_precision" not in structure:
        errors.append("METRIC_MISSING:merge_precision")
    if "invalid_accepted_rate" not in flat:
        errors.append("METRIC_MISSING:flat_invalid_accepted_rate")
    if "invalid_rejection_rate" not in ceiling:
        errors.append("METRIC_MISSING:ceiling_invalid_rejection_rate")
    fixtures = payload.get("fixtures", [])
    if len(fixtures) != FIXTURE_COUNT:
        errors.append(f"FIXTURE_COUNT:{len(fixtures)}")
    ids = [str(row.get("fixture_id", "")) for row in fixtures]
    if any(not fid.startswith("V356-") for fid in ids) or len(ids) != len(set(ids)):
        errors.append("FIXTURE_IDS")
    return tuple(dict.fromkeys(errors))


def evaluate_feasibility_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Preregistered gate evaluation over aggregated condition-B metrics."""
    producer = metrics.get("condition_B_producer", {})
    structure = metrics.get("structure_means", {})
    flat = metrics.get("condition_C_flat_control", {})
    replication = metrics.get("replication_stability", {})
    checks = {
        "VALID_CONTRACT_ACCEPTANCE>=95": (
            producer.get("valid_acceptance", 0.0) >= GATE["min_valid_acceptance"]
        ),
        "INVALID_REJECTION>=95": (
            producer.get("invalid_rejection", 0.0) >= GATE["min_invalid_rejection"]
        ),
        "UNSAFE_STRUCTURE_ACCEPTANCE=0": (
            producer.get("unsafe_structure_acceptance", -1)
            == GATE["max_unsafe_structure_acceptance"]
        ),
        "OWNERSHIP_PRECISION>=90": (
            producer.get("ownership_precision", 0.0)
            >= GATE["min_cell_ownership_precision"]
        ),
        "MERGE_PRECISION>=90": (
            structure.get("merge_precision", 0.0) >= GATE["min_merged_region_precision"]
        ),
        "STRUCTURED_BEATS_FLAT": (
            flat.get("invalid_accepted_rate", 1.0)
            > producer.get("unsafe_structure_acceptance", 1.0)
        ),
        "REPLICATION_STABLE": (
            replication.get("stable", 0) >= replication.get("applicable", 1)
        ),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "gate_version": "v356-feasibility-gate-preregistered",
    }


def claim_type_distribution(fixtures: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter = Counter()
    for row in fixtures:
        counter[row.get("category", "?")] += 0  # keep categories visible
        for claim in row.get("claims", []):
            counter[claim.get("claim_type", "?")] += 1
    return dict(counter)
