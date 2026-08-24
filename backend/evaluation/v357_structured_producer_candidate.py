"""Public V3.57 structured table producer candidate protocol.

Records the preregistered DEV-benchmark shape and acceptance gates for the
V3.57 candidate phase. Real-page ground truth lives in the gitignored
private benchmark area together with document provenance; this module only
holds version identifiers, gate thresholds and validation helpers.
"""

from __future__ import annotations

from typing import Any


CANDIDATE_VERSION = "structured-table-producer-v357-candidate"
BENCHMARK_VERSION = "v357-dev-realpage-benchmark-v1"

# Preregistered DEV acceptance gates (protocol 13.6).
GATE = {
    "min_ownership_precision": 0.90,
    "max_unsafe_acceptance": 0,
    "min_invalid_rejection": 0.95,
    "min_valid_recall_fraction": 0.50,
}

FAILURE_TAXONOMY_CODES = (
    "NO_TEXT_GRID",
    "HEADER_BAND_UNSPLIT",
    "VERTICAL_MERGE_FOUND",
    "BINDING_NOT_FOUND",
    "AMBIGUOUS_LOCATION",
)

REQUIRED_ANNOTATION_FIELDS = {
    "doc",
    "page_index",
    "columns",
    "rows",
}

REQUIRED_RESULT_KEYS = {
    "benchmark_version",
    "metrics",
    "gates",
}


def validate_annotation_entry(entry: dict[str, Any]) -> tuple[str, ...]:
    """Structural checks for one annotated real-page table."""
    errors: list[str] = []
    missing = REQUIRED_ANNOTATION_FIELDS - entry.keys()
    if missing:
        errors.append("MISSING_FIELDS:" + ",".join(sorted(missing)))
        return tuple(errors)
    rows = entry["rows"]
    if not isinstance(rows, list) or not rows:
        errors.append("ROWS_EMPTY")
    columns = entry["columns"]
    if not isinstance(columns, list) or not columns:
        errors.append("COLUMNS_EMPTY")
    n_rows = len(rows)
    for merge in entry.get("merges", []):
        covered = merge.get("covered_row_indices", [])
        if not covered or any(not isinstance(i, int) or i < 0 or i >= n_rows for i in covered):
            errors.append("MERGE_INDEX_INVALID")
            break
        if merge.get("anchor_row_index") != min(covered):
            errors.append("MERGE_ANCHOR_NOT_FIRST")
            break
    for row_index, row in enumerate(rows):
        cells = row.get("cells", {})
        for key in cells:
            if not isinstance(key, str) or not key.isdigit():
                errors.append(f"CELL_KEY_INVALID:{row_index}:{key}")
                break
    return tuple(dict.fromkeys(errors))


def evaluate_candidate_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Preregistered DEV gate evaluation."""
    checks = {
        "OWNERSHIP_PRECISION>=90": (
            metrics.get("ownership_precision_proxy", 0.0)
            >= GATE["min_ownership_precision"]
        ),
        "UNSAFE_ACCEPTANCE=0": (
            metrics.get("unsafe_acceptance", -1) == GATE["max_unsafe_acceptance"]
        ),
        "INVALID_REJECTION>=95": (
            metrics.get("invalid_rejection", 0.0) >= GATE["min_invalid_rejection"]
        ),
        "USABLE_RECALL": (
            metrics.get("valid_accepted", 0)
            >= GATE["min_valid_recall_fraction"] * max(metrics.get("valid_total", 1), 1)
        ),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "gate_version": "v357-dev-gate-preregistered",
    }
