"""Public V3.53 read-only vertical merged-cell contract reassessment."""

from __future__ import annotations

from collections import Counter
from typing import Any


BENCHMARK_VERSION = "v353-vertical-merged-cell-coverage-contract-reassessment-v1"
SOURCE_BENCHMARK_VERSION = "v352-independent-table-structure-proof-contract-generalization-v1"
SOURCE_FIXTURE_COUNT = 64
SOURCE_VERTICAL_FIXTURE_COUNT = 8
SOURCE_VERTICAL_VALID_COUNT = 4
SOURCE_VERTICAL_INVALID_COUNT = 4
SOURCE_FALSE_REJECTION_COUNT = 4
SOURCE_UNSAFE_ACCEPTANCE_COUNT = 0
SOURCE_FILE_SHA256 = {
    "independent_fixtures.json": "95e9d21a469541bbfd9ca1359b9e754449c6abfff55151de43e01c1f90bee380",
    "generalization_results.json": "05ca09f213d66f4e0b766473805c86d5e648b1b3bcdf66726a004afd5cf52df1",
}
CONTRACT_SHA256 = "8bbfcb4c37bb734081429cb37710d8dcad9f06c042b1127017db8dcee4c1ebad"

COVERAGE_CLASSES = (
    "VALID_DESCENDANT_ROW_NOT_EXPRESSIBLE",
    "OUTSIDE_COVERAGE_REJECTED",
    "MODEL_SCOPE_MISMATCH_REJECTED",
    "CONFLICTING_OWNER_REJECTED",
    "MALFORMED_SPAN_REJECTED",
)
RESPONSIBILITY_BY_CLASS = {
    "VALID_DESCENDANT_ROW_NOT_EXPRESSIBLE": "PROOF_CONTRACT_SCHEMA",
    "OUTSIDE_COVERAGE_REJECTED": "EVIDENCE_PROOF_VALIDATION",
    "MODEL_SCOPE_MISMATCH_REJECTED": "EVIDENCE_PROOF_VALIDATION",
    "CONFLICTING_OWNER_REJECTED": "EVIDENCE_PROOF_VALIDATION",
    "MALFORMED_SPAN_REJECTED": "EVIDENCE_PROOF_VALIDATION",
}
MINIMUM_EXTENSION_FIELDS = frozenset({
    "coverage_id",
    "coverage_owner_scope_id",
    "coverage_anchor_cell_id",
    "covered_row_ids",
    "conflicting_coverage_ids",
})
EXTENSION_INVARIANTS = (
    "coverage identifiers are non-empty and stable",
    "coverage_owner_scope_id equals the proof model_scope_id",
    "coverage_anchor_cell_id remains in the same document and table region",
    "covered_row_ids is explicit, non-empty, unique, and includes the anchor row",
    "the value_row_id is a member of covered_row_ids",
    "conflicting_coverage_ids is empty",
    "coverage membership is byte-stable across chunk replication",
)
RETAINED_GUARDS = (
    "direct model_row_id equality remains valid when no merged coverage is supplied",
    "model and parameter scope matching remains exact",
    "column and header lineage checks remain unchanged",
    "conflicting owners continue to invalidate a proof",
    "merged_cell_span remains advisory and malformed spans remain invalid",
)
SAFETY_PROPERTIES = (
    "explicit_row_membership",
    "stable_identifier_lineage",
    "preserves_physical_anchor",
    "overlap_conflict_expressible",
    "no_proximity_inference",
)
REPRESENTATION_OPTIONS = {
    "SPAN_ARITHMETIC": {
        "explicit_row_membership": False,
        "stable_identifier_lineage": False,
        "preserves_physical_anchor": True,
        "overlap_conflict_expressible": False,
        "no_proximity_inference": False,
        "minimal": True,
    },
    "DUPLICATE_MODEL_ROW_ID": {
        "explicit_row_membership": True,
        "stable_identifier_lineage": False,
        "preserves_physical_anchor": False,
        "overlap_conflict_expressible": False,
        "no_proximity_inference": True,
        "minimal": True,
    },
    "ANCHOR_PLUS_COVERED_ROW_IDS": {
        "explicit_row_membership": True,
        "stable_identifier_lineage": True,
        "preserves_physical_anchor": True,
        "overlap_conflict_expressible": True,
        "no_proximity_inference": True,
        "minimal": True,
    },
    "FULL_CELL_OWNERSHIP_GRAPH": {
        "explicit_row_membership": True,
        "stable_identifier_lineage": True,
        "preserves_physical_anchor": True,
        "overlap_conflict_expressible": True,
        "no_proximity_inference": True,
        "minimal": False,
    },
}


def validate_source_manifest(manifest: dict) -> tuple[str, ...]:
    checks = {
        "BENCHMARK_VERSION": manifest.get("benchmark_version") == BENCHMARK_VERSION,
        "SOURCE_BENCHMARK_VERSION": manifest.get("source_benchmark_version") == SOURCE_BENCHMARK_VERSION,
        "SOURCE_FIXTURE_COUNT": manifest.get("source_fixture_count") == SOURCE_FIXTURE_COUNT,
        "SOURCE_FILE_SHA256": manifest.get("source_file_sha256") == SOURCE_FILE_SHA256,
        "CONTRACT_SHA256": manifest.get("contract_sha256") == CONTRACT_SHA256,
        "READ_ONLY": manifest.get("read_only_reassessment") is True,
        "V352_RERUN_FORBIDDEN": manifest.get("reran_v352") is False,
        "CONTRACT_CHANGE_FORBIDDEN": manifest.get("modified_v351_contract") is False,
        "PARSER_CHANGE_FORBIDDEN": manifest.get("modified_parser") is False,
    }
    return tuple(name for name, passed in checks.items() if not passed)


def validate_annotations(
    annotations: list[dict], source_records: list[dict],
) -> tuple[str, ...]:
    errors: list[str] = []
    vertical = {
        row["fixture_id"]: row for row in source_records
        if row.get("case") == "VERTICAL_MERGED_MODEL_CELL"
    }
    ids = [str(row.get("fixture_id", "")) for row in annotations]
    if not all(ids) or len(ids) != len(set(ids)) or set(ids) != set(vertical):
        errors.append("ANNOTATION_COVERAGE")
    required = {
        "fixture_id", "coverage_class", "responsibility", "reason",
        "required_extension_fields",
    }
    for index, row in enumerate(annotations):
        source = vertical.get(str(row.get("fixture_id", "")), {})
        coverage_class = row.get("coverage_class")
        fields = row.get("required_extension_fields")
        if required - row.keys():
            errors.append(f"ANNOTATION_FIELDS:{index}")
        if coverage_class not in COVERAGE_CLASSES:
            errors.append(f"ANNOTATION_CLASS:{index}")
        if row.get("responsibility") != RESPONSIBILITY_BY_CLASS.get(coverage_class):
            errors.append(f"ANNOTATION_OWNER:{index}")
        if not isinstance(fields, list) or not set(fields or ()).issubset(MINIMUM_EXTENSION_FIELDS):
            errors.append(f"ANNOTATION_EXTENSION_FIELDS:{index}")
        if coverage_class == "VALID_DESCENDANT_ROW_NOT_EXPRESSIBLE":
            if set(fields or ()) != MINIMUM_EXTENSION_FIELDS:
                errors.append(f"ANNOTATION_MINIMUM_EXTENSION:{index}")
            if source.get("expected") != "VALID" or source.get("observed") != "INVALID":
                errors.append(f"ANNOTATION_FALSE_REJECTION:{index}")
        elif source.get("expected") != "INVALID" or source.get("observed") != "INVALID":
            errors.append(f"ANNOTATION_SAFE_REJECTION:{index}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"ANNOTATION_REASON:{index}")
    if len(vertical) != SOURCE_VERTICAL_FIXTURE_COUNT:
        errors.append(f"SOURCE_VERTICAL_COUNT:{len(vertical)}")
    return tuple(errors)


def assess_representation_options() -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for name, properties in REPRESENTATION_OPTIONS.items():
        safe = all(properties[property] for property in SAFETY_PROPERTIES)
        rows[name] = {
            **properties,
            "safe": safe,
            "selected": safe and properties["minimal"],
        }
    selected = [name for name, row in rows.items() if row["selected"]]
    return {"options": rows, "selected": selected, "unique_minimum_safe": len(selected) == 1}


def summarize(annotations: list[dict], source_records: list[dict]) -> dict[str, Any]:
    vertical = [
        row for row in source_records if row.get("case") == "VERTICAL_MERGED_MODEL_CELL"
    ]
    return {
        "vertical_fixtures": len(vertical),
        "vertical_valid": sum(row["expected"] == "VALID" for row in vertical),
        "vertical_invalid": sum(row["expected"] == "INVALID" for row in vertical),
        "false_rejections": sum(
            row["expected"] == "VALID" and row["observed"] == "INVALID"
            for row in vertical
        ),
        "unsafe_acceptances": sum(
            row["expected"] == "INVALID" and row["observed"] == "VALID"
            for row in vertical
        ),
        "schema_gap_annotations": sum(
            row["responsibility"] == "PROOF_CONTRACT_SCHEMA" for row in annotations
        ),
        "safe_guard_annotations": sum(
            row["responsibility"] == "EVIDENCE_PROOF_VALIDATION" for row in annotations
        ),
        "class_counts": dict(sorted(Counter(
            row["coverage_class"] for row in annotations
        ).items())),
    }


def decide(
    summary: dict, option_assessment: dict, *, integrity: bool,
    annotations_complete: bool, source_reconciled: bool,
) -> dict[str, Any]:
    checks = {
        "integrity": integrity,
        "annotations_complete": annotations_complete,
        "source_reconciled": source_reconciled,
        "all_vertical_fixtures_classified": summary.get("vertical_fixtures") == SOURCE_VERTICAL_FIXTURE_COUNT,
        "all_false_rejections_explained": summary.get("false_rejections") == SOURCE_FALSE_REJECTION_COUNT,
        "unsafe_acceptance_zero": summary.get("unsafe_acceptances") == SOURCE_UNSAFE_ACCEPTANCE_COUNT,
        "schema_gap_exact": summary.get("schema_gap_annotations") == SOURCE_VERTICAL_VALID_COUNT,
        "safe_guards_preserved": summary.get("safe_guard_annotations") == SOURCE_VERTICAL_INVALID_COUNT,
        "unique_minimum_safe_representation": option_assessment.get("unique_minimum_safe") is True,
        "selected_anchor_plus_covered_rows": option_assessment.get("selected") == ["ANCHOR_PLUS_COVERED_ROW_IDS"],
    }
    complete = all(checks.values())
    return {
        "status": "CONTRACT_REASSESSMENT_COMPLETE" if complete else "RUNTIME_INVALID",
        "contract_decision": (
            "EXPLICIT_SCOPE_COVERAGE_REQUIRED" if complete else "UNRESOLVED"
        ),
        "minimum_safe_representation": (
            "ANCHOR_PLUS_COVERED_ROW_IDS" if complete else None
        ),
        "recommendation": (
            "DEFINE_V354_MERGED_CELL_COVERAGE_CONTRACT_CANDIDATE"
            if complete else "REPAIR_V353_REASSESSMENT"
        ),
        "checks": checks,
    }
