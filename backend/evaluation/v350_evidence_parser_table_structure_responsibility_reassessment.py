"""Public contract for the read-only V3.50 table-structure reassessment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v350-evidence-parser-table-structure-responsibility-reassessment-v1"
SOURCE_BENCHMARK_VERSION = "v349-table-ownership-boundary-analysis-dev-v1"
SOURCE_QUERY_COUNT = 60
SOURCE_ERROR_COUNT = 28
SOURCE_FALSE_ANSWER_COUNT = 1
SOURCE_FALSE_REFUSAL_COUNT = 27
SOURCE_OWNERSHIP_ACTION_COUNT = 15
SOURCE_BINDING_COUNT = 4
SOURCE_VETO_COUNT = 11
SOURCE_UNSAFE_BINDING_COUNT = 1
SOURCE_UNSAFE_VETO_COUNT = 5
SOURCE_DATASET_SHA256 = "78c7ed34b26d35f2903855c4ce7ae814e5f98a8ccb7fdb058be0daa8c8f9ae8c"
SOURCE_FILE_SHA256 = {
    "dev_benchmark.json": "e99e3ad1134dbe6ee997e09d16a0ef044d7d7c78a65a6c298698d17c9511516b",
    "baseline_results.json": "735751e4577124656a9e77b6284168ca01c33ff490365e6fc4ed3bfd6a782c40",
    "candidate_results.json": "77a5a0d6476baca90bac7d2cde9165001126845e1a38c49a2f0b021aef45dee6",
}
CANDIDATE_VERSION = "evidence-v349-table-ownership-candidate"
CANDIDATE_SHA256 = "32f274276bc1ddf10a0338d312254a85e377db8f03aa3f534c49c56a704c4484"

STRUCTURE_PROOF_FIELDS = frozenset({
    "table_region_id",
    "row_id",
    "column_id",
    "header_path",
    "cell_role",
    "model_scope_id",
    "parameter_scope_id",
    "value_scope_id",
    "qualifier_scope_id",
    "section_scope_id",
    "merged_cell_span",
    "reference_target",
    "conflicting_owner_ids",
})

ACTION_CLASSES = (
    "FLATTENED_DIRECT_ROW_SUFFICIENT",
    "EXPLICIT_REFERENCE_SUFFICIENT",
    "CONSERVATIVE_UNSUPPORTED_SAFE",
    "ROW_CELL_LINEAGE_MISSING",
    "COLUMN_HEADER_LINEAGE_MISSING",
    "RETRIEVAL_EVIDENCE_MISSING",
)
RESPONSIBILITY_BY_CLASS = {
    "FLATTENED_DIRECT_ROW_SUFFICIENT": "EVIDENCE_POLICY",
    "EXPLICIT_REFERENCE_SUFFICIENT": "EVIDENCE_POLICY",
    "CONSERVATIVE_UNSUPPORTED_SAFE": "EVIDENCE_POLICY",
    "ROW_CELL_LINEAGE_MISSING": "TABLE_STRUCTURE_PROVENANCE",
    "COLUMN_HEADER_LINEAGE_MISSING": "TABLE_STRUCTURE_PROVENANCE",
    "RETRIEVAL_EVIDENCE_MISSING": "RETRIEVAL_SELECTION",
}
STRUCTURE_CLASSES = frozenset({
    "ROW_CELL_LINEAGE_MISSING",
    "COLUMN_HEADER_LINEAGE_MISSING",
})


@dataclass(frozen=True)
class ErrorAttribution:
    query_id: str
    error_type: str
    failure_class: str
    responsibility: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "error_type": self.error_type,
            "failure_class": self.failure_class,
            "responsibility": self.responsibility,
            "reason": self.reason,
        }


def validate_source_manifest(manifest: dict) -> tuple[str, ...]:
    checks = {
        "BENCHMARK_VERSION": manifest.get("benchmark_version") == BENCHMARK_VERSION,
        "SOURCE_BENCHMARK_VERSION": manifest.get("source_benchmark_version") == SOURCE_BENCHMARK_VERSION,
        "SOURCE_QUERY_COUNT": manifest.get("source_query_count") == SOURCE_QUERY_COUNT,
        "SOURCE_DATASET_SHA256": manifest.get("source_dataset_sha256") == SOURCE_DATASET_SHA256,
        "SOURCE_FILE_SHA256": manifest.get("source_file_sha256") == SOURCE_FILE_SHA256,
        "CANDIDATE_VERSION": manifest.get("candidate_version") == CANDIDATE_VERSION,
        "CANDIDATE_SHA256": manifest.get("candidate_sha256") == CANDIDATE_SHA256,
        "READ_ONLY_REQUIRED": manifest.get("read_only_reassessment") is True,
        "V349_RERUN_FORBIDDEN": manifest.get("reran_v349") is False,
        "PARSER_CHANGE_FORBIDDEN": manifest.get("modified_parser") is False,
    }
    return tuple(name for name, passed in checks.items() if not passed)


def _action_ids(records: list[dict]) -> set[str]:
    return {
        str(row["query_id"]) for row in records
        if row.get("ownership_relation") != "UNSUPPORTED" or row.get("action") == "VETO"
    }


def validate_action_annotations(
    annotations: list[dict], source_records: list[dict],
) -> tuple[str, ...]:
    errors: list[str] = []
    source_by_id = {str(row["query_id"]): row for row in source_records}
    expected_ids = _action_ids(source_records)
    ids = [str(row.get("query_id", "")) for row in annotations]
    if not all(ids) or len(ids) != len(set(ids)) or set(ids) != expected_ids:
        errors.append("ANNOTATION_COVERAGE")
    required = {
        "query_id", "action_kind", "correct", "failure_class", "responsibility",
        "required_proof_fields", "reason",
    }
    for index, row in enumerate(annotations):
        query_id = str(row.get("query_id", ""))
        source = source_by_id.get(query_id, {})
        failure_class = row.get("failure_class")
        fields = row.get("required_proof_fields")
        expected_action = (
            "BINDING" if source.get("ownership_relation") != "UNSUPPORTED" else "VETO"
        )
        if required - row.keys():
            errors.append(f"ANNOTATION_FIELDS:{index}")
        if row.get("action_kind") != expected_action:
            errors.append(f"ANNOTATION_ACTION:{index}")
        if row.get("correct") is not (source.get("decision") == source.get("expected")):
            errors.append(f"ANNOTATION_CORRECTNESS:{index}")
        if failure_class not in ACTION_CLASSES:
            errors.append(f"ANNOTATION_CLASS:{index}")
        if row.get("responsibility") != RESPONSIBILITY_BY_CLASS.get(failure_class):
            errors.append(f"ANNOTATION_OWNER:{index}")
        if not isinstance(fields, list) or not set(fields or ()).issubset(STRUCTURE_PROOF_FIELDS):
            errors.append(f"ANNOTATION_PROOF_FIELDS:{index}")
        if failure_class in STRUCTURE_CLASSES and not fields:
            errors.append(f"STRUCTURE_PROOF_REQUIRED:{index}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"ANNOTATION_REASON:{index}")
    if len(expected_ids) != SOURCE_OWNERSHIP_ACTION_COUNT:
        errors.append(f"SOURCE_OWNERSHIP_ACTIONS:{len(expected_ids)}")
    return tuple(errors)


def classify_error(
    candidate: dict, baseline: dict, action_annotation: dict | None,
) -> ErrorAttribution:
    expected = candidate.get("expected")
    decision = candidate.get("decision")
    if expected == decision:
        raise ValueError("NOT_ERROR")
    query_id = str(candidate["query_id"])
    if action_annotation is not None:
        return ErrorAttribution(
            query_id,
            "FALSE_ANSWER" if expected == "ABSTAIN" else "FALSE_REFUSAL",
            str(action_annotation["failure_class"]),
            str(action_annotation["responsibility"]),
            str(action_annotation["reason"]),
        )
    if expected == "ANSWER" and decision == "ABSTAIN":
        if not baseline.get("relevant_evidence_retrieved", False):
            return ErrorAttribution(
                query_id, "FALSE_REFUSAL", "INHERITED_RETRIEVAL_MISSING",
                "RETRIEVAL_SELECTION", "REGISTERED_EVIDENCE_NOT_SELECTED",
            )
        return ErrorAttribution(
            query_id, "FALSE_REFUSAL", "INHERITED_EVIDENCE_REFUSAL",
            "FROZEN_EVIDENCE_CHAIN", str(baseline.get("reason_code", "")),
        )
    raise ValueError("UNANNOTATED_FALSE_ANSWER")


def summarize(errors: list[dict], actions: list[dict]) -> dict[str, Any]:
    bindings = [row for row in actions if row["action_kind"] == "BINDING"]
    vetoes = [row for row in actions if row["action_kind"] == "VETO"]
    return {
        "errors": len(errors),
        "false_answers": sum(row["error_type"] == "FALSE_ANSWER" for row in errors),
        "false_refusals": sum(row["error_type"] == "FALSE_REFUSAL" for row in errors),
        "ownership_actions": len(actions),
        "bindings": len(bindings),
        "vetoes": len(vetoes),
        "unsafe_bindings": sum(not row["correct"] for row in bindings),
        "unsafe_vetoes": sum(not row["correct"] for row in vetoes),
        "correct_actions": sum(row["correct"] for row in actions),
        "action_responsibility_counts": dict(sorted(Counter(
            row["responsibility"] for row in actions
        ).items())),
        "error_responsibility_counts": dict(sorted(Counter(
            row["responsibility"] for row in errors
        ).items())),
        "action_failure_counts": dict(sorted(Counter(
            row["failure_class"] for row in actions
        ).items())),
    }


def decide(
    summary: dict, *, integrity: bool, reconciled: bool,
    annotations_complete: bool,
) -> dict[str, Any]:
    checks = {
        "integrity": integrity,
        "metrics_reconciled": reconciled,
        "annotations_complete": annotations_complete,
        "all_errors_classified": summary.get("errors") == SOURCE_ERROR_COUNT,
        "all_false_answers_classified": summary.get("false_answers") == SOURCE_FALSE_ANSWER_COUNT,
        "all_false_refusals_classified": summary.get("false_refusals") == SOURCE_FALSE_REFUSAL_COUNT,
        "all_actions_classified": summary.get("ownership_actions") == SOURCE_OWNERSHIP_ACTION_COUNT,
        "bindings_reconciled": summary.get("bindings") == SOURCE_BINDING_COUNT,
        "vetoes_reconciled": summary.get("vetoes") == SOURCE_VETO_COUNT,
        "unsafe_bindings_reconciled": summary.get("unsafe_bindings") == SOURCE_UNSAFE_BINDING_COUNT,
        "unsafe_vetoes_reconciled": summary.get("unsafe_vetoes") == SOURCE_UNSAFE_VETO_COUNT,
    }
    valid = all(checks.values())
    return {
        "status": "RESPONSIBILITY_REASSESSMENT_COMPLETE" if valid else "RUNTIME_INVALID",
        "responsibility_decision": (
            "SPLIT_STRUCTURE_PRODUCER_AND_EVIDENCE_VALIDATOR"
            if valid else "UNRESOLVED"
        ),
        "recommendation": (
            "DEFINE_TABLE_STRUCTURE_PROOF_CONTRACT"
            if valid else "REPAIR_V350_REASSESSMENT"
        ),
        "checks": checks,
    }
