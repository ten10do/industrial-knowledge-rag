"""Public contract for the read-only V3.48 responsibility reassessment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v348-bounded-claim-binding-responsibility-reassessment-v1"
SOURCE_BENCHMARK_VERSION = "v347-bounded-evidence-claim-binding-dev-v1"
SOURCE_QUERY_COUNT = 60
SOURCE_ERROR_COUNT = 27
SOURCE_FALSE_ANSWER_COUNT = 17
SOURCE_FALSE_REFUSAL_COUNT = 10
SOURCE_UNSAFE_VETO_COUNT = 3
SOURCE_DATASET_SHA256 = "2ef494fff413e68fe2166b02cab4f3fb7527c51e6010063f167517ac1bb91093"
SOURCE_FILE_SHA256 = {
    "dev_benchmark.json": "31576f6b73fad803397a34f93eb6a1f188d57fb964517ab9db90b1508e2a47ea",
    "baseline_results.json": "b7aa17a9f0591048c5a4a49883a381ffab4f7df0f0837fcb29637ebdb97d3125",
    "candidate_results.json": "0f13ca6d159e313f841bf21474eecee855d49943340d1b8320ead44829598921",
}
CANDIDATE_VERSION = "evidence-v347-bounded-claim-binding-candidate"
CANDIDATE_SHA256 = "07f0fe553973375e0750ee77267c25d97db7c97255b255aea2398b54e3beed68"
STRUCTURAL_THRESHOLD = 0.50

FALSE_ANSWER_CLASSES = (
    "RETRIEVAL_COUNTERCLAIM_MISSING",
    "TARGET_METADATA_BINDING_GAP",
    "ATTRIBUTE_ROLE_COLLISION",
    "QUALIFIER_CELL_BINDING_GAP",
    "SECTION_ATTRIBUTE_BINDING_GAP",
    "ATTRIBUTE_VOCABULARY_GAP",
    "REFERENCE_LINE_BINDING_GAP",
)
UNSAFE_VETO_CLASSES = (
    "SECTION_DISTANCE_PROXY_ERROR",
    "ATTRIBUTE_ROLE_COLLISION_VETO",
)
FALSE_REFUSAL_CLASSES = (
    "RETRIEVAL_MISSING",
    "INHERITED_EVIDENCE_REFUSAL",
    "CLAIM_BINDING_UNSAFE_VETO",
)
STRUCTURAL_CLASSES = frozenset({
    "ATTRIBUTE_ROLE_COLLISION",
    "QUALIFIER_CELL_BINDING_GAP",
    "SECTION_ATTRIBUTE_BINDING_GAP",
    "REFERENCE_LINE_BINDING_GAP",
    *UNSAFE_VETO_CLASSES,
})
RESPONSIBILITY_BY_CLASS = {
    "RETRIEVAL_COUNTERCLAIM_MISSING": "RETRIEVAL_TO_EVIDENCE_SELECTION",
    "TARGET_METADATA_BINDING_GAP": "EVIDENCE_CLAIM_BINDING",
    "ATTRIBUTE_ROLE_COLLISION": "TABLE_STRUCTURE_BOUNDARY",
    "QUALIFIER_CELL_BINDING_GAP": "TABLE_STRUCTURE_BOUNDARY",
    "SECTION_ATTRIBUTE_BINDING_GAP": "TABLE_STRUCTURE_BOUNDARY",
    "ATTRIBUTE_VOCABULARY_GAP": "EVIDENCE_CLAIM_BINDING",
    "REFERENCE_LINE_BINDING_GAP": "TABLE_STRUCTURE_BOUNDARY",
    "SECTION_DISTANCE_PROXY_ERROR": "TABLE_STRUCTURE_BOUNDARY",
    "ATTRIBUTE_ROLE_COLLISION_VETO": "TABLE_STRUCTURE_BOUNDARY",
}


@dataclass(frozen=True)
class Attribution:
    query_id: str
    error_type: str
    failure_class: str
    responsibility: str
    reason: str
    signal_in_selected_candidates: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "error_type": self.error_type,
            "failure_class": self.failure_class,
            "responsibility": self.responsibility,
            "reason": self.reason,
            "signal_in_selected_candidates": self.signal_in_selected_candidates,
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
        "V347_RERUN_FORBIDDEN": manifest.get("reran_v347") is False,
    }
    return tuple(name for name, passed in checks.items() if not passed)


def _validate_annotations(
    annotations: list[dict], expected_ids: set[str], allowed_classes: tuple[str, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    ids = [str(row.get("query_id", "")) for row in annotations]
    if not all(ids) or len(ids) != len(set(ids)) or set(ids) != expected_ids:
        errors.append("ANNOTATION_COVERAGE")
    required = {
        "query_id", "failure_class", "responsibility", "reason",
        "signal_in_selected_candidates",
    }
    for index, row in enumerate(annotations):
        failure_class = row.get("failure_class")
        if required - row.keys():
            errors.append(f"ANNOTATION_FIELDS:{index}")
        if failure_class not in allowed_classes:
            errors.append(f"ANNOTATION_CLASS:{index}")
        if row.get("responsibility") != RESPONSIBILITY_BY_CLASS.get(failure_class):
            errors.append(f"ANNOTATION_OWNER:{index}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"ANNOTATION_REASON:{index}")
        if row.get("signal_in_selected_candidates") not in {True, False}:
            errors.append(f"ANNOTATION_SIGNAL:{index}")
    return tuple(errors)


def validate_false_answer_annotations(
    annotations: list[dict], source_records: list[dict],
) -> tuple[str, ...]:
    ids = {
        row["query_id"] for row in source_records
        if row.get("expected") == "ABSTAIN" and row.get("decision") == "ANSWER"
    }
    errors = list(_validate_annotations(annotations, ids, FALSE_ANSWER_CLASSES))
    if len(ids) != SOURCE_FALSE_ANSWER_COUNT:
        errors.append(f"SOURCE_FALSE_ANSWERS:{len(ids)}")
    return tuple(errors)


def validate_unsafe_veto_annotations(
    annotations: list[dict], source_records: list[dict],
) -> tuple[str, ...]:
    ids = {
        row["query_id"] for row in source_records
        if row.get("expected") == "ANSWER" and row.get("baseline_decision") == "ANSWER"
        and row.get("decision") == "ABSTAIN" and row.get("action") == "VETO"
    }
    errors = list(_validate_annotations(annotations, ids, UNSAFE_VETO_CLASSES))
    if len(ids) != SOURCE_UNSAFE_VETO_COUNT:
        errors.append(f"SOURCE_UNSAFE_VETOES:{len(ids)}")
    return tuple(errors)


def classify_false_refusal(candidate: dict, baseline: dict) -> Attribution:
    if candidate.get("expected") != "ANSWER" or candidate.get("decision") != "ABSTAIN":
        raise ValueError("NOT_FALSE_REFUSAL")
    if candidate.get("action") == "VETO":
        return Attribution(
            str(candidate["query_id"]), "FALSE_REFUSAL", "CLAIM_BINDING_UNSAFE_VETO",
            "EVIDENCE_CLAIM_BINDING", str(candidate.get("reason_code", "")), True,
        )
    if not baseline.get("relevant_evidence_retrieved", False):
        return Attribution(
            str(candidate["query_id"]), "FALSE_REFUSAL", "RETRIEVAL_MISSING",
            "RETRIEVAL", "GOLD_CHUNK_NOT_SELECTED", False,
        )
    return Attribution(
        str(candidate["query_id"]), "FALSE_REFUSAL", "INHERITED_EVIDENCE_REFUSAL",
        "FROZEN_EVIDENCE_CHAIN", str(baseline.get("reason_code", "")), True,
    )


def summarize(records: list[dict], unsafe_vetoes: list[dict]) -> dict[str, Any]:
    counts = Counter(row["failure_class"] for row in records)
    false_answers = [row for row in records if row["error_type"] == "FALSE_ANSWER"]
    false_refusals = [row for row in records if row["error_type"] == "FALSE_REFUSAL"]
    actionable = len(false_answers) + len(unsafe_vetoes)
    structural = sum(row["failure_class"] in STRUCTURAL_CLASSES for row in false_answers)
    structural += sum(row["failure_class"] in STRUCTURAL_CLASSES for row in unsafe_vetoes)
    return {
        "errors": len(records),
        "false_answers": len(false_answers),
        "false_refusals": len(false_refusals),
        "unsafe_vetoes": len(unsafe_vetoes),
        "selected_signal_present_false_answers": sum(
            row["signal_in_selected_candidates"] for row in false_answers
        ),
        "selected_signal_missing_false_answers": sum(
            not row["signal_in_selected_candidates"] for row in false_answers
        ),
        "structural_actionable_errors": structural,
        "actionable_errors": actionable,
        "structural_actionable_share": structural / actionable if actionable else 0.0,
        "failure_counts": dict(sorted(counts.items())),
        "unsafe_veto_counts": dict(sorted(Counter(
            row["failure_class"] for row in unsafe_vetoes
        ).items())),
    }


def decide(summary: dict, *, integrity: bool, reconciled: bool) -> dict[str, Any]:
    checks = {
        "integrity": integrity,
        "metrics_reconciled": reconciled,
        "all_errors_classified": summary.get("errors") == SOURCE_ERROR_COUNT,
        "all_false_answers_classified": summary.get("false_answers") == SOURCE_FALSE_ANSWER_COUNT,
        "all_false_refusals_classified": summary.get("false_refusals") == SOURCE_FALSE_REFUSAL_COUNT,
        "all_unsafe_vetoes_explained": summary.get("unsafe_vetoes") == SOURCE_UNSAFE_VETO_COUNT,
    }
    if not all(checks.values()):
        status = "RUNTIME_INVALID"
        recommendation = "REPAIR_V348_REASSESSMENT"
    else:
        status = "RESPONSIBILITY_REASSESSMENT_COMPLETE"
        recommendation = (
            "ENTER_EVIDENCE_TABLE_OWNERSHIP_BOUNDARY_ANALYSIS"
            if float(summary.get("structural_actionable_share", 0.0)) >= STRUCTURAL_THRESHOLD
            else "CONTINUE_BOUNDED_EVIDENCE_CLAIM_BINDING_REVIEW"
        )
    return {
        "status": status,
        "recommendation": recommendation,
        "structural_threshold": STRUCTURAL_THRESHOLD,
        "checks": checks,
    }
