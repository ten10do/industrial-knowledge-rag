"""Public contract for the V3.46 claim-binding reassessment.

V3.46 is read-only over frozen V3.45 artifacts.  It attributes observed
decision errors without rerunning retrieval or any reasoning layer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v346-evidence-decision-claim-binding-reassessment-v1"
SOURCE_BENCHMARK_VERSION = "v345-evidence-decision-scope-refinement-dev-v1"
SOURCE_QUERY_COUNT = 60
SOURCE_ERROR_COUNT = 19
SOURCE_FALSE_ANSWER_COUNT = 9
SOURCE_FALSE_REFUSAL_COUNT = 10
SOURCE_DATASET_SHA256 = "206906c050eadb1e0eec823353938f6a653a9d603b445bb911a5b81b38eb0a78"
SOURCE_FILE_SHA256 = {
    "dev_benchmark.json": "e83908242fa520a4dd49bfec0ef61c1ad13e5f09f62e6693747ec64bb96fb871",
    "baseline_results.json": "6e8dbe806811dda3cc5bc91a07806cb3d191c08cabf6266001ac4c920e818564",
    "candidate_results.json": "33627122dfd7f7ae9d0113ea4ecb44c4df75d7268bb62018d163be8784d07b85",
}
CANDIDATE_VERSION = "evidence-v345-decision-scope-candidate"
CANDIDATE_SHA256 = "e9065905b0150f6ddbc216a956dad14424fbd0f9cd03f75f66817f4f96dcfee5"
REPRESENTATION_THRESHOLD = 0.50

FALSE_ANSWER_CLASSES = (
    "ATTRIBUTE_VOCABULARY_GAP",
    "SELECTED_CLAIM_WINDOW_GAP",
    "RELATION_QUALIFIER_GAP",
    "SECTION_OWNERSHIP_GAP",
    "REFERENCE_OWNERSHIP_GAP",
)
FALSE_REFUSAL_CLASSES = (
    "RETRIEVAL_MISSING",
    "IDENTITY_FALSE_REJECTION",
    "ROUTING_OUTSIDE_VERIFICATION",
)
REPRESENTATION_GAPS = frozenset({
    "ATTRIBUTE_VOCABULARY_GAP",
    "RELATION_QUALIFIER_GAP",
    "SECTION_OWNERSHIP_GAP",
    "REFERENCE_OWNERSHIP_GAP",
})
SELECTION_GAPS = frozenset({"SELECTED_CLAIM_WINDOW_GAP"})


@dataclass(frozen=True)
class Attribution:
    query_id: str
    error_type: str
    failure_class: str
    responsibility: str
    reason: str
    signal_in_selected_candidates: bool | None = None

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
    errors: list[str] = []
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if manifest.get("source_benchmark_version") != SOURCE_BENCHMARK_VERSION:
        errors.append("SOURCE_BENCHMARK_VERSION")
    if manifest.get("source_query_count") != SOURCE_QUERY_COUNT:
        errors.append("SOURCE_QUERY_COUNT")
    if manifest.get("source_dataset_sha256") != SOURCE_DATASET_SHA256:
        errors.append("SOURCE_DATASET_SHA256")
    if manifest.get("source_file_sha256") != SOURCE_FILE_SHA256:
        errors.append("SOURCE_FILE_SHA256")
    if manifest.get("candidate_version") != CANDIDATE_VERSION:
        errors.append("CANDIDATE_VERSION")
    if manifest.get("candidate_sha256") != CANDIDATE_SHA256:
        errors.append("CANDIDATE_SHA256")
    if manifest.get("read_only_reassessment") is not True:
        errors.append("READ_ONLY_REQUIRED")
    if manifest.get("reran_v345") is not False:
        errors.append("V345_RERUN_FORBIDDEN")
    return tuple(errors)


def validate_false_answer_annotations(
    annotations: list[dict], source_records: list[dict],
) -> tuple[str, ...]:
    errors: list[str] = []
    false_answers = {
        row["query_id"]: row for row in source_records
        if row.get("expected") == "ABSTAIN" and row.get("decision") == "ANSWER"
    }
    if len(false_answers) != SOURCE_FALSE_ANSWER_COUNT:
        errors.append(f"SOURCE_FALSE_ANSWERS:{len(false_answers)}")
    ids = [str(row.get("query_id", "")) for row in annotations]
    if len(ids) != SOURCE_FALSE_ANSWER_COUNT or not all(ids) or len(ids) != len(set(ids)):
        errors.append("ANNOTATION_IDS")
    if set(ids) != set(false_answers):
        errors.append("ANNOTATION_COVERAGE")
    required = {
        "query_id", "failure_class", "responsibility", "reason",
        "signal_in_selected_candidates",
    }
    for index, row in enumerate(annotations):
        if required - row.keys():
            errors.append(f"ANNOTATION_FIELDS:{index}")
        if row.get("failure_class") not in FALSE_ANSWER_CLASSES:
            errors.append(f"ANNOTATION_CLASS:{index}")
        if row.get("responsibility") not in {
            "EVIDENCE_CLAIM_REPRESENTATION", "RETRIEVAL_TO_EVIDENCE_SELECTION",
        }:
            errors.append(f"ANNOTATION_RESPONSIBILITY:{index}")
        expected_owner = (
            "EVIDENCE_CLAIM_REPRESENTATION"
            if row.get("failure_class") in REPRESENTATION_GAPS
            else "RETRIEVAL_TO_EVIDENCE_SELECTION"
        )
        if row.get("responsibility") != expected_owner:
            errors.append(f"ANNOTATION_OWNER_MISMATCH:{index}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"ANNOTATION_REASON:{index}")
        if row.get("signal_in_selected_candidates") not in {True, False}:
            errors.append(f"ANNOTATION_SIGNAL:{index}")
    return tuple(errors)


def classify_false_refusal(record: dict) -> Attribution:
    if record.get("expected") != "ANSWER" or record.get("decision") != "ABSTAIN":
        raise ValueError("NOT_FALSE_REFUSAL")
    if not record.get("relevant_evidence_retrieved", False):
        return Attribution(
            str(record["query_id"]), "FALSE_REFUSAL", "RETRIEVAL_MISSING",
            "RETRIEVAL", "GOLD_CHUNK_NOT_SELECTED", False,
        )
    if record.get("identity_result") != "COMPATIBLE":
        return Attribution(
            str(record["query_id"]), "FALSE_REFUSAL", "IDENTITY_FALSE_REJECTION",
            "IDENTITY", "RETRIEVED_GOLD_REJECTED_BY_IDENTITY", True,
        )
    if record.get("query_path") != "VERIFICATION":
        return Attribution(
            str(record["query_id"]), "FALSE_REFUSAL", "ROUTING_OUTSIDE_VERIFICATION",
            "QUERY_ROUTING", str(record.get("reason", "NON_VERIFICATION_PATH")), True,
        )
    raise ValueError(f"UNATTRIBUTED_FALSE_REFUSAL:{record.get('query_id')}")


def summarize(records: list[dict]) -> dict[str, Any]:
    counts = Counter(row["failure_class"] for row in records)
    false_answers = [row for row in records if row["error_type"] == "FALSE_ANSWER"]
    false_refusals = [row for row in records if row["error_type"] == "FALSE_REFUSAL"]
    representation = sum(counts[name] for name in REPRESENTATION_GAPS)
    selection = sum(counts[name] for name in SELECTION_GAPS)
    return {
        "errors": len(records),
        "false_answers": len(false_answers),
        "false_refusals": len(false_refusals),
        "representation_false_answers": representation,
        "selection_false_answers": selection,
        "representation_fa_share": representation / len(false_answers) if false_answers else 0.0,
        "selection_fa_share": selection / len(false_answers) if false_answers else 0.0,
        "selected_signal_present_false_answers": sum(
            row.get("signal_in_selected_candidates") is True for row in false_answers
        ),
        "failure_counts": {
            name: counts[name]
            for name in FALSE_ANSWER_CLASSES + FALSE_REFUSAL_CLASSES
        },
    }


def decide(summary: dict, *, integrity: bool, reconciled: bool) -> dict[str, Any]:
    checks = {
        "integrity": integrity,
        "metrics_reconciled": reconciled,
        "all_errors_classified": summary.get("errors") == SOURCE_ERROR_COUNT,
        "all_false_answers_classified": (
            summary.get("false_answers") == SOURCE_FALSE_ANSWER_COUNT
        ),
        "all_false_refusals_classified": (
            summary.get("false_refusals") == SOURCE_FALSE_REFUSAL_COUNT
        ),
    }
    if not all(checks.values()):
        status = "RUNTIME_INVALID"
        recommendation = "REPAIR_V346_REASSESSMENT"
    else:
        status = "CLAIM_BINDING_REASSESSMENT_COMPLETE"
        if float(summary.get("representation_fa_share", 0.0)) >= REPRESENTATION_THRESHOLD:
            recommendation = "ENTER_BOUNDED_EVIDENCE_CLAIM_BINDING_DESIGN"
        elif float(summary.get("selection_fa_share", 0.0)) > REPRESENTATION_THRESHOLD:
            recommendation = "ENTER_RETRIEVAL_TO_EVIDENCE_SELECTION_REASSESSMENT"
        else:
            recommendation = "CONTINUE_EVIDENCE_RESPONSIBILITY_REVIEW"
    return {
        "status": status,
        "recommendation": recommendation,
        "threshold": REPRESENTATION_THRESHOLD,
        "checks": checks,
    }
