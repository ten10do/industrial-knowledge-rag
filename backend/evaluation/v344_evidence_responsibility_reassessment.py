"""Public V3.44 Evidence responsibility reassessment contract.

The reassessment is read-only over frozen V3.43 artifacts.  It separates the
upgrade-only V3.42 candidate's responsibility from the architectural Evidence
decision responsibility established in V3.29.  It has no runtime authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v344-evidence-responsibility-reassessment-v1"
SOURCE_BENCHMARK_VERSION = "v343-evidence-sufficiency-independent-dev-v1"
SOURCE_QUERY_COUNT = 60
SOURCE_DATASET_SHA256 = "31bb4f1b0c84efd761cf22c4d169d0b294eca3a7b54da4860e9b60a645fe994c"
SOURCE_FILE_SHA256 = {
    "independent_benchmark.json": "0390dd49dfe22c3e010a6cfe8386c681fcb93e311c15d7a0f8132e052fe93909",
    "baseline_results.json": "7d388e1ea3a7625c9dc64086d3b0ce79b3cefb3ff883876b074bfba2af7e8151",
    "candidate_results.json": "d9a2b24cc9e698d7c89b9907534247e9134da6b4fbcf77dca2294f9ed5aaa584",
}
CANDIDATE_VERSION = "evidence-v342-sufficiency-candidate"
CANDIDATE_SHA256 = "f02f39035ae1c88e7b2d65a5939bc4321739e6b561db18d5b89d63d13f18dcfc"
ATTRIBUTION_THRESHOLD = 0.30

FAILURE_TAXONOMY = (
    "PARSER_LIMIT",
    "ROUTING_OUTSIDE_VERIFICATION",
    "RETRIEVAL_MISSING",
    "IDENTITY_FALSE_REJECTION",
    "EVIDENCE_RELATION_BINDING_GAP",
    "EVIDENCE_VALUE_BINDING_GAP",
    "EVIDENCE_OTHER_GAP",
    "INHERITED_FALSE_ANSWER",
    "CANDIDATE_UNSAFE_RELAX",
    "SAFE_RECOVERY",
    "CORRECT_ANSWER",
    "CORRECT_ABSTAIN",
)

EVIDENCE_CANDIDATE_FAILURES = frozenset({
    "EVIDENCE_RELATION_BINDING_GAP",
    "EVIDENCE_VALUE_BINDING_GAP",
    "EVIDENCE_OTHER_GAP",
})
UPSTREAM_FALSE_REFUSALS = frozenset({
    "ROUTING_OUTSIDE_VERIFICATION",
    "RETRIEVAL_MISSING",
    "IDENTITY_FALSE_REJECTION",
})


@dataclass(frozen=True)
class Attribution:
    failure_class: str
    candidate_owner: str
    architectural_owner: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "failure_class": self.failure_class,
            "candidate_owner": self.candidate_owner,
            "architectural_owner": self.architectural_owner,
            "reason": self.reason,
        }


def validate_source_manifest(manifest: dict) -> tuple[str, ...]:
    errors: list[str] = []
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if manifest.get("source_benchmark_version") != SOURCE_BENCHMARK_VERSION:
        errors.append("SOURCE_BENCHMARK_VERSION")
    if manifest.get("source_dataset_sha256") != SOURCE_DATASET_SHA256:
        errors.append("SOURCE_DATASET_SHA256")
    if manifest.get("source_file_sha256") != SOURCE_FILE_SHA256:
        errors.append("SOURCE_FILE_SHA256")
    if manifest.get("candidate_version") != CANDIDATE_VERSION:
        errors.append("CANDIDATE_VERSION")
    if manifest.get("candidate_sha256") != CANDIDATE_SHA256:
        errors.append("CANDIDATE_SHA256")
    if manifest.get("source_query_count") != SOURCE_QUERY_COUNT:
        errors.append("SOURCE_QUERY_COUNT")
    if manifest.get("read_only_reassessment") is not True:
        errors.append("READ_ONLY_REQUIRED")
    if manifest.get("reran_v343") is not False:
        errors.append("V343_RERUN_FORBIDDEN")
    return tuple(errors)


def classify_record(record: dict) -> Attribution:
    expected = record.get("expected")
    baseline = record.get("baseline_decision")
    decision = record.get("decision")
    if expected == "ANSWER" and decision == "ANSWER":
        if baseline == "ABSTAIN":
            return Attribution("SAFE_RECOVERY", "V342_EVIDENCE_SUFFICIENCY", "NONE", str(record.get("reason", "")))
        return Attribution("CORRECT_ANSWER", "NONE", "NONE", "PRESERVED_CORRECT_ANSWER")
    if expected == "ABSTAIN" and decision == "ABSTAIN":
        return Attribution("CORRECT_ABSTAIN", "NONE", "NONE", "PRESERVED_CORRECT_ABSTAIN")
    if expected == "ABSTAIN" and decision == "ANSWER":
        if baseline == "ANSWER":
            return Attribution(
                "INHERITED_FALSE_ANSWER", "OUTSIDE_V342_UPGRADE_ONLY_SCOPE",
                "EVIDENCE_DECISION_BASELINE", "V341_ANSWER_PRESERVED",
            )
        return Attribution(
            "CANDIDATE_UNSAFE_RELAX", "V342_EVIDENCE_SUFFICIENCY",
            "EVIDENCE_SUFFICIENCY", str(record.get("reason", "")),
        )
    if expected != "ANSWER" or decision != "ABSTAIN":
        raise ValueError("UNCLASSIFIABLE_DECISION")
    if not record.get("parser_recoverable", True):
        return Attribution("PARSER_LIMIT", "OUTSIDE_V342_SCOPE", "PARSER", "PARSER_LIMIT")
    if not record.get("relevant_evidence_retrieved", False):
        return Attribution("RETRIEVAL_MISSING", "OUTSIDE_V342_SCOPE", "RETRIEVAL", "GOLD_CHUNK_NOT_SELECTED")
    if record.get("identity_result") == "INCOMPATIBLE":
        return Attribution(
            "IDENTITY_FALSE_REJECTION", "OUTSIDE_V342_SCOPE", "IDENTITY",
            "GOLD_COMPATIBLE_IDENTITY_REJECTED",
        )
    if record.get("query_path") != "VERIFICATION":
        return Attribution(
            "ROUTING_OUTSIDE_VERIFICATION", "OUTSIDE_V342_SCOPE", "QUERY_ROUTING",
            str(record.get("reason", "NON_VERIFICATION_PATH_PRESERVED")),
        )
    reason = str(record.get("reason", ""))
    if reason == "ATTRIBUTE_RELATION_MISSING":
        failure = "EVIDENCE_RELATION_BINDING_GAP"
    elif reason == "VALUE_OR_ACTION_NOT_BOUND":
        failure = "EVIDENCE_VALUE_BINDING_GAP"
    else:
        failure = "EVIDENCE_OTHER_GAP"
    return Attribution(failure, "V342_EVIDENCE_SUFFICIENCY", "EVIDENCE_SUFFICIENCY", reason)


def summarize(records: list[dict]) -> dict[str, Any]:
    counts = Counter(record["failure_class"] for record in records)
    residual_fr = sum(
        record.get("expected") == "ANSWER" and record.get("decision") == "ABSTAIN"
        for record in records
    )
    evidence_fr = sum(counts[name] for name in EVIDENCE_CANDIDATE_FAILURES)
    upstream_fr = sum(counts[name] for name in UPSTREAM_FALSE_REFUSALS)
    inherited_fa = counts["INHERITED_FALSE_ANSWER"]
    unsafe = counts["CANDIDATE_UNSAFE_RELAX"]
    return {
        "queries": len(records),
        "residual_false_refusals": residual_fr,
        "evidence_candidate_false_refusals": evidence_fr,
        "upstream_false_refusals": upstream_fr,
        "evidence_candidate_fr_share": evidence_fr / residual_fr if residual_fr else 0.0,
        "upstream_fr_share": upstream_fr / residual_fr if residual_fr else 0.0,
        "inherited_false_answers": inherited_fa,
        "candidate_unsafe_relax": unsafe,
        "safe_recoveries": counts["SAFE_RECOVERY"],
        "candidate_changes": counts["SAFE_RECOVERY"] + unsafe,
        "architectural_evidence_errors": evidence_fr + inherited_fa + unsafe,
        "failure_counts": {name: counts[name] for name in FAILURE_TAXONOMY},
    }


def decide(summary: dict, *, integrity: bool, reconciled: bool) -> dict[str, Any]:
    evidence_share = float(summary.get("evidence_candidate_fr_share", 0.0))
    upstream_share = float(summary.get("upstream_fr_share", 0.0))
    inherited_fa = int(summary.get("inherited_false_answers", 0))
    unsafe = int(summary.get("candidate_unsafe_relax", 0))
    checks = {
        "integrity": integrity,
        "metrics_reconciled": reconciled,
        "all_records_classified": int(summary.get("queries", 0)) == SOURCE_QUERY_COUNT,
        "candidate_unsafe_relax_zero": unsafe == 0,
    }
    if not all(checks.values()):
        status = "RUNTIME_INVALID"
        recommendation = "REPAIR_REASSESSMENT_RUNTIME"
    else:
        status = "RESPONSIBILITY_REASSESSMENT_COMPLETE"
        recommendation = (
            "ENTER_EVIDENCE_DECISION_SCOPE_REDESIGN"
            if inherited_fa > 0 and upstream_share > ATTRIBUTION_THRESHOLD
            else "CONTINUE_LOCAL_EVIDENCE_SUFFICIENCY_REFINEMENT"
        )
    return {
        "status": status,
        "recommendation": recommendation,
        "checks": checks,
        "local_sufficiency_triggered": evidence_share > ATTRIBUTION_THRESHOLD,
        "upstream_majority": upstream_share > ATTRIBUTION_THRESHOLD,
        "inherited_false_answer_scope_gap": inherited_fa > 0,
        "threshold": ATTRIBUTION_THRESHOLD,
    }
