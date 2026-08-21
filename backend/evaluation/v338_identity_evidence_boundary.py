"""Public contract for the V3.38 Identity/Evidence boundary diagnosis.

The DEV documents, queries, annotations, retrieved text, and per-query results
remain private and gitignored.  This module contains only schema validation,
failure attribution, aggregation, and the pre-registered recommendation rule.
It has no runtime decision authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


BENCHMARK_VERSION = "v338-identity-evidence-fr-diagnosis-dev-v1"
QUERY_COUNT = 60
ANSWER_COUNT = 40
ABSTAIN_COUNT = 20
ATTRIBUTION_THRESHOLD = 0.30

COVERAGE = (
    "SAME_FAMILY_COMPATIBLE",
    "MODULE_OWNERSHIP",
    "PARAMETER_INHERITANCE",
    "TABLE_SPECIFICATION",
    "CROSS_REFERENCE",
    "CONFIGURATION_RELATIONSHIP",
)

FAILURE_TAXONOMY = (
    "IDENTITY_ERROR",
    "RETRIEVAL_MISSING",
    "EVIDENCE_TOO_STRICT",
    "PARSER_LIMIT",
    "UNSUPPORTED",
)

RECOMMENDATIONS = {
    "IDENTITY_ERROR": "IDENTITY_REFINEMENT",
    "RETRIEVAL_MISSING": "RETRIEVAL_IMPROVEMENT",
    "EVIDENCE_TOO_STRICT": "EVIDENCE_SUFFICIENCY_REFINEMENT",
    "PARSER_LIMIT": "PARSER_TABLE_ANALYSIS",
}


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def validate_dataset(payload: dict) -> ValidationReport:
    errors: list[str] = []
    rows = payload.get("queries", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if payload.get("uses_v335_data") is not False:
        errors.append("V335_EXCLUSION")
    if payload.get("uses_v337_data") is not False:
        errors.append("V337_EXCLUSION")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")
    answers = sum(row.get("expected") == "ANSWER" for row in rows)
    abstains = sum(row.get("expected") == "ABSTAIN" for row in rows)
    if answers != ANSWER_COUNT:
        errors.append(f"ANSWER_COUNT:{answers}")
    if abstains != ABSTAIN_COUNT:
        errors.append(f"ABSTAIN_COUNT:{abstains}")
    required = {
        "query_id", "query", "document_id", "expected", "coverage",
        "relevant_chunk_ids", "confidence", "new_document",
    }
    for index, row in enumerate(rows):
        missing = sorted(key for key in required if key not in row)
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(missing)}")
        if row.get("expected") not in {"ANSWER", "ABSTAIN"}:
            errors.append(f"EXPECTED:{index}")
        if row.get("coverage") not in COVERAGE:
            errors.append(f"COVERAGE:{index}")
        if row.get("confidence") != "HIGH":
            errors.append(f"CONFIDENCE:{index}")
        if row.get("new_document") is not True:
            errors.append(f"NEW_DOCUMENT:{index}")
        if (row.get("expected") == "ANSWER") != bool(row.get("relevant_chunk_ids")):
            errors.append(f"RELEVANCE_LABEL:{index}")
    ids = [str(row.get("query_id", "")) for row in rows]
    texts = [" ".join(str(row.get("query", "")).casefold().split()) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_TEXTS")
    covered = Counter(row.get("coverage") for row in rows)
    errors.extend(f"MISSING_COVERAGE:{name}" for name in COVERAGE if not covered[name])
    return ValidationReport(tuple(errors))


def classify_failure(record: dict) -> tuple[str, str]:
    """Attribute one observation using frozen labels and runtime diagnostics."""
    expected = record.get("expected")
    predicted = record.get("final_decision")
    if expected == "ABSTAIN":
        return "UNSUPPORTED", "EXPECTED_UNSUPPORTED"
    if predicted == "ANSWER":
        return "UNSUPPORTED", "NO_FAILURE"
    if not record.get("relevant_evidence_parsed", False):
        return "PARSER_LIMIT", str(record.get("parser_reason") or "RELEVANT_STRUCTURE_NOT_RECOVERED")
    if not record.get("relevant_evidence_retrieved", False):
        return "RETRIEVAL_MISSING", "GOLD_CHUNK_NOT_IN_RETRIEVAL_CANDIDATES"
    if record.get("identity_result") == "INCOMPATIBLE":
        return "IDENTITY_ERROR", str(record.get("identity_reason") or "COMPATIBLE_IDENTITY_REJECTED")
    return "EVIDENCE_TOO_STRICT", str(record.get("evidence_reason") or "SUFFICIENT_EVIDENCE_REFUSED")


def summarize(records: list[dict]) -> dict[str, Any]:
    failures = [row for row in records if row.get("expected") == "ANSWER" and row.get("final_decision") == "ABSTAIN"]
    counts = Counter(row.get("failure_class") for row in failures)
    denominator = len(failures)
    return {
        "queries": len(records),
        "answerable": sum(row.get("expected") == "ANSWER" for row in records),
        "abstain": sum(row.get("expected") == "ABSTAIN" for row in records),
        "false_refusals": denominator,
        "false_answers": sum(row.get("expected") == "ABSTAIN" and row.get("final_decision") == "ANSWER" for row in records),
        "failure_counts": {name: counts[name] for name in FAILURE_TAXONOMY},
        "failure_shares": {
            name: counts[name] / denominator if denominator else 0.0
            for name in FAILURE_TAXONOMY
        },
    }


def recommend(summary: dict) -> dict[str, Any]:
    shares = summary.get("failure_shares", {})
    triggered = [
        name for name in RECOMMENDATIONS
        if float(shares.get(name, 0.0)) > ATTRIBUTION_THRESHOLD
    ]
    primary = max(triggered, key=lambda name: float(shares[name])) if triggered else ""
    return {
        "threshold": ATTRIBUTION_THRESHOLD,
        "triggered": triggered,
        "primary_failure_class": primary or None,
        "recommendation": RECOMMENDATIONS.get(primary, "NO_REFINEMENT_TRIGGERED"),
    }
