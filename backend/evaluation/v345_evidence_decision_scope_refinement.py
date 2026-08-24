"""Public V3.45 Evidence decision-scope DEV benchmark contract.

The official manuals, query text, evidence excerpts, retrieval candidates, and
predictions stay in the gitignored private benchmark directory.  This module
only validates and scores the frozen V3.45 experiment.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from backend.evaluation.v342_evidence_sufficiency_identity_aware import metrics


BENCHMARK_VERSION = "v345-evidence-decision-scope-refinement-dev-v1"
BASELINE_VERSION = "evidence-v342-sufficiency-candidate"
CANDIDATE_VERSION = "evidence-v345-decision-scope-candidate"
CANDIDATE_SHA256 = "e9065905b0150f6ddbc216a956dad14424fbd0f9cd03f75f66817f4f96dcfee5"
QUERY_COUNT = 60
ANSWER_COUNT = 30
ABSTAIN_COUNT = 30
DOCUMENT_COUNT = 2

EVIDENCE_CASES = (
    "IDENTITY_COMPATIBLE_RECOVERY",
    "TECHNICAL_SYNONYM",
    "ABBREVIATION",
    "PARAMETER_TABLE_RELATION",
    "SECTION_INHERITED_EVIDENCE",
    "CROSS_REFERENCE_EVIDENCE",
)

HARD_NEGATIVE_TYPES = (
    "SAME_FAMILY_WRONG_PARAMETER",
    "SAME_MANUFACTURER_WRONG_MODEL",
    "SAME_VALUE_WRONG_ATTRIBUTE",
    "IDENTITY_CORRECT_EVIDENCE_INSUFFICIENT",
)


def validate_dataset(payload: dict) -> tuple[str, ...]:
    errors: list[str] = []
    rows = payload.get("queries", [])
    documents = payload.get("documents", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    if payload.get("baseline_version") != BASELINE_VERSION:
        errors.append("BASELINE_VERSION")
    if payload.get("candidate_version") != CANDIDATE_VERSION:
        errors.append("CANDIDATE_VERSION")
    if payload.get("candidate_sha256_at_freeze") != CANDIDATE_SHA256:
        errors.append("CANDIDATE_SHA256")
    for key in (
        "uses_a_to_h_data", "uses_j_data", "uses_k_data",
        "uses_historical_sealed_data", "uses_v342_documents",
        "uses_v343_documents",
    ):
        if payload.get(key) is not False:
            errors.append(f"FORBIDDEN_SOURCE:{key}")
    if len(documents) != DOCUMENT_COUNT:
        errors.append(f"DOCUMENT_COUNT:{len(documents)}")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")

    document_ids = {str(item.get("document_id", "")) for item in documents}
    hashes = [str(item.get("sha256", "")).casefold() for item in documents]
    if "" in document_ids or len(document_ids) != len(documents):
        errors.append("DOCUMENT_IDS")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        errors.append("DOCUMENT_HASHES")
    if any(item.get("official_english_pdf") is not True for item in documents):
        errors.append("OFFICIAL_ENGLISH_PDF")
    if any(item.get("prior_document_overlap") is not False for item in documents):
        errors.append("PRIOR_DOCUMENT_OVERLAP")

    required = {
        "query_id", "query", "document_id", "expected", "evidence_case",
        "evidence_relation_expected", "target", "relation", "attribute",
        "value_or_action", "relevant_chunk_ids", "confidence",
        "identity_expected", "identity_compatible", "hard_negative_type",
        "parser_recoverable",
    }
    cases: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    ids: list[str] = []
    texts: list[str] = []
    for index, row in enumerate(rows):
        qid = str(row.get("query_id", ""))
        ids.append(qid)
        texts.append(" ".join(str(row.get("query", "")).casefold().split()))
        missing = required - row.keys()
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(sorted(missing))}")
        if row.get("document_id") not in document_ids:
            errors.append(f"UNKNOWN_DOCUMENT:{qid}")
        case = str(row.get("evidence_case", ""))
        if case not in EVIDENCE_CASES:
            errors.append(f"EVIDENCE_CASE:{qid}")
        cases[case] += 1
        expected = row.get("expected")
        if expected == "ANSWER":
            answers[case] += 1
            if row.get("hard_negative_type") is not None:
                errors.append(f"ANSWER_HARD_NEGATIVE:{qid}")
            if not row.get("relevant_chunk_ids"):
                errors.append(f"ANSWER_EVIDENCE:{qid}")
            if not all(str(row.get(key, "")).strip() for key in (
                "target", "relation", "attribute", "value_or_action",
            )):
                errors.append(f"ANSWER_COMPONENTS:{qid}")
        elif expected == "ABSTAIN":
            hard_type = row.get("hard_negative_type")
            if hard_type not in HARD_NEGATIVE_TYPES:
                errors.append(f"HARD_NEGATIVE:{qid}")
            else:
                negatives[hard_type] += 1
            if row.get("relevant_chunk_ids"):
                errors.append(f"ABSTAIN_EVIDENCE:{qid}")
        else:
            errors.append(f"EXPECTED:{qid}")
        if row.get("confidence") != "HIGH":
            errors.append(f"CONFIDENCE:{qid}")
        if row.get("identity_expected") not in {"COMPATIBLE", "INCOMPATIBLE"}:
            errors.append(f"IDENTITY:{qid}")
        if row.get("identity_compatible") != (row.get("identity_expected") == "COMPATIBLE"):
            errors.append(f"IDENTITY_COMPATIBILITY:{qid}")

    if sum(row.get("expected") == "ANSWER" for row in rows) != ANSWER_COUNT:
        errors.append("ANSWER_COUNT")
    if sum(row.get("expected") == "ABSTAIN" for row in rows) != ABSTAIN_COUNT:
        errors.append("ABSTAIN_COUNT")
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_TEXTS")
    for case in EVIDENCE_CASES:
        if cases[case] != 10:
            errors.append(f"CASE_COUNT:{case}:{cases[case]}")
        if answers[case] != 5:
            errors.append(f"CASE_ANSWER_COUNT:{case}:{answers[case]}")
    if set(negatives) != set(HARD_NEGATIVE_TYPES):
        errors.append("HARD_NEGATIVE_COVERAGE")
    return tuple(errors)


def acceptance(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    unsafe_veto: int,
    unsafe_relax: int,
    baseline_hard_negative_fa: int,
    candidate_hard_negative_fa: int,
    baseline_replay_mismatches: int,
    runtime_integrity: bool,
    candidate_frozen: bool,
    one_shot: bool,
) -> dict[str, Any]:
    baseline_fa = int(baseline["false_answers"])
    fa_reduction = (
        (baseline_fa - int(candidate["false_answers"])) / baseline_fa
        if baseline_fa else 0.0
    )
    fr_increase_pp = (
        float(candidate["false_refusal_rate"])
        - float(baseline["false_refusal_rate"])
    )
    checks = {
        "fa_reduction_at_least_50pct": fa_reduction >= 0.5,
        "fr_increase_at_most_5pp": fr_increase_pp <= 0.05,
        "unsafe_veto_zero": unsafe_veto == 0,
        "unsafe_relax_zero": unsafe_relax == 0,
        "identity_hard_negative_fa_not_increased": (
            candidate_hard_negative_fa <= baseline_hard_negative_fa
        ),
        "baseline_replay_mismatch_zero": baseline_replay_mismatches == 0,
        "runtime_integrity": runtime_integrity,
        "candidate_frozen": candidate_frozen,
        "one_shot": one_shot,
    }
    return {
        "status": "DEV_READY" if all(checks.values()) else "PARTIAL",
        "fa_reduction": fa_reduction,
        "fr_increase_pp": fr_increase_pp,
        "checks": checks,
    }


__all__ = [
    "ABSTAIN_COUNT", "ANSWER_COUNT", "BASELINE_VERSION", "BENCHMARK_VERSION",
    "CANDIDATE_SHA256", "CANDIDATE_VERSION", "DOCUMENT_COUNT", "EVIDENCE_CASES",
    "HARD_NEGATIVE_TYPES", "QUERY_COUNT", "acceptance", "metrics", "validate_dataset",
]
