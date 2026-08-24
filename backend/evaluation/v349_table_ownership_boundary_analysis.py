"""Public V3.49 Evidence table-ownership DEV contract."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from backend.evaluation.v342_evidence_sufficiency_identity_aware import metrics


BENCHMARK_VERSION = "v349-table-ownership-boundary-analysis-dev-v1"
BASELINE_VERSION = "evidence-v347-bounded-claim-binding-candidate"
CANDIDATE_VERSION = "evidence-v349-table-ownership-candidate"
CANDIDATE_SHA256 = "32f274276bc1ddf10a0338d312254a85e377db8f03aa3f534c49c56a704c4484"
QUERY_COUNT = 60
ANSWER_COUNT = 30
ABSTAIN_COUNT = 30
DOCUMENT_COUNT = 2

TABLE_CASE_COUNTS = {
    "SIMPLE_PARAMETER_TABLE": (8, 4),
    "MULTI_MODEL_TABLE": (8, 4),
    "MERGED_HEADER": (8, 4),
    "MULTI_COLUMN_SPECIFICATION": (8, 4),
    "CROSS_REFERENCE_TABLE": (7, 4),
    "DEFAULT_VALUE_TABLE": (7, 4),
    "CONFIGURATION_TABLE": (7, 3),
    "SAFETY_LIMIT_TABLE": (7, 3),
}
HARD_NEGATIVE_TYPES = (
    "SAME_VALUE_WRONG_MODEL",
    "SAME_PARAMETER_WRONG_COLUMN",
    "SAME_TABLE_NEARBY_DIFFERENT_SCOPE",
    "SAME_SECTION_DIFFERENT_DEVICE",
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
        "uses_v331_j", "uses_v333_k", "uses_v335_l", "uses_v342_dev",
        "uses_v348_reassessment_data", "uses_historical_sealed_data",
    ):
        if payload.get(key) is not False:
            errors.append(f"FORBIDDEN_SOURCE:{key}")
    if len(documents) != DOCUMENT_COUNT:
        errors.append(f"DOCUMENT_COUNT:{len(documents)}")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")
    document_ids = {str(row.get("document_id", "")) for row in documents}
    hashes = [str(row.get("sha256", "")).casefold() for row in documents]
    if "" in document_ids or len(document_ids) != len(documents):
        errors.append("DOCUMENT_IDS")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        errors.append("DOCUMENT_HASHES")
    if any(row.get("official_english_pdf") is not True for row in documents):
        errors.append("OFFICIAL_ENGLISH_PDF")
    if any(row.get("prior_document_overlap") is not False for row in documents):
        errors.append("PRIOR_DOCUMENT_OVERLAP")

    required = {
        "query_id", "query", "document_id", "expected", "table_case",
        "target", "relation", "attribute", "value_or_action", "section",
        "ownership_relation_expected", "relevant_chunk_ids", "confidence",
        "hard_negative_type", "parser_recoverable",
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
        if required - row.keys():
            errors.append(f"FIELDS:{index}")
        if row.get("document_id") not in document_ids:
            errors.append(f"UNKNOWN_DOCUMENT:{qid}")
        case = str(row.get("table_case", ""))
        if case not in TABLE_CASE_COUNTS:
            errors.append(f"TABLE_CASE:{qid}")
        cases[case] += 1
        expected = row.get("expected")
        if expected == "ANSWER":
            answers[case] += 1
            if row.get("hard_negative_type") is not None:
                errors.append(f"ANSWER_HARD_NEGATIVE:{qid}")
            if not row.get("relevant_chunk_ids"):
                errors.append(f"ANSWER_EVIDENCE:{qid}")
            if row.get("ownership_relation_expected") == "UNSUPPORTED":
                errors.append(f"ANSWER_OWNERSHIP:{qid}")
        elif expected == "ABSTAIN":
            hard = row.get("hard_negative_type")
            if hard not in HARD_NEGATIVE_TYPES:
                errors.append(f"HARD_NEGATIVE:{qid}")
            else:
                negatives[hard] += 1
            if row.get("relevant_chunk_ids"):
                errors.append(f"ABSTAIN_EVIDENCE:{qid}")
            if row.get("ownership_relation_expected") != "UNSUPPORTED":
                errors.append(f"ABSTAIN_OWNERSHIP:{qid}")
        else:
            errors.append(f"EXPECTED:{qid}")
        if not all(str(row.get(key, "")).strip() for key in (
            "target", "relation", "attribute", "value_or_action",
        )):
            errors.append(f"CLAIM_COMPONENTS:{qid}")
        if row.get("confidence") != "HIGH":
            errors.append(f"CONFIDENCE:{qid}")
    if sum(row.get("expected") == "ANSWER" for row in rows) != ANSWER_COUNT:
        errors.append("ANSWER_COUNT")
    if sum(row.get("expected") == "ABSTAIN" for row in rows) != ABSTAIN_COUNT:
        errors.append("ABSTAIN_COUNT")
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_TEXTS")
    for case, (count, answer_count) in TABLE_CASE_COUNTS.items():
        if cases[case] != count:
            errors.append(f"CASE_COUNT:{case}:{cases[case]}")
        if answers[case] != answer_count:
            errors.append(f"CASE_ANSWER_COUNT:{case}:{answers[case]}")
    if set(negatives) != set(HARD_NEGATIVE_TYPES):
        errors.append("HARD_NEGATIVE_COVERAGE")
    return tuple(errors)


def ownership_metrics(records: list[dict]) -> dict[str, Any]:
    bindings = [row for row in records if row.get("ownership_relation") != "UNSUPPORTED"]
    true_bindings = sum(row.get("expected") == "ANSWER" for row in bindings)
    unsafe = sum(row.get("expected") == "ABSTAIN" for row in bindings)
    answerable = sum(row.get("expected") == "ANSWER" for row in records)
    return {
        "ownership_precision": true_bindings / len(bindings) if bindings else 0.0,
        "ownership_recall": true_bindings / answerable if answerable else 0.0,
        "unsafe_ownership_binding": unsafe,
        "wrong_table_attribution": unsafe,
        "bindings": len(bindings),
    }


def acceptance(
    baseline: dict[str, Any], candidate: dict[str, Any], ownership: dict[str, Any], *,
    baseline_replay_mismatches: int, runtime_integrity: bool,
    candidate_frozen: bool, one_shot: bool,
) -> dict[str, Any]:
    baseline_fa = int(baseline["false_answers"])
    fa_reduction = (
        (baseline_fa - int(candidate["false_answers"])) / baseline_fa
        if baseline_fa else 0.0
    )
    fr_increase_pp = float(candidate["false_refusal_rate"]) - float(baseline["false_refusal_rate"])
    checks = {
        "unsafe_ownership_binding_zero": ownership["unsafe_ownership_binding"] == 0,
        "fa_reduction_at_least_30pct": fa_reduction >= 0.30,
        "fr_increase_at_most_5pp": fr_increase_pp <= 0.05,
        "ownership_precision_at_least_90pct": ownership["ownership_precision"] >= 0.90,
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
