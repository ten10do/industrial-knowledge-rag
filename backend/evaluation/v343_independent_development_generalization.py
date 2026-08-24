"""Public contract for V3.43 independent development generalization.

Private manuals, queries, evidence, retrieval candidates, and predictions stay
under ``benchmark_private``.  This module only validates the frozen protocol
and scores the unchanged V3.42 candidate.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from backend.evaluation.v342_evidence_sufficiency_identity_aware import (
    EVIDENCE_CASES,
    HARD_NEGATIVE_TYPES,
    acceptance as v342_acceptance,
    metrics,
)


BENCHMARK_VERSION = "v343-evidence-sufficiency-independent-dev-v1"
CANDIDATE_VERSION = "evidence-v342-sufficiency-candidate"
CANDIDATE_SHA256 = "f02f39035ae1c88e7b2d65a5939bc4321739e6b561db18d5b89d63d13f18dcfc"
QUERY_COUNT = 60
ANSWER_COUNT = 30
ABSTAIN_COUNT = 30
DOCUMENT_COUNT = 2
READY_STATUS = "EVIDENCE_GENERALIZATION_READY"
PARTIAL_STATUS = "PARTIAL"


def normalize_query(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def query_fingerprint(value: str) -> str:
    return hashlib.sha256(normalize_query(value).encode("utf-8")).hexdigest()


def validate_dataset(payload: dict) -> tuple[str, ...]:
    errors: list[str] = []
    rows = payload.get("queries", [])
    documents = payload.get("documents", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION")
    for key in (
        "uses_a_to_h_data", "uses_j_data", "uses_k_data",
        "uses_historical_sealed_data", "uses_v342_documents",
    ):
        if payload.get(key) is not False:
            errors.append(f"FORBIDDEN_SOURCE:{key}")
    if payload.get("candidate_version") != CANDIDATE_VERSION:
        errors.append("CANDIDATE_VERSION")
    if payload.get("candidate_sha256_at_freeze") != CANDIDATE_SHA256:
        errors.append("CANDIDATE_HASH")
    if len(documents) != DOCUMENT_COUNT:
        errors.append(f"DOCUMENT_COUNT:{len(documents)}")
    if len(rows) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(rows)}")

    document_ids = {str(item.get("document_id", "")) for item in documents}
    urls = [str(item.get("official_url", "")).casefold() for item in documents]
    hashes = [str(item.get("sha256", "")).casefold() for item in documents]
    if "" in document_ids or len(document_ids) != len(documents):
        errors.append("DOCUMENT_IDS")
    if not all(urls) or len(urls) != len(set(urls)):
        errors.append("DOCUMENT_URLS")
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
    normalized: list[str] = []
    prior_queries = set(payload.get("v342_query_fingerprints", []))
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            errors.append(f"FIELDS:{index}:{','.join(sorted(missing))}")
        qid = str(row.get("query_id", ""))
        text = normalize_query(str(row.get("query", "")))
        ids.append(qid)
        normalized.append(text)
        if query_fingerprint(str(row.get("query", ""))) in prior_queries:
            errors.append(f"QUERY_LEAKAGE:{qid}")
        if row.get("document_id") not in document_ids:
            errors.append(f"UNKNOWN_DOCUMENT:{qid}")
        case = str(row.get("evidence_case", ""))
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

    if len([row for row in rows if row.get("expected") == "ANSWER"]) != ANSWER_COUNT:
        errors.append("ANSWER_COUNT")
    if len([row for row in rows if row.get("expected") == "ABSTAIN"]) != ABSTAIN_COUNT:
        errors.append("ABSTAIN_COUNT")
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS")
    if not all(normalized) or len(normalized) != len(set(normalized)):
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
    unsafe_relax: int,
    baseline_hard_negative_fa: int,
    candidate_hard_negative_fa: int,
    v341_regressions: int,
    runtime_integrity: bool,
    candidate_frozen: bool,
    document_disjoint: bool,
    query_disjoint: bool,
    one_shot: bool,
) -> dict[str, Any]:
    result = v342_acceptance(
        baseline, candidate,
        unsafe_relax=unsafe_relax,
        baseline_hard_negative_fa=baseline_hard_negative_fa,
        candidate_hard_negative_fa=candidate_hard_negative_fa,
        v341_regressions=v341_regressions,
        runtime_integrity=runtime_integrity,
    )
    checks = dict(result["checks"])
    checks.update({
        "candidate_frozen": candidate_frozen,
        "document_disjoint": document_disjoint,
        "query_disjoint": query_disjoint,
        "one_shot": one_shot,
    })
    return {
        **result,
        "status": READY_STATUS if all(checks.values()) else PARTIAL_STATUS,
        "checks": checks,
    }
