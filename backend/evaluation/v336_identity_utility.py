"""Public contract for the V3.36 Identity Utility DEV reassessment."""

from __future__ import annotations

from collections import Counter


BENCHMARK_VERSION = "v336-identity-utility-dev-v1"
BASELINE_VERSION = "identity-aware-evidence-v334-candidate"
CANDIDATE_VERSION = "identity-v336-candidate"
MIN_QUERIES = 60
MAX_FA_INCREASE = 0.05
MIN_FR_REDUCTION = 0.50
MAX_HARD_NEGATIVE_FA = 0.10

TAXONOMY = (
    "TRUE_IDENTITY_CONFLICT",
    "COMPATIBLE_IDENTITY_VARIATION",
    "SUBMODULE_IDENTITY",
    "DOCUMENT_SCOPE_IDENTITY",
    "PRODUCT_FAMILY_RELATION",
)
IDENTITY_SLICES = ("family", "model", "module", "firmware", "option")


def validate_benchmark(payload: dict) -> list[str]:
    errors: list[str] = []
    queries = payload.get("queries", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION_MISMATCH")
    if len(queries) < MIN_QUERIES:
        errors.append(f"QUERY_COUNT:{len(queries)}")

    labels = Counter(item.get("expected") for item in queries)
    if labels["ANSWER"] != labels["ABSTAIN"] or labels["ANSWER"] * 2 != len(queries):
        errors.append(f"LABEL_BALANCE:{labels['ANSWER']}/{labels['ABSTAIN']}")
    if any(item.get("expected") not in {"ANSWER", "ABSTAIN"} for item in queries):
        errors.append("INVALID_EXPECTED_LABEL")

    taxonomy = Counter(item.get("taxonomy") for item in queries)
    for category in TAXONOMY:
        if not taxonomy[category]:
            errors.append(f"MISSING_TAXONOMY:{category}")
    slices = Counter(item.get("identity_slice") for item in queries)
    for identity_slice in IDENTITY_SLICES:
        if not slices[identity_slice]:
            errors.append(f"MISSING_IDENTITY_SLICE:{identity_slice}")

    ids = [str(item.get("query_id", "")) for item in queries]
    texts = [" ".join(str(item.get("query", "")).casefold().split()) for item in queries]
    documents = [str(item.get("document_id", "")) for item in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_ID_DUPLICATE_OR_MISSING")
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERY_DUPLICATE_OR_MISSING")
    if not all(documents) or len(documents) != len(set(documents)):
        errors.append("DEV_DOCUMENTS_NOT_QUERY_DISJOINT")
    if set(documents) & set(payload.get("forbidden_document_ids", [])):
        errors.append("FORBIDDEN_DOCUMENT_OVERLAP")
    if payload.get("uses_v335_sealed_data") is not False:
        errors.append("V335_SEALED_EXCLUSION_NOT_DECLARED")
    if any(not item.get("hard_near_miss") for item in queries if item.get("expected") == "ABSTAIN"):
        errors.append("NEGATIVE_NOT_HARD_NEAR_MISS")
    return errors


def score(records: list[dict]) -> dict[str, float | int]:
    answerable = [item for item in records if item["expected"] == "ANSWER"]
    abstainable = [item for item in records if item["expected"] == "ABSTAIN"]
    false_answers = sum(item["predicted"] == "ANSWER" for item in abstainable)
    false_refusals = sum(item["predicted"] == "ABSTAIN" for item in answerable)
    correct = sum(item["predicted"] == item["expected"] for item in records)
    return {
        "n": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "answerable_recall": 1 - false_refusals / len(answerable) if answerable else 0.0,
        "abstain_recall": 1 - false_answers / len(abstainable) if abstainable else 0.0,
        "false_answer_rate": false_answers / len(abstainable) if abstainable else 0.0,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else 0.0,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
    }


def decide(before: dict, after: dict, hard_negative: dict, *, runtime_valid: bool) -> dict:
    fa_increase = after["false_answer_rate"] - before["false_answer_rate"]
    before_fr = before["false_refusal_rate"]
    fr_reduction = (
        (before_fr - after["false_refusal_rate"]) / before_fr
        if before_fr else 0.0
    )
    ready = (
        fa_increase <= MAX_FA_INCREASE
        and fr_reduction >= MIN_FR_REDUCTION
        and hard_negative["false_answer_rate"] <= MAX_HARD_NEGATIVE_FA
        and runtime_valid
    )
    return {
        "decision": "DEV_READY" if ready else "PARTIAL",
        "false_answer_increase": fa_increase,
        "false_refusal_reduction": fr_reduction,
        "hard_negative_false_answer_rate": hard_negative["false_answer_rate"],
        "runtime_valid": runtime_valid,
    }


def validate_audit(records: list[dict], query_count: int) -> list[str]:
    errors: list[str] = []
    if len(records) != query_count:
        errors.append(f"AUDIT_COUNT:{len(records)}/{query_count}")
    required = {
        "query_id", "identity_before", "identity_after", "decision",
        "reason_code", "expected", "taxonomy", "identity_slice",
    }
    for index, record in enumerate(records):
        missing = sorted(required - {key for key, value in record.items() if value not in (None, "")})
        if missing:
            errors.append(f"AUDIT_FIELDS:{index}:{','.join(missing)}")
    return errors
