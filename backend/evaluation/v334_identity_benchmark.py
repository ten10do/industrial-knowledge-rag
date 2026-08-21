"""Public validation and scoring for the private V3.34 identity DEV benchmark."""

from __future__ import annotations

from collections import Counter


BENCHMARK_VERSION = "v334-identity-dev-v1"
CANDIDATE_VERSION = "identity-aware-evidence-v334-candidate"
MIN_QUERY_COUNT = 50
MAX_EXPERIMENTS = 3
MAX_FALSE_REFUSAL_INCREASE = 0.05
MIN_MEANINGFUL_FA_REDUCTION = 0.10

CATEGORY_MINIMUMS = {
    "FAMILY_MODEL": 20,
    "MODULE_CONTROLLER": 10,
    "FIRMWARE_VERSION": 5,
    "OPTION_ACCESSORY": 5,
    "PARAMETER_SCOPE": 10,
}
EXPECTED_LABELS = {"ANSWER", "ABSTAIN"}
FINAL_STATUSES = {"IDENTITY_GENERALIZATION_READY", "PARTIAL", "FAILED"}


def validate_benchmark(payload: dict) -> list[str]:
    errors: list[str] = []
    queries = payload.get("queries", [])
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("BENCHMARK_VERSION_MISMATCH")
    if len(queries) < MIN_QUERY_COUNT:
        errors.append("QUERY_COUNT_BELOW_50")

    counts = Counter(str(item.get("category", "")) for item in queries)
    for category, minimum in CATEGORY_MINIMUMS.items():
        if counts[category] < minimum:
            errors.append(f"CATEGORY_BELOW_MINIMUM:{category}")

    ids = [str(item.get("query_id", "")) for item in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_IDS_MISSING_OR_DUPLICATED")
    texts = [" ".join(str(item.get("query", "")).casefold().split()) for item in queries]
    if not all(texts) or len(texts) != len(set(texts)):
        errors.append("QUERIES_MISSING_OR_DUPLICATED")
    if any(item.get("expected") not in EXPECTED_LABELS for item in queries):
        errors.append("INVALID_EXPECTED_LABEL")
    if {item.get("expected") for item in queries} != EXPECTED_LABELS:
        errors.append("POSITIVE_NEGATIVE_COVERAGE_MISSING")
    if any(not item.get("hard_near_miss") for item in queries if item.get("expected") == "ABSTAIN"):
        errors.append("NEGATIVE_NOT_HARD_NEAR_MISS")

    benchmark_documents = {str(item.get("document_id", "")) for item in queries}
    prior_documents = {str(item) for item in payload.get("forbidden_document_ids", [])}
    if "" in benchmark_documents:
        errors.append("DOCUMENT_ID_MISSING")
    if benchmark_documents & prior_documents:
        errors.append("PRIOR_DOCUMENT_OVERLAP")
    if len(benchmark_documents) != len(queries):
        errors.append("DEV_DOCUMENTS_NOT_QUERY_DISJOINT")
    if payload.get("uses_v333_k_check") is not False:
        errors.append("V333_K_CHECK_EXCLUSION_NOT_DECLARED")
    return errors


def score_predictions(records: list[dict]) -> dict:
    total = len(records)
    answerable = [item for item in records if item["expected"] == "ANSWER"]
    abstainable = [item for item in records if item["expected"] == "ABSTAIN"]
    false_answers = sum(item["predicted"] == "ANSWER" for item in abstainable)
    false_refusals = sum(item["predicted"] == "ABSTAIN" for item in answerable)
    correct = sum(item["predicted"] == item["expected"] for item in records)
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "answerable_recall": 1.0 - false_refusals / len(answerable) if answerable else 0.0,
        "abstention_recall": 1.0 - false_answers / len(abstainable) if abstainable else 0.0,
        "false_answer_rate": false_answers / len(abstainable) if abstainable else 0.0,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else 0.0,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
    }


def compare_metrics(baseline: dict, candidate: dict) -> dict:
    fa_reduction = baseline["false_answer_rate"] - candidate["false_answer_rate"]
    fr_increase = candidate["false_refusal_rate"] - baseline["false_refusal_rate"]
    if fa_reduction >= MIN_MEANINGFUL_FA_REDUCTION and fr_increase <= MAX_FALSE_REFUSAL_INCREASE:
        status = "IDENTITY_GENERALIZATION_READY"
    elif fa_reduction > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"
    return {
        "status": status,
        "false_answer_reduction": fa_reduction,
        "false_refusal_increase": fr_increase,
        "fa_reduced": fa_reduction > 0,
        "fr_tradeoff_within_limit": fr_increase <= MAX_FALSE_REFUSAL_INCREASE,
    }


def validate_experiment_log(experiments: list[dict]) -> list[str]:
    errors: list[str] = []
    if not experiments:
        errors.append("NO_EXPERIMENT_RECORDED")
    if len(experiments) > MAX_EXPERIMENTS:
        errors.append("EXPERIMENT_LIMIT_EXCEEDED")
    ids = [str(item.get("experiment_id", "")) for item in experiments]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("EXPERIMENT_IDS_MISSING_OR_DUPLICATED")
    required = {"hypothesis", "change", "metrics", "decision"}
    for index, experiment in enumerate(experiments):
        missing = sorted(required - set(experiment))
        if missing:
            errors.append(f"EXPERIMENT_FIELDS_MISSING:{index}:{','.join(missing)}")
    return errors
