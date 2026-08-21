"""Tests for the public V3.34 benchmark contract and acceptance policy."""

from __future__ import annotations

from backend.evaluation.v334_identity_benchmark import (
    BENCHMARK_VERSION,
    CANDIDATE_VERSION,
    CATEGORY_MINIMUMS,
    MAX_EXPERIMENTS,
    compare_metrics,
    score_predictions,
    validate_benchmark,
    validate_experiment_log,
)


def _benchmark():
    queries = []
    index = 0
    for category, count in CATEGORY_MINIMUMS.items():
        for offset in range(count):
            index += 1
            expected = "ANSWER" if offset % 2 == 0 else "ABSTAIN"
            queries.append({
                "query_id": f"V334-{index:03d}",
                "query": f"Unique identity query {category} {offset}",
                "category": category,
                "expected": expected,
                "hard_near_miss": expected == "ABSTAIN",
                "document_id": f"v334-dev-doc-{index:03d}",
            })
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "uses_v333_k_check": False,
        "forbidden_document_ids": ["prior-doc"],
        "queries": queries,
    }


def test_versions_and_experiment_cap_are_frozen():
    assert CANDIDATE_VERSION == "identity-aware-evidence-v334-candidate"
    assert MAX_EXPERIMENTS == 3


def test_valid_benchmark_passes():
    assert validate_benchmark(_benchmark()) == []


def test_prior_document_overlap_is_rejected():
    payload = _benchmark()
    payload["forbidden_document_ids"] = [payload["queries"][0]["document_id"]]
    assert "PRIOR_DOCUMENT_OVERLAP" in validate_benchmark(payload)


def test_duplicate_dev_documents_are_rejected():
    payload = _benchmark()
    payload["queries"][1]["document_id"] = payload["queries"][0]["document_id"]
    assert "DEV_DOCUMENTS_NOT_QUERY_DISJOINT" in validate_benchmark(payload)


def test_scoring_uses_answerable_and_abstainable_denominators():
    metrics = score_predictions([
        {"expected": "ANSWER", "predicted": "ANSWER"},
        {"expected": "ANSWER", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ANSWER"},
    ])
    assert metrics["accuracy"] == 0.5
    assert metrics["answerable_recall"] == 0.5
    assert metrics["abstention_recall"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5


def test_ready_requires_meaningful_fa_drop_and_bounded_fr_tradeoff():
    baseline = {"false_answer_rate": 0.6, "false_refusal_rate": 0.1}
    candidate = {"false_answer_rate": 0.3, "false_refusal_rate": 0.14}
    comparison = compare_metrics(baseline, candidate)
    assert comparison["status"] == "IDENTITY_GENERALIZATION_READY"
    assert comparison["false_answer_reduction"] == 0.3


def test_large_false_refusal_tradeoff_is_partial():
    baseline = {"false_answer_rate": 0.6, "false_refusal_rate": 0.1}
    candidate = {"false_answer_rate": 0.3, "false_refusal_rate": 0.2}
    assert compare_metrics(baseline, candidate)["status"] == "PARTIAL"


def test_no_false_answer_improvement_fails():
    baseline = {"false_answer_rate": 0.4, "false_refusal_rate": 0.1}
    candidate = {"false_answer_rate": 0.4, "false_refusal_rate": 0.1}
    assert compare_metrics(baseline, candidate)["status"] == "FAILED"


def test_experiment_log_enforces_cap_and_fields():
    valid = [{
        "experiment_id": "V334-E1", "hypothesis": "h", "change": "c",
        "metrics": {}, "decision": "accept",
    }]
    assert validate_experiment_log(valid) == []
    assert "EXPERIMENT_LIMIT_EXCEEDED" in validate_experiment_log(valid * 4)
