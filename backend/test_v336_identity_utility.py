"""Tests for the V3.36 Identity Utility DEV contract."""

from backend.evaluation.v336_identity_utility import (
    BENCHMARK_VERSION,
    IDENTITY_SLICES,
    TAXONOMY,
    decide,
    score,
    validate_audit,
    validate_benchmark,
)


def _benchmark():
    queries = []
    for index in range(60):
        queries.append({
            "query_id": f"U-{index:03d}",
            "query": f"Unique utility query {index}",
            "document_id": f"utility-doc-{index:03d}",
            "expected": "ANSWER" if index < 30 else "ABSTAIN",
            "taxonomy": TAXONOMY[index % len(TAXONOMY)],
            "identity_slice": IDENTITY_SLICES[index % len(IDENTITY_SLICES)],
            "hard_near_miss": index >= 30,
        })
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "uses_v335_sealed_data": False,
        "forbidden_document_ids": [],
        "queries": queries,
    }


def test_valid_balanced_benchmark_passes():
    assert validate_benchmark(_benchmark()) == []


def test_sealed_data_and_document_reuse_are_rejected():
    payload = _benchmark()
    payload["uses_v335_sealed_data"] = True
    payload["queries"][1]["document_id"] = payload["queries"][0]["document_id"]
    errors = validate_benchmark(payload)
    assert "V335_SEALED_EXCLUSION_NOT_DECLARED" in errors
    assert "DEV_DOCUMENTS_NOT_QUERY_DISJOINT" in errors


def test_score_uses_class_denominators():
    metrics = score([
        {"expected": "ANSWER", "predicted": "ANSWER"},
        {"expected": "ANSWER", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ANSWER"},
    ])
    assert metrics["accuracy"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5


def test_ready_requires_utility_and_safety_conditions():
    before = {"false_answer_rate": 0.0, "false_refusal_rate": 0.4}
    after = {"false_answer_rate": 0.05, "false_refusal_rate": 0.2}
    hard = {"false_answer_rate": 0.1}
    assert decide(before, after, hard, runtime_valid=True)["decision"] == "DEV_READY"
    after["false_refusal_rate"] = 0.21
    assert decide(before, after, hard, runtime_valid=True)["decision"] == "PARTIAL"
    assert decide(before, after, hard, runtime_valid=False)["decision"] == "PARTIAL"


def test_per_query_audit_fields_are_mandatory():
    record = {
        "query_id": "U-001", "identity_before": "INCOMPATIBLE",
        "identity_after": "COMPATIBLE", "decision": "ANSWER",
        "reason_code": "MODEL_VARIANT_DESCENDANT", "expected": "ANSWER",
        "taxonomy": TAXONOMY[0], "identity_slice": "model",
    }
    assert validate_audit([record], 1) == []
    del record["reason_code"]
    assert validate_audit([record], 1) == ["AUDIT_FIELDS:0:reason_code"]
