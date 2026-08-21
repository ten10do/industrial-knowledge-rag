from backend.evaluation.v338_identity_evidence_boundary import (
    BENCHMARK_VERSION,
    COVERAGE,
    classify_failure,
    recommend,
    summarize,
    validate_dataset,
)


def _dataset():
    rows = []
    for index in range(60):
        answer = index < 40
        rows.append({
            "query_id": f"FR-{index:03d}",
            "query": f"Unique diagnosis query {index}",
            "document_id": f"new-doc-{index % 5}",
            "expected": "ANSWER" if answer else "ABSTAIN",
            "coverage": COVERAGE[index % len(COVERAGE)],
            "relevant_chunk_ids": [f"chunk-{index}"] if answer else [],
            "confidence": "HIGH",
            "new_document": True,
        })
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "uses_v335_data": False,
        "uses_v337_data": False,
        "queries": rows,
    }


def test_valid_dataset_passes():
    assert validate_dataset(_dataset()).ok


def test_sealed_data_and_balance_are_enforced():
    payload = _dataset()
    payload["uses_v337_data"] = True
    payload["queries"][0]["expected"] = "ABSTAIN"
    payload["queries"][0]["relevant_chunk_ids"] = []
    errors = validate_dataset(payload).errors
    assert "V337_EXCLUSION" in errors
    assert "ANSWER_COUNT:39" in errors


def test_failure_precedence_is_parser_then_retrieval_then_identity_then_evidence():
    base = {"expected": "ANSWER", "final_decision": "ABSTAIN"}
    assert classify_failure(base)[0] == "PARSER_LIMIT"
    assert classify_failure({**base, "relevant_evidence_parsed": True})[0] == "RETRIEVAL_MISSING"
    identity = {**base, "relevant_evidence_parsed": True, "relevant_evidence_retrieved": True, "identity_result": "INCOMPATIBLE"}
    assert classify_failure(identity)[0] == "IDENTITY_ERROR"
    assert classify_failure({**identity, "identity_result": "COMPATIBLE"})[0] == "EVIDENCE_TOO_STRICT"


def test_recommendation_uses_strict_thirty_percent_threshold():
    records = []
    for index in range(10):
        failure = "EVIDENCE_TOO_STRICT" if index < 4 else "RETRIEVAL_MISSING"
        records.append({"expected": "ANSWER", "final_decision": "ABSTAIN", "failure_class": failure})
    summary = summarize(records)
    decision = recommend(summary)
    assert set(decision["triggered"]) == {"EVIDENCE_TOO_STRICT", "RETRIEVAL_MISSING"}
    assert decision["primary_failure_class"] == "RETRIEVAL_MISSING"
    assert decision["recommendation"] == "RETRIEVAL_IMPROVEMENT"
