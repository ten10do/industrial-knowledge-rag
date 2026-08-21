from backend.evaluation.v339_evidence_sufficiency import (
    BENCHMARK_VERSION,
    COVERAGE,
    acceptance,
    classify_baseline,
    metrics,
    validate_dataset,
)


def _dataset():
    rows = []
    for index in range(80):
        expected = "ANSWER" if index % 10 < 5 else "ABSTAIN"
        rows.append({
            "query_id": f"V339-{index:03d}",
            "query": f"Unique evidence sufficiency query {index}",
            "document_id": f"new-doc-{index % 4}",
            "expected": expected,
            "coverage": COVERAGE[index // 10],
            "relevant_chunk_ids": [f"chunk-{index}"] if expected == "ANSWER" else [],
            "confidence": "HIGH",
            "new_document": True,
            "identity_expected": "COMPATIBLE",
            "scope_ambiguous": False,
            "parser_recoverable": True,
        })
    return {"benchmark_version": BENCHMARK_VERSION, "uses_v335_data": False, "uses_v337_data": False, "queries": rows}


def test_dataset_contract():
    assert validate_dataset(_dataset()).ok


def test_dataset_rejects_sealed_data_and_slice_imbalance():
    payload = _dataset()
    payload["uses_v335_data"] = True
    payload["queries"][0]["coverage"] = COVERAGE[1]
    errors = validate_dataset(payload).errors
    assert "V335_EXCLUSION" in errors
    assert any(error.startswith("COVERAGE_COUNT:") for error in errors)


def test_baseline_taxonomy_precedence():
    assert classify_baseline({"expected": "ABSTAIN", "decision": "ABSTAIN"})[0] == "UNSAFE_RELAX"
    assert classify_baseline({"expected": "ANSWER", "decision": "ANSWER"})[0] == "UNSAFE_RELAX"
    base = {"expected": "ANSWER", "decision": "ABSTAIN"}
    assert classify_baseline({**base, "parser_recoverable": False})[0] == "PARSER_LIMIT"
    assert classify_baseline({**base, "parser_recoverable": True})[0] == "MISSING_EVIDENCE"
    assert classify_baseline({**base, "parser_recoverable": True, "relevant_evidence_retrieved": True, "scope_ambiguous": True})[0] == "SCOPE_AMBIGUITY"
    assert classify_baseline({**base, "parser_recoverable": True, "relevant_evidence_retrieved": True})[0] == "SAFE_RELAX_CANDIDATE"


def test_identity_incompatible_refusal_is_not_a_safe_relax_candidate():
    record = {
        "expected": "ANSWER",
        "decision": "ABSTAIN",
        "identity_result": "INCOMPATIBLE",
        "parser_recoverable": True,
        "relevant_evidence_retrieved": True,
        "scope_ambiguous": False,
    }
    assert classify_baseline(record) == ("UNSAFE_RELAX", "IDENTITY_BOUNDARY_NOT_RELAXABLE")


def test_metrics_and_acceptance_policy():
    rows = [
        *({"expected": "ANSWER", "baseline": "ABSTAIN", "candidate": "ANSWER"} for _ in range(10)),
        *({"expected": "ABSTAIN", "baseline": "ABSTAIN", "candidate": "ABSTAIN"} for _ in range(10)),
    ]
    baseline = metrics(rows, "baseline")
    candidate = metrics(rows, "candidate")
    result = acceptance(
        baseline, candidate, unsafe_relax=0,
        baseline_hard_negative_fa=0, candidate_hard_negative_fa=0,
        runtime_integrity=True,
    )
    assert result["status"] == "DEV_READY"
