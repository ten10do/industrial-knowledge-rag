from backend.evaluation import v343_independent_development_generalization as v


def _row(case: str, index: int) -> dict:
    answer = index < 5
    hard = None if answer else v.HARD_NEGATIVE_TYPES[index % len(v.HARD_NEGATIVE_TYPES)]
    identity = "INCOMPATIBLE" if hard == "SAME_MANUFACTURER_WRONG_MODEL" else "COMPATIBLE"
    return {
        "query_id": f"V343-{case}-{index}",
        "query": f"Unique independent proposition {case} {index}",
        "document_id": "new-doc-a" if index % 2 else "new-doc-b",
        "expected": "ANSWER" if answer else "ABSTAIN",
        "evidence_case": case,
        "evidence_relation_expected": "DIRECT_SUPPORTED" if answer else "UNSAFE",
        "target": "model", "relation": "asserts", "attribute": "attribute",
        "value_or_action": "value", "relevant_chunk_ids": ["chunk"] if answer else [],
        "confidence": "HIGH", "identity_expected": identity,
        "identity_compatible": identity == "COMPATIBLE", "hard_negative_type": hard,
        "parser_recoverable": True,
    }


def _payload() -> dict:
    return {
        "benchmark_version": v.BENCHMARK_VERSION,
        "candidate_version": v.CANDIDATE_VERSION,
        "candidate_sha256_at_freeze": v.CANDIDATE_SHA256,
        "uses_a_to_h_data": False, "uses_j_data": False, "uses_k_data": False,
        "uses_historical_sealed_data": False, "uses_v342_documents": False,
        "v342_query_fingerprints": [],
        "documents": [
            {"document_id": "new-doc-a", "official_url": "https://a.example/manual.pdf",
             "sha256": "a" * 64, "official_english_pdf": True, "prior_document_overlap": False},
            {"document_id": "new-doc-b", "official_url": "https://b.example/manual.pdf",
             "sha256": "b" * 64, "official_english_pdf": True, "prior_document_overlap": False},
        ],
        "queries": [_row(case, index) for case in v.EVIDENCE_CASES for index in range(10)],
    }


def test_contract_accepts_balanced_independent_dataset():
    assert v.validate_dataset(_payload()) == ()


def test_contract_rejects_prior_query_and_candidate_change():
    payload = _payload()
    payload["candidate_sha256_at_freeze"] = "0" * 64
    payload["v342_query_fingerprints"] = [v.query_fingerprint(payload["queries"][0]["query"])]
    errors = v.validate_dataset(payload)
    assert "CANDIDATE_HASH" in errors
    assert any(error.startswith("QUERY_LEAKAGE:") for error in errors)


def test_generalization_acceptance_extends_v342_gate():
    baseline = {"false_refusals": 10, "false_answer_rate": 0.10}
    candidate = {"false_refusals": 5, "false_answer_rate": 0.15}
    result = v.acceptance(
        baseline, candidate, unsafe_relax=0,
        baseline_hard_negative_fa=2, candidate_hard_negative_fa=2,
        v341_regressions=0, runtime_integrity=True, candidate_frozen=True,
        document_disjoint=True, query_disjoint=True, one_shot=True,
    )
    assert result["status"] == v.READY_STATUS
    assert not v.acceptance(
        baseline, candidate, unsafe_relax=0,
        baseline_hard_negative_fa=2, candidate_hard_negative_fa=2,
        v341_regressions=0, runtime_integrity=True, candidate_frozen=False,
        document_disjoint=True, query_disjoint=True, one_shot=True,
    )["status"] == v.READY_STATUS


def test_metrics_are_reused_without_policy_change():
    rows = [
        {"expected": "ANSWER", "decision": "ANSWER"},
        {"expected": "ABSTAIN", "decision": "ABSTAIN"},
    ]
    assert v.metrics(rows)["accuracy"] == 1.0
