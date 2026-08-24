from backend.evaluation import v347_bounded_evidence_claim_binding as v


def _row(case: str, index: int) -> dict:
    answer = index < 5
    hard = None if answer else v.HARD_NEGATIVE_TYPES[index % len(v.HARD_NEGATIVE_TYPES)]
    identity = "INCOMPATIBLE" if hard == "SAME_MANUFACTURER_WRONG_MODEL" else "COMPATIBLE"
    return {
        "query_id": f"V347-{case}-{index}",
        "query": f"Unique V347 claim {case} {index}",
        "document_id": "doc-a" if index % 2 else "doc-b",
        "expected": "ANSWER" if answer else "ABSTAIN", "claim_case": case,
        "target": "model", "relation": "asserts", "attribute": "attribute",
        "value_or_action": "value", "relevant_chunk_ids": ["chunk"] if answer else [],
        "confidence": "HIGH", "identity_expected": identity,
        "identity_compatible": identity == "COMPATIBLE", "hard_negative_type": hard,
        "parser_recoverable": True,
    }


def _payload() -> dict:
    return {
        "benchmark_version": v.BENCHMARK_VERSION,
        "baseline_version": v.BASELINE_VERSION,
        "candidate_version": v.CANDIDATE_VERSION,
        "candidate_sha256_at_freeze": v.CANDIDATE_SHA256,
        "uses_a_to_h_data": False, "uses_j_data": False, "uses_k_data": False,
        "uses_historical_sealed_data": False, "uses_v342_documents": False,
        "uses_v343_documents": False, "uses_v345_documents": False,
        "documents": [
            {"document_id": "doc-a", "sha256": "a" * 64,
             "official_english_pdf": True, "prior_document_overlap": False},
            {"document_id": "doc-b", "sha256": "b" * 64,
             "official_english_pdf": True, "prior_document_overlap": False},
        ],
        "queries": [_row(case, index) for case in v.CLAIM_CASES for index in range(10)],
    }


def test_contract_accepts_balanced_new_dev_dataset():
    assert v.validate_dataset(_payload()) == ()


def test_contract_rejects_candidate_change_and_prior_sources():
    payload = _payload()
    payload["candidate_sha256_at_freeze"] = "0" * 64
    payload["uses_v345_documents"] = True
    errors = v.validate_dataset(payload)
    assert "CANDIDATE_SHA256" in errors
    assert "FORBIDDEN_SOURCE:uses_v345_documents" in errors


def test_acceptance_requires_material_safe_false_answer_reduction():
    baseline = {"false_answers": 8, "false_refusal_rate": 0.10}
    candidate = {"false_answers": 4, "false_refusal_rate": 0.15}
    result = v.acceptance(
        baseline, candidate, unsafe_veto=0, unsafe_relax=0,
        baseline_hard_negative_fa=8, candidate_hard_negative_fa=4,
        baseline_replay_mismatches=0, runtime_integrity=True,
        candidate_frozen=True, one_shot=True,
    )
    assert result["status"] == "DEV_READY"
    assert v.acceptance(
        baseline, candidate, unsafe_veto=1, unsafe_relax=0,
        baseline_hard_negative_fa=8, candidate_hard_negative_fa=4,
        baseline_replay_mismatches=0, runtime_integrity=True,
        candidate_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
