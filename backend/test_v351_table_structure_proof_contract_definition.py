from backend.evaluation import v351_table_structure_proof_contract_definition as v


def test_registered_fixture_distribution_is_balanced_and_complete():
    assert v.FIXTURE_COUNT == 36
    assert v.VALID_COUNT == v.INVALID_COUNT == 18
    assert v.SYNTHETIC_COUNT + v.FROZEN_V349_COUNT == v.FIXTURE_COUNT
    assert len(v.FROZEN_V349_IDS) == v.FROZEN_V349_COUNT
    assert "UNSUPPORTED" in v.ALL_RELATIONS


def test_metrics_count_unsafe_acceptance_and_frozen_correctness():
    rows = [
        {"expected": "VALID", "observed": "VALID", "origin": "V349_FROZEN", "reason_codes": [], "expected_reason_code": None},
        {"expected": "INVALID", "observed": "VALID", "origin": "SYNTHETIC", "reason_codes": [], "expected_reason_code": "MODEL_SCOPE_MISMATCH"},
        {"expected": "INVALID", "observed": "INVALID", "origin": "SYNTHETIC", "reason_codes": ["CONFLICTING_OWNER"], "expected_reason_code": "CONFLICTING_OWNER"},
    ]
    result = v.contract_metrics(rows)
    assert result["accuracy"] == 2 / 3
    assert result["unsafe_acceptance_count"] == 1
    assert result["reason_target_miss_count"] == 1
    assert result["frozen_v349_correct"] == 1


def test_contract_ready_requires_every_safety_and_integrity_check():
    metrics = {
        "accuracy": 1.0, "valid_recall": 1.0, "invalid_rejection": 1.0,
        "unsafe_acceptance_count": 0, "false_rejection_count": 0,
        "reason_target_miss_count": 0,
        "frozen_v349_correct": 6, "frozen_v349_total": 6,
    }
    result = v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        contract_frozen=True, one_shot=True,
    )
    assert result["status"] == "CONTRACT_READY"
    metrics["unsafe_acceptance_count"] = 1
    assert v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        contract_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
