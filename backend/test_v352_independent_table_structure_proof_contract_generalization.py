from backend.evaluation import v352_independent_table_structure_proof_contract_generalization as v


def test_independent_fixture_contract_is_sixty_four_and_balanced_by_case():
    assert v.FIXTURE_COUNT == 64
    assert v.VALID_COUNT == v.INVALID_COUNT == 32
    assert sum(count for count, _ in v.CASE_COUNTS.values()) == 64
    assert all(pair == (8, 4) for pair in v.CASE_COUNTS.values())


def test_metrics_expose_case_level_false_rejection():
    records = []
    for case in v.CASE_COUNTS:
        records.extend([
            {"case": case, "expected": "VALID", "observed": "INVALID", "reason_codes": [], "expected_reason_code": None},
            {"case": case, "expected": "INVALID", "observed": "INVALID", "reason_codes": ["X"], "expected_reason_code": "X"},
        ])
    metrics = v.generalization_metrics(records)
    assert metrics["unsafe_acceptance_count"] == 0
    assert metrics["false_rejection_count"] == len(v.CASE_COUNTS)
    assert all(row["valid_recall"] == 0.0 for row in metrics["case_metrics"].values())


def test_generalization_ready_requires_global_and_per_case_gates():
    cases = {
        case: {"valid_recall": 1.0, "unsafe_acceptance_count": 0}
        for case in v.CASE_COUNTS
    }
    metrics = {
        "accuracy": 0.95, "valid_recall": 0.90, "invalid_rejection": 1.0,
        "unsafe_acceptance_count": 0, "reason_target_accuracy": 0.95,
        "case_metrics": cases,
    }
    result = v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        contract_frozen=True, one_shot=True,
    )
    assert result["status"] == "GENERALIZATION_READY"
    cases["VERTICAL_MERGED_MODEL_CELL"]["valid_recall"] = 0.5
    assert v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        contract_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
