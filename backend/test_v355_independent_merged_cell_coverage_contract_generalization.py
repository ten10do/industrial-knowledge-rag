from backend.evaluation import v355_independent_merged_cell_coverage_contract_generalization as v


def test_protocol_is_sixty_four_and_balanced():
    assert v.FIXTURE_COUNT == 64
    assert v.VALID_COUNT == v.INVALID_COUNT == 32
    assert sum(count for count, _ in v.CASE_COUNTS.values()) == 64
    assert sum(valid for _, valid in v.CASE_COUNTS.values()) == 32


def test_metrics_expose_case_recall_safety_and_base_regression():
    records = []
    for case, (count, valid_count) in v.CASE_COUNTS.items():
        records.extend([
            {"case": case, "expected": "VALID", "observed": "VALID", "correct": True,
             "reason_codes": [], "expected_reason_code": None, "coverage_used": case != "BASE_RELATION_REGRESSION"}
            for _ in range(valid_count)
        ])
        records.extend([
            {"case": case, "expected": "INVALID", "observed": "INVALID", "correct": True,
             "reason_codes": ["X"], "expected_reason_code": "X", "coverage_used": False}
            for _ in range(count - valid_count)
        ])
    metrics = v.generalization_metrics(records)
    assert metrics["accuracy"] == 1.0
    assert metrics["unsafe_acceptance_count"] == 0
    assert metrics["base_relation_regression_count"] == 0
    assert all(row["valid_recall"] == 1.0 for row in metrics["case_metrics"].values())


def test_ready_requires_global_per_case_and_regression_gates():
    cases = {
        case: {"valid_recall": 1.0, "unsafe_acceptance_count": 0}
        for case in v.CASE_COUNTS
    }
    metrics = {
        "accuracy": 0.95, "valid_recall": 0.90, "invalid_rejection": 1.0,
        "unsafe_acceptance_count": 0, "reason_target_accuracy": 0.95,
        "base_relation_regression_count": 0, "case_metrics": cases,
    }
    result = v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        candidate_frozen=True, one_shot=True,
    )
    assert result["status"] == "GENERALIZATION_READY"
    cases["NESTED_SAME_OWNER"]["unsafe_acceptance_count"] = 1
    assert v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        candidate_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
