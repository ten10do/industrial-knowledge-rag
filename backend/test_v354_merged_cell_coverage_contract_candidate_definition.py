from backend.evaluation import v354_merged_cell_coverage_contract_candidate_definition as v


def test_protocol_is_seventy_two_and_balanced_by_case():
    assert v.FIXTURE_COUNT == 72
    assert v.VALID_COUNT == v.INVALID_COUNT == 36
    assert sum(count for count, _ in v.CASE_COUNTS.values()) == 72
    assert sum(valid for _, valid in v.CASE_COUNTS.values()) == 36


def test_metrics_expose_unsafe_acceptance_and_regressions():
    records = []
    for case, (_, valid_count) in v.CASE_COUNTS.items():
        records.extend([
            {
                "case": case, "expected": "VALID", "observed": "VALID",
                "correct": True, "reason_codes": [], "expected_reason_code": None,
                "coverage_used": case not in {"REGRESSION_V351_RELATIONS", "REGRESSION_V352_NONVERTICAL"},
            }
            for _ in range(valid_count)
        ])
        records.extend([
            {
                "case": case, "expected": "INVALID", "observed": "INVALID",
                "correct": True, "reason_codes": ["X"], "expected_reason_code": "X",
                "coverage_used": False,
            }
            for _ in range(v.CASE_COUNTS[case][0] - valid_count)
        ])
    metrics = v.candidate_metrics(records)
    assert metrics["unsafe_acceptance_count"] == 0
    assert metrics["frozen_vertical_valid_recovered"] == 4
    assert metrics["v351_regression_count"] == 0
    assert metrics["v352_nonvertical_regression_count"] == 0


def test_ready_decision_requires_every_safety_and_regression_gate():
    cases = {case: {"unsafe_acceptance_count": 0} for case in v.CASE_COUNTS}
    metrics = {
        "accuracy": 0.98, "valid_recall": 0.95, "invalid_rejection": 1.0,
        "unsafe_acceptance_count": 0, "reason_target_accuracy": 0.95,
        "frozen_vertical_valid_recovered": 4,
        "frozen_vertical_invalid_preserved": 4,
        "v351_regression_count": 0, "v352_nonvertical_regression_count": 0,
        "case_metrics": cases,
    }
    result = v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        candidate_frozen=True, one_shot=True,
    )
    assert result["status"] == "CONTRACT_CANDIDATE_READY"
    metrics["unsafe_acceptance_count"] = 1
    assert v.decide(
        metrics, dataset_valid=True, source_integrity=True,
        candidate_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
