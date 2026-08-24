from backend.evaluation import v349_table_ownership_boundary_analysis as v


def test_case_distribution_contract_is_exactly_sixty_and_balanced():
    assert sum(count for count, _ in v.TABLE_CASE_COUNTS.values()) == 60
    assert sum(answers for _, answers in v.TABLE_CASE_COUNTS.values()) == 30


def test_ownership_metrics_count_unsafe_bindings_and_recall():
    rows = [
        {"expected": "ANSWER", "ownership_relation": "DIRECT_ROW"},
        {"expected": "ANSWER", "ownership_relation": "UNSUPPORTED"},
        {"expected": "ABSTAIN", "ownership_relation": "COLUMN_BOUND"},
        {"expected": "ABSTAIN", "ownership_relation": "UNSUPPORTED"},
    ]
    result = v.ownership_metrics(rows)
    assert result["ownership_precision"] == 0.5
    assert result["ownership_recall"] == 0.5
    assert result["unsafe_ownership_binding"] == 1
    assert result["wrong_table_attribution"] == 1


def test_acceptance_requires_all_registered_safety_and_utility_gates():
    baseline = {"false_answers": 10, "false_refusal_rate": 0.10}
    candidate = {"false_answers": 7, "false_refusal_rate": 0.15}
    ownership = {"unsafe_ownership_binding": 0, "ownership_precision": 0.90}
    result = v.acceptance(
        baseline, candidate, ownership, baseline_replay_mismatches=0,
        runtime_integrity=True, candidate_frozen=True, one_shot=True,
    )
    assert result["status"] == "DEV_READY"
    ownership["unsafe_ownership_binding"] = 1
    assert v.acceptance(
        baseline, candidate, ownership, baseline_replay_mismatches=0,
        runtime_integrity=True, candidate_frozen=True, one_shot=True,
    )["status"] == "PARTIAL"
