from backend.evaluation.v337_sealed_identity_revalidation import (
    decide,
    score,
    validate_corpus,
    validate_queries,
)


def test_score_balanced_records():
    rows = [
        {"expected": "ANSWER", "predicted": "ANSWER"},
        {"expected": "ANSWER", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ANSWER"},
    ]
    metrics = score(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5


def test_decide_passes_registered_policy():
    baseline = {"false_refusal_rate": 0.4}
    candidate = {"false_refusal_rate": 0.2, "false_answer_rate": 0.1, "accuracy": 0.8}
    hard_negative = {"false_answer_rate": 0.1}
    assert decide(baseline, candidate, hard_negative, runtime_valid=True)["decision"] == "SEALED_IDENTITY_PASS"


def test_decide_rejects_insufficient_relative_fr_reduction():
    baseline = {"false_refusal_rate": 0.4}
    candidate = {"false_refusal_rate": 0.25, "false_answer_rate": 0.0, "accuracy": 0.9}
    hard_negative = {"false_answer_rate": 0.0}
    assert decide(baseline, candidate, hard_negative, runtime_valid=True)["decision"] == "SEALED_IDENTITY_FAIL"


def test_empty_inputs_fail_validation():
    assert not validate_corpus([], [], set()).ok
    assert not validate_queries([], set()).ok
