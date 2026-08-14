"""Tests for V3.15 calibration and frozen-replay reporting."""

from copy import deepcopy

from backend.evaluation.v315_support_precision import (
    REQUIRED_MANUFACTURERS,
    V315_FAILURE_TAXONOMY,
    abc_report,
    calibration_distribution,
    failure_delta,
)


def _result(rows, metrics, elapsed=0.1):
    return {
        "rows": rows,
        "metrics": {"support": metrics},
        "replay_elapsed_seconds": elapsed,
    }


def test_calibration_distribution_covers_required_balance():
    queries = []
    manufacturers = sorted(REQUIRED_MANUFACTURERS)
    for index in range(24):
        queries.append({
            "expected_support": "SUPPORTED" if index < 12 else "INSUFFICIENT",
            "manufacturer": manufacturers[index % 4],
            "category": "positive" if index < 12 else "hard_negative",
        })
    distribution = calibration_distribution({"queries": queries})
    assert distribution["support"] == {"INSUFFICIENT": 12, "SUPPORTED": 12}
    assert distribution["manufacturer"] == {name: 6 for name in manufacturers}


def test_failure_taxonomy_only_contains_observed_classes():
    assert V315_FAILURE_TAXONOMY == {
        "RETRIEVAL_CONTEXT_ERROR",
        "COMPATIBILITY_COVERAGE_FAILURE",
        "VALUE_COVERAGE_FAILURE",
        "ATTRIBUTE_COVERAGE_FAILURE",
        "GENERIC_CONCEPT_OVERMATCH",
        "PARTIAL_SUPPORT_ACCEPTED",
    }


def test_failure_delta_separates_fixed_and_introduced_errors():
    before = {"rows": [
        {"query_id": "fixed-fs", "expected_supported": False, "predicted_supported": True},
        {"query_id": "new-fi", "expected_supported": True, "predicted_supported": True},
    ]}
    after = {"rows": [
        {"query_id": "fixed-fs", "expected_supported": False, "predicted_supported": False},
        {"query_id": "new-fi", "expected_supported": True, "predicted_supported": False},
    ]}
    assert failure_delta(before, after) == {
        "false_support_fixed": ["fixed-fs"],
        "false_support_introduced": [],
        "false_insufficient_fixed": [],
        "false_insufficient_introduced": ["new-fi"],
    }


def test_abc_report_records_ranges_runtime_and_offline_isolation():
    metrics = {
        "support_accuracy": 1.0, "supported_recall": 1.0,
        "unsupported_recall": 1.0, "false_support_rate": 0.0,
        "false_insufficient_rate": 0.0,
    }
    rows = [{"query_id": "ok", "expected_supported": True, "predicted_supported": True}]
    before = {corpus: _result(deepcopy(rows), deepcopy(metrics)) for corpus in "ABC"}
    after = {corpus: _result(deepcopy(rows), deepcopy(metrics), index / 10) for index, corpus in enumerate("ABC", 1)}
    report = abc_report(before, after)
    assert report["generalization_range"]["after"]["false_support_rate"]["range"] == 0
    assert report["replay_runtime_seconds"] == {"A": .1, "B": .2, "C": .3}
    assert set(report["live_retrieval"].values()) == {"NO"}
