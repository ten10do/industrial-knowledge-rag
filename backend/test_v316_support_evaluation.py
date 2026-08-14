"""Tests for V3.16 recall calibration and replay reporting."""

from backend.evaluation.v316_support_recall import (
    REQUIRED_MANUFACTURERS,
    V316_FAILURE_CLASSES,
    calibration_distribution,
    calibration_failure_matrix,
    final_report,
)


def test_recall_distribution_tracks_balance_semantics_and_classes():
    manufacturers = sorted(REQUIRED_MANUFACTURERS)
    classes = sorted(V316_FAILURE_CLASSES)
    queries = [{
        "expected_support": "SUPPORTED" if index < 18 else "INSUFFICIENT",
        "manufacturer": manufacturers[index % len(manufacturers)],
        "category": "semantic" if index < 8 else "safety_negative",
        "failure_class": classes[index % len(classes)],
        "semantic_positive": index < 8,
    } for index in range(30)]
    distribution = calibration_distribution({"queries": queries})
    assert distribution["support"] == {"INSUFFICIENT": 12, "SUPPORTED": 18}
    assert distribution["semantic_positives"] == 8
    assert set(distribution["manufacturer"]) == REQUIRED_MANUFACTURERS


def test_calibration_failure_matrix_reports_both_error_directions():
    report = {"rows": [
        {"calibration_id": "positive", "failure_class": "OVER_CONSTRAINED_VALUE", "expected_support": "SUPPORTED", "predicted_support": "INSUFFICIENT"},
        {"calibration_id": "negative", "failure_class": "PARTIAL_SUPPORT_ACCEPTED", "expected_support": "INSUFFICIENT", "predicted_support": "SUPPORTED"},
    ]}
    matrix = calibration_failure_matrix(report)
    assert matrix["OVER_CONSTRAINED_VALUE"]["false_insufficient"] == ["positive"]
    assert matrix["PARTIAL_SUPPORT_ACCEPTED"]["false_support"] == ["negative"]


def test_final_report_adds_combined_runtime_and_offline_isolation():
    metrics = {name: 1.0 if "recall" in name or name == "support_accuracy" else 0.0 for name in (
        "support_accuracy", "supported_recall", "unsupported_recall", "false_support_rate", "false_insufficient_rate",
    )}
    row = {"query_id": "ok", "expected_supported": True, "predicted_supported": True}
    before = {corpus: {"rows": [row], "metrics": {"support": metrics}, "replay_elapsed_seconds": 0.0} for corpus in "ABC"}
    after = {corpus: {"rows": [row], "metrics": {"support": metrics}, "replay_elapsed_seconds": seconds} for corpus, seconds in zip("ABC", (.1, .2, .3))}
    report = final_report(before, after)
    assert abs(report["combined_replay_runtime_seconds"] - .6) < 1e-12
    assert set(report["live_retrieval"].values()) == {"NO"}
