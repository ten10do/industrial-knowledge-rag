"""Public validation checks for the private V3.20 calibration harness."""

from __future__ import annotations

from copy import deepcopy

import pytest

from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v320_evidence_contract import calibration_metrics, validate_calibration
from backend.evaluation.v320_frozen_replay import assemble_report


FAILURE_CLASSES = (
    "identifier", "protocol", "attribute", "value", "action",
    "requirement", "semantic", "multi_chunk", "cross_scope", "qualifier",
)
MANUFACTURERS = ("Rockwell", "ABB", "Omron", "Beckhoff")


def _manifest() -> dict:
    queries = []
    candidates = {}
    for index in range(24):
        chunk_id = f"chunk-{index:02}"
        candidates[chunk_id] = {"content": "fixed candidate", "metadata": {"chunk_id": chunk_id}}
        for answerable, suffix in ((True, "p"), (False, "n")):
            queries.append({
                "query_id": f"c{index:02}{suffix}",
                "pair_id": f"pair-{index:02}",
                "query": f"new paired calibration query {index} {suffix}",
                "answerable": answerable,
                "manufacturer": MANUFACTURERS[index % len(MANUFACTURERS)],
                "failure_class": FAILURE_CLASSES[index % len(FAILURE_CLASSES)],
                "candidate_chunk_ids": [chunk_id],
                "semantic_positive": answerable and index < 10,
                "multi_chunk_positive": answerable and index < 5,
                "unsafe_multi_chunk_negative": not answerable and index < 5,
            })
    manifest = {"calibration_id": "synthetic-v320", "queries": queries, "candidates": candidates}
    manifest["freeze"] = {
        "query_sha256": hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in queries]),
        "annotation_sha256": hash_json(queries),
        "manifest_sha256": hash_json(manifest),
    }
    return manifest


def test_calibration_distribution_pairs_and_freeze_are_enforced():
    manifest = _manifest()
    distribution = validate_calibration(manifest)
    assert distribution["queries"] == 48
    assert distribution["answerable"] == distribution["abstain"] == 24
    assert distribution["pairs"] == 24
    assert distribution["semantic_positive"] == 10
    assert distribution["multi_chunk_positive"] == 5
    assert distribution["unsafe_multi_chunk_negative"] == 5

    changed = deepcopy(manifest)
    changed["queries"][0]["query"] = "mutation after freeze"
    with pytest.raises(ValueError, match="QUERY_HASH"):
        validate_calibration(changed)


def test_calibration_metrics_report_both_safety_and_utility_errors():
    rows = [
        {"query_id": "p1", "answerable": True, "decision": "ANSWER"},
        {"query_id": "p2", "answerable": True, "decision": "ABSTAIN"},
        {"query_id": "n1", "answerable": False, "decision": "ABSTAIN"},
        {"query_id": "n2", "answerable": False, "decision": "ANSWER"},
    ]
    metrics = calibration_metrics(rows)
    assert metrics["decision_accuracy"] == 0.5
    assert metrics["answerable_recall"] == 0.5
    assert metrics["abstain_recall"] == 0.5
    assert metrics["false_answer_ids"] == ["n2"]
    assert metrics["false_refusal_ids"] == ["p2"]


def _replay_result(corpus: str, positive_decision: str, negative_decision: str) -> dict:
    rows = [
        {"query_id": f"{corpus.lower()}p", "answerable": True, "expected_supported": True,
         "base_decision": positive_decision, "final_decision": positive_decision,
         "ground_truth": {"failure_class": "SEMANTIC"}},
        {"query_id": f"{corpus.lower()}n", "answerable": False, "expected_supported": False,
         "base_decision": negative_decision, "final_decision": negative_decision,
         "ground_truth": {"failure_class": "PROTOCOL_TOPIC_OVERMATCH"}},
    ]
    false_answers = [rows[1]["query_id"]] if negative_decision == "ANSWER" else []
    false_refusals = [rows[0]["query_id"]] if positive_decision == "ABSTAIN" else []
    return {
        "artifact_id": f"artifact-{corpus}", "artifact_hash": "a" * 64,
        "replay_elapsed_seconds": 0.1, "rows": rows,
        "metrics": {"evidence": {
            "decision_accuracy": (2 - len(false_answers) - len(false_refusals)) / 2,
            "answerable_recall": 1 - len(false_refusals), "ood_recall": 1 - len(false_answers),
            "false_answer_rate": len(false_answers), "false_refusal_rate": len(false_refusals),
            "false_answer_ids": false_answers, "false_refusal_ids": false_refusals,
        }},
    }


def test_frozen_report_tracks_ranges_failure_classes_and_regressions():
    before = {corpus: _replay_result(corpus, "ANSWER", "ABSTAIN") for corpus in "ABCDE"}
    after = deepcopy(before)
    after["E"] = _replay_result("E", "ABSTAIN", "ABSTAIN")
    report = assemble_report(before, after)
    assert report["matrix"]["E"]["evidence_regressions"] == ["ep"]
    assert report["matrix"]["A"]["false_answer_plus_false_refusal_after"] == 0
    assert report["generalization_range"]["after"]["answerable_recall"]["spread"] == 1
    assert report["failure_class_matrix"]["after"]["SEMANTIC"]["false_refusal_ids"] == ["E:ep"]
    assert all(not value for value in report["live_retrieval_isolation"]["D"].values())
