from __future__ import annotations

from backend.evaluation.v310_runner import (
    aggregate_combined, aggregate_combined_latency, fixed_latency_subset,
    timeout_root_cause_audit,
)


def _corpus(recall: float, failure: str) -> dict:
    retrieval = {
        "overall": {"count": 2, "hit_rate_at_1": recall, "hit_rate_at_3": recall, "recall_at_5": recall, "mrr": recall},
        "failure_summary": {failure: 1}, "rows": [],
    }
    section = {
        "correct_section_hit_at_1": recall, "correct_section_recall_at_5": recall,
        "section_expansion_win_rate": recall, "section_expansion_loss_rate": 1 - recall,
    }
    evidence = {metric: recall for metric in ("decision_accuracy", "ood_recall", "answerable_recall", "false_answer_rate", "false_refusal_rate")}
    evidence["rows"] = [{"answerable": False, "decision": "ANSWER"}]
    support = {metric: recall for metric in ("support_accuracy", "supported_recall", "unsupported_recall", "false_support_rate", "false_insufficient_rate")}
    support["rows"] = [{"expected_supported": False, "predicted_supported": True}]
    return {"validity": "VALID", "retrieval": {"p1": retrieval, "p2": retrieval}, "section": section, "evidence": evidence, "support": support}


def test_combined_aggregation_uses_saved_rows_and_retains_corpus_failure_counts():
    combined = aggregate_combined(_corpus(.5, "RECALL_FAILURE"), _corpus(1.0, "MODEL_CONFUSION"))
    assert combined["validity"] == "VALID"
    assert combined["retrieval"]["p2"]["recall_at_5"] == .75
    assert combined["failure_matrix"]["A"]["recall"] == 1
    assert combined["failure_matrix"]["B"]["model_confusion"] == 1


def test_timeout_root_cause_audit_identifies_combined_index_rebuild_without_claiming_model_reload():
    audit = {item["stage"]: item for item in timeout_root_cause_audit()}
    assert audit["INDEX_BUILD"]["avoidable"] == "yes; Combined correctness aggregates saved A/B rows."
    assert audit["MODEL_LOAD"]["avoidable"] == "already reused per process; no model behavior change."


def test_fixed_latency_subset_is_category_balanced_and_combined_latency_uses_query_samples():
    queries = [
        {"query_id": f"{category}-{index}", "category": category}
        for category in ("identifier", "procedure", "semantic", "ood") for index in range(2)
    ]
    subset = fixed_latency_subset(queries)
    assert len(subset) == 8
    rows = [{"stages_ms": {"total": value, "reranker": value - 1}} for value in (1, 3)]
    latency = {"validity": "VALID", "reports": {"p1": {"rows": rows}, "p2": {"rows": rows}}}
    combined = aggregate_combined_latency(latency, latency)
    assert combined["reports"]["p1"]["sample_count"] == 4
    assert combined["reports"]["p1"]["median_ms"] == 2.0
