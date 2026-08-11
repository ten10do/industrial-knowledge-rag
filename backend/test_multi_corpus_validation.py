from __future__ import annotations

import pytest

from backend.evaluation.multi_corpus_validation import (
    combined_failure_taxonomy,
    combined_metrics,
    comparison_coverage,
    failure_only_rows,
    generalization_gap,
    per_category_metrics,
    validate_corpus_separation,
)


def _manifest(document_id: str, query_id: str) -> dict:
    return {
        "documents": [{"document_id": document_id}],
        "queries": [{"query_id": query_id, "relevant_document_ids": [document_id]}],
    }


def _report(count: int, recall: float, failures: dict | None = None) -> dict:
    return {
        "overall": {"count": count, "hit_rate_at_1": recall / 2, "hit_rate_at_3": recall, "recall_at_5": recall, "mrr": recall},
        "category_metrics": {"procedure": {"count": count, "recall_at_5": recall}},
        "comparison_coverage_at_5": None,
        "failure_summary": failures or {},
        "rows": [{"query_id": "ok", "failure_type": None}, {"query_id": "bad", "failure_type": "RECALL_FAILURE"}],
    }


def test_multi_corpus_separation_rejects_reused_ids_and_cross_labels():
    assert validate_corpus_separation({"A": _manifest("a-doc", "a-query"), "B": _manifest("b-doc", "b-query")}) == {
        "corpora": 2, "documents": 2, "queries": 2,
    }
    with pytest.raises(ValueError, match="disjoint"):
        validate_corpus_separation({"A": _manifest("a-doc", "a-query"), "B": _manifest("a-doc", "b-query")})
    invalid = _manifest("b-doc", "b-query")
    invalid["queries"][0]["relevant_document_ids"] = ["a-doc"]
    with pytest.raises(ValueError, match="cross-corpus"):
        validate_corpus_separation({"A": _manifest("a-doc", "a-query"), "B": invalid})


def test_combined_gap_categories_comparison_and_failure_only_trace():
    a, b = _report(2, 0.5, {"RECALL_FAILURE": 1}), _report(6, 1.0, {"MODEL_CONFUSION": 2})
    assert combined_metrics({"A": a, "B": b}) == {
        "count": 8, "hit_rate_at_1": 0.4375, "hit_rate_at_3": 0.875, "recall_at_5": 0.875, "mrr": 0.875,
    }
    assert generalization_gap(a, b) == 0.5
    assert per_category_metrics({"A": a}) == {"A": a["category_metrics"]}
    assert comparison_coverage({"A": a}) == {"A": None}
    assert combined_failure_taxonomy({"A": a, "B": b}) == {"MODEL_CONFUSION": 2, "RECALL_FAILURE": 1}
    assert failure_only_rows(a) == [{"query_id": "bad", "failure_type": "RECALL_FAILURE"}]
