from __future__ import annotations

import pytest

from backend.evaluation.multi_corpus_validation import (
    comparison_label_coverage,
    comparison_metrics,
    combined_failure_taxonomy,
    combined_metrics,
    cross_corpus_failure_matrix,
    enriched_annotation_hash,
    comparison_coverage,
    failure_only_rows,
    generalization_gap,
    generalization_gaps,
    per_category_metrics,
    query_text_hash,
    unified_latency_summary,
    validate_annotation_enrichment,
    validate_corpus_separation,
)


def _manifest(document_id: str, query_id: str) -> dict:
    return {
        "documents": [{"document_id": document_id}],
        "queries": [{"query_id": query_id, "query": "frozen query", "relevant_document_ids": [document_id]}],
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


def test_enrichment_keeps_query_text_frozen_and_validates_section_evidence():
    manifest = _manifest("b-doc", "b-query")
    query = manifest["queries"][0]
    query.update({
        "answerable": True, "expected_section": "Commissioning", "expected_subsection": "Start",
        "expected_sections": [{"document_id": "b-doc", "page": 3, "ground_truth_section": "Commissioning", "parser_section": "Cover", "relevant_chunk_ids": ["chunk-b"]}],
        "evidence_label": "ANSWER", "support_label": "SUPPORTED", "support_gate_truth": "SUPPORTED", "ood_type": "",
    })
    manifest["annotation_enrichment"] = {
        "annotation_version": "v3.10", "original_annotation_sha256": "original", "enrichment_timestamp": "2026-08-11T00:00:00Z",
        "original_query_sha256": query_text_hash(manifest), "enriched_annotation_sha256": "",
    }
    manifest["annotation_enrichment"]["enriched_annotation_sha256"] = enriched_annotation_hash(manifest)
    validate_annotation_enrichment(manifest)
    query["query"] = "changed"
    with pytest.raises(ValueError, match="query text hash"):
        validate_annotation_enrichment(manifest)


def test_comparison_labels_and_unified_latency_policy_are_checked():
    comparison = {"category": "comparison", "answerable": True, "relevant_document_ids": ["a", "b"], "relevant_chunk_ids": ["x", "y"]}
    assert comparison_label_coverage([comparison]) == {"queries": 1, "multi_document_queries": 1}
    with pytest.raises(ValueError, match="multiple documents"):
        comparison_label_coverage([{**comparison, "relevant_document_ids": ["a"]}])
    samples = {
        "A": {"warmup_runs": 1, "measured_runs": 3, "tracing_enabled": False, "candidate_k": 7, "reranker_pool_k": 7, "median_ms": 10.0, "p95_ms": 12.0},
        "B": {"warmup_runs": 1, "measured_runs": 3, "tracing_enabled": False, "candidate_k": 7, "reranker_pool_k": 7, "median_ms": 20.0, "p95_ms": 24.0},
    }
    assert unified_latency_summary(samples)["B"] == {"median_ms": 20.0, "p95_ms": 24.0}
    samples["B"]["candidate_k"] = 5
    with pytest.raises(ValueError, match="shared policy"):
        unified_latency_summary(samples)


def test_comparison_metrics_keep_chunk_and_document_coverage_separate():
    query = {
        "query_id": "compare", "category": "comparison", "answerable": True,
        "relevant_document_ids": ["a", "b"], "relevant_chunk_ids": ["a-1", "b-1"],
    }
    report = {"rows": [{
        "query_id": "compare", "candidate_ids": ["a-1", "b-2"],
        "candidates": [{"document_id": "a"}, {"document_id": "b"}],
    }]}
    assert comparison_metrics([query], report) == {
        "queries": 1, "relevant_set_coverage_at_5": 0.0,
        "relevant_document_coverage_at_5": 1.0,
    }


def test_generalization_gaps_and_failure_matrix_do_not_average_corpora():
    a, b = _report(2, 0.5, {"RECALL_FAILURE": 1}), _report(6, 1.0, {"MODEL_CONFUSION": 2})
    section_a = {"correct_section_hit_at_1": .5, "correct_section_recall_at_5": .5, "section_expansion_win_rate": 0, "section_expansion_loss_rate": .5}
    section_b = {"correct_section_hit_at_1": .75, "correct_section_recall_at_5": 1, "section_expansion_win_rate": .25, "section_expansion_loss_rate": 0}
    evidence_a = {metric: .5 for metric in ("decision_accuracy", "ood_recall", "answerable_recall", "false_answer_rate", "false_refusal_rate")}
    evidence_b = {metric: 1.0 for metric in evidence_a}
    support_a = {metric: .5 for metric in ("support_accuracy", "supported_recall", "unsupported_recall", "false_support_rate", "false_insufficient_rate")}
    support_b = {metric: 1.0 for metric in support_a}
    gaps = generalization_gaps(a, b, section_a, section_b, evidence_a, evidence_b, support_a, support_b)
    assert gaps["retrieval"]["recall_at_5"] == .5
    assert gaps["section"]["correct_section_hit_at_1"] == .25
    assert gaps["evidence"]["decision_accuracy"] == .5
    assert gaps["support"]["support_accuracy"] == .5
    matrix = cross_corpus_failure_matrix({
        "A": {"retrieval": a, "evidence": {"rows": [{"answerable": False, "decision": "ANSWER"}]}, "support": {"rows": []}},
        "B": {"retrieval": b, "evidence": {"rows": [{"answerable": True, "decision": "ABSTAIN"}]}, "support": {"rows": []}},
    })
    assert matrix["A"]["recall"] == 1
    assert matrix["A"]["evidence_false_answer"] == 1
    assert matrix["B"]["model_confusion"] == 2
    assert matrix["B"]["evidence_false_refusal"] == 1
