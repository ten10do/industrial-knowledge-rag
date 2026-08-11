"""Corpus-independent helpers for frozen multi-corpus retrieval validation."""

from __future__ import annotations

from collections import Counter
from typing import Any


OVERALL_METRICS = (
    "hit_rate_at_1",
    "hit_rate_at_3",
    "recall_at_5",
    "mrr",
)


def validate_corpus_separation(manifests: dict[str, dict]) -> dict[str, int]:
    """Reject reused document/query IDs and cross-corpus relevance labels."""
    seen_documents: set[str] = set()
    seen_queries: set[str] = set()
    document_count = query_count = 0
    for name, manifest in manifests.items():
        documents = {item["document_id"] for item in manifest["documents"]}
        queries = {item["query_id"] for item in manifest["queries"]}
        if not documents or not queries:
            raise ValueError(f"Corpus {name} requires documents and queries.")
        if seen_documents & documents or seen_queries & queries:
            raise ValueError("Corpus document IDs and query IDs must be disjoint.")
        for query in manifest["queries"]:
            if not set(query["relevant_document_ids"]).issubset(documents):
                raise ValueError(f"Corpus {name} contains a cross-corpus relevance label.")
        seen_documents.update(documents)
        seen_queries.update(queries)
        document_count += len(documents)
        query_count += len(queries)
    return {"corpora": len(manifests), "documents": document_count, "queries": query_count}


def combined_metrics(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Count-weighted aggregate for independently frozen corpus reports."""
    overall = [report["overall"] for report in reports.values()]
    total = sum(item["count"] for item in overall)
    if not total:
        raise ValueError("Combined metrics require answerable retrieval rows.")
    combined = {"count": total}
    for metric in OVERALL_METRICS:
        combined[metric] = sum(item[metric] * item["count"] for item in overall) / total
    return combined


def generalization_gap(corpus_a: dict[str, Any], corpus_b: dict[str, Any], metric: str = "recall_at_5") -> float:
    """Return Corpus B minus Corpus A for one retrieval metric."""
    return corpus_b["overall"][metric] - corpus_a["overall"][metric]


def per_category_metrics(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep category scores separate; absent categories are not converted to zero."""
    return {
        corpus: report.get("category_metrics", {})
        for corpus, report in reports.items()
    }


def comparison_coverage(reports: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {
        corpus: report.get("comparison_coverage_at_5")
        for corpus, report in reports.items()
    }


def combined_failure_taxonomy(reports: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for report in reports.values():
        totals.update(report.get("failure_summary", {}))
    return dict(sorted(totals.items()))


def failure_only_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Make failure tracing opt-in without retaining successful-query detail."""
    return [row for row in report.get("rows", []) if row.get("failure_type")]
