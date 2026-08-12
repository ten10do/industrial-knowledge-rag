"""Corpus-independent helpers for frozen multi-corpus retrieval validation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


OVERALL_METRICS = (
    "hit_rate_at_1",
    "hit_rate_at_3",
    "recall_at_5",
    "mrr",
)
OOD_TYPES = {
    "unknown_identifier", "unknown_model", "unsupported_parameter",
    "unsupported_procedure", "protocol_mismatch", "cross_equipment",
    "plausible_hallucination", "other",
}
SUPPORT_LABELS = {"SUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"}
EVIDENCE_METRICS = (
    "decision_accuracy", "ood_recall", "answerable_recall",
    "false_answer_rate", "false_refusal_rate",
)
SUPPORT_METRICS = (
    "support_accuracy", "supported_recall", "unsupported_recall",
    "false_support_rate", "false_insufficient_rate",
)
SECTION_METRICS = (
    "correct_section_hit_at_1", "correct_section_recall_at_5",
    "section_expansion_win_rate", "section_expansion_loss_rate",
)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def query_text_hash(manifest: dict[str, Any]) -> str:
    """Hash only stable query IDs and text, excluding all annotation enrichment."""
    return _hash_payload([
        {"query_id": item["query_id"], "query": item["query"]}
        for item in manifest["queries"]
    ])


def enriched_annotation_hash(manifest: dict[str, Any]) -> str:
    """Hash the complete document/query annotations, excluding mutable timestamps."""
    return _hash_payload({"documents": manifest["documents"], "queries": manifest["queries"]})


def validate_annotation_enrichment(manifest: dict[str, Any]) -> None:
    """Validate V3.10 labels without depending on retrieval output."""
    enrichment = manifest.get("annotation_enrichment")
    if not enrichment:
        raise ValueError("V3.10 annotation enrichment is required.")
    required = {
        "annotation_version", "original_query_sha256", "enriched_annotation_sha256",
        "enrichment_timestamp", "original_annotation_sha256",
    }
    if not required.issubset(enrichment):
        raise ValueError("V3.10 enrichment metadata is incomplete.")
    if enrichment["original_query_sha256"] != query_text_hash(manifest):
        raise ValueError("V3.10 query text hash changed.")
    if enrichment["enriched_annotation_sha256"] != enriched_annotation_hash(manifest):
        raise ValueError("V3.10 enriched annotation hash is stale.")
    for query in manifest["queries"]:
        evidence = query.get("evidence_label")
        support = query.get("support_label")
        gate_truth = query.get("support_gate_truth")
        if evidence not in {"ANSWER", "ABSTAIN"} or support not in SUPPORT_LABELS:
            raise ValueError("V3.10 evidence or support label is invalid.")
        if gate_truth not in {"SUPPORTED", "INSUFFICIENT"}:
            raise ValueError("V3.10 support gate truth is invalid.")
        if query["answerable"] != (evidence == "ANSWER"):
            raise ValueError("V3.10 evidence labels must match answerable ground truth.")
        if gate_truth == "SUPPORTED" and support != "SUPPORTED":
            raise ValueError("Only SUPPORTED labels can pass the support gate.")
        if query["answerable"]:
            sections = query.get("expected_sections", [])
            if not query.get("expected_section") or not isinstance(query.get("expected_subsection"), str) or not sections:
                raise ValueError("V3.10 answerable queries require expected section labels.")
            for section in sections:
                fields = {"document_id", "page", "ground_truth_section", "parser_section", "relevant_chunk_ids"}
                if not fields.issubset(section) or not section["relevant_chunk_ids"]:
                    raise ValueError("V3.10 section evidence is incomplete.")
            if query.get("ood_type"):
                raise ValueError("Answerable queries cannot have an OOD type.")
        elif query.get("ood_type") not in OOD_TYPES:
            raise ValueError("V3.10 OOD queries require a known OOD type.")


def comparison_label_coverage(queries: list[dict[str, Any]]) -> dict[str, int]:
    """Count only genuine multi-document comparison annotations."""
    comparisons = [item for item in queries if item.get("category") == "comparison" and item.get("answerable")]
    invalid = [
        item for item in comparisons
        if len(set(item.get("relevant_document_ids", []))) < 2 or len(set(item.get("relevant_chunk_ids", []))) < 2
    ]
    if invalid:
        raise ValueError("Comparison labels require multiple documents and chunks.")
    return {"queries": len(comparisons), "multi_document_queries": len(comparisons) - len(invalid)}


def comparison_metrics(queries: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, float | int | None]:
    """Measure top-five coverage for source-grounded comparison labels.

    Relevant-set coverage requires every labelled chunk. Relevant-document
    coverage is reported separately so a missing chunk is not mistaken for a
    missing source document.
    """
    comparison_label_coverage(queries)
    comparisons = [
        query for query in queries
        if query.get("category") == "comparison" and query.get("answerable")
    ]
    if not comparisons:
        return {
            "queries": 0,
            "relevant_set_coverage_at_5": None,
            "relevant_document_coverage_at_5": None,
        }
    rows_by_id = {row["query_id"]: row for row in report.get("rows", [])}
    if set(query["query_id"] for query in comparisons) - set(rows_by_id):
        raise ValueError("Comparison report is missing labelled query rows.")

    relevant_set_hits = relevant_document_hits = 0
    for query in comparisons:
        row = rows_by_id[query["query_id"]]
        candidates = row.get("candidates", [])[:5]
        candidate_chunks = set(row.get("candidate_ids", [])) or {
            item.get("chunk_id") for item in candidates
        }
        candidate_documents = {item.get("document_id") for item in candidates}
        relevant_set_hits += set(query["relevant_chunk_ids"]).issubset(candidate_chunks)
        relevant_document_hits += set(query["relevant_document_ids"]).issubset(candidate_documents)
    count = len(comparisons)
    return {
        "queries": count,
        "relevant_set_coverage_at_5": relevant_set_hits / count,
        "relevant_document_coverage_at_5": relevant_document_hits / count,
    }


def unified_latency_summary(samples: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Validate a common measurement policy before presenting cross-corpus latency."""
    policies = {
        (item["warmup_runs"], item["measured_runs"], item["tracing_enabled"], item["candidate_k"], item["reranker_pool_k"])
        for item in samples.values()
    }
    if len(policies) != 1:
        raise ValueError("Latency samples must use one shared policy.")
    return {
        name: {"median_ms": item["median_ms"], "p95_ms": item["p95_ms"]}
        for name, item in samples.items()
    }


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


def metric_gap(corpus_a: dict[str, Any], corpus_b: dict[str, Any], metric: str) -> float | None:
    """Return Corpus B minus Corpus A, preserving unavailable metrics as None."""
    a_value, b_value = corpus_a.get(metric), corpus_b.get(metric)
    return b_value - a_value if a_value is not None and b_value is not None else None


def generalization_gaps(
    retrieval_a: dict[str, Any],
    retrieval_b: dict[str, Any],
    section_a: dict[str, Any],
    section_b: dict[str, Any],
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    support_a: dict[str, Any],
    support_b: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    """Produce non-averaged Corpus B minus Corpus A generalization gaps."""
    return {
        "retrieval": {metric: generalization_gap(retrieval_a, retrieval_b, metric) for metric in OVERALL_METRICS},
        "section": {metric: metric_gap(section_a, section_b, metric) for metric in SECTION_METRICS},
        "evidence": {metric: metric_gap(evidence_a, evidence_b, metric) for metric in EVIDENCE_METRICS},
        "support": {metric: metric_gap(support_a, support_b, metric) for metric in SUPPORT_METRICS},
    }


def cross_corpus_failure_matrix(corpora: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count failure families independently for each corpus; never hide B in an average."""
    matrix: dict[str, dict[str, int]] = {}
    for name, data in corpora.items():
        retrieval = data["retrieval"]
        failures = retrieval.get("failure_summary", {})
        evidence = data.get("evidence", {})
        support = data.get("support", {})
        evidence_rows = evidence.get("rows", [])
        support_rows = support.get("rows", [])
        matrix[name] = {
            "recall": failures.get("RECALL_FAILURE", 0),
            "ranking": failures.get("RANKING_FAILURE", 0),
            "section": failures.get("SECTION_CONFUSION", 0),
            "model_confusion": failures.get("MODEL_CONFUSION", 0),
            "identifier": failures.get("IDENTIFIER_CONFUSION", 0),
            "evidence_false_answer": sum(
                not row.get("answerable") and row.get("decision") == "ANSWER"
                for row in evidence_rows
            ),
            "evidence_false_refusal": sum(
                row.get("answerable") and row.get("decision") == "ABSTAIN"
                for row in evidence_rows
            ),
            "support_false_positive": sum(
                not row.get("expected_supported") and row.get("predicted_supported")
                for row in support_rows
            ),
            "support_false_negative": sum(
                row.get("expected_supported") and not row.get("predicted_supported")
                for row in support_rows
            ),
        }
    return matrix


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
