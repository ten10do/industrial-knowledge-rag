"""Local-only V3.1 validation over vendor documents parsed by Industrial Ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.benchmark_schema import evaluate_rows, rank_of
from backend.evaluation.full_vector_benchmark import (
    FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
    full_vector_knowledge_base,
)
from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base
from backend.evaluation.retrieval_observability import analyze_observability, write_trace_artifacts
from backend.retrieval import RetrievalCandidate, RetrievalResult, analyze_query
from backend.retrieval.evidence_support import skipped_support, validate_evidence_support
from backend.retrieval.section import normalize_section
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.product_identity import identity_from_metadata


PRIVATE_REQUIRED_DOCUMENT_FIELDS = {
    "document_id", "file", "source_name", "source_type", "manufacturer",
    "equipment_type", "equipment_model", "document_type", "language", "version",
    "publish_date", "commit_allowed",
}
PRIVATE_REQUIRED_QUERY_FIELDS = {
    "query_id", "query", "category", "answerable", "relevant_chunk_ids",
    "relevant_document_ids", "expected_model", "expected_error_code",
    "expected_section", "difficulty",
}
PRIVATE_CATEGORIES = {
    "identifier", "fault", "parameter", "semantic", "procedure", "safety",
    "maintenance", "mixed", "comparison", "ood",
}
CORE_PRIVATE_MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")
PRIVATE_MODES = (*CORE_PRIVATE_MODES, "hybrid_section_rerank")
FALSE_REFUSAL_REASONS = {
    "WEAK_RETRIEVAL_EVIDENCE": "DISTANCE_THRESHOLD",
    "MODEL_MISMATCH": "METADATA_MISMATCH",
    "INSUFFICIENT_EVIDENCE": "UNSUPPORTED_DETAIL_RULE",
    "NO_CANDIDATE": "NO_CANDIDATE",
    "UNKNOWN_IDENTIFIER": "IDENTIFIER_RULE",
}


def annotation_hash(manifest: dict) -> str:
    """Hash only the frozen labels, not local paths or downloaded file bytes."""
    frozen = {"documents": manifest["documents"], "queries": manifest["queries"]}
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_hash(calibration: dict) -> str:
    payload = json.dumps(calibration["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def development_hash(development: dict) -> str:
    payload = json.dumps(development["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def support_calibration_hash(calibration: dict) -> str:
    payload = json.dumps(calibration["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def section_development_hash(development: dict) -> str:
    payload = json.dumps(development["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_private_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents, queries = manifest.get("documents", []), manifest.get("queries", [])
    if not documents or not queries:
        raise ValueError("Private manifest requires documents and queries.")
    document_ids = set()
    for item in documents:
        if not PRIVATE_REQUIRED_DOCUMENT_FIELDS.issubset(item):
            raise ValueError("Private manifest document metadata is incomplete.")
        if item["document_id"] in document_ids or item["commit_allowed"] is not False:
            raise ValueError("Private documents must have unique IDs and commit_allowed=false.")
        source = Path(item["file"])
        if source.is_absolute() or not item["file"]:
            raise ValueError("Private document file must be a non-empty relative path.")
        document_ids.add(item["document_id"])
    query_ids = set()
    for item in queries:
        if not PRIVATE_REQUIRED_QUERY_FIELDS.issubset(item):
            raise ValueError("Private query annotation is incomplete.")
        if item["query_id"] in query_ids or item["category"] not in PRIVATE_CATEGORIES:
            raise ValueError("Private query ID or category is invalid.")
        if item["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("Private query difficulty is invalid.")
        if not set(item["relevant_document_ids"]).issubset(document_ids):
            raise ValueError("Private query references an unknown document.")
        if item["answerable"] != bool(item["relevant_chunk_ids"]):
            raise ValueError("Private answerable flag must match relevant chunks.")
        query_ids.add(item["query_id"])
    frozen_hash = manifest.get("freeze", {}).get("annotation_sha256")
    if frozen_hash and frozen_hash != annotation_hash(manifest):
        raise ValueError("Private annotation hash does not match the frozen manifest.")
    return manifest


def load_private_calibration(path: Path, manifest: dict, chunk_ids: set[str]) -> dict:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    queries = calibration.get("queries", [])
    if not 20 <= len(queries) <= 30:
        raise ValueError("V3.2 calibration requires 20 to 30 independent queries.")
    if sum(bool(item.get("answerable")) for item in queries) < 12:
        raise ValueError("V3.2 calibration requires at least 12 answerable queries.")
    if sum(not bool(item.get("answerable")) for item in queries) < 8:
        raise ValueError("V3.2 calibration requires at least 8 OOD queries.")
    required = {
        "query_id", "query", "category", "answerable", "relevant_chunk_ids",
        "expected_model", "expected_error_code", "ood_type",
    }
    query_ids = set()
    frozen_queries = {item["query"].strip().casefold() for item in manifest["queries"]}
    for item in queries:
        if not required.issubset(item) or item["query_id"] in query_ids:
            raise ValueError("V3.2 calibration query schema or ID is invalid.")
        if item["query"].strip().casefold() in frozen_queries:
            raise ValueError("Calibration queries must be independent from V3.1 evaluation queries.")
        if item["answerable"] != bool(item["relevant_chunk_ids"]):
            raise ValueError("Calibration answerable flag must match relevant chunks.")
        if not set(item["relevant_chunk_ids"]).issubset(chunk_ids):
            raise ValueError("Calibration query references an unknown chunk.")
        query_ids.add(item["query_id"])
    if calibration.get("corpus_annotation_sha256") != annotation_hash(manifest):
        raise ValueError("Calibration corpus annotation hash does not match V3.1.")
    frozen_hash = calibration.get("freeze", {}).get("calibration_sha256")
    if frozen_hash and frozen_hash != calibration_hash(calibration):
        raise ValueError("Calibration hash does not match the frozen query set.")
    return calibration


def load_private_development(path: Path, manifest: dict, chunk_ids: set[str]) -> dict:
    development = json.loads(path.read_text(encoding="utf-8"))
    queries = development.get("queries", [])
    if not 15 <= len(queries) <= 25:
        raise ValueError("V3.3 development set requires 15 to 25 independent queries.")
    required = {
        "query_id", "query", "category", "answerable", "relevant_chunk_ids",
        "expected_model", "expected_error_code", "expected_scope",
    }
    frozen_queries = {item["query"].strip().casefold() for item in manifest["queries"]}
    query_ids = set()
    for item in queries:
        if not required.issubset(item) or item["query_id"] in query_ids:
            raise ValueError("V3.3 development query schema or ID is invalid.")
        if item["query"].strip().casefold() in frozen_queries:
            raise ValueError("V3.3 development queries must differ from V3.1 frozen queries.")
        if item["answerable"] != bool(item["relevant_chunk_ids"]):
            raise ValueError("V3.3 development answerable flag must match relevant chunks.")
        if not set(item["relevant_chunk_ids"]).issubset(chunk_ids):
            raise ValueError("V3.3 development query references an unknown chunk.")
        query_ids.add(item["query_id"])
    if development.get("corpus_annotation_sha256") != annotation_hash(manifest):
        raise ValueError("V3.3 development corpus hash does not match V3.1.")
    frozen_hash = development.get("freeze", {}).get("development_sha256")
    if frozen_hash and frozen_hash != development_hash(development):
        raise ValueError("V3.3 development hash does not match its query set.")
    return development


def load_support_calibration(path: Path, manifest: dict, chunk_ids: set[str]) -> dict:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    queries = calibration.get("queries", [])
    if not 20 <= len(queries) <= 30:
        raise ValueError("V3.4 support calibration requires 20 to 30 independent queries.")
    if sum(bool(item.get("supported")) for item in queries) < 12:
        raise ValueError("V3.4 support calibration requires at least 12 supported queries.")
    if sum(not bool(item.get("supported")) for item in queries) < 8:
        raise ValueError("V3.4 support calibration requires at least 8 unsupported queries.")
    required = {
        "query_id", "query", "category", "supported", "relevant_chunk_ids",
        "expected_base_decision",
    }
    frozen_queries = {item["query"].strip().casefold() for item in manifest["queries"]}
    query_ids = set()
    for item in queries:
        if not required.issubset(item) or item["query_id"] in query_ids:
            raise ValueError("V3.4 support calibration query schema or ID is invalid.")
        if item["query"].strip().casefold() in frozen_queries:
            raise ValueError("V3.4 support calibration must differ from V3.1 frozen queries.")
        if item["expected_base_decision"] not in {"ANSWER", "ABSTAIN"}:
            raise ValueError("V3.4 expected base decision is invalid.")
        if not set(item["relevant_chunk_ids"]).issubset(chunk_ids):
            raise ValueError("V3.4 support calibration references an unknown chunk.")
        if not item["supported"] and item["relevant_chunk_ids"]:
            raise ValueError("Unsupported V3.4 queries must not claim supporting chunks.")
        query_ids.add(item["query_id"])
    if calibration.get("corpus_annotation_sha256") != annotation_hash(manifest):
        raise ValueError("V3.4 support calibration corpus hash does not match V3.1.")
    frozen_hash = calibration.get("freeze", {}).get("support_calibration_sha256")
    if frozen_hash and frozen_hash != support_calibration_hash(calibration):
        raise ValueError("V3.4 support calibration hash does not match its query set.")
    return calibration


def load_section_development(path: Path, manifest: dict, chunk_ids: set[str]) -> dict:
    development = json.loads(path.read_text(encoding="utf-8"))
    queries = development.get("queries", [])
    if not 20 <= len(queries) <= 30:
        raise ValueError("V3.5 section development requires 20 to 30 independent queries.")
    required = {
        "query_id", "query", "category", "answerable", "supported",
        "relevant_chunk_ids", "expected_model", "expected_section", "hard_positive",
    }
    frozen_queries = {item["query"].strip().casefold() for item in manifest["queries"]}
    query_ids = set()
    for item in queries:
        if not required.issubset(item) or item["query_id"] in query_ids:
            raise ValueError("V3.5 section development schema or ID is invalid.")
        if item["query"].strip().casefold() in frozen_queries:
            raise ValueError("V3.5 section development must differ from frozen queries.")
        if item["answerable"] != bool(item["relevant_chunk_ids"]):
            raise ValueError("V3.5 answerable flag must match relevant chunks.")
        if not set(item["relevant_chunk_ids"]).issubset(chunk_ids):
            raise ValueError("V3.5 section development references an unknown chunk.")
        if item["supported"] and not item["answerable"]:
            raise ValueError("Unsupported V3.5 queries cannot be labeled supported.")
        query_ids.add(item["query_id"])
    if development.get("corpus_annotation_sha256") != annotation_hash(manifest):
        raise ValueError("V3.5 section development corpus hash does not match V3.1.")
    frozen_hash = development.get("freeze", {}).get("section_development_sha256")
    if frozen_hash and frozen_hash != section_development_hash(development):
        raise ValueError("V3.5 section development hash does not match its query set.")
    return development


def _resolve_local_file(manifest_path: Path, value: str) -> Path:
    root = manifest_path.parent.resolve()
    source = (root / value).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError(f"Private benchmark file is unavailable: {value}")
    return source


def ingest_private_documents(manifest_path: Path, manifest: dict) -> tuple[list[Document], dict]:
    """Use the application PDF parser and Industrial Chunker; never a benchmark loader."""
    documents, audit = [], {}
    for entry in manifest["documents"]:
        source = _resolve_local_file(manifest_path, entry["file"])
        if source.suffix.lower() != ".pdf":
            raise ValueError("Private corpus currently accepts text-based PDF files only.")
        parsed_pages = rag_core.load_pdf(source)
        chunks = rag_core.split_documents(parsed_pages)
        if not chunks:
            raise ValueError(f"Industrial Ingestion produced no chunks for {entry['document_id']}.")
        for chunk in chunks:
            metadata = chunk.metadata
            aliases = entry.get("model_aliases", "")
            if isinstance(aliases, list):
                aliases = "|".join(str(item) for item in aliases)
            metadata.update({
                "document_id": entry["document_id"],
                "source": entry["source_name"],
                "source_name": entry["source_name"],
                "source_type": entry["source_type"],
                "manufacturer": entry["manufacturer"],
                "equipment_type": entry["equipment_type"],
                "equipment_model": entry["equipment_model"],
                "product_family": entry.get("product_family", ""),
                "product_series": entry.get("product_series", ""),
                "model_aliases": aliases,
                "document_type": entry["document_type"],
                "language": entry["language"],
                "document_version": entry["version"],
                "publish_date": entry["publish_date"],
            })
        sample = chunks[:10]
        issues = []
        if any(not item.metadata.get("section") for item in sample):
            issues.append("missing_section")
        if any(item.metadata.get("page") is None for item in sample):
            issues.append("missing_page")
        if any(not item.metadata.get("chunk_id") for item in sample):
            issues.append("missing_chunk_id")
        audit[entry["document_id"]] = {
            "pages": len(parsed_pages),
            "chunks": len(chunks),
            "sample": [{
                "chunk_id": item.metadata["chunk_id"],
                "page": item.metadata.get("page"),
                "section": item.metadata.get("section", ""),
                "knowledge_type": item.metadata.get("knowledge_type", ""),
                "error_code": item.metadata.get("error_code", ""),
                "content_length": len(item.page_content),
            } for item in sample],
            "issues": issues,
        }
        documents.extend(chunks)
    return documents, audit


def _candidate_rows(candidates: list) -> list[dict]:
    return [{
        "rank": rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.metadata.get("document_id", ""),
        "source": candidate.metadata.get("source", ""),
        "page": candidate.metadata.get("page"),
        "section": candidate.metadata.get("section", ""),
        "equipment_model": candidate.metadata.get("equipment_model", ""),
        "error_code": candidate.metadata.get("error_code", ""),
        "lexical_rank": getattr(candidate, "lexical_rank", None),
        "vector_rank": getattr(candidate, "vector_rank", None),
        "fusion_rank": getattr(candidate, "pre_rerank_rank", None) or getattr(candidate, "final_rank", None),
        "vector_distance": candidate.vector_score,
        "lexical_score": candidate.lexical_score,
        "pre_rerank_rank": candidate.pre_rerank_rank,
        "rerank_rank": candidate.rerank_rank,
        "identity_relation": getattr(candidate, "identity_relation", "UNKNOWN"),
        "scope_match": getattr(candidate, "scope_match", "none"),
        "scope_level": getattr(candidate, "scope_level", "GLOBAL_SCOPE"),
        "section_expanded": getattr(candidate, "section_expanded", False),
        "section_rank": getattr(candidate, "section_rank", None),
        "neighbor_distance": getattr(candidate, "neighbor_distance", None),
        "pre_section_rank": getattr(candidate, "pre_section_rank", None),
        "section_candidate_source": getattr(candidate, "section_candidate_source", ""),
    } for rank, candidate in enumerate(candidates, start=1)]


def _query_for_schema(query: dict) -> dict:
    return {
        **query,
        "query_type": query["category"],
        "expected_equipment_model": query["expected_model"],
        "expected_error_code": query.get("expected_error_code", ""),
    }


def _summary_metrics(queries: list[dict], rows: list[dict]) -> dict:
    report = evaluate_rows([_query_for_schema(item) for item in queries], rows)
    model_queries = [item for item in queries if item["answerable"] and item["expected_model"]]
    by_id = {row["query_id"]: row for row in rows}
    model_confusion = [
        item for item in model_queries
        if (top := (by_id[item["query_id"]]["candidates"] or [{}])[0]).get("equipment_model")
        and top.get("equipment_model") != item["expected_model"]
    ]
    comparisons = [item for item in queries if item["answerable"] and item["category"] == "comparison"]
    identifier = defaultdict(list)
    for item in queries:
        if item["answerable"] and item["category"] == "identifier":
            identifier["fault_code" if item.get("expected_error_code", "").upper().startswith(("F", "A")) else "parameter_or_register"].append(item)

    def top1_rate(items: list[dict]) -> float | None:
        return (sum(by_id[item["query_id"]]["rank"] == 1 for item in items) / len(items)) if items else None

    exact_model_queries = [
        item for item in queries
        if item["answerable"]
        and by_id[item["query_id"]].get("retrieval_scope", {}).get("requested_scope") == "EXACT_MODEL_SCOPE"
    ]
    identifier_queries = [item for item in queries if item["answerable"] and item.get("expected_error_code")]
    scoped_rows = [
        by_id[item["query_id"]] for item in queries
        if by_id[item["query_id"]].get("retrieval_scope", {}).get("requested_scope")
        in {"EXACT_MODEL_SCOPE", "SERIES_SCOPE", "FAMILY_SCOPE", "MULTI_IDENTITY_SCOPE"}
    ]
    scoped_candidates = [candidate for row in scoped_rows for candidate in row["candidates"][:5]]
    exact_model_confusions = [
        item for item in exact_model_queries
        if (top := (by_id[item["query_id"]]["candidates"] or [{}])[0]).get("equipment_model")
        and item.get("expected_model")
        and top.get("equipment_model") != item["expected_model"]
    ]
    scope_labeled = [item for item in queries if item.get("expected_scope")]

    def recall_at_5(items: list[dict]) -> float | None:
        return (sum((by_id[item["query_id"]]["rank"] or 99) <= 5 for item in items) / len(items)) if items else None

    return {
        **report,
        "comparison_coverage_at_5": (
            sum(set(by_id[item["query_id"]]["candidate_ids"][:5]) >= set(item["relevant_chunk_ids"]) for item in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "identifier_safety": {
            "fault_code_exact_hit_at_1": top1_rate(identifier["fault_code"]),
            "parameter_or_register_hit_at_1": top1_rate(identifier["parameter_or_register"]),
            "equipment_model_exact_hit_at_1": top1_rate(model_queries),
        },
        "model_confusion": {
            "count": len(model_confusion),
            "rate": len(model_confusion) / len(model_queries) if model_queries else None,
            "query_ids": [item["query_id"] for item in model_confusion],
        },
        "model_aware_metrics": {
            "exact_model_query_count": len(exact_model_queries),
            "exact_model_hit_at_1": top1_rate(exact_model_queries),
            "exact_model_recall_at_5": recall_at_5(exact_model_queries),
            "identifier_query_count": len(identifier_queries),
            "identifier_hit_at_1": top1_rate(identifier_queries),
            "identifier_recall_at_5": recall_at_5(identifier_queries),
            "equipment_model_confusion_rate": (
                len(exact_model_confusions) / len(exact_model_queries) if exact_model_queries else None
            ),
            "scope_precision": (
                sum(candidate.get("scope_match") == "primary" for candidate in scoped_candidates) / len(scoped_candidates)
                if scoped_candidates else None
            ),
            "scope_fallback_rate": (
                sum(row.get("retrieval_scope", {}).get("fallback_used", False) for row in rows) / len(rows)
                if rows else None
            ),
            "scope_decision_accuracy": (
                sum(
                    by_id[item["query_id"]].get("retrieval_scope", {}).get("requested_scope")
                    == item["expected_scope"]
                    for item in scope_labeled
                ) / len(scope_labeled)
                if scope_labeled else None
            ),
        },
    }


@contextmanager
def _section_mode(enabled: bool):
    previous = os.environ.get("SECTION_EXPANSION_ENABLED")
    os.environ["SECTION_EXPANSION_ENABLED"] = "true" if enabled else "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SECTION_EXPANSION_ENABLED", None)
        else:
            os.environ["SECTION_EXPANSION_ENABLED"] = previous


def _run_mode(
    mode: str,
    queries: list[dict],
    light_rag,
    light_id,
    reranker,
    *,
    trace_enabled: bool = False,
    section_strategy: str = "current",
    candidate_k: int = 5,
) -> dict:
    rows, latencies, rerank_rows, reranker_candidate_counts = [], [], [], []
    section_enabled = mode == "hybrid_section_rerank"
    with _section_mode(section_enabled):
        for query in queries:
            started = time.perf_counter()
            if mode == "bm25":
                result = light_rag.retrieve_docs(query["query"], k=5, knowledge_base_id=light_id, retrieval_mode="lexical")
                candidates = light_rag.filter_relevant_docs(result)
            else:
                retrieval_mode = "hybrid" if mode in {"hybrid_rerank", "hybrid_section_rerank"} else mode
                result = rag_core.retrieve_docs(
                    query["query"], k=candidate_k,
                    knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                    retrieval_mode=retrieval_mode,
                    trace_enabled=trace_enabled,
                    trace_query_id=query["query_id"],
                    section_merge_strategy=section_strategy if section_enabled else None,
                )
                candidates = rag_core.filter_relevant_docs(result)
            before = _candidate_rows(candidates.candidates)
            if mode in {"hybrid_rerank", "hybrid_section_rerank"}:
                outcome = reranker.rerank(query["query"], candidates, top_k=3)
                selected = outcome.result.candidates
            else:
                outcome, selected = None, candidates.candidates
            trace = getattr((outcome.result if outcome else candidates), "trace", None)
            if trace:
                trace.finalize(selected)
            candidate_rows = _candidate_rows(selected)
            latencies.append((time.perf_counter() - started) * 1000)
            before_rank = rank_of(before, query["relevant_chunk_ids"])
            rank = rank_of(candidate_rows, query["relevant_chunk_ids"])
            if outcome:
                reranker_candidate_counts.append(outcome.candidate_count)
                rerank_rows.append({
                    "query_id": query["query_id"], "before_rank": before_rank, "after_rank": rank,
                    "answerable": query["answerable"],
                    "candidate_missing": before_rank is None,
                })
            rows.append({
                "query_id": query["query_id"], "query": query["query"],
                "rank": rank, "refused": not candidate_rows, "candidates": candidate_rows,
                "candidate_ids": [item["chunk_id"] for item in candidate_rows],
                "retrieval_scope": (
                    candidates.scope_decision.as_dict()
                    if getattr(candidates, "scope_decision", None) else {}
                ),
                "section_retrieval": (
                    candidates.section_report.as_dict()
                    if getattr(candidates, "section_report", None) else None
                ),
                "trace": trace.as_dict() if trace else None,
            })
    report = _summary_metrics(queries, rows)
    report["latency_ms_median"] = statistics.median(latencies)
    report["latency_ms_p95"] = sorted(latencies)[max(0, round(len(latencies) * .95) - 1)]
    if rerank_rows:
        report["rerank_analysis"] = _rerank_analysis(rerank_rows)
        report["reranker_candidate_count_median"] = statistics.median(reranker_candidate_counts)
    return report


def _assert_trace_equivalence(plain: dict, traced: dict) -> None:
    plain_rows = {item["query_id"]: item.get("candidate_ids", []) for item in plain.get("rows", [])}
    traced_rows = {item["query_id"]: item.get("candidate_ids", []) for item in traced.get("rows", [])}
    if plain_rows != traced_rows:
        changed = sorted(key for key in plain_rows if plain_rows.get(key) != traced_rows.get(key))
        raise RuntimeError(f"Tracing changed retrieval output for queries: {changed}")


def _trace_overhead(plain: dict, traced: dict) -> dict:
    before = float(plain.get("latency_ms_median", 0.0))
    after = float(traced.get("latency_ms_median", 0.0))
    return {
        "off_median_ms": before,
        "on_median_ms": after,
        "delta_ms": after - before,
        "delta_rate": ((after - before) / before if before else 0.0),
    }


def _measure_trace_overhead(queries: list[dict], light_rag, light_id, reranker) -> dict:
    """Alternate warm OFF/ON requests so cache order does not masquerade as trace cost."""
    del light_rag, light_id
    sample = queries[:3]
    if not sample:
        return {"status": "NOT_RUN"}

    def run(query: dict, enabled: bool) -> float:
        with _section_mode(True):
            started = time.perf_counter()
            result = rag_core.retrieve_docs(
                query["query"], k=5,
                knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                retrieval_mode="hybrid",
                trace_enabled=enabled,
                trace_query_id=query["query_id"],
            )
            candidates = rag_core.filter_relevant_docs(result)
            outcome = reranker.rerank(query["query"], candidates, top_k=3)
            trace = getattr(outcome.result, "trace", None)
            if trace:
                trace.finalize(outcome.result.candidates)
            return (time.perf_counter() - started) * 1000

    run(sample[0], False)
    run(sample[0], True)
    timings = {False: [], True: []}
    for index, query in enumerate(sample):
        order = (False, True) if index % 2 == 0 else (True, False)
        for enabled in order:
            timings[enabled].append(run(query, enabled))
    before = statistics.median(timings[False])
    after = statistics.median(timings[True])
    return {
        "sample_count": len(sample),
        "off_median_ms": before,
        "on_median_ms": after,
        "delta_ms": after - before,
        "delta_rate": ((after - before) / before if before else 0.0),
    }


def _observability_summary(report: dict) -> dict:
    return {
        key: value for key, value in report.items()
        if key not in {"baseline_candidate_traces", "section_candidate_traces"}
    }


def _rerank_analysis(rows: list[dict]) -> dict:
    answerable_rows = [item for item in rows if item.get("answerable", True)]
    comparable = [item for item in answerable_rows if not item["candidate_missing"]]
    improved = sum(bool(item["after_rank"] and item["after_rank"] < item["before_rank"]) for item in comparable)
    degraded = sum(item["after_rank"] is None or item["after_rank"] > item["before_rank"] for item in comparable)
    same = len(comparable) - improved - degraded
    deltas = [item["before_rank"] - item["after_rank"] for item in comparable if item["after_rank"]]
    return {
        "improved": improved, "same": same, "degraded": degraded,
        "candidate_missing": len(answerable_rows) - len(comparable),
        "win_rate": improved / len(comparable) if comparable else 0.0,
        "tie_rate": same / len(comparable) if comparable else 0.0,
        "loss_rate": degraded / len(comparable) if comparable else 0.0,
        "mean_rank_delta": statistics.mean(deltas) if deltas else 0.0,
        "rows": rows,
    }


def _section_metrics(queries: list[dict], baseline: dict, section: dict) -> dict:
    before = {row["query_id"]: row for row in baseline["rows"]}
    after = {row["query_id"]: row for row in section["rows"]}
    labeled = [item for item in queries if item.get("answerable") and item.get("expected_section")]

    def section_rank(item: dict, row: dict) -> int | None:
        expected = normalize_section(item.get("expected_section", ""))
        for rank, candidate in enumerate(row.get("candidates", [])[:5], start=1):
            if normalize_section(candidate.get("section", "")) == expected:
                return rank
        return None

    before_section = {item["query_id"]: section_rank(item, before[item["query_id"]]) for item in labeled}
    after_section = {item["query_id"]: section_rank(item, after[item["query_id"]]) for item in labeled}
    answerable = [item for item in queries if item.get("answerable")]
    wins = losses = 0
    for item in answerable:
        old = before[item["query_id"]].get("rank")
        new = after[item["query_id"]].get("rank")
        if new is not None and (old is None or new < old):
            wins += 1
        elif old is not None and (new is None or new > old):
            losses += 1
    return {
        "correct_section_hit_at_1": (
            sum(rank == 1 for rank in after_section.values()) / len(labeled) if labeled else None
        ),
        "correct_section_recall_at_5": (
            sum(rank is not None and rank <= 5 for rank in after_section.values()) / len(labeled) if labeled else None
        ),
        "relevant_chunk_recall_at_5": (
            sum((after[item["query_id"]].get("rank") or 99) <= 5 for item in answerable) / len(answerable)
            if answerable else None
        ),
        "section_expansion_win_rate": wins / len(answerable) if answerable else 0.0,
        "section_expansion_loss_rate": losses / len(answerable) if answerable else 0.0,
        "wins": wins,
        "losses": losses,
        "rows": [
            {
                "query_id": item["query_id"],
                "before_section_rank": before_section[item["query_id"]],
                "after_section_rank": after_section[item["query_id"]],
                "before_chunk_rank": before[item["query_id"]].get("rank"),
                "after_chunk_rank": after[item["query_id"]].get("rank"),
            }
            for item in labeled
        ],
    }


def _report_support(queries: list[dict], report: dict, documents: list[Document]) -> dict:
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    report_rows = {item["query_id"]: item for item in report["rows"]}
    rows = []
    for query in queries:
        candidate_rows = report_rows[query["query_id"]].get("candidates", [])
        candidates = [
            RetrievalCandidate(
                document=documents_by_chunk[item["chunk_id"]],
                retrieval_source=item.get("section_candidate_source") or "hybrid",
                final_rank=item.get("rank"),
                identity_relation=item.get("identity_relation", "UNKNOWN"),
                scope_match=item.get("scope_match", "none"),
                scope_level=item.get("scope_level", "GLOBAL_SCOPE"),
            )
            for item in candidate_rows
            if item.get("chunk_id") in documents_by_chunk
        ]
        result = RetrievalResult(
            candidates,
            query_analysis=analyze_query(query["query"], documents),
            corpus_documents=documents,
            retrieval_mode="hybrid",
        )
        support = validate_evidence_support(query["query"], result, documents)
        rows.append({
            "query_id": query["query_id"],
            "expected_supported": bool(query.get("supported", query.get("answerable"))),
            "status": support.status,
            "missing_requirements": list(support.missing_requirements),
            "supporting_chunks": list(support.supporting_chunks),
        })
    return {"rows": rows}


def _support_recovery(queries: list[dict], baseline: dict, section: dict) -> dict:
    before = {item["query_id"]: item for item in baseline["rows"]}
    after = {item["query_id"]: item for item in section["rows"]}
    def expected_supported(item: dict) -> bool:
        return bool(item.get("supported", item.get("answerable")))

    def support_status(item: dict) -> str:
        return str(item.get("status") or item.get("support", {}).get("status", "UNKNOWN"))

    recoverable = [
        item for item in queries
        if expected_supported(item) and support_status(before[item["query_id"]]) != "SUPPORTED"
    ]
    recovered = [
        item for item in recoverable
        if support_status(after[item["query_id"]]) == "SUPPORTED"
    ]
    lost = [
        item for item in queries
        if expected_supported(item)
        and support_status(before[item["query_id"]]) == "SUPPORTED"
        and support_status(after[item["query_id"]]) != "SUPPORTED"
    ]
    negatives = [item for item in queries if not expected_supported(item)]
    baseline_false_support = [
        item for item in negatives if support_status(before[item["query_id"]]) == "SUPPORTED"
    ]
    false_support = [
        item for item in negatives if support_status(after[item["query_id"]]) == "SUPPORTED"
    ]
    introduced_false_support = [
        item for item in false_support
        if support_status(before[item["query_id"]]) != "SUPPORTED"
    ]
    return {
        "recoverable_count": len(recoverable),
        "recovered_count": len(recovered),
        "support_recovery_rate": len(recovered) / len(recoverable) if recoverable else 0.0,
        "recovered_query_ids": [item["query_id"] for item in recovered],
        "support_loss_count": len(lost),
        "support_loss_query_ids": [item["query_id"] for item in lost],
        "baseline_false_support_count": len(baseline_false_support),
        "false_support_count": len(false_support),
        "false_support_rate": len(false_support) / len(negatives) if negatives else 0.0,
        "false_support_query_ids": [item["query_id"] for item in false_support],
        "introduced_false_support_count": len(introduced_false_support),
        "introduced_false_support_query_ids": [item["query_id"] for item in introduced_false_support],
    }


def _over_filter_report(before: dict, after: dict) -> dict:
    before_rows = {item["query_id"]: item for item in before.get("rows", [])}
    examples = []
    for row in after.get("rows", []):
        baseline = before_rows.get(row["query_id"], {})
        if baseline.get("rank") is not None and row.get("rank") is None:
            row["failure_type"] = "OVER_FILTER_FAILURE"
            examples.append(row["query_id"])
    after["failure_summary"]["OVER_FILTER_FAILURE"] = len(examples)
    return {"count": len(examples), "examples": examples}


def _model_confusion_audit(manifest: dict, baseline: dict, documents: list[Document]) -> list[dict]:
    queries = {item["query_id"]: item for item in manifest["queries"]}
    reports = baseline["reports"]
    mode_rows = {
        mode: {item["query_id"]: item for item in report["rows"]}
        for mode, report in reports.items()
    }
    evidence_rows = {item["query_id"]: item for item in baseline["evidence"]["rows"]}
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    audit = []
    for rerank_row in reports["hybrid_rerank"]["rows"]:
        if rerank_row.get("failure_type") != "MODEL_CONFUSION":
            continue
        query_id = rerank_row["query_id"]
        query = queries[query_id]
        hybrid = mode_rows["hybrid"][query_id]
        lexical = mode_rows["bm25"][query_id]
        vector = mode_rows["vector"][query_id]
        rerank_by_chunk = {item["chunk_id"]: item["rank"] for item in rerank_row["candidates"]}
        lexical_by_chunk = {item["chunk_id"]: item["rank"] for item in lexical["candidates"]}
        vector_by_chunk = {item["chunk_id"]: item["rank"] for item in vector["candidates"]}
        candidates = []
        for candidate in hybrid["candidates"][:5]:
            identity = identity_from_metadata({"equipment_model": candidate.get("equipment_model", "")})
            candidates.append({
                "chunk_id": candidate["chunk_id"],
                "candidate_identity": identity.as_dict(),
                "lexical_rank": lexical_by_chunk.get(candidate["chunk_id"]),
                "vector_rank": vector_by_chunk.get(candidate["chunk_id"]),
                "fusion_rank": candidate["rank"],
                "rerank_rank": rerank_by_chunk.get(candidate["chunk_id"]),
            })
        relevant_documents = [
            documents_by_chunk[chunk_id]
            for chunk_id in query["relevant_chunk_ids"]
            if chunk_id in documents_by_chunk
        ]
        parsed = evidence_rows[query_id].get("parsed_query_metadata", {})
        relevant_in_hybrid = hybrid.get("rank") is not None
        if not parsed.get("equipment_model") and any(char.isdigit() for char in query["query"]):
            category = "C_MODEL_MISSING_FROM_QUERY_PARSER"
        elif not parsed.get("equipment_model") and not parsed.get("product_series") and not parsed.get("product_family"):
            category = "F_AMBIGUOUS_PRODUCT_IDENTITY"
        elif not relevant_in_hybrid:
            category = "D_RELEVANT_MODEL_CANDIDATE_NEVER_RETRIEVED"
        else:
            category = "A_EXACT_MODEL_QUERY_WRONG_MODEL"
        audit.append({
            "query_id": query_id,
            "query": query["query"],
            "target_identity": (
                identity_from_metadata(relevant_documents[0].metadata).as_dict()
                if relevant_documents else {"equipment_model": query["expected_model"]}
            ),
            "parsed_query_identity": parsed,
            "classification": category,
            "top_5_candidates": candidates,
        })
    return audit


def _evidence_row(query: dict, result, evidence, documents_by_chunk: dict[str, Document]) -> dict:
    analysis = getattr(result, "query_analysis", None)
    row = {
        "query_id": query["query_id"],
        "query": query["query"],
        "category": query.get("category", ""),
        "ood_type": query.get("ood_type", ""),
        "answerable": query["answerable"],
        "parsed_query_metadata": asdict(analysis) if analysis else {},
        "retrieval_scope": (
            result.scope_decision.as_dict()
            if getattr(result, "scope_decision", None) else {}
        ),
        "relevant_chunk_metadata": [
            {
                "chunk_id": chunk_id,
                "document_id": document.metadata.get("document_id", ""),
                "manufacturer": document.metadata.get("manufacturer", ""),
                "equipment_model": document.metadata.get("equipment_model", ""),
                "product_family": document.metadata.get("product_family", ""),
                "product_series": document.metadata.get("product_series", ""),
                "section": document.metadata.get("section", ""),
                "page": document.metadata.get("page"),
                "error_code": document.metadata.get("error_code", ""),
            }
            for chunk_id in query["relevant_chunk_ids"]
            if (document := documents_by_chunk.get(chunk_id)) is not None
        ],
        "top_candidate": _candidate_rows(result.candidates)[:1],
        **evidence.as_dict(),
    }
    if query["answerable"] and row["decision"] == "ABSTAIN":
        row["false_refusal_type"] = FALSE_REFUSAL_REASONS.get(row["reason"], "OTHER")
        row["expected_evidence"] = {
            "chunk_ids": query["relevant_chunk_ids"],
            "document_ids": query.get("relevant_document_ids", []),
            "model": query["expected_model"],
            "error_code": query["expected_error_code"],
            "section": query.get("expected_section", ""),
        }
    return row


def _summarize_evidence(rows: list[dict]) -> dict:
    answerable = [item for item in rows if item["answerable"]]
    ood = [item for item in rows if not item["answerable"]]
    false_refusal = [item for item in answerable if item["decision"] == "ABSTAIN"]
    false_answer = [item for item in ood if item["decision"] == "ANSWER"]
    ood_categories = {}
    for category in sorted({item.get("ood_type") or "unclassified" for item in ood}):
        category_rows = [item for item in ood if (item.get("ood_type") or "unclassified") == category]
        ood_categories[category] = {
            "count": len(category_rows),
            "recall": sum(item["decision"] == "ABSTAIN" for item in category_rows) / len(category_rows),
        }
    return {
        "decision_accuracy": sum((item["decision"] == "ANSWER") == item["answerable"] for item in rows) / len(rows),
        "ood_recall": sum(item["decision"] == "ABSTAIN" for item in ood) / len(ood) if ood else None,
        "answerable_recall": sum(item["decision"] == "ANSWER" for item in answerable) / len(answerable),
        "false_answer_rate": len(false_answer) / len(ood) if ood else None,
        "false_refusal_rate": len(false_refusal) / len(answerable),
        "false_refusals": false_refusal,
        "false_answers": false_answer,
        "false_refusal_taxonomy": dict(Counter(item["false_refusal_type"] for item in false_refusal)),
        "ood_category_metrics": ood_categories,
        "rows": rows,
    }


def _evidence_report(queries: list[dict], documents: list[Document]) -> dict:
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    rows = []
    for query in queries:
        result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode="hybrid")
        evidence = rag_core.analyze_evidence(query["query"], result, "hybrid")
        rows.append(_evidence_row(query, result, evidence, documents_by_chunk))
    return _summarize_evidence(rows)


def _support_candidate_rows(queries: list[dict], documents: list[Document], reranker) -> tuple[list[dict], dict[str, dict]]:
    rows, base_rows = [], {}
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    for query in queries:
        raw = rag_core.retrieve_docs(
            query["query"], k=5,
            knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
            retrieval_mode="hybrid",
        )
        base = rag_core.analyze_evidence(query["query"], raw, "hybrid")
        base_rows[query["query_id"]] = {"decision": base.decision, "reason": base.reason}
        candidates = rag_core.filter_relevant_docs(raw)
        if base.decision == "ANSWER":
            candidates = reranker.rerank(query["query"], candidates, top_k=3).result
        else:
            candidates = RetrievalResult(
                [], query_analysis=getattr(raw, "query_analysis", None),
                corpus_documents=documents, retrieval_mode="hybrid",
                scope_decision=getattr(raw, "scope_decision", None),
            )
        rows.append({
            "query_id": query["query_id"],
            "candidates": _candidate_rows(candidates.candidates),
        })
    return rows, base_rows


def _support_report(
    queries: list[dict],
    candidate_rows: list[dict],
    base_rows: dict[str, dict],
    documents: list[Document],
) -> dict:
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    candidates_by_id = {item["query_id"]: item["candidates"] for item in candidate_rows}
    rows, latencies = [], []
    for query in queries:
        base = base_rows[query["query_id"]]
        candidate_data = candidates_by_id.get(query["query_id"], [])
        candidates = [
            RetrievalCandidate(
                document=documents_by_chunk[item["chunk_id"]],
                retrieval_source="hybrid",
                final_rank=rank,
                pre_rerank_rank=item.get("pre_rerank_rank"),
                rerank_rank=item.get("rerank_rank"),
                identity_relation=item.get("identity_relation", "UNKNOWN"),
                scope_match=item.get("scope_match", "none"),
                scope_level=item.get("scope_level", "GLOBAL_SCOPE"),
            )
            for rank, item in enumerate(candidate_data, start=1)
            if item["chunk_id"] in documents_by_chunk
        ]
        result = RetrievalResult(
            candidates,
            query_analysis=analyze_query(query["query"], documents),
            corpus_documents=documents,
            retrieval_mode="hybrid",
        )
        started = time.perf_counter()
        support = (
            validate_evidence_support(query["query"], result, documents)
            if base["decision"] == "ANSWER"
            else skipped_support()
        )
        latencies.append((time.perf_counter() - started) * 1000)
        expected = bool(query.get("supported", query.get("answerable", False)))
        gate_refused = base["decision"] == "ABSTAIN" or support.status == "INSUFFICIENT"
        predicted_supported = not gate_refused
        if base["decision"] == "ABSTAIN":
            audit_status = "UNSUPPORTED" if not expected else "AMBIGUOUS"
        elif support.status == "SUPPORTED":
            audit_status = "SUPPORTED"
        elif support.status == "INSUFFICIENT":
            audit_status = "PARTIALLY_SUPPORTED" if expected else "UNSUPPORTED"
        else:
            audit_status = "AMBIGUOUS"
        rows.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "expected_supported": expected,
            "base_decision": base["decision"],
            "base_reason": base["reason"],
            "support": support.as_dict(),
            "final_decision": "ABSTAIN" if gate_refused else "ANSWER",
            "predicted_supported": predicted_supported,
            "audit_status": audit_status,
            "top_candidates": [{
                **item,
                "candidate_evidence": " ".join(
                    documents_by_chunk[item["chunk_id"]].page_content.split()
                )[:500] if item["chunk_id"] in documents_by_chunk else "",
            } for item in candidate_data],
        })
    supported = [item for item in rows if item["expected_supported"]]
    unsupported = [item for item in rows if not item["expected_supported"]]
    correct = sum(item["predicted_supported"] == item["expected_supported"] for item in rows)
    false_support = sum(item["predicted_supported"] for item in unsupported)
    false_insufficient = sum(not item["predicted_supported"] for item in supported)
    return {
        "support_accuracy": correct / len(rows) if rows else None,
        "unsupported_recall": (
            sum(not item["predicted_supported"] for item in unsupported) / len(unsupported)
            if unsupported else None
        ),
        "supported_recall": (
            sum(item["predicted_supported"] for item in supported) / len(supported)
            if supported else None
        ),
        "false_support_rate": false_support / len(unsupported) if unsupported else None,
        "false_insufficient_rate": false_insufficient / len(supported) if supported else None,
        "rule_latency_ms_median": statistics.median(latencies) if latencies else None,
        "status_distribution": dict(Counter(item["support"]["status"] for item in rows)),
        "rows": rows,
    }


def _frozen_support_report(
    queries: list[dict],
    retrieval_report: dict,
    evidence_report: dict,
    documents: list[Document],
) -> dict:
    base_rows = {
        item["query_id"]: {"decision": item["decision"], "reason": item["reason"]}
        for item in evidence_report["rows"]
    }
    return _support_report(queries, retrieval_report["rows"], base_rows, documents)


def _support_gate_evidence_summary(base: dict, support: dict) -> dict:
    rows_by_id = {item["query_id"]: item for item in support["rows"]}
    rows = []
    for item in base["rows"]:
        support_row = rows_by_id[item["query_id"]]
        row = {
            **item,
            "decision": support_row["final_decision"],
            "support": support_row["support"],
        }
        if row["answerable"] and row["decision"] == "ABSTAIN" and "false_refusal_type" not in row:
            row["false_refusal_type"] = "SUPPORT_INSUFFICIENT"
            row["expected_evidence"] = {
                "chunk_ids": [],
                "document_ids": [],
                "model": "",
                "error_code": "",
                "section": "",
            }
        rows.append(row)
    return _summarize_evidence(rows)


def _calibration_report(queries: list[dict], documents: list[Document]) -> dict:
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    before_rows, after_rows = [], []
    for query in queries:
        result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode="hybrid")
        before = analyze_retrieval_evidence(
            query["query"], result, documents, "hybrid", identity_matching=False,
        )
        after = analyze_retrieval_evidence(query["query"], result, documents, "hybrid")
        before_rows.append(_evidence_row(query, result, before, documents_by_chunk))
        after_rows.append(_evidence_row(query, result, after, documents_by_chunk))
    before_report, after_report = _summarize_evidence(before_rows), _summarize_evidence(after_rows)
    metric_names = (
        "decision_accuracy", "ood_recall", "answerable_recall",
        "false_answer_rate", "false_refusal_rate",
    )
    return {
        "before": before_report,
        "after": after_report,
        "delta": {name: after_report[name] - before_report[name] for name in metric_names},
    }


def run_private_calibration(manifest_path: Path) -> dict:
    manifest = load_private_manifest(manifest_path)
    ingested, _ = ingest_private_documents(manifest_path, manifest)
    calibration_path = manifest_path.parent / "annotations" / "v32_calibration.json"
    calibration = load_private_calibration(
        calibration_path,
        manifest,
        {item.metadata["chunk_id"] for item in ingested},
    )
    with full_vector_knowledge_base(ingested):
        report = _calibration_report(calibration["queries"], ingested)
    return {
        "name": calibration.get("name", "v3.2-private-calibration"),
        "corpus_annotation_sha256": annotation_hash(manifest),
        "documents": len(manifest["documents"]),
        "answerable": sum(item["answerable"] for item in calibration["queries"]),
        "ood": sum(not item["answerable"] for item in calibration["queries"]),
        "total": len(calibration["queries"]),
        "hash": calibration_hash(calibration),
        **report,
    }


def run_private_benchmark(manifest_path: Path) -> dict:
    manifest = load_private_manifest(manifest_path)
    ingested, audit = ingest_private_documents(manifest_path, manifest)
    ingested_chunk_ids = {item.metadata["chunk_id"] for item in ingested}
    unknown = set().union(*(set(item["relevant_chunk_ids"]) for item in manifest["queries"])) - ingested_chunk_ids
    if unknown:
        raise ValueError(f"Annotation references chunks absent after Industrial Ingestion: {sorted(unknown)}")
    calibration_path = manifest_path.parent / "annotations" / "v32_calibration.json"
    calibration = (
        load_private_calibration(calibration_path, manifest, ingested_chunk_ids)
        if calibration_path.exists()
        else None
    )
    development_path = manifest_path.parent / "annotations" / "v33_development.json"
    development = (
        load_private_development(development_path, manifest, ingested_chunk_ids)
        if development_path.exists()
        else None
    )
    support_calibration_path = manifest_path.parent / "annotations" / "v34_support_calibration.json"
    support_calibration = (
        load_support_calibration(support_calibration_path, manifest, ingested_chunk_ids)
        if support_calibration_path.exists()
        else None
    )
    section_development_path = manifest_path.parent / "annotations" / "v35_section_development.json"
    section_development = (
        load_section_development(section_development_path, manifest, ingested_chunk_ids)
        if section_development_path.exists()
        else None
    )
    benchmark = {"documents": [{"content": item.page_content, "metadata": dict(item.metadata)} for item in ingested]}
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=5, top_k=3, device="cpu"))
    with ExitStack() as stack:
        light_rag, light_id = stack.enter_context(benchmark_knowledge_base(benchmark))
        stack.enter_context(full_vector_knowledge_base(ingested))
        reports = {mode: _run_mode(mode, manifest["queries"], light_rag, light_id, reranker) for mode in PRIVATE_MODES}
        evidence = _evidence_report(manifest["queries"], ingested)
        calibration_report = _calibration_report(calibration["queries"], ingested) if calibration else None
        development_reports = (
            {
                mode: _run_mode(mode, development["queries"], light_rag, light_id, reranker)
                for mode in ("hybrid", "hybrid_rerank")
            }
            if development else None
        )
        section_development_reports = (
            {
                mode: _run_mode(mode, section_development["queries"], light_rag, light_id, reranker)
                for mode in ("hybrid_rerank", "hybrid_section_rerank")
            }
            if section_development else None
        )
        trace_frozen_reports = {
            mode: _run_mode(
                mode, manifest["queries"], light_rag, light_id, reranker,
                trace_enabled=True,
            )
            for mode in ("hybrid_rerank", "hybrid_section_rerank")
        }
        trace_section_development_reports = (
            {
                mode: _run_mode(
                    mode, section_development["queries"], light_rag, light_id, reranker,
                    trace_enabled=True,
                )
                for mode in ("hybrid_rerank", "hybrid_section_rerank")
            }
            if section_development else None
        )
        for mode, traced in trace_frozen_reports.items():
            _assert_trace_equivalence(reports[mode], traced)
        if trace_section_development_reports:
            for mode, traced in trace_section_development_reports.items():
                _assert_trace_equivalence(section_development_reports[mode], traced)
        measured_trace_overhead = _measure_trace_overhead(
            manifest["queries"], light_rag, light_id, reranker,
        )
        support_candidate_rows, support_base_rows = (
            _support_candidate_rows(support_calibration["queries"], ingested, reranker)
            if support_calibration else (None, None)
        )
    frozen_observability = analyze_observability(
        manifest["queries"],
        trace_frozen_reports["hybrid_rerank"],
        trace_frozen_reports["hybrid_section_rerank"],
    )
    development_observability = (
        analyze_observability(
            section_development["queries"],
            trace_section_development_reports["hybrid_rerank"],
            trace_section_development_reports["hybrid_section_rerank"],
        )
        if section_development and trace_section_development_reports else None
    )
    trace_artifacts = write_trace_artifacts(
        manifest_path.parent / "annotations" / "v36_runtime",
        {
            "frozen": frozen_observability,
            **({"development": development_observability} if development_observability else {}),
        },
    )
    frozen_support = _frozen_support_report(
        manifest["queries"], reports["hybrid_rerank"], evidence, ingested,
    )
    section_frozen_support = _frozen_support_report(
        manifest["queries"], reports["hybrid_section_rerank"], evidence, ingested,
    )
    support_calibration_report = (
        _support_report(
            support_calibration["queries"], support_candidate_rows,
            support_base_rows, ingested,
        )
        if support_calibration else None
    )
    v34_evidence = _support_gate_evidence_summary(evidence, frozen_support)
    baseline_path = manifest_path.parent / "annotations" / "v32_final.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    over_filter = (
        {
            mode: _over_filter_report(baseline["reports"][mode], reports[mode])
            for mode in CORE_PRIVATE_MODES
        }
        if baseline else {"status": "NOT_RUN"}
    )
    confusion_audit = _model_confusion_audit(manifest, baseline, ingested) if baseline else []
    counts = Counter(item["document_type"] for item in manifest["documents"])
    return {
        "dataset": manifest.get("name", "private-real-corpus"),
        "status": "READY" if len(manifest["documents"]) >= 3 and len(manifest["queries"]) >= 30 else "PARTIAL",
        "real_corpus_gate": "PASS" if len(manifest["documents"]) >= 3 and len(manifest["queries"]) >= 30 else "PARTIAL",
        "freeze": {"annotation_sha256": annotation_hash(manifest), **manifest.get("freeze", {})},
        "corpus": {
            "documents": len(manifest["documents"]), "pages": sum(item["pages"] for item in audit.values()),
            "chunks": len(ingested), "manufacturers": len({item["manufacturer"] for item in manifest["documents"]}),
            "equipment_models": len({item["equipment_model"] for item in manifest["documents"]}),
            "document_type_distribution": dict(counts),
            "language_distribution": dict(Counter(item["language"] for item in manifest["documents"])),
        },
        "parser_audit": audit,
        "query_distribution": dict(Counter(item["category"] for item in manifest["queries"])),
        "reports": reports,
        "over_filter": over_filter,
        "model_confusion_audit": confusion_audit,
        "baseline": ({
            "source": baseline_path.name,
            "reports": {
                mode: {
                    "overall": baseline["reports"][mode]["overall"],
                    "failure_summary": baseline["reports"][mode]["failure_summary"],
                    "latency_ms_median": baseline["reports"][mode]["latency_ms_median"],
                    "comparison_coverage_at_5": baseline["reports"][mode].get("comparison_coverage_at_5"),
                }
                for mode in CORE_PRIVATE_MODES
            },
            "evidence": {
                key: baseline["evidence"][key]
                for key in (
                    "decision_accuracy", "ood_recall", "answerable_recall",
                    "false_answer_rate", "false_refusal_rate",
                )
            },
        } if baseline else {"status": "NOT_RUN"}),
        "evidence": evidence,
        "v34_evidence": v34_evidence,
        "support_frozen": frozen_support,
        "section_support_frozen": section_frozen_support,
        "section_support_recovery": _support_recovery(
            manifest["queries"], frozen_support, section_frozen_support,
        ),
        "section_metrics": _section_metrics(
            manifest["queries"], reports["hybrid_rerank"], reports["hybrid_section_rerank"],
        ),
        "v36_observability": {
            "frozen": _observability_summary(frozen_observability),
            "development": (
                _observability_summary(development_observability)
                if development_observability else {"status": "NOT_RUN"}
            ),
            "trace_overhead": {
                "measured": measured_trace_overhead,
                "full_run_order_effect": _trace_overhead(
                    reports["hybrid_section_rerank"], trace_frozen_reports["hybrid_section_rerank"],
                ),
                "development": (
                    _trace_overhead(
                        section_development_reports["hybrid_section_rerank"],
                        trace_section_development_reports["hybrid_section_rerank"],
                    )
                    if section_development_reports and trace_section_development_reports
                    else {"status": "NOT_RUN"}
                ),
            },
            "artifacts": trace_artifacts,
        },
        "calibration": ({
            "name": calibration.get("name", "v3.2-private-calibration"),
            "documents": len(manifest["documents"]),
            "answerable": sum(item["answerable"] for item in calibration["queries"]),
            "ood": sum(not item["answerable"] for item in calibration["queries"]),
            "total": len(calibration["queries"]),
            "hash": calibration_hash(calibration),
            **calibration_report,
        } if calibration else {"status": "NOT_RUN"}),
        "development": ({
            "name": development.get("name", "v3.3-private-development"),
            "count": len(development["queries"]),
            "categories": dict(Counter(item["category"] for item in development["queries"])),
            "hash": development_hash(development),
            "reports": development_reports,
        } if development else {"status": "NOT_RUN"}),
        "support_calibration": ({
            "name": support_calibration.get("name", "v3.4-evidence-support-calibration"),
            "supported": sum(item["supported"] for item in support_calibration["queries"]),
            "unsupported": sum(not item["supported"] for item in support_calibration["queries"]),
            "total": len(support_calibration["queries"]),
            "hash": support_calibration_hash(support_calibration),
            **support_calibration_report,
        } if support_calibration else {"status": "NOT_RUN"}),
        "section_development": ({
            "name": section_development.get("name", "v3.5-section-development"),
            "count": len(section_development["queries"]),
            "categories": dict(Counter(item["category"] for item in section_development["queries"])),
            "hard_positives": dict(Counter(item["hard_positive"] for item in section_development["queries"])),
            "hash": section_development_hash(section_development),
            "reports": section_development_reports,
            "section_metrics": _section_metrics(
                section_development["queries"],
                section_development_reports["hybrid_rerank"],
                section_development_reports["hybrid_section_rerank"],
            ),
            "support": (
                lambda before, after: {
                    "baseline": before,
                    "section": after,
                    "recovery": _support_recovery(section_development["queries"], before, after),
                }
            )(
                _report_support(section_development["queries"], section_development_reports["hybrid_rerank"], ingested),
                _report_support(section_development["queries"], section_development_reports["hybrid_section_rerank"], ingested),
            ),
        } if section_development else {"status": "NOT_RUN"}),
    }
