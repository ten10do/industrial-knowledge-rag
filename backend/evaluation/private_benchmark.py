"""Local-only V3.1 validation over vendor documents parsed by Industrial Ingestion."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.benchmark_schema import evaluate_rows, rank_of
from backend.evaluation.full_vector_benchmark import (
    FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
    full_vector_knowledge_base,
)
from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig


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
PRIVATE_MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")
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
            metadata.update({
                "document_id": entry["document_id"],
                "source": entry["source_name"],
                "source_name": entry["source_name"],
                "source_type": entry["source_type"],
                "manufacturer": entry["manufacturer"],
                "equipment_type": entry["equipment_type"],
                "equipment_model": entry["equipment_model"],
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
        "section": candidate.metadata.get("section", ""),
        "equipment_model": candidate.metadata.get("equipment_model", ""),
        "error_code": candidate.metadata.get("error_code", ""),
        "vector_distance": candidate.vector_score,
        "lexical_score": candidate.lexical_score,
        "pre_rerank_rank": candidate.pre_rerank_rank,
        "rerank_rank": candidate.rerank_rank,
    } for rank, candidate in enumerate(candidates, start=1)]


def _query_for_schema(query: dict) -> dict:
    return {
        **query,
        "query_type": query["category"],
        "expected_equipment_model": query["expected_model"],
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
            identifier["fault_code" if item["expected_error_code"].upper().startswith(("F", "A")) else "parameter_or_register"].append(item)

    def top1_rate(items: list[dict]) -> float | None:
        return (sum(by_id[item["query_id"]]["rank"] == 1 for item in items) / len(items)) if items else None

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
    }


def _run_mode(mode: str, queries: list[dict], light_rag, light_id, reranker) -> dict:
    rows, latencies, rerank_rows = [], [], []
    for query in queries:
        started = time.perf_counter()
        if mode == "bm25":
            result = light_rag.retrieve_docs(query["query"], k=5, knowledge_base_id=light_id, retrieval_mode="lexical")
            candidates = light_rag.filter_relevant_docs(result)
        else:
            retrieval_mode = "hybrid" if mode == "hybrid_rerank" else mode
            result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode=retrieval_mode)
            candidates = rag_core.filter_relevant_docs(result)
        before = _candidate_rows(candidates.candidates)
        if mode == "hybrid_rerank":
            outcome = reranker.rerank(query["query"], candidates, top_k=3)
            selected = outcome.result.candidates
        else:
            outcome, selected = None, candidates.candidates
        candidate_rows = _candidate_rows(selected)
        latencies.append((time.perf_counter() - started) * 1000)
        before_rank = rank_of(before, query["relevant_chunk_ids"])
        rank = rank_of(candidate_rows, query["relevant_chunk_ids"])
        if outcome:
            rerank_rows.append({
                "query_id": query["query_id"], "before_rank": before_rank, "after_rank": rank,
                "candidate_missing": before_rank is None,
            })
        rows.append({
            "query_id": query["query_id"], "query": query["query"],
            "rank": rank, "refused": not candidate_rows, "candidates": candidate_rows,
            "candidate_ids": [item["chunk_id"] for item in candidate_rows],
        })
    report = _summary_metrics(queries, rows)
    report["latency_ms_median"] = statistics.median(latencies)
    report["latency_ms_p95"] = sorted(latencies)[max(0, round(len(latencies) * .95) - 1)]
    if rerank_rows:
        comparable = [item for item in rerank_rows if not item["candidate_missing"]]
        improved = sum(item["after_rank"] and item["after_rank"] < item["before_rank"] for item in comparable)
        degraded = sum(item["after_rank"] is None or item["after_rank"] > item["before_rank"] for item in comparable)
        same = len(comparable) - improved - degraded
        deltas = [item["before_rank"] - item["after_rank"] for item in comparable if item["after_rank"]]
        report["rerank_analysis"] = {
            "improved": improved, "same": same, "degraded": degraded,
            "candidate_missing": len(rerank_rows) - len(comparable),
            "win_rate": improved / len(comparable) if comparable else 0.0,
            "tie_rate": same / len(comparable) if comparable else 0.0,
            "loss_rate": degraded / len(comparable) if comparable else 0.0,
            "mean_rank_delta": statistics.mean(deltas) if deltas else 0.0,
            "rows": rerank_rows,
        }
    return report


def _evidence_report(queries: list[dict], documents: list[Document]) -> dict:
    rows = []
    for query in queries:
        result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode="hybrid")
        evidence = rag_core.analyze_evidence(query["query"], result, "hybrid")
        row = {"query_id": query["query_id"], "query": query["query"], "answerable": query["answerable"], **evidence.as_dict()}
        if query["answerable"] and row["decision"] == "ABSTAIN":
            row["false_refusal_type"] = FALSE_REFUSAL_REASONS.get(row["reason"], "OTHER")
            row["top_candidate"] = _candidate_rows(result.candidates)[:1]
            row["expected_evidence"] = {
                "chunk_ids": query["relevant_chunk_ids"],
                "document_ids": query["relevant_document_ids"],
                "model": query["expected_model"],
                "error_code": query["expected_error_code"],
                "section": query["expected_section"],
            }
        rows.append(row)
    answerable = [item for item in rows if item["answerable"]]
    ood = [item for item in rows if not item["answerable"]]
    false_refusal = [item for item in answerable if item["decision"] == "ABSTAIN"]
    false_answer = [item for item in ood if item["decision"] == "ANSWER"]
    return {
        "decision_accuracy": sum((item["decision"] == "ANSWER") == item["answerable"] for item in rows) / len(rows),
        "ood_recall": sum(item["decision"] == "ABSTAIN" for item in ood) / len(ood) if ood else None,
        "answerable_recall": sum(item["decision"] == "ANSWER" for item in answerable) / len(answerable),
        "false_answer_rate": len(false_answer) / len(ood) if ood else None,
        "false_refusal_rate": len(false_refusal) / len(answerable),
        "false_refusals": false_refusal,
        "false_refusal_taxonomy": dict(Counter(item["false_refusal_type"] for item in false_refusal)),
    }


def run_private_benchmark(manifest_path: Path) -> dict:
    manifest = load_private_manifest(manifest_path)
    ingested, audit = ingest_private_documents(manifest_path, manifest)
    unknown = set().union(*(set(item["relevant_chunk_ids"]) for item in manifest["queries"])) - {item.metadata["chunk_id"] for item in ingested}
    if unknown:
        raise ValueError(f"Annotation references chunks absent after Industrial Ingestion: {sorted(unknown)}")
    benchmark = {"documents": [{"content": item.page_content, "metadata": dict(item.metadata)} for item in ingested]}
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=5, top_k=3, device="cpu"))
    with ExitStack() as stack:
        light_rag, light_id = stack.enter_context(benchmark_knowledge_base(benchmark))
        stack.enter_context(full_vector_knowledge_base(ingested))
        reports = {mode: _run_mode(mode, manifest["queries"], light_rag, light_id, reranker) for mode in PRIVATE_MODES}
        evidence = _evidence_report(manifest["queries"], ingested)
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
        "evidence": evidence,
    }
