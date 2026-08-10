"""Local-only V3.1 validation over vendor documents parsed by Industrial Ingestion."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
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
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.evidence import analyze_retrieval_evidence


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


def calibration_hash(calibration: dict) -> str:
    payload = json.dumps(calibration["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
                "answerable": query["answerable"],
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
        report["rerank_analysis"] = _rerank_analysis(rerank_rows)
    return report


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


def _evidence_row(query: dict, result, evidence, documents_by_chunk: dict[str, Document]) -> dict:
    analysis = getattr(result, "query_analysis", None)
    row = {
        "query_id": query["query_id"],
        "query": query["query"],
        "category": query.get("category", ""),
        "ood_type": query.get("ood_type", ""),
        "answerable": query["answerable"],
        "parsed_query_metadata": asdict(analysis) if analysis else {},
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
    benchmark = {"documents": [{"content": item.page_content, "metadata": dict(item.metadata)} for item in ingested]}
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=5, top_k=3, device="cpu"))
    with ExitStack() as stack:
        light_rag, light_id = stack.enter_context(benchmark_knowledge_base(benchmark))
        stack.enter_context(full_vector_knowledge_base(ingested))
        reports = {mode: _run_mode(mode, manifest["queries"], light_rag, light_id, reranker) for mode in PRIVATE_MODES}
        evidence = _evidence_report(manifest["queries"], ingested)
        calibration_report = _calibration_report(calibration["queries"], ingested) if calibration else None
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
        "calibration": ({
            "name": calibration.get("name", "v3.2-private-calibration"),
            "documents": len(manifest["documents"]),
            "answerable": sum(item["answerable"] for item in calibration["queries"]),
            "ood": sum(not item["answerable"] for item in calibration["queries"]),
            "total": len(calibration["queries"]),
            "hash": calibration_hash(calibration),
            **calibration_report,
        } if calibration else {"status": "NOT_RUN"}),
    }
