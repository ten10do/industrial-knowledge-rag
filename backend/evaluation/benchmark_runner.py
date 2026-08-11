"""Unified, repeatable runner for synthetic, challenge, and private benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.benchmark_schema import evaluate_rows, load_manifest, rank_of
from backend.evaluation.full_vector_benchmark import (
    FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
    full_vector_knowledge_base,
)
from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base, run_benchmark


CHALLENGE_PATH = Path(__file__).resolve().parent / "fixtures" / "industrial_challenge.json"
PRIVATE_PATH = Path(__file__).resolve().parent / "benchmark_private" / "manifest.json"
MODES = ("tfidf", "bm25", "vector", "hybrid")


def _file_content(manifest_path: Path, document: dict) -> str:
    source = (manifest_path.parent / document["file"]).resolve()
    root = manifest_path.parent.resolve()
    if root not in source.parents or not source.exists():
        raise ValueError(f"Private benchmark file is unavailable: {document['file']}")
    if source.suffix.lower() == ".pdf":
        return "\n".join(page.page_content for page in rag_core.load_pdf(source))
    return source.read_text(encoding="utf-8")


def _benchmark_documents(manifest_path: Path, manifest: dict) -> dict:
    return {
        "documents": [
            {
                "content": document.get("content") or _file_content(manifest_path, document),
                "metadata": {
                    key: value for key, value in document.items() if key != "content"
                },
            }
            for document in manifest["documents"]
        ]
    }


def _candidate_rows(candidates: list) -> list[dict]:
    return [
        {
            "rank": candidate.final_rank or rank,
            "chunk_id": candidate.chunk_id,
            "document_id": candidate.metadata.get("document_id", ""),
            "section": candidate.metadata.get("section", ""),
            "equipment_model": candidate.metadata.get("equipment_model", ""),
            "error_code": candidate.metadata.get("error_code", ""),
            "retrieval_source": candidate.retrieval_source,
            "lexical_rank": candidate.lexical_rank,
            "vector_rank": candidate.vector_rank,
            "fusion_score": candidate.fusion_score,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def run_manifest_benchmark(manifest_path: Path, modes: tuple[str, ...] = MODES) -> dict:
    """Run each retrieval mode on exactly the same annotated chunks and queries."""
    manifest = load_manifest(manifest_path)
    benchmark = _benchmark_documents(manifest_path, manifest)
    reports = {}
    with ExitStack() as stack:
        light_rag, light_id = stack.enter_context(benchmark_knowledge_base(benchmark))
        full_documents = [
            Document(page_content=item["content"], metadata=dict(item["metadata"]))
            for item in benchmark["documents"]
        ]
        stack.enter_context(full_vector_knowledge_base(full_documents))
        for mode in modes:
            rows, latencies = [], []
            for query in manifest["queries"]:
                started = time.perf_counter()
                if mode == "tfidf":
                    raw = light_rag.retrieve_docs(query["query"], k=5, knowledge_base_id=light_id, retrieval_mode="vector")
                    relevant = light_rag.filter_relevant_docs(raw)
                elif mode == "bm25":
                    raw = light_rag.retrieve_docs(query["query"], k=5, knowledge_base_id=light_id, retrieval_mode="lexical")
                    relevant = light_rag.filter_relevant_docs(raw)
                else:
                    raw = rag_core.retrieve_docs(
                        query["query"], k=5,
                        knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                        retrieval_mode=mode,
                    )
                    relevant = rag_core.filter_relevant_docs(raw)
                latencies.append((time.perf_counter() - started) * 1000)
                candidates = _candidate_rows(relevant.candidates)
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "expected_chunk_ids": query["relevant_chunk_ids"],
                        "rank": rank_of(candidates, query["relevant_chunk_ids"]),
                        "refused": not candidates,
                        "candidates": candidates,
                    }
                )
            report = evaluate_rows(manifest["queries"], rows)
            report["latency_ms_median"] = statistics.median(latencies)
            report["failed_queries"] = [row for row in report["rows"] if row["failure_type"]]
            reports[mode] = report
    return {
        "dataset": manifest["name"],
        "dataset_path": str(manifest_path),
        "document_count": len(manifest["documents"]),
        "query_count": len(manifest["queries"]),
        "backend": "TF-IDF/BM25 light index; HuggingFace embeddings + Chroma for vector/hybrid",
        "reports": reports,
    }


def run_dataset(dataset: str, modes: tuple[str, ...] = MODES) -> dict:
    if dataset == "challenge":
        return run_manifest_benchmark(CHALLENGE_PATH, modes)
    if dataset == "private":
        if not PRIVATE_PATH.exists():
            return {
                "dataset": "private",
                "status": "REAL_CORPUS_GATE_NOT_RUN",
                "reason": "No local ignored manifest at backend/evaluation/benchmark_private/manifest.json.",
            }
        from backend.evaluation.private_benchmark import run_private_benchmark

        if modes != MODES:
            raise ValueError("Private real-corpus validation must run every frozen pipeline.")
        return run_private_benchmark(PRIVATE_PATH)
    synthetic = run_benchmark()
    requested = {"tfidf": "legacy_tfidf", "bm25": "bm25", "hybrid": "hybrid"}
    return {
        "dataset": "synthetic",
        "note": "Legacy deterministic fixture; it does not represent real-world accuracy.",
        "reports": {mode: synthetic["reports"][requested[mode]] for mode in modes if mode in requested},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "challenge", "private"), required=True)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--output", type=Path, help="Optional caller-selected JSON output path.")
    args = parser.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    report = run_dataset(args.dataset, modes)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
