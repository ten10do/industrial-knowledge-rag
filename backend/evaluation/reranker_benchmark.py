"""V3 Cross-Encoder A/B benchmark over the frozen V2.6 challenge set."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.benchmark_runner import CHALLENGE_PATH, _benchmark_documents
from backend.evaluation.benchmark_schema import load_manifest, rank_of
from backend.evaluation.full_vector_benchmark import (
    FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
    full_vector_knowledge_base,
)
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SOURCES = ("vector", "hybrid")
EXPERIMENTS = ((5, 3), (5, 5), (10, 3), (10, 5))


@contextmanager
def _candidate_limits(candidate_k: int):
    names = ("LEXICAL_TOP_K", "VECTOR_TOP_K", "HYBRID_TOP_K")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(max(10, candidate_k) if name != "HYBRID_TOP_K" else candidate_k)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _candidate_row(candidate, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.metadata.get("document_id", ""),
        "section": candidate.metadata.get("section", ""),
        "equipment_model": candidate.metadata.get("equipment_model", ""),
        "error_code": candidate.metadata.get("error_code", ""),
        "original_rank": candidate.pre_rerank_rank or candidate.final_rank or rank,
        "lexical_rank": candidate.lexical_rank,
        "vector_rank": candidate.vector_rank,
        "fusion_score": candidate.fusion_score,
        "rerank_score": candidate.rerank_score,
        "rerank_rank": candidate.rerank_rank,
    }


def _metrics(rows: list[dict], rank_key: str) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    ranks = [row[rank_key] for row in answerable]
    return {
        "count": len(answerable),
        "hit_rate_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "hit_rate_at_3": sum((rank or 99) <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum((rank or 99) <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks if rank) / len(ranks),
    }


def _category_delta(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["answerable"]:
            grouped[row["category"]].append(row)
    return {
        category: {
            "count": len(items),
            "before_hit_at_1": sum(item["rank_before"] == 1 for item in items) / len(items),
            "after_hit_at_1": sum(item["rank_after"] == 1 for item in items) / len(items),
            "delta": (
                sum(item["rank_after"] == 1 for item in items)
                - sum(item["rank_before"] == 1 for item in items)
            ) / len(items),
        }
        for category, items in sorted(grouped.items())
    }


def _rank_result(before: int | None, after: int | None) -> str:
    if before is None:
        return "MISSING"
    if after is None or after > before:
        return "DEGRADED"
    if after < before:
        return "IMPROVED"
    return "SAME"


def _run_experiment(manifest: dict, source: str, candidate_k: int, final_k: int, reranker) -> dict:
    rows, retrieval_latencies, rerank_latencies, end_to_end = [], [], [], []
    with _candidate_limits(candidate_k):
        for query in manifest["queries"]:
            started = time.perf_counter()
            retrieval_started = time.perf_counter()
            raw = rag_core.retrieve_docs(
                query["query"], k=candidate_k,
                knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                retrieval_mode=source,
            )
            retrieval_latencies.append((time.perf_counter() - retrieval_started) * 1000)
            candidates = rag_core.filter_relevant_docs(raw)
            before = [_candidate_row(item, rank) for rank, item in enumerate(candidates.candidates, start=1)]
            rank_before = rank_of(before, query["relevant_chunk_ids"])
            evidence = rag_core.analyze_evidence(query["query"], raw, source)
            reranker_called = False
            if evidence.decision == "ANSWER":
                rerank_started = time.perf_counter()
                outcome = reranker.rerank(query["query"], candidates, top_k=final_k)
                rerank_latencies.append((time.perf_counter() - rerank_started) * 1000)
                reranker_called = True
                after_candidates = outcome.result.candidates
            else:
                outcome = None
                after_candidates = candidates.candidates[:final_k]
            after = [_candidate_row(item, rank) for rank, item in enumerate(after_candidates, start=1)]
            rank_after = rank_of(after, query["relevant_chunk_ids"])
            result = _rank_result(rank_before, rank_after)
            taxonomy = {
                "IMPROVED": "RERANK_IMPROVED",
                "SAME": "RERANK_UNCHANGED",
                "DEGRADED": "RERANK_DEGRADED",
                "MISSING": "RERANK_CANDIDATE_MISSING",
            }[result]
            rows.append({
                "query_id": query["query_id"],
                "query": query["query"],
                "category": query["category"],
                "answerable": query["answerable"],
                "expected_chunks": query["relevant_chunk_ids"],
                "candidate_source": source,
                "candidate_recall": rank_before is not None,
                "before_top_5": before[:5],
                "after_top_5": after[:5],
                "rank_before": rank_before,
                "rank_after": rank_after,
                "rank_delta": (rank_before - rank_after) if rank_before and rank_after else None,
                "result": result,
                "taxonomy": taxonomy,
                "evidence_decision": evidence.decision,
                "reranker_called": reranker_called,
                "reranker_effective": outcome.reranker_effective if outcome else False,
            })
            end_to_end.append((time.perf_counter() - started) * 1000)
    answerable = [row for row in rows if row["answerable"]]
    counts = {
        "improved": sum(row["result"] == "IMPROVED" for row in answerable),
        "same": sum(row["result"] == "SAME" for row in answerable),
        "degraded": sum(row["result"] == "DEGRADED" for row in answerable),
        "missing": sum(row["result"] == "MISSING" for row in answerable),
    }
    comparable = [row["rank_delta"] for row in answerable if row["rank_delta"] is not None]
    return {
        "source": source,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "candidate_recall": sum(row["candidate_recall"] for row in answerable) / len(answerable),
        "before_metrics": _metrics(rows, "rank_before"),
        "after_metrics": _metrics(rows, "rank_after"),
        "rank_analysis": {
            **counts,
            "win_rate": counts["improved"] / len(answerable),
            "tie_rate": counts["same"] / len(answerable),
            "loss_rate": counts["degraded"] / len(answerable),
            "mean_rank_delta": statistics.mean(comparable) if comparable else 0.0,
        },
        "category_delta": _category_delta(rows),
        "latency_ms": {
            "retrieval_median": statistics.median(retrieval_latencies),
            "rerank_median": statistics.median(rerank_latencies) if rerank_latencies else 0.0,
            "end_to_end_median": statistics.median(end_to_end),
        },
        "rows": rows,
    }


def run_real_reranker_smoke() -> dict:
    reranker = CrossEncoderReranker(
        RerankerConfig(enabled=True, model_name=MODEL_NAME, candidate_k=5, top_k=3, device="cpu")
    )
    started = time.perf_counter()
    reranker.load()
    cold_load_seconds = time.perf_counter() - started
    documents = [
        Document(page_content="G120 F0001 input phase failure", metadata={"chunk_id": "F0001"}),
        Document(page_content="G120 F0002 DC-link overvoltage during deceleration", metadata={"chunk_id": "F0002"}),
        Document(page_content="S7-1200 MW20 memory word", metadata={"chunk_id": "MW20"}),
    ]
    from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult

    result = RetrievalResult([
        RetrievalCandidate(document=document, retrieval_source="vector", final_rank=rank)
        for rank, document in enumerate(documents, start=1)
    ])
    outcome = reranker.rerank("G120 减速时报 F0002 直流母线过压", result, top_k=3)
    return {
        "model": MODEL_NAME,
        "device": "cpu",
        "cold_load_seconds": cold_load_seconds,
        "real_scores": [item.rerank_score for item in outcome.result.candidates],
        "ranking": [item.chunk_id for item in outcome.result.candidates],
        "effective": outcome.reranker_effective,
    }


def run_reranker_benchmark() -> dict:
    manifest = load_manifest(CHALLENGE_PATH)
    benchmark = _benchmark_documents(CHALLENGE_PATH, manifest)
    documents = [
        Document(page_content=item["content"], metadata=dict(item["metadata"]))
        for item in benchmark["documents"]
    ]
    reranker = CrossEncoderReranker(
        RerankerConfig(enabled=True, model_name=MODEL_NAME, candidate_k=10, top_k=5, device="cpu")
    )
    cold_started = time.perf_counter()
    reranker.load()
    cold_load_seconds = time.perf_counter() - cold_started
    with ExitStack() as stack:
        stack.enter_context(full_vector_knowledge_base(documents))
        experiments = [
            _run_experiment(manifest, source, candidate_k, final_k, reranker)
            for source in SOURCES
            for candidate_k, final_k in EXPERIMENTS
        ]
    return {
        "model": MODEL_NAME,
        "device": "cpu",
        "cold_load_seconds": cold_load_seconds,
        "experiments": experiments,
    }


if __name__ == "__main__":
    print(json.dumps(run_reranker_benchmark(), ensure_ascii=False, indent=2))
