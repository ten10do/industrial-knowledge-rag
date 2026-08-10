"""Real HuggingFace + Chroma validation for the V2 retrieval fixture."""

from __future__ import annotations

import json
import statistics
import time
import gc
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.retrieval_benchmark import (
    benchmark_knowledge_base,
    load_benchmark,
)
from backend.ingestion import PageText, ingest_pages


FULL_BENCHMARK_KNOWLEDGE_BASE_ID = "v25-full-vector-benchmark"
SMOKE_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "ingestion" / "fixtures"
    / "industrial_documents.json"
)


def _document(item: dict) -> Document:
    return Document(page_content=item["content"], metadata=dict(item["metadata"]))


@contextmanager
def full_vector_knowledge_base(documents: list[Document]):
    """Create a real, temporary Chroma collection without touching runtime DBs."""
    previous_persist_dir = rag_core.PERSIST_DIR
    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        rag_core.PERSIST_DIR = Path(temp_dir) / "vector_db"
        persist_dir = rag_core.get_persist_dir(FULL_BENCHMARK_KNOWLEDGE_BASE_ID)
        vector_db = rag_core.build_vector_db(documents, persist_dir)
        try:
            yield persist_dir, vector_db
        finally:
            try:
                vector_db.delete_collection()
            except Exception:
                pass
            client = getattr(vector_db, "_client", None)
            if client is not None:
                try:
                    client.clear_system_cache()
                    client.close()
                except Exception:
                    pass
            del vector_db
            gc.collect()
            rag_core.PERSIST_DIR = previous_persist_dir


def _rank(candidates, expected_chunk_id: str) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate.chunk_id == expected_chunk_id:
            return index
    return None


def _candidate_rows(candidates: list) -> list[dict]:
    return [
        {
            "chunk_id": candidate.chunk_id,
            "distance": candidate.vector_score,
            "section": candidate.metadata.get("section", ""),
            "equipment_model": candidate.metadata.get("equipment_model", ""),
            "error_code": candidate.metadata.get("error_code", ""),
        }
        for candidate in candidates
    ]


def _evaluate_mode(retriever, cases: list[dict], mode: str) -> dict:
    rows, latencies = [], []
    for case in cases:
        started = time.perf_counter()
        retrieved = retriever(case["question"], mode)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = retrieved["relevant"]
        rows.append(
            {
                "id": case["id"],
                "rank": _rank(relevant.candidates, case["expected_chunk_id"]),
                "refused": not relevant.candidates,
                "top": (
                    relevant.candidates[0].metadata
                    if relevant.candidates
                    else {}
                ),
                "vector_debug": _candidate_rows(retrieved["raw"].candidates),
            }
        )
    answerable = [case for case in cases if not case["should_refuse"]]
    rows_by_id = {row["id"]: row for row in rows}
    identifier_cases = [case for case in answerable if case["expected_error_code"]]
    model_cases = [case for case in answerable if case["expected_equipment_model"]]
    semantic_cases = [case for case in answerable if case["id"].startswith("semantic")]
    hit_at_1 = sum(rows_by_id[case["id"]]["rank"] == 1 for case in answerable)
    hit_at_3 = sum(
        (rows_by_id[case["id"]]["rank"] or 99) <= 3 for case in answerable
    )
    reciprocal_rank = sum(
        1 / rows_by_id[case["id"]]["rank"]
        for case in answerable
        if rows_by_id[case["id"]]["rank"]
    )
    metrics = {
        "hit_rate_at_1": hit_at_1 / len(answerable),
        "hit_rate_at_3": hit_at_3 / len(answerable),
        "mrr": reciprocal_rank / len(answerable),
        "exact_identifier_hit_at_1": sum(
            rows_by_id[case["id"]]["top"].get("error_code")
            == case["expected_error_code"]
            for case in identifier_cases
        ) / len(identifier_cases),
        "fault_code_exact_match_rate": sum(
            rows_by_id[case["id"]]["top"].get("error_code")
            == case["expected_error_code"]
            for case in identifier_cases
        ) / len(identifier_cases),
        "equipment_model_exact_match_rate": sum(
            rows_by_id[case["id"]]["top"].get("equipment_model")
            == case["expected_equipment_model"]
            for case in model_cases
        ) / len(model_cases),
        "model_confusion_rate": sum(
            bool(rows_by_id[case["id"]]["top"].get("equipment_model"))
            and rows_by_id[case["id"]]["top"].get("equipment_model")
            != case["expected_equipment_model"]
            for case in model_cases
        ) / len(model_cases),
        "semantic_query_hit_at_1": sum(
            rows_by_id[case["id"]]["rank"] == 1 for case in semantic_cases
        ) / len(semantic_cases),
        "ood_refusal_accuracy": sum(
            rows_by_id[case["id"]]["refused"]
            for case in cases
            if case["should_refuse"]
        ) / len([case for case in cases if case["should_refuse"]]),
    }
    return {
        "metrics": metrics,
        "latency_ms_median": statistics.median(latencies),
        "rows": rows,
    }


def run_full_vector_benchmark() -> dict:
    benchmark = load_benchmark()
    documents = [_document(item) for item in benchmark["documents"]]
    cold_started = time.perf_counter()
    embedding = rag_core.get_embedding_model()
    embedding.embed_query("工业知识库 cold model validation")
    cold_model_load_seconds = time.perf_counter() - cold_started

    with benchmark_knowledge_base(benchmark) as (light_rag_core, light_id):
        with full_vector_knowledge_base(documents) as (persist_dir, vector_db):
            persisted = vector_db.get(include=["documents", "metadatas"])
            if len(persisted.get("documents", [])) != len(documents):
                raise RuntimeError("Chroma did not persist every benchmark chunk.")

            def retrieve_light(question: str, mode: str):
                raw = light_rag_core.retrieve_docs(
                    question, k=3, knowledge_base_id=light_id,
                    retrieval_mode=mode,
                )
                return {"raw": raw, "relevant": light_rag_core.filter_relevant_docs(raw)}

            def retrieve_full(question: str, mode: str):
                raw = rag_core.retrieve_docs(
                    question, k=3,
                    knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                    retrieval_mode=mode,
                )
                return {"raw": raw, "relevant": rag_core.filter_relevant_docs(raw)}

            reports = {
                "legacy_tfidf": _evaluate_mode(retrieve_light, benchmark["cases"], "vector"),
                "bm25": _evaluate_mode(retrieve_light, benchmark["cases"], "lexical"),
                "dense_vector": _evaluate_mode(retrieve_full, benchmark["cases"], "vector"),
                "hybrid": _evaluate_mode(retrieve_full, benchmark["cases"], "hybrid"),
            }

            hybrid_rows = {row["id"]: row for row in reports["hybrid"]["rows"]}
            singles = {
                name: {row["id"]: row for row in report["rows"]}
                for name, report in reports.items()
                if name != "hybrid"
            }
            wins = {"improved": 0, "tied": 0, "declined": 0}
            per_query = []
            for case in benchmark["cases"]:
                if case["should_refuse"]:
                    continue
                ranks = {
                    name: rows[case["id"]]["rank"]
                    for name, rows in singles.items()
                }
                hybrid_rank = hybrid_rows[case["id"]]["rank"]
                best_single = min((rank or 99) for rank in ranks.values())
                comparison = (hybrid_rank or 99) - best_single
                wins["improved" if comparison < 0 else "tied" if comparison == 0 else "declined"] += 1
                per_query.append(
                    {
                        "id": case["id"],
                        "query": case["question"],
                        "expected_chunk_id": case["expected_chunk_id"],
                        "tfidf_rank": ranks["legacy_tfidf"],
                        "bm25_rank": ranks["bm25"],
                        "vector_rank": ranks["dense_vector"],
                        "hybrid_rank": hybrid_rank,
                    }
                )
    return {
        "backend": "HuggingFace Embeddings + Chroma",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dimension": len(embedding.embed_query("dimension check")),
        "cold_model_load_seconds": cold_model_load_seconds,
        "reports": reports,
        "hybrid_win_rate": wins,
        "per_query": per_query,
        "temporary_chroma": True,
    }


def run_full_vector_benchmark_with_rrf(rrf_k: int) -> dict:
    previous_value = os.environ.get("RRF_K")
    os.environ["RRF_K"] = str(rrf_k)
    try:
        report = run_full_vector_benchmark()
        report["rrf_k"] = rrf_k
        return report
    finally:
        if previous_value is None:
            os.environ.pop("RRF_K", None)
        else:
            os.environ["RRF_K"] = previous_value


def run_real_full_smoke_test() -> dict:
    """Exercise industrial ingestion/chunking -> real embeddings -> temporary Chroma."""
    fixture = json.loads(SMOKE_FIXTURE_PATH.read_text(encoding="utf-8"))
    source = fixture["documents"][1]
    chunks = ingest_pages(
        source["file_name"],
        [PageText(index, text) for index, text in enumerate(source["pages"])],
    )
    documents = [
        Document(page_content=chunk.page_content, metadata=dict(chunk.metadata))
        for chunk in chunks
    ]
    with full_vector_knowledge_base(documents) as (_, vector_db):
        results = vector_db.similarity_search_with_score("F0002 故障处理", k=2)
        return {
            "embedding_loaded": bool(rag_core.get_embedding_model()),
            "chunk_count": len(documents),
            "stored_count": len(vector_db.get(include=["documents"])["documents"]),
            "result_count": len(results),
            "top_metadata": dict(results[0][0].metadata) if results else {},
            "top_distance": float(results[0][1]) if results else None,
            "temporary_chroma": True,
        }


if __name__ == "__main__":
    print(json.dumps(run_full_vector_benchmark(), ensure_ascii=False, indent=2))
