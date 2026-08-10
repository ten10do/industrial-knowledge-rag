from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from backend.light_rag_core import LightDocument, _fit_index


BENCHMARK_PATH = Path(__file__).resolve().parent / "fixtures" / "retrieval_benchmark.json"
MODES = {"legacy_tfidf": "vector", "bm25": "lexical", "hybrid": "hybrid"}


def load_benchmark(path: Path = BENCHMARK_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        benchmark = json.load(handle)
    if len(benchmark.get("documents", [])) < 6 or len(benchmark.get("cases", [])) < 7:
        raise ValueError("V2 benchmark requires six documents and seven cases.")
    return benchmark


@contextmanager
def benchmark_knowledge_base(benchmark: dict):
    from backend import light_rag_core

    knowledge_base_id = "v2-retrieval-benchmark"
    previous = light_rag_core._knowledge_bases.get(knowledge_base_id)
    documents = [
        LightDocument(item["content"], dict(item["metadata"]))
        for item in benchmark["documents"]
    ]
    light_rag_core._knowledge_bases[knowledge_base_id] = _fit_index(documents)
    try:
        yield light_rag_core, knowledge_base_id
    finally:
        if previous is None:
            light_rag_core._knowledge_bases.pop(knowledge_base_id, None)
        else:
            light_rag_core._knowledge_bases[knowledge_base_id] = previous


def _metrics(cases: list[dict], results: list[dict]) -> dict:
    answerable = [case for case in cases if not case["should_refuse"]]
    by_id = {result["id"]: result for result in results}
    hit1 = hit3 = identifier_hit = fault_hit = model_hit = model_confusion = 0
    reciprocal_rank = 0.0
    ood_cases = [case for case in cases if case["should_refuse"]]
    identifier_cases = [
        case for case in answerable if case["expected_error_code"]
    ]
    model_cases = [
        case for case in answerable if case["expected_equipment_model"]
    ]
    ood_refused = 0
    for case in cases:
        result = by_id[case["id"]]
        if case["should_refuse"]:
            ood_refused += int(result["refused"])
            continue
        chunks = [candidate["chunk_id"] for candidate in result["candidates"]]
        top = result["candidates"][0] if result["candidates"] else {}
        hit1 += int(bool(chunks) and chunks[0] == case["expected_chunk_id"])
        hit3 += int(case["expected_chunk_id"] in chunks[:3])
        if case["expected_chunk_id"] in chunks:
            reciprocal_rank += 1 / (chunks.index(case["expected_chunk_id"]) + 1)
        if case["expected_error_code"]:
            identifier_hit += int(top.get("error_code") == case["expected_error_code"])
            fault_hit += int(top.get("error_code") == case["expected_error_code"])
        if case["expected_equipment_model"]:
            model_hit += int(top.get("equipment_model") == case["expected_equipment_model"])
            model_confusion += int(
                bool(top.get("equipment_model"))
                and top.get("equipment_model") != case["expected_equipment_model"]
            )
    return {
        "hit_rate_at_1": hit1 / len(answerable),
        "hit_rate_at_3": hit3 / len(answerable),
        "mrr": reciprocal_rank / len(answerable),
        "exact_identifier_hit_at_1": identifier_hit / len(identifier_cases),
        "fault_code_exact_match_rate": fault_hit / len(identifier_cases),
        "equipment_model_exact_match_rate": model_hit / len(model_cases),
        "model_confusion_rate": model_confusion / len(model_cases),
        "ood_refusal_accuracy": ood_refused / len(ood_cases),
    }


def run_benchmark(path: Path = BENCHMARK_PATH) -> dict:
    benchmark = load_benchmark(path)
    reports = {}
    with benchmark_knowledge_base(benchmark) as (light_rag_core, knowledge_base_id):
        for name, mode in MODES.items():
            results, latencies = [], []
            for case in benchmark["cases"]:
                started = time.perf_counter()
                retrieved = light_rag_core.retrieve_docs(
                    case["question"], k=3, knowledge_base_id=knowledge_base_id,
                    retrieval_mode=mode,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                relevant = light_rag_core.filter_relevant_docs(retrieved)
                candidates = [
                    {
                        "chunk_id": candidate.chunk_id,
                        "error_code": candidate.metadata.get("error_code", ""),
                        "equipment_model": candidate.metadata.get("equipment_model", ""),
                        "retrieval_source": candidate.retrieval_source,
                    }
                    for candidate in relevant.candidates
                ]
                results.append({"id": case["id"], "refused": not candidates, "candidates": candidates})
            reports[name] = {
                "metrics": _metrics(benchmark["cases"], results),
                "latency_ms_median": statistics.median(latencies),
                "results": results,
            }
    return {"benchmark": "light BM25 + legacy TF-IDF only", "reports": reports}


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2))
