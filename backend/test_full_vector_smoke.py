import os

import pytest

from backend.evaluation.full_vector_benchmark import run_real_full_smoke_test
from backend.evaluation.retrieval_benchmark import load_benchmark


def test_full_benchmark_extends_semantic_cases_without_changing_shared_documents():
    benchmark = load_benchmark()
    assert len(benchmark["documents"]) == 6
    assert len(benchmark["cases"]) >= 10
    assert sum(case["id"].startswith("semantic") for case in benchmark["cases"]) >= 3


@pytest.mark.skipif(
    os.getenv("RUN_FULL_VECTOR_SMOKE") != "1",
    reason="requires a real cached/downloadable HuggingFace embedding model",
)
def test_real_full_vector_smoke_preserves_industrial_metadata():
    report = run_real_full_smoke_test()
    assert report["embedding_loaded"]
    assert report["chunk_count"] > 0
    assert report["stored_count"] == report["chunk_count"]
    assert report["result_count"] > 0
    assert report["top_metadata"].get("chunk_id")
    assert report["temporary_chroma"]
