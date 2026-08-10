from types import SimpleNamespace

from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base, load_benchmark, run_benchmark
from backend.retrieval import BM25Index, RetrievalCandidate, rrf_fuse
from backend.retrieval.filters import analyze_query, filter_documents
from backend.retrieval.tokenizer import tokenize


def test_tokenizer_preserves_industrial_identifiers():
    tokens = tokenize("F0002 S7-1200 0x8001 MW20 SINAMICS G120")
    assert {"f0002", "s7-1200", "0x8001", "mw20", "sinamics", "g120"}.issubset(tokens)


def test_bm25_ranks_exact_fault_code_and_mixed_language():
    index = BM25Index(["F0001 输入缺相", "F0002 DC link overvoltage 直流母线过压"])
    assert index.score("F0002 的原因是什么")[1] > index.score("F0002 的原因是什么")[0]


def test_query_analysis_and_metadata_filter_fallback():
    documents = [
        SimpleNamespace(metadata={"error_code": "F0002", "equipment_model": "S7-1200"}),
        SimpleNamespace(metadata={"error_code": "F0001", "equipment_model": "S7-1500"}),
    ]
    analysis = analyze_query("S7-1200 的 F0002", documents)
    filtered, applied = filter_documents(documents, analysis)
    assert analysis.error_code == "F0002"
    assert analysis.equipment_model == "S7-1200"
    assert applied and filtered == documents[:1]
    broader, applied = filter_documents(documents, analyze_query("F9999", documents))
    assert not applied and broader == documents


def test_rrf_deduplicates_by_chunk_id_and_keeps_ranks():
    first = SimpleNamespace(metadata={"chunk_id": "one"})
    second = SimpleNamespace(metadata={"chunk_id": "two"})
    merged = rrf_fuse(
        [RetrievalCandidate(first, "lexical", lexical_rank=1), RetrievalCandidate(second, "lexical", lexical_rank=2)],
        [RetrievalCandidate(second, "vector", vector_rank=1), RetrievalCandidate(first, "vector", vector_rank=2)],
        top_k=2,
    )
    assert {candidate.chunk_id for candidate in merged} == {"one", "two"}
    assert all(candidate.lexical_rank and candidate.vector_rank for candidate in merged)
    assert [candidate.final_rank for candidate in merged] == [1, 2]


def test_light_hybrid_preserves_metadata_and_refuses_unknown_identifier():
    benchmark = load_benchmark()
    with benchmark_knowledge_base(benchmark) as (light_rag_core, knowledge_base_id):
        result = light_rag_core.retrieve_docs("SINAMICS G120 的 F0002", k=3, knowledge_base_id=knowledge_base_id, retrieval_mode="hybrid")
        assert result.candidates[0].chunk_id == "g120-f0002"
        assert result.candidates[0].retrieval_source == "hybrid"
        assert light_rag_core.has_relevant_docs(result)
        unknown = light_rag_core.retrieve_docs("F9999 应该如何处理", k=3, knowledge_base_id=knowledge_base_id, retrieval_mode="hybrid")
        assert not unknown and not light_rag_core.has_relevant_docs(unknown)


def test_v2_benchmark_reports_real_light_modes_only():
    report = run_benchmark()
    assert set(report["reports"]) == {"legacy_tfidf", "bm25", "hybrid"}
    assert report["reports"]["bm25"]["metrics"]["ood_refusal_accuracy"] == 1.0
