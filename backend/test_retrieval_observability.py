from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from backend.evaluation.retrieval_observability import analyze_observability, overlay_relevance
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.fusion import rrf_fuse
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.scope import RetrievalScopeDecision, ScopeTier
from backend.retrieval.section import SectionConfig, expand_section_candidates
from backend.retrieval.trace import RetrievalTrace, create_trace


def _doc(chunk_id: str, content: str, *, section: str = "General", index: int = 0, code: str = ""):
    return Document(page_content=content, metadata={
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "page": index,
        "chunk_index": index,
        "section": section,
        "subsection": "",
        "equipment_model": "Drive 100",
        "product_family": "Drive",
        "error_code": code,
    })


def _candidate(document, source="lexical", rank=1):
    return RetrievalCandidate(
        document=document,
        retrieval_source=source,
        lexical_rank=(rank if source == "lexical" else None),
        vector_rank=(rank if source == "vector" else None),
        lexical_score=(1.0 if source == "lexical" else None),
        vector_score=(0.1 if source == "vector" else None),
        final_rank=rank,
    )


def _scope(documents, *, identifiers=()):
    return RetrievalScopeDecision(
        requested_scope="EXACT_MODEL_SCOPE",
        effective_scope="EXACT_MODEL_SCOPE",
        tiers=(ScopeTier("EXACT_MODEL_SCOPE", tuple(documents)),),
        candidate_counts={"exact_candidates": len(documents)},
        identifiers=tuple(identifiers),
        identifier_found=bool(identifiers),
    )


def test_trace_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_TRACE_ENABLED", raising=False)
    assert create_trace("query") is None
    monkeypatch.setenv("RETRIEVAL_TRACE_ENABLED", "true")
    assert isinstance(create_trace("query"), RetrievalTrace)


def test_lifecycle_scope_and_schema_fields_are_recorded():
    document = _doc("chunk-a", "startup procedure")
    candidate = _candidate(document)
    trace = RetrievalTrace("Drive 100 startup", "q1")
    analysis = SimpleNamespace(
        knowledge_type="procedure", manufacturer="ACME", product_family="Drive",
        product_series="", equipment_model="Drive 100", identifiers=("S03",),
    )
    scope = _scope([document], identifiers=("S03",))
    trace.configure_query(analysis, scope)
    for stage, event in (
        ("RETRIEVAL", "RETRIEVED_BM25"),
        ("SCOPE", "SCOPE_ACCEPTED"),
        ("RRF", "MERGED_RRF"),
        ("SECTION_MERGE", "SECTION_DUPLICATE"),
    ):
        trace.mark_stage(stage, [candidate], event)
    trace.budget(candidate, selected=True, priority=1, lane="BASE", protected=True)
    trace.mark_stage("BUDGET", [candidate])
    trace.finalize([candidate])
    payload = trace.as_dict()
    item = payload["candidates"][0]
    assert item["chunk_id"] == "chunk-a"
    assert item["budget_selected"] is True
    assert item["identifier_protected"] is True
    assert payload["scope"]["requested_scope"] == "EXACT_MODEL_SCOPE"
    assert payload["identifier_protection"]["protection_applied"] is True
    assert payload["candidate_counts_by_stage"]["FINAL"] == 1


def test_rrf_trace_records_dedup_and_global_budget_without_changing_order():
    first, second = _doc("first", "one"), _doc("second", "two")
    plain = rrf_fuse(
        [_candidate(first, "lexical", 1), _candidate(second, "lexical", 2)],
        [_candidate(first, "vector", 1)],
        top_k=1,
    )
    trace = RetrievalTrace("query")
    traced = rrf_fuse(
        [_candidate(first, "lexical", 1), _candidate(second, "lexical", 2)],
        [_candidate(first, "vector", 1)],
        top_k=1,
        trace=trace,
    )
    assert [item.chunk_id for item in traced] == [item.chunk_id for item in plain]
    payload = {item["chunk_id"]: item for item in trace.as_dict()["candidates"]}
    assert any(event["event"] == "DEDUPLICATED" for event in payload["first"]["events"])
    assert payload["second"]["drop_reason"] == "GLOBAL_BUDGET_DISPLACED"


def test_section_trace_records_provenance_budget_reject_and_displacement():
    base_docs = [_doc(f"base-{index}", "unrelated reference", index=index) for index in range(5)]
    expanded = _doc("expanded", "configure startup commissioning", section="Commissioning", index=9)
    documents = [*base_docs, expanded]
    scope = _scope(documents)
    config = SectionConfig(enabled=True, neighbor_window=0, candidate_k=1, max_expanded=1)
    plain, _ = expand_section_candidates(
        "configure startup", [_candidate(item, rank=index) for index, item in enumerate(base_docs, 1)],
        documents, scope, budget=5, cache_key="plain", config=config,
    )
    trace = RetrievalTrace("configure startup")
    traced, _ = expand_section_candidates(
        "configure startup", [_candidate(item, rank=index) for index, item in enumerate(base_docs, 1)],
        documents, scope, budget=5, cache_key="traced", config=config, trace=trace,
    )
    assert [item.chunk_id for item in traced] == [item.chunk_id for item in plain]
    payload = trace.as_dict()
    candidates = {item["chunk_id"]: item for item in payload["candidates"]}
    assert candidates["expanded"]["expansion_type"] == "SECTION_INDEX_MATCH"
    assert candidates["expanded"]["section_score_breakdown"]["total"] > 0
    assert candidates["base-4"]["budget_reject_reason"] == "ORIGINAL_BUDGET_FULL"
    assert candidates["expanded"]["candidate_source"] == "SECTION_EXPANDED"
    assert candidates["expanded"]["preservation_class"] == "EXPANSION"
    assert candidates["expanded"]["budget_reason"] == "EXPANSION_SLOT"
    assert payload["displacements"][0] == {
        "displaced_chunk": "base-4",
        "replacement_chunk": "expanded",
        "reason": "SECTION_BUDGET_DISPLACED",
        "displaced_relevant": None,
        "replacement_relevant": None,
        "classification": "UNKNOWN_DISPLACEMENT",
    }


def test_identifier_candidate_protection_is_visible_in_section_budget_trace():
    exact = _doc("s03", "Fault S03", section="Troubleshooting", index=1, code="S03")
    expanded = _doc("startup", "configure startup", section="Commissioning", index=2)
    exact_candidate = _candidate(exact)
    exact_candidate.exact_metadata_match = True
    trace = RetrievalTrace("Drive 100 S03 startup")
    result, _ = expand_section_candidates(
        "Drive 100 S03 startup", [exact_candidate], [exact, expanded], _scope([exact, expanded], identifiers=("S03",)),
        budget=1, cache_key="identifier", config=SectionConfig(enabled=True, neighbor_window=0, candidate_k=1, max_expanded=1),
        trace=trace,
    )
    assert [item.chunk_id for item in result] == ["s03"]
    protection = trace.as_dict()["identifier_protection"]
    assert protection["protected_candidates"] == ["s03"]
    assert protection["protection_applied"] is True


def test_reranker_trace_records_selected_and_truncated_candidates():
    documents = [_doc(f"c{index}", f"content {index}") for index in range(3)]
    candidates = [_candidate(item, rank=index) for index, item in enumerate(documents, 1)]
    trace = RetrievalTrace("query")
    result = RetrievalResult(candidates, trace=trace)
    model = SimpleNamespace(predict=lambda pairs, show_progress_bar=False: [0.1, 0.9, 0.2])
    reranker = CrossEncoderReranker(
        RerankerConfig(enabled=True, candidate_k=3, top_k=2),
        model_factory=lambda *_: model,
    )
    outcome = reranker.rerank("query", result)
    assert [item.chunk_id for item in outcome.result.candidates] == ["c1", "c2"]
    payload = {item["chunk_id"]: item for item in trace.as_dict()["candidates"]}
    assert payload["c0"]["drop_reason"] == "RERANK_TRUNCATED"
    assert payload["c1"]["rerank_rank"] == 1


def test_relevance_overlay_classifies_harmful_and_beneficial_displacements():
    trace = RetrievalTrace("query", "q1")
    relevant = _candidate(_doc("relevant", "correct"))
    wrong = _candidate(_doc("wrong", "wrong"))
    trace.event(relevant, "MERGED_RRF", "RRF")
    trace.event(wrong, "SECTION_ADDED", "SECTION_MERGE")
    trace.displacement(relevant, wrong, "SECTION_BUDGET_DISPLACED")
    trace.displacement(wrong, relevant, "SECTION_BUDGET_DISPLACED")
    result = overlay_relevance(trace.as_dict(), {
        "relevant_chunk_ids": ["relevant"], "expected_section": "General", "expected_model": "Drive 100",
    })
    assert [item["classification"] for item in result["displacements"]] == [
        "HARMFUL_DISPLACEMENT", "BENEFICIAL_DISPLACEMENT",
    ]


def test_analysis_finds_reranker_as_first_failure_and_counterfactual_regression():
    query = {
        "query_id": "q1", "query": "query", "answerable": True,
        "relevant_chunk_ids": ["relevant"], "expected_section": "General", "expected_model": "Drive 100",
    }
    relevant = _candidate(_doc("relevant", "correct"))
    baseline = RetrievalTrace("query", "q1")
    section = RetrievalTrace("query", "q1")
    for trace in (baseline, section):
        for stage in ("RETRIEVAL", "SCOPE", "RRF", "SECTION_MERGE", "BUDGET"):
            trace.mark_stage(stage, [relevant])
    baseline.mark_stage("RERANK", [relevant])
    baseline.finalize([relevant])
    section.drop(relevant, "RERANK_DROPPED", "RERANK", "RERANK_TRUNCATED")
    section.mark_stage("RERANK", [])
    section.finalize([])
    report = analyze_observability(
        [query],
        {"rows": [{"query_id": "q1", "trace": baseline.as_dict()}]},
        {"rows": [{"query_id": "q1", "trace": section.as_dict()}]},
    )
    assert report["first_failure_stage"] == {"RERANK": 1}
    assert report["counterfactual"]["SECTION_CAUSED_REGRESSION"] == 1
    assert report["reranker_drop_relevant_count"] == 1


def test_recovered_candidate_is_not_reported_as_an_early_failure():
    query = {
        "query_id": "q1", "query": "query", "answerable": True,
        "relevant_chunk_ids": ["relevant"], "expected_section": "General",
    }
    relevant = _candidate(_doc("relevant", "correct"))
    trace = RetrievalTrace("query", "q1")
    trace.mark_stage("RETRIEVAL", [])
    trace.mark_stage("SCOPE", [])
    trace.mark_stage("RRF", [])
    for stage in ("SECTION_MERGE", "BUDGET", "RERANK"):
        trace.mark_stage(stage, [relevant])
    trace.finalize([relevant])
    report = analyze_observability(
        [query],
        {"rows": [{"query_id": "q1", "trace": trace.as_dict()}]},
        {"rows": [{"query_id": "q1", "trace": trace.as_dict()}]},
    )
    assert report["first_failure_stage"] == {"SUCCESS": 1}
