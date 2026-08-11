from types import SimpleNamespace

from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base
from backend.retrieval import RetrievalCandidate, RetrievalScope, analyze_query, build_retrieval_scope, rrf_fuse
from backend.retrieval.scope import collect_scoped_candidates


def _doc(chunk_id, model="", content="", *, family="", series="", aliases=""):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "manufacturer": "Vendor",
            "equipment_model": model,
            "product_family": family,
            "product_series": series,
            "model_aliases": aliases,
        },
    )


def _scope(query, documents):
    analysis = analyze_query(query, documents)
    return analysis, build_retrieval_scope(query, documents, analysis)


def test_scope_decisions_cover_exact_series_family_global_unknown_and_multi():
    documents = [
        _doc("a", "Drive 100", family="Drive"),
        _doc("b", "Drive 200", family="Drive"),
        _doc("series", "Flex 500-series (501/502)", family="Flex", series="Flex 500"),
    ]
    assert _scope("Drive 100 startup", documents)[1].requested_scope == RetrievalScope.EXACT_MODEL_SCOPE.value
    assert _scope("Flex 500 series wiring", documents)[1].requested_scope == RetrievalScope.SERIES_SCOPE.value
    assert _scope("Drive maintenance", documents)[1].requested_scope == RetrievalScope.FAMILY_SCOPE.value
    assert _scope("controller maintenance", documents)[1].requested_scope == RetrievalScope.GLOBAL_SCOPE.value
    unknown = _scope("Drive 900 startup", documents)[1]
    assert unknown.requested_scope == RetrievalScope.UNKNOWN_SCOPE.value
    assert unknown.fallback_used and unknown.fallback_reason == "unknown_model"
    multi = _scope("Drive 100 vs Drive 200", documents)[1]
    assert multi.requested_scope == RetrievalScope.MULTI_IDENTITY_SCOPE.value
    assert {item.metadata["chunk_id"] for item in multi.tiers[0].documents} == {"a", "b"}


def test_bare_model_resolution_is_unique_and_conservative():
    documents = [_doc("a", "Drive 100", family="Drive"), _doc("b", "Pump 200", family="Pump")]
    analysis, decision = _scope("100 startup sequence", documents)
    assert analysis.equipment_model == "Drive 100"
    assert decision.requested_scope == RetrievalScope.EXACT_MODEL_SCOPE.value

    ambiguous = [_doc("a", "Drive 100", family="Drive"), _doc("b", "Pump 100", family="Pump")]
    analysis, decision = _scope("100 startup sequence", ambiguous)
    assert not analysis.equipment_model
    assert decision.requested_scope == RetrievalScope.GLOBAL_SCOPE.value


def test_series_manual_is_compatible_with_member_model_query():
    documents = [
        _doc("series", "PowerFlex 520-series (523/525)", family="PowerFlex", series="PowerFlex 520"),
        _doc("sibling", "PowerFlex 527", family="PowerFlex"),
    ]
    analysis, decision = _scope("PowerFlex 525 setup", documents)
    assert analysis.identity_confidence == "EXACT_MODEL"
    assert [item.metadata["chunk_id"] for item in decision.tiers[0].documents] == ["series"]


def test_identifier_scope_protects_exact_chunks_and_uses_observable_fallback():
    documents = [
        _doc("exact", "Drive 100", "Fault S03 means overspeed.", family="Drive"),
        _doc("same-model", "Drive 100", "General troubleshooting.", family="Drive"),
        _doc("sibling", "Drive 200", "Other fault.", family="Drive"),
    ]
    analysis, decision = _scope("Drive 100 S03 condition", documents)
    assert analysis.identifiers == ("S03",)
    assert decision.identifier_found
    assert [item.metadata["chunk_id"] for item in decision.tiers[0].documents] == ["exact"]

    def retrieve(scoped, limit):
        return [RetrievalCandidate(item, "lexical", lexical_score=1.0) for item in scoped[:limit]]

    candidates = collect_scoped_candidates(decision, 2, retrieve)
    assert [item.chunk_id for item in candidates] == ["exact", "same-model"]
    assert [item.scope_match for item in candidates] == ["primary", "fallback"]
    assert decision.fallback_used and decision.fallback_reason == "insufficient_identifier_candidates"


def test_unknown_model_never_claims_a_known_sibling_as_an_exact_scope_match():
    documents = [
        _doc("known", "CompactLogix 5380", "Known sibling manual.", family="CompactLogix"),
        _doc("family", "CompactLogix", "General controller manual.", family="CompactLogix"),
    ]
    analysis, decision = _scope("CompactLogix 5390 wiring", documents)

    assert analysis.identity_confidence == "EXACT_MODEL"
    assert decision.requested_scope == RetrievalScope.UNKNOWN_SCOPE.value
    assert decision.tiers[0].level == RetrievalScope.GLOBAL_SCOPE.value

    def retrieve(scoped, limit):
        return [RetrievalCandidate(item, "lexical", lexical_score=1.0) for item in scoped[:limit]]

    candidates = collect_scoped_candidates(decision, 2, retrieve)
    assert all(item.identity_relation != "EXACT_MODEL" for item in candidates)


def test_scope_fallback_is_a_candidate_pool_decision_not_an_rrf_formula_change():
    primary = RetrievalCandidate(_doc("primary", "Drive 100", family="Drive"), "lexical", lexical_rank=1)
    fallback = RetrievalCandidate(_doc("fallback", "Drive 200", family="Drive"), "vector", vector_rank=1)
    primary.scope_match = "primary"
    fallback.scope_match = "fallback"

    # The RRF contribution remains equal. Scope eligibility prevents a sibling
    # fallback from being treated as a primary exact-model candidate.
    fused = rrf_fuse([primary], [fallback], top_k=2)
    assert [item.chunk_id for item in fused] == ["primary", "fallback"]


def test_light_hybrid_scopes_both_channels_before_fusion_and_does_not_over_filter():
    benchmark = {
        "documents": [
            {"content": "network ring recovery procedure", "metadata": {"chunk_id": "right", "manufacturer": "Vendor", "equipment_model": "Drive 100", "product_family": "Drive"}},
            {"content": "network ring recovery procedure", "metadata": {"chunk_id": "wrong", "manufacturer": "Vendor", "equipment_model": "Drive 200", "product_family": "Drive"}},
            {"content": "S07 overload threshold", "metadata": {"chunk_id": "identifier", "manufacturer": "Vendor", "equipment_model": "Drive 100", "product_family": "Drive"}},
        ]
    }
    with benchmark_knowledge_base(benchmark) as (light, knowledge_base_id):
        result = light.retrieve_docs("Drive 100 network ring recovery", k=2, knowledge_base_id=knowledge_base_id, retrieval_mode="hybrid")
        assert result.candidates[0].chunk_id == "right"
        assert all(item.metadata["equipment_model"] == "Drive 100" for item in result.candidates)
        assert all(item.scope_match == "primary" for item in result.candidates)

        identifier = light.retrieve_docs("Drive 100 S07 threshold", k=2, knowledge_base_id=knowledge_base_id, retrieval_mode="hybrid")
        assert identifier.candidates[0].chunk_id == "identifier"
        assert {item.chunk_id for item in identifier.candidates} >= {"identifier", "right"}
