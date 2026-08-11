from types import SimpleNamespace

from backend import rag_core
from backend.retrieval import (
    RetrievalCandidate,
    RetrievalResult,
    SectionConfig,
    analyze_query,
    build_retrieval_scope,
    expand_section_candidates,
    infer_section_hint,
    load_section_config,
    normalize_section,
    validate_evidence_support,
)
from backend.retrieval.section import SECTION_MERGE_APPEND, SECTION_MERGE_RESERVED
from backend.retrieval.trace import RetrievalTrace


def _doc(chunk_id, content, *, model="Drive 100", section="Chapter 5 Commissioning", index=0):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": f"manual-{model.casefold().replace(' ', '-')}",
            "manufacturer": "Vendor",
            "equipment_model": model,
            "product_family": "Drive",
            "section": section,
            "subsection": "",
            "page": index,
            "page_start": index,
            "page_end": index,
            "chunk_index": index,
        },
    )


def _expand(query, documents, base, *, budget=5, window=1, candidate_k=2, max_expanded=3):
    analysis = analyze_query(query, documents)
    scope = build_retrieval_scope(query, documents, analysis)
    candidates = [
        RetrievalCandidate(
            document=document,
            retrieval_source="hybrid",
            final_rank=rank,
            exact_metadata_match=True,
            scope_match="primary",
        )
        for rank, document in enumerate(base, start=1)
    ]
    return expand_section_candidates(
        query,
        candidates,
        documents,
        scope,
        budget=budget,
        cache_key=f"test-{id(documents)}",
        config=SectionConfig(True, window, candidate_k, max_expanded),
    )


def test_section_normalization_handles_unicode_spacing_numbering_and_separators():
    values = {
        normalize_section("Chapter 5  Commissioning"),
        normalize_section("Chapter\u00a05/uni00A0Commissioning"),
        normalize_section("5 - Commissioning"),
    }
    assert values == {"5 commissioning"}
    assert normalize_section("Appendix B.  Safety > Power") == "appendix b safety / power"


def test_section_hint_uses_small_aliases_and_dynamic_vocabulary():
    hint = infer_section_hint("How do I start up and configure the drive?", {"commissioning", "configuration"})
    assert "procedure" in hint.matched_aliases
    assert "network" in hint.matched_aliases
    assert "configuration" in hint.vocabulary_matches


def test_section_retrieval_production_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SECTION_EXPANSION_ENABLED", raising=False)
    config, error = load_section_config()
    assert not error
    assert config.enabled is False


def test_same_section_neighbor_recovers_split_procedure_and_excludes_other_sections_and_models():
    documents = [
        _doc("before", "Prepare the device before startup.", index=1),
        _doc("anchor", "Configure the address during commissioning.", index=2),
        _doc("after", "Verify operation and authorize the device for use.", index=3),
        _doc("other-section", "Unrelated maintenance procedure.", section="Chapter 7 Maintenance", index=4),
        _doc("other-model", "Configure the address during commissioning.", model="Drive 200", index=2),
    ]
    result, report = _expand("Drive 100 how to configure and verify startup", documents, [documents[1]], candidate_k=1)
    ids = [candidate.chunk_id for candidate in result]
    assert {"before", "anchor", "after"}.issubset(ids)
    assert "other-section" not in ids
    assert "other-model" not in ids
    assert report.section_expansion_used
    assert all(candidate.metadata["equipment_model"] == "Drive 100" for candidate in result)


def test_no_section_metadata_falls_back_to_original_retrieval():
    document = _doc("base", "General startup details.", section="")
    result, report = _expand("Drive 100 startup", [document], [document])
    assert [candidate.chunk_id for candidate in result] == ["base"]
    assert not report.section_effective
    assert report.section_fallback_reason == "section_metadata_unavailable"


def test_candidate_merge_deduplicates_and_enforces_budget():
    documents = [_doc(str(index), f"Configure startup step {index}.", index=index) for index in range(8)]
    result, report = _expand("Drive 100 configure startup", documents, documents[:5], budget=5, candidate_k=1, max_expanded=2)
    ids = [candidate.chunk_id for candidate in result]
    assert len(ids) == len(set(ids)) == 5
    assert report.section_candidates_added <= 2
    assert report.candidate_budget_overflow >= 0


def test_expanded_candidate_survives_relevance_filter_for_reranking():
    document = _doc("section-only", "Commissioning verification procedure.")
    candidate = RetrievalCandidate(document=document, retrieval_source="section", section_expanded=True)
    result = RetrievalResult([candidate], corpus_documents=[document], retrieval_mode="hybrid")
    filtered = rag_core.filter_relevant_docs(result)
    assert [item.chunk_id for item in filtered.candidates] == ["section-only"]


def test_identifier_candidate_is_preserved_when_section_candidates_are_added():
    exact = _doc("s03", "Fault S03 means motor overspeed.", section="Chapter 7 Troubleshooting", index=9)
    documents = [
        _doc("startup", "Configure startup network address.", index=1),
        _doc("verify", "Verify startup operation.", index=2),
        exact,
    ]
    result, _ = _expand("Drive 100 S03 startup fault", documents, [documents[0], exact], budget=2, max_expanded=2)
    assert "s03" in [candidate.chunk_id for candidate in result]


def test_section_candidates_recover_support_without_false_support():
    wrong = _doc("wrong", "General motor parameter programming.", section="Chapter 3 Parameters", index=10)
    correct = _doc(
        "discharge",
        "After mains power is removed, wait three minutes for DC bus capacitors to discharge. Verify DC voltage is zero.",
        section="Appendix B.",
        index=20,
    )
    query = "Drive 100 after removing mains power, when are the DC bus capacitors discharged?"
    result, _ = _expand(query, [wrong, correct], [wrong], candidate_k=2)
    support_result = RetrievalResult(result, query_analysis=analyze_query(query, [wrong, correct]))
    support = validate_evidence_support(query, support_result, [wrong, correct])
    assert support.status == "SUPPORTED"
    assert "discharge" in support.supporting_chunks

    unsupported_query = "Drive 100 how to configure PROFINET device name and commission it?"
    network = _doc("ethernet", "Configure the EtherNet/IP address and verify network operation.", index=30)
    unsupported, _ = _expand(unsupported_query, [network], [network], candidate_k=1)
    guard_result = RetrievalResult(unsupported, query_analysis=analyze_query(unsupported_query, [network]))
    guard = validate_evidence_support(unsupported_query, guard_result, [network])
    assert guard.status == "INSUFFICIENT"
    assert "protocol:profinet" in guard.missing_requirements


def test_append_merge_preserves_original_budget_and_admits_section_recovery():
    base = [_doc(f"base-{index}", "unrelated reference", index=index) for index in range(5)]
    recovered = _doc("recovered", "configure startup commissioning", section="Commissioning", index=9)
    documents = [*base, recovered]
    analysis = analyze_query("Drive 100 configure startup", documents)
    scope = build_retrieval_scope("Drive 100 configure startup", documents, analysis)
    candidates = [RetrievalCandidate(document=item, retrieval_source="hybrid", final_rank=index)
                  for index, item in enumerate(base, 1)]
    trace = RetrievalTrace("Drive 100 configure startup")
    result, _ = expand_section_candidates(
        "Drive 100 configure startup", candidates, documents, scope, budget=5,
        cache_key="append-preservation", config=SectionConfig(True, 0, 1, 2),
        merge_strategy=SECTION_MERGE_APPEND, trace=trace,
    )
    assert [item.chunk_id for item in result] == [*(item.metadata["chunk_id"] for item in base), "recovered"]
    payload = {item["chunk_id"]: item for item in trace.as_dict()["candidates"]}
    assert payload["base-4"]["budget_selected"] is True
    assert payload["base-4"]["budget_reason"] == "ORIGINAL_RESERVED"
    assert payload["recovered"]["budget_reason"] == "EXPANSION_SLOT"
    assert trace.as_dict()["displacements"] == []


def test_reserved_merge_emits_explicit_displacement_and_deduplicates_both_source():
    base = [_doc(f"base-{index}", "unrelated reference", index=index) for index in range(5)]
    anchor = _doc("anchor", "configure startup commissioning", section="Commissioning", index=9)
    documents = [*base, anchor]
    analysis = analyze_query("Drive 100 configure startup", documents)
    scope = build_retrieval_scope("Drive 100 configure startup", documents, analysis)
    candidates = [
        RetrievalCandidate(document=anchor, retrieval_source="hybrid", final_rank=1),
        *[RetrievalCandidate(document=item, retrieval_source="hybrid", final_rank=index + 2)
          for index, item in enumerate(base)],
    ]
    trace = RetrievalTrace("Drive 100 configure startup")
    result, _ = expand_section_candidates(
        "Drive 100 configure startup", candidates, documents, scope, budget=5,
        cache_key="reserved-both", config=SectionConfig(True, 0, 1, 1),
        merge_strategy=SECTION_MERGE_RESERVED, trace=trace,
    )
    assert len({item.chunk_id for item in result}) == len(result)
    payload = {item["chunk_id"]: item for item in trace.as_dict()["candidates"]}
    assert payload["anchor"]["candidate_source"] == "BOTH"
    assert payload["base-4"]["budget_selected"] is False
    assert payload["base-4"]["budget_reject_reason"] == "ORIGINAL_BUDGET_FULL"
