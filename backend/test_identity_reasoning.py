"""Deterministic tests for the V3.34 identity-aware Evidence candidate."""

from __future__ import annotations

from types import SimpleNamespace

import backend.retrieval.identity_reasoning as identity_reasoning
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_mixed import analyze_mixed_evidence
from backend.retrieval.identity_reasoning import (
    IDENTITY_AWARE_CANDIDATE_STATUS,
    IDENTITY_AWARE_CANDIDATE_VERSION,
    IdentityCompatibility,
    ScopeLevel,
    analyze_identity_aware_evidence,
    analyze_identity_boundary,
    extract_claim_identity,
    extract_query_identity,
    identity_compatibility,
)


def _document(text: str, **metadata):
    values = {
        "chunk_id": "chunk-1",
        "document_id": "dev-doc-1",
        "manufacturer": "Acme Controls",
        "product_family": "CX",
        "product_series": "CX series",
        "equipment_model": "CX series",
        "model_aliases": "CX5140|CX5130",
        "section": "Capabilities",
        "page": 4,
    }
    values.update(metadata)
    return SimpleNamespace(page_content=text, metadata=values)


def _result(document) -> RetrievalResult:
    candidate = RetrievalCandidate(
        document=document,
        retrieval_source="dev",
        lexical_score=5.0,
        vector_score=0.1,
        final_rank=1,
    )
    return RetrievalResult(
        [candidate], corpus_documents=[document], retrieval_mode="hybrid",
    )


def test_candidate_identity_is_explicitly_versioned():
    assert IDENTITY_AWARE_CANDIDATE_VERSION == "identity-aware-evidence-v334-candidate"
    assert IDENTITY_AWARE_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"


def test_query_member_alias_is_model_not_series_document_identity():
    document = _document("The CX series supports EtherCAT.")
    identity = extract_query_identity("What protocol does CX5140 support?", [document])
    assert identity.model == "CX5140"
    assert identity.scope_level == ScopeLevel.MODEL.value


def test_claim_scope_comes_from_sentence_not_series_alias_metadata():
    document = _document("The CX series supports EtherCAT.")
    identity = extract_claim_identity(document.page_content, document.metadata)
    assert identity.model == ""
    assert identity.series == "CX series"
    assert identity.scope_level == ScopeLevel.SERIES.value


def test_family_or_series_evidence_cannot_cover_model_query():
    document = _document("The CX series supports EtherCAT.")
    query = extract_query_identity("What protocol does CX5140 support?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    status, reason = identity_compatibility(query, claim)
    assert status == IdentityCompatibility.INCOMPATIBLE
    assert reason == "BROADER_EVIDENCE_SCOPE"


def test_exact_model_evidence_covers_exact_model_query():
    document = _document("CX5140 supports EtherCAT.")
    query = extract_query_identity("What protocol does CX5140 support?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.COMPATIBLE, "EXACT_MODEL",
    )


def test_descendant_model_evidence_can_cover_series_query():
    document = _document("CX5140 supports EtherCAT.")
    query = extract_query_identity("Does the CX series support EtherCAT?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.COMPATIBLE, "MODEL_DESCENDS_FROM_SERIES",
    )


def test_module_query_rejects_controller_claim():
    document = _document(
        "The CX controller supports EtherCAT.", module_model="EL6751",
    )
    query = extract_query_identity("Does the EL6751 module support EtherCAT?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert query.scope_level == ScopeLevel.MODULE.value
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.INCOMPATIBLE, "MODULE_TO_CONTROLLER_LEAKAGE",
    )


def test_exact_module_claim_is_compatible():
    document = _document(
        "The EL6751 module supports EtherCAT.", module_model="EL6751",
    )
    query = extract_query_identity("Does the EL6751 module support EtherCAT?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.COMPATIBLE, "EXACT_MODULE",
    )


def test_firmware_mismatch_is_incompatible():
    document = _document("CX5140 firmware 3.0 provides web diagnostics.")
    query = extract_query_identity("Does CX5140 firmware 2.1 provide web diagnostics?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.INCOMPATIBLE, "FIRMWARE_MISMATCH",
    )


def test_option_query_rejects_base_series_claim():
    document = _document(
        "The CX series provides redundant networking.", option_code="OP10",
    )
    query = extract_query_identity("Does the OP10 option provide redundant networking?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.INCOMPATIBLE, "OPTION_SCOPE_MISMATCH",
    )


def test_parameter_scope_mismatch_is_incompatible():
    document = _document("CX5140 parameter P042 controls acceleration.")
    query = extract_query_identity("Does parameter P041 control acceleration on CX5140?", [document])
    claim = extract_claim_identity(document.page_content, document.metadata)
    assert identity_compatibility(query, claim) == (
        IdentityCompatibility.INCOMPATIBLE, "PARAMETER_SCOPE_MISMATCH",
    )


def test_exact_firmware_option_and_parameter_scopes_remain_compatible():
    cases = (
        (
            "What protocol does CX5140 firmware 2.1 support?",
            "CX5140 firmware 2.1 supports EtherCAT.", {}, "EXACT_MODEL",
        ),
        (
            "What protocol does OP10 support?",
            "Option OP10 supports EtherCAT.",
            {"option_code": "OP10", "model_aliases": "CX5140|OP10"}, "EXACT_OPTION",
        ),
        (
            "What protocol is selected by parameter P041 on CX5140?",
            "CX5140 parameter P041 selects EtherCAT.", {}, "EXACT_MODEL",
        ),
    )
    for query_text, claim_text, metadata, reason in cases:
        document = _document(claim_text, **metadata)
        query = extract_query_identity(query_text, [document])
        claim = extract_claim_identity(document.page_content, document.metadata)
        assert identity_compatibility(query, claim) == (
            IdentityCompatibility.COMPATIBLE, reason,
        )


def test_identity_boundary_abstains_before_existing_contract(monkeypatch):
    document = _document("The CX series supports EtherCAT.")
    result = _result(document)

    def must_not_run(*args, **kwargs):
        raise AssertionError("existing Evidence must not run after a reliable mismatch")

    monkeypatch.setattr(identity_reasoning, "analyze_mixed_evidence", must_not_run)
    decision = analyze_identity_aware_evidence(
        "What protocol does CX5140 support?", result, [document], "hybrid",
        apply_open_sufficiency=False,
    )
    assert decision.decision == "ABSTAIN"
    assert decision.reason == "IDENTITY_SCOPE_MISMATCH"
    assert decision.final_decision_source == "IDENTITY_BOUNDARY"
    assert not decision.delegated_to_existing_evidence


def test_compatible_case_delegates_without_changing_existing_decision():
    document = _document("CX5140 supports EtherCAT.")
    result = _result(document)
    query = "What protocol does CX5140 support?"
    baseline = analyze_mixed_evidence(
        query, result, [document], "hybrid", apply_open_sufficiency=False,
    )
    candidate = analyze_identity_aware_evidence(
        query, result, [document], "hybrid", apply_open_sufficiency=False,
    )
    assert candidate.identity_boundary["status"] == "COMPATIBLE"
    assert candidate.delegated_to_existing_evidence
    assert (candidate.decision, candidate.reason) == (baseline.decision, baseline.reason)


def test_unknown_case_delegates_without_changing_existing_decision():
    document = _document(
        "Disconnect input power before servicing.",
        manufacturer="", product_family="", product_series="",
        equipment_model="", model_aliases="",
    )
    result = _result(document)
    query = "Should input power be disconnected before servicing?"
    boundary = analyze_identity_boundary(query, result, [document])
    baseline = analyze_mixed_evidence(
        query, result, [document], "hybrid", apply_open_sufficiency=False,
    )
    candidate = analyze_identity_aware_evidence(
        query, result, [document], "hybrid", apply_open_sufficiency=False,
    )
    assert boundary.status == "UNKNOWN"
    assert candidate.delegated_to_existing_evidence
    assert (candidate.decision, candidate.reason) == (baseline.decision, baseline.reason)
