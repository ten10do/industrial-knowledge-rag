from types import SimpleNamespace

from langchain_core.documents import Document

from backend.evaluation import v342_evidence_sufficiency_identity_aware as benchmark
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_sufficiency_v342 import (
    EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION,
    EvidenceSufficiencyRelation,
    analyze_evidence_sufficiency,
    classify_evidence_sufficiency_relation,
)


def _candidate(text: str, *, model: str = "EL18xx", aliases: str | None = None,
               document_id: str = "beckhoff-el18xx", manufacturer: str = "Beckhoff"):
    aliases = aliases or "EL1804|EL1814|EL1808|EL1809|EL1819"
    document = Document(
        page_content=text,
        metadata={
            "chunk_id": "chunk-1", "document_id": document_id,
            "manufacturer": manufacturer, "equipment_model": model,
            "product_family": model, "product_series": "EtherCAT Terminals",
            "model_aliases": aliases,
        },
    )
    return RetrievalCandidate(document=document, retrieval_source="hybrid")


def _v341(*, decision: str = "ABSTAIN", identity: str = "COMPATIBLE",
          expanded: bool = False, query_path: str = "VERIFICATION"):
    value = SimpleNamespace(
        decision=decision, reason="V341_REASON", final_decision_source="V341",
        query_path=query_path, identity_result=identity, expanded=expanded,
    )
    value.as_dict = lambda: {
        "decision": decision, "reason": value.reason,
        "identity_result": identity, "expanded": expanded,
    }
    return value


def test_versions_and_required_relation_values():
    assert EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION == "evidence-v342-sufficiency-candidate"
    assert {item.value for item in EvidenceSufficiencyRelation} == {
        "DIRECT_SUPPORTED", "SEMANTIC_SUPPORTED", "REFERENCE_SUPPORTED",
        "INSUFFICIENT", "UNSAFE",
    }
    assert EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value == (
        benchmark.EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value
    )


def test_direct_target_attribute_and_value_are_bound_in_one_chunk():
    relation = classify_evidence_sufficiency_relation(
        "Is the EL1809 input filter 3 ms?",
        [_candidate("Technical data EL1809 EL1819\nInput filter 3 ms 10 us")],
    )
    assert relation.relation == "DIRECT_SUPPORTED"
    assert relation.attribute_anchor == "input_filter"
    assert relation.value_action_anchor == "3 ms"


def test_technical_synonym_is_semantic_support():
    relation = classify_evidence_sufficiency_relation(
        "Is the EL1809 electrical separation 500 V?",
        [_candidate("Technical data EL1809 EL1819\nElectrical isolation 500 V")],
    )
    assert relation.relation == "SEMANTIC_SUPPORTED"
    assert relation.semantic_match


def test_abbreviation_definition_requires_both_acronym_and_expansion():
    candidate = _candidate(
        "Abbreviations\nDPP\nSPDU\nDirect Parameter Page\nService Protocol Data Unit",
        model="BNI IOL-302-002-E012", aliases="BNI IOL-302-002-E012|BNI00AR",
        document_id="balluff", manufacturer="Balluff",
    )
    supported = classify_evidence_sufficiency_relation(
        "For BNI IOL-302-002-E012, what does DPP stand for?", [candidate],
    )
    assert supported.relation == "SEMANTIC_SUPPORTED"
    assert supported.reason_code == "ABBREVIATION_DEFINITION_SUPPORTED"


def test_explicit_cross_reference_target_is_required():
    candidate = _candidate(
        "Technical data EL1809 EL1819\nMounting [ } 55 ] on 35 mm mounting rail"
    )
    supported = classify_evidence_sufficiency_relation(
        "Does EL1809 mounting refer to page 55?", [candidate],
    )
    wrong = classify_evidence_sufficiency_relation(
        "Does EL1809 mounting refer to page 67?", [candidate],
    )
    assert supported.relation == "REFERENCE_SUPPORTED"
    assert wrong.relation == "UNSAFE"
    assert wrong.reason_code == "REFERENCE_TARGET_MISMATCH"


def test_unknown_model_cannot_borrow_from_same_manufacturer():
    relation = classify_evidence_sufficiency_relation(
        "Is the EL1909 input filter 3 ms?",
        [_candidate("Technical data EL1809 EL1819\nInput filter 3 ms 10 us")],
    )
    assert relation.relation == "UNSAFE"
    assert relation.reason_code == "CROSS_MODEL_LEAKAGE_BLOCKED"


def test_sibling_model_column_value_is_blocked():
    relation = classify_evidence_sufficiency_relation(
        "Is the EL1819 input filter 3 ms?",
        [_candidate("Technical data EL1809 EL1819\nInput filter 3 ms 10 us")],
    )
    assert relation.relation == "UNSAFE"
    assert relation.reason_code == "SIBLING_MODEL_VALUE_BLOCKED"


def test_same_value_under_wrong_attribute_is_not_support():
    relation = classify_evidence_sufficiency_relation(
        "Is the EL1809 weight 100 g?",
        [_candidate("Technical data EL1809 EL1819\nCurrent consumption via E-bus 100 mA\nWeight 65 g")],
    )
    assert relation.relation == "UNSAFE"
    assert relation.reason_code == "ATTRIBUTE_VALUE_MISMATCH"


def test_section_scope_mismatch_blocks_parameter_borrowing():
    candidate = _candidate(
        "Configuration: Extended with BNI IOL-302-002-E012\nInversion of the inputs 4 bytes",
        model="BNI IOL-302-002-E012", aliases="BNI IOL-302-002-E012|BNI00AR",
        document_id="balluff", manufacturer="Balluff",
    )
    relation = classify_evidence_sufficiency_relation(
        'Under Configuration "Extension Off", is BNI IOL-302-002-E012 input inversion 4 bytes?',
        [candidate],
    )
    assert relation.relation == "UNSAFE"
    assert relation.reason_code == "SECTION_SCOPE_MISMATCH"


def test_identity_incompatible_and_nonverification_paths_are_preserved(monkeypatch):
    result = RetrievalResult([_candidate("EL1809 Input filter 3 ms")])
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v342.analyze_identity_claim_evidence",
        lambda *args, **kwargs: _v341(identity="INCOMPATIBLE"),
    )
    incompatible = analyze_evidence_sufficiency("Is the EL1809 input filter 3 ms?", result, [], "hybrid")
    assert incompatible.decision == "ABSTAIN"
    assert incompatible.reason == "IDENTITY_BOUNDARY_PRESERVED"

    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v342.analyze_identity_claim_evidence",
        lambda *args, **kwargs: _v341(query_path="OPEN"),
    )
    open_path = analyze_evidence_sufficiency("Is the EL1809 input filter 3 ms?", result, [], "hybrid")
    assert open_path.decision == "ABSTAIN"
    assert open_path.reason == "NON_VERIFICATION_PATH_PRESERVED"


def test_compatible_v341_refusal_can_be_relaxed_but_answer_is_unchanged(monkeypatch):
    result = RetrievalResult([_candidate("Technical data EL1809 EL1819\nInput filter 3 ms 10 us")])
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v342.analyze_identity_claim_evidence",
        lambda *args, **kwargs: _v341(),
    )
    rescued = analyze_evidence_sufficiency("Is the EL1809 input filter 3 ms?", result, [], "hybrid")
    assert rescued.decision == "ANSWER"
    assert rescued.relaxed
    assert rescued.final_decision_source == "V342_EVIDENCE_SUFFICIENCY"

    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v342.analyze_identity_claim_evidence",
        lambda *args, **kwargs: _v341(decision="ANSWER"),
    )
    unchanged = analyze_evidence_sufficiency("Is the EL1809 input filter 3 ms?", result, [], "hybrid")
    assert unchanged.decision == "ANSWER"
    assert not unchanged.relaxed


def _benchmark_row(case: str, index: int) -> dict:
    answer = index < 5
    hard_type = None if answer else benchmark.HARD_NEGATIVE_TYPES[index % 4]
    identity = "INCOMPATIBLE" if hard_type == "SAME_MANUFACTURER_WRONG_MODEL" else "COMPATIBLE"
    return {
        "query_id": f"V342-{case}-{index}", "query": f"Unique {case} proposition {index}",
        "document_id": "new-official-document", "expected": "ANSWER" if answer else "ABSTAIN",
        "evidence_case": case,
        "evidence_relation_expected": "DIRECT_SUPPORTED" if answer else "UNSAFE",
        "target": "model", "relation": "asserts", "attribute": "attribute",
        "value_or_action": "value", "relevant_chunk_ids": ["chunk"] if answer else [],
        "confidence": "HIGH", "new_document": True, "identity_expected": identity,
        "identity_compatible": identity == "COMPATIBLE", "hard_negative_type": hard_type,
        "parser_recoverable": True,
    }


def test_benchmark_contract_and_acceptance_policy():
    payload = {
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "uses_a_to_h_data": False, "uses_j_data": False, "uses_k_data": False,
        "uses_historical_sealed_data": False,
        "queries": [
            _benchmark_row(case, index)
            for case in benchmark.EVIDENCE_CASES for index in range(10)
        ],
    }
    assert benchmark.validate_dataset(payload).ok
    baseline = {"false_refusals": 10, "false_answer_rate": 0.1}
    candidate = {"false_refusals": 5, "false_answer_rate": 0.15}
    result = benchmark.acceptance(
        baseline, candidate, unsafe_relax=0,
        baseline_hard_negative_fa=2, candidate_hard_negative_fa=2,
        v341_regressions=0, runtime_integrity=True,
    )
    assert result["status"] == "DEV_READY"


def test_benchmark_rejects_forbidden_corpus_flag():
    payload = {
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "uses_a_to_h_data": True, "uses_j_data": False, "uses_k_data": False,
        "uses_historical_sealed_data": False,
        "queries": [
            _benchmark_row(case, index)
            for case in benchmark.EVIDENCE_CASES for index in range(10)
        ],
    }
    assert "FORBIDDEN_CORPUS:uses_a_to_h_data" in benchmark.validate_dataset(payload).errors
