"""Tests for the V3.40 coverage-relation evidence sufficiency candidate."""

from types import SimpleNamespace

from langchain_core.documents import Document

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_sufficiency_v340 import (
    EVIDENCE_COVERAGE_CANDIDATE_VERSION,
    RELAX_CONFIDENCE_FLOOR,
    analyze_coverage_evidence_sufficiency,
    classify_coverage_relation,
)


def _candidate(text: str, *, model: str = "MX-100", manufacturer: str = "Acme",
               chunk: str = "c1", series: str = "MX-100 series"):
    document = Document(
        page_content=text,
        metadata={
            "chunk_id": chunk,
            "document_id": "manual-1",
            "manufacturer": manufacturer,
            "product_family": "MX",
            "product_series": series,
            "equipment_model": model,
        },
    )
    return RetrievalCandidate(document=document, retrieval_source="hybrid")


def _baseline(*, identity: str = "COMPATIBLE", delegated: bool = True, decision: str = "ABSTAIN"):
    value = SimpleNamespace(
        query_path="VERIFICATION",
        decision=decision,
        reason="RULE",
        final_decision_source="RULE",
        identity_boundary={"status": identity},
        delegated_to_existing_evidence=delegated,
        existing_evidence={"base_rule_reason": "MISSING_VALUE_EVIDENCE"},
    )
    value.as_dict = lambda: {
        "decision": decision,
        "identity_boundary": value.identity_boundary,
        "delegated_to_existing_evidence": delegated,
    }
    return value


def test_candidate_version_is_explicit():
    assert EVIDENCE_COVERAGE_CANDIDATE_VERSION == "evidence-v340-coverage-candidate"
    assert 0.0 < RELAX_CONFIDENCE_FLOOR < 1.0


# ---------------------------------------------------------------------------
# DIRECT_PARAMETER
# ---------------------------------------------------------------------------

def test_direct_parameter_block_is_supported():
    relation = classify_coverage_relation(
        "Is the MX-100 supply voltage range 15 to 32 VDC?",
        [_candidate("MX-100\nSupply voltage range : 15~32Vdc")],
    )
    assert relation.relation == "DIRECT"
    assert relation.reason_code == "DIRECT_PARAMETER_SUPPORTED"
    assert relation.confidence >= RELAX_CONFIDENCE_FLOOR


def test_same_model_wrong_value_is_a_value_scope_conflict():
    relation = classify_coverage_relation(
        "Is the MX-100 supply voltage range 15 to 32 VDC?",
        [_candidate("MX-100\nSupply voltage range : 15~28.8Vdc")],
    )
    assert relation.relation == "UNSUPPORTED"
    assert relation.reason_code == "VALUE_SCOPE_CONFLICT"


def test_sibling_model_value_assertion_is_a_negative_scope_conflict():
    relation = classify_coverage_relation(
        "Is the MX-100 supply voltage range 15 to 32 VDC?",
        [_candidate("MX-200\nSupply voltage range : 15~32Vdc", model="MX-200",
                    series="MX-100 series")],
    )
    assert relation.relation == "UNSUPPORTED"
    assert relation.reason_code == "NEGATIVE_SCOPE_CONFLICT"


# ---------------------------------------------------------------------------
# PRODUCT_FAMILY_INHERITANCE
# ---------------------------------------------------------------------------

def test_family_claim_without_manual_quantifier_is_unsupported():
    relation = classify_coverage_relation(
        "Does every MX-100 variant include a CANbus port?",
        [_candidate("The MX-100 includes a CANbus port.")],
    )
    assert relation.relation == "UNSUPPORTED"
    assert relation.reason_code == "FAMILY_INHERITANCE_UNSUPPORTED"


def test_family_claim_with_explicit_quantifier_is_inherited():
    relation = classify_coverage_relation(
        "May one CANbus port be added to every MX-100 model?",
        [_candidate("One CANbus port may be added to all models.")],
    )
    assert relation.relation == "INHERITED"
    assert relation.reason_code == "FAMILY_INHERITANCE_SUPPORTED"
    assert "FAMILY_QUANTIFIER" in relation.anchors


# ---------------------------------------------------------------------------
# CROSS_SECTION_REFERENCE
# ---------------------------------------------------------------------------

def test_cross_section_reference_with_explicit_link_is_supported():
    relation = classify_coverage_relation(
        "Does the MX-100 manual refer EtherCAT connection details to manual W505?",
        [_candidate(
            "For details on the built-in EtherCAT port, refer to the Built-in "
            "EtherCAT User's Manual (Cat. No. W505)."
        )],
    )
    assert relation.relation == "REFERENCED"
    assert relation.reason_code == "CROSS_SECTION_LINK_EXPLICIT"


def test_cross_section_reference_without_link_is_refused():
    relation = classify_coverage_relation(
        "Does the MX-100 manual refer EtherCAT connection details to manual W509?",
        [_candidate("EtherCAT connection uses the standard built-in port.")],
    )
    assert relation.relation == "UNSUPPORTED"
    assert relation.reason_code in ("CROSS_SECTION_LINK_MISSING", "LEXICAL_COVERAGE_INSUFFICIENT")


# ---------------------------------------------------------------------------
# MODULE_PARENT_RELATION / CONFIGURATION_DEPENDENCY
# ---------------------------------------------------------------------------

def test_module_parent_compatibility_is_dependent():
    relation = classify_coverage_relation(
        "Can the MX-100 controller support the EXP-8 expansion module?",
        [_candidate("The MX-100 controller supports the EXP-8 expansion module.")],
    )
    assert relation.relation == "DEPENDENT"
    assert relation.reason_code == "MODULE_PARENT_SUPPORTED"


def test_configuration_dependency_with_explicit_requirement_is_supported():
    relation = classify_coverage_relation(
        "Does the MX-100 PID loop require parameter P00?",
        [_candidate("The PID loop requires parameter P00 to be set before operation.")],
    )
    assert relation.relation == "DEPENDENT"
    assert relation.reason_code == "CONFIGURATION_DEPENDENCY_SUPPORTED"


def test_configuration_dependency_without_requirement_statement_is_refused():
    relation = classify_coverage_relation(
        "Does the MX-100 PID loop require parameter P00?",
        [_candidate("The PID loop can be tuned from the keypad.")],
    )
    assert relation.relation == "UNSUPPORTED"
    assert relation.reason_code in ("CONFIGURATION_DEPENDENCY_MISSING", "LEXICAL_COVERAGE_INSUFFICIENT")


# ---------------------------------------------------------------------------
# Inherited guards and the decision wrapper
# ---------------------------------------------------------------------------

def test_negation_and_procedure_queries_are_never_relaxed():
    negated = classify_coverage_relation(
        "Can the MX-100 operate without protective grounding?",
        [_candidate("MX-100 grounding terminal is protective.")],
    )
    procedural = classify_coverage_relation(
        "Should MX-100 wiring be performed after power off?",
        [_candidate("MX-100 wiring section.")],
    )
    assert negated.reason_code == "NEGATED_OR_EXCLUSIVE_PROPOSITION"
    assert procedural.reason_code == "PROCEDURE_RELAXATION_FORBIDDEN"


def test_identity_boundary_is_never_bypassed(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v340.analyze_identity_aware_evidence",
        lambda *args, **kwargs: _baseline(identity="INCOMPATIBLE", delegated=False),
    )
    result = RetrievalResult([_candidate("MX-100 supply voltage range 15~32Vdc")])
    decision = analyze_coverage_evidence_sufficiency(
        "Is the MX-100 supply voltage range 15 to 32 VDC?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert not decision.relaxed
    assert decision.reason == "IDENTITY_BOUNDARY_PRESERVED"


def test_compatible_soft_abstention_is_relaxed_through_typed_relation(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v340.analyze_identity_aware_evidence",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate("MX-100\nSupply voltage range : 15~32Vdc")])
    decision = analyze_coverage_evidence_sufficiency(
        "Is the MX-100 supply voltage range 15 to 32 VDC?", result, [], "hybrid",
    )
    assert decision.decision == "ANSWER"
    assert decision.relaxed
    assert decision.final_decision_source == "V340_COVERAGE"
    assert decision.relation["relation"] == "DIRECT"
    assert decision.relation["reason_code"] == "DIRECT_PARAMETER_SUPPORTED"


def test_scope_conflict_blocks_the_relax(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v340.analyze_identity_aware_evidence",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate("MX-100\nSupply voltage range : 15~28.8Vdc")])
    decision = analyze_coverage_evidence_sufficiency(
        "Is the MX-100 supply voltage range 15 to 32 VDC?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert not decision.relaxed
    assert decision.reason == "VALUE_SCOPE_CONFLICT"