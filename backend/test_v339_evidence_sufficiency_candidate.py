from types import SimpleNamespace

from langchain_core.documents import Document

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_sufficiency_v339 import (
    EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION,
    analyze_evidence_sufficiency,
    evaluate_compatible_evidence_relation,
)


def _candidate(text: str, *, model: str = "URD-0800", manufacturer: str = "Unitronics", chunk: str = "c1"):
    document = Document(
        page_content=text,
        metadata={
            "chunk_id": chunk,
            "document_id": "manual-1",
            "manufacturer": manufacturer,
            "product_family": "UniStream",
            "product_series": "URB Remote I/O",
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
    assert EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION == "evidence-v339-sufficiency-candidate"


def test_same_parameter_block_with_model_value_and_unit_is_supported():
    candidate = _candidate(
        "URD-0800 (DI08) - 8 Digital Inputs\n"
        "Input Signal Delay OFF to ON: 0.3 ms Max\n"
        "ON to OFF: 0.3 ms Max"
    )
    result = evaluate_compatible_evidence_relation(
        "Is the URD-0800 input signal delay at most 0.3 ms in either direction?",
        [candidate],
    )
    assert result.supported
    assert result.relation == "SAME_PARAMETER_BLOCK"


def test_different_model_and_conflicting_value_are_not_supported():
    wrong_model = _candidate(
        "URD-0808 input signal delay OFF to ON: 0.3 ms Max", model="URD-0808",
    )
    wrong_value = _candidate(
        "URD-0800 input signal delay OFF to ON: 30 ms Max",
    )
    query = "Is the URD-0800 input signal delay at most 0.3 ms?"
    assert not evaluate_compatible_evidence_relation(query, [wrong_model]).supported
    assert not evaluate_compatible_evidence_relation(query, [wrong_value]).supported


def test_family_alias_does_not_authorize_a_different_model_value():
    candidate = _candidate(
        "URD-0032NG-4 output voltage range is 15 to 32 VDC.",
        model="UniStream Remote I/O",
    )
    candidate.document.metadata["model_aliases"] = "URB-TCP|URB-TCP2|URD-0032NG-4"
    assert not evaluate_compatible_evidence_relation(
        "Is the URB-TCP2 system-power supply range 15 to 32 VDC?", [candidate],
    ).supported


def test_model_and_value_cannot_be_joined_across_candidates():
    model = _candidate("URD-0800 digital input module", chunk="model")
    value = _candidate("Input signal delay is 0.3 ms maximum", model="URD-0808", chunk="value")
    result = evaluate_compatible_evidence_relation(
        "Is the URD-0800 input signal delay at most 0.3 ms?", [model, value],
    )
    assert not result.supported


def test_procedure_and_negative_relaxations_are_forbidden():
    candidate = _candidate("URD-0800 wiring may be performed with power on.")
    assert not evaluate_compatible_evidence_relation(
        "Should URD-0800 wiring be performed after power is off?", [candidate],
    ).supported
    assert not evaluate_compatible_evidence_relation(
        "Can URD-0800 operate without protective grounding?", [candidate],
    ).supported


def test_explicit_contradiction_is_not_relaxed():
    candidate = _candidate(
        "A CJ-series Power Supply Unit cannot be used to power an NJ-system Expansion Rack.",
        model="NJ-series CPU Unit",
        manufacturer="Omron",
    )
    assert not evaluate_compatible_evidence_relation(
        "May a CJ-series Power Supply Unit power an NJ-system Expansion Rack?", [candidate],
    ).supported


def test_adverse_scope_is_not_interpreted_as_permission():
    candidate = _candidate(
        "A CJ-series supply may power the rack without error detection, but operation may be unstable "
        "due to insufficient supplied power.",
        model="NJ-series CPU Unit",
        manufacturer="Omron",
    )
    assert not evaluate_compatible_evidence_relation(
        "May a CJ-series Power Supply Unit power an NJ-system Expansion Rack?", [candidate],
    ).supported


def test_identity_boundary_is_never_bypassed(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v339.analyze_identity_aware_evidence",
        lambda *args, **kwargs: _baseline(identity="INCOMPATIBLE", delegated=False),
    )
    result = RetrievalResult([_candidate("URD-0800 input signal delay is 0.3 ms")])
    decision = analyze_evidence_sufficiency(
        "Is the URD-0800 input signal delay 0.3 ms?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert not decision.relaxed
    assert decision.reason == "IDENTITY_BOUNDARY_PRESERVED"


def test_compatible_soft_abstention_can_be_relaxed(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_sufficiency_v339.analyze_identity_aware_evidence",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate(
        "URD-0800 (DI08)\nInput Signal Delay OFF to ON: 0.3 ms Max\nON to OFF: 0.3 ms Max"
    )])
    decision = analyze_evidence_sufficiency(
        "Is the URD-0800 input signal delay at most 0.3 ms in either direction?",
        result,
        [],
        "hybrid",
    )
    assert decision.decision == "ANSWER"
    assert decision.relaxed
    assert decision.final_decision_source == "V339_SUFFICIENCY"
