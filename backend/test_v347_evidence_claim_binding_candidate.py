from types import SimpleNamespace

from langchain_core.documents import Document

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_claim_binding_v347 import (
    EVIDENCE_CLAIM_BINDING_CANDIDATE_VERSION,
    analyze_evidence_claim_binding,
)


def _candidate(text: str):
    return RetrievalCandidate(
        document=Document(page_content=text, metadata={
            "chunk_id": "chunk-1", "document_id": "new-doc",
            "manufacturer": "Vendor", "equipment_model": "DX-100",
            "product_family": "DX", "product_series": "DX-100",
            "model_aliases": ["DX-100"],
        }),
        retrieval_source="hybrid",
    )


def _baseline(decision="ANSWER"):
    value = SimpleNamespace(
        decision=decision, reason_code="BASELINE", confidence=1.0,
        final_decision_source="V345",
    )
    value.as_dict = lambda: {"decision": decision, "reason_code": "BASELINE"}
    return value


def _patch(monkeypatch, decision="ANSWER"):
    monkeypatch.setattr(
        "backend.retrieval.evidence_claim_binding_v347.v345.analyze_evidence_decision_scope",
        lambda *args, **kwargs: _baseline(decision),
    )


def test_version_is_independent_candidate():
    assert EVIDENCE_CLAIM_BINDING_CANDIDATE_VERSION == "evidence-v347-bounded-claim-binding-candidate"


def test_weigh_alias_vetoes_only_explicit_owned_countervalue(monkeypatch):
    _patch(monkeypatch)
    result = RetrievalResult([_candidate("DX-100\nWeight 65 g")])
    decision = analyze_evidence_claim_binding(
        "Does DX-100 weigh 70 g?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert decision.action == "VETO"
    assert decision.reason_code == "EXPLICIT_ATTRIBUTE_VALUE_CONFLICT"


def test_pwr_led_row_ownership_beats_other_red_led(monkeypatch):
    _patch(monkeypatch)
    result = RetrievalResult([_candidate(
        "DX-100\nPWR LED Yes; green\nERROR LED Yes; red",
    )])
    decision = analyze_evidence_claim_binding(
        "Is the DX-100 PWR LED red?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert decision.relation["observed_values"] == ("green",)


def test_factory_qualifier_binds_nearest_explicit_setting(monkeypatch):
    _patch(monkeypatch)
    result = RetrievalResult([_candidate(
        "DX-100\nInput filter time 1 us, 8 us (factory setting), 16 us",
    )])
    conflict = analyze_evidence_claim_binding(
        "Is the DX-100 factory input filter setting 1 us?", result, [], "hybrid",
    )
    supported = analyze_evidence_claim_binding(
        "Is the DX-100 factory input filter setting 8 us?", result, [], "hybrid",
    )
    assert conflict.decision == "ABSTAIN"
    assert conflict.reason_code == "EXPLICIT_QUALIFIER_VALUE_CONFLICT"
    assert supported.decision == "ANSWER"


def test_reference_page_requires_same_attribute_row(monkeypatch):
    _patch(monkeypatch)
    result = RetrievalResult([_candidate(
        "DX-100\nReadback time guidance: see Page 60\nDiagnostics: see Page 61",
    )])
    conflict = analyze_evidence_claim_binding(
        "Does DX-100 readback-time guidance refer to Page 61?", result, [], "hybrid",
    )
    assert conflict.decision == "ABSTAIN"
    assert conflict.reason_code == "EXPLICIT_REFERENCE_OWNERSHIP_CONFLICT"


def test_missing_signal_and_every_baseline_abstain_are_preserved(monkeypatch):
    _patch(monkeypatch)
    result = RetrievalResult([_candidate("DX-100 general description")])
    missing = analyze_evidence_claim_binding("Does DX-100 weigh 70 g?", result, [], "hybrid")
    assert missing.decision == "ANSWER"
    assert missing.action == "PRESERVE"

    _patch(monkeypatch, "ABSTAIN")
    abstain = analyze_evidence_claim_binding("Does DX-100 weigh 70 g?", result, [], "hybrid")
    assert abstain.decision == "ABSTAIN"
    assert abstain.action == "PRESERVE"
