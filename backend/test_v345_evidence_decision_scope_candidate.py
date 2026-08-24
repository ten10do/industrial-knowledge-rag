from types import SimpleNamespace

from langchain_core.documents import Document

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_decision_scope_v345 import (
    EVIDENCE_DECISION_SCOPE_CANDIDATE_VERSION,
    analyze_evidence_decision_scope,
)


def _candidate(text: str):
    return RetrievalCandidate(
        document=Document(page_content=text, metadata={
            "chunk_id": "chunk-1", "document_id": "new-doc",
            "manufacturer": "Festo", "equipment_model": "CPX-E-16DI",
            "product_family": "CPX-E", "product_series": "CPX-E",
            "model_aliases": "CPX-E-16DI",
        }),
        retrieval_source="hybrid",
    )


def _baseline(*, decision="ANSWER", relaxed=False, reason="RULE", expanded=False):
    value = SimpleNamespace(
        decision=decision, relaxed=relaxed, reason=reason, confidence=1.0,
        final_decision_source="V342", relation={}, baseline={"expanded": expanded},
    )
    value.as_dict = lambda: {
        "decision": decision, "relaxed": relaxed, "reason": reason,
        "expanded": expanded,
    }
    return value


def test_version_is_independent_candidate():
    assert EVIDENCE_DECISION_SCOPE_CANDIDATE_VERSION == "evidence-v345-decision-scope-candidate"


def test_explicit_same_attribute_value_conflict_vetoes_existing_answer(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_decision_scope_v345.v342.analyze_evidence_sufficiency",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate("Technical data CPX-E-16DI\nWeight 670 g")])
    decision = analyze_evidence_decision_scope(
        "Does CPX-E-16DI have a weight of 130 g?", result, [], "hybrid",
    )
    assert decision.decision == "ABSTAIN"
    assert decision.action == "VETO"
    assert decision.reason_code == "EXPLICIT_ATTRIBUTE_VALUE_CONFLICT"


def test_supported_existing_answer_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_decision_scope_v345.v342.analyze_evidence_sufficiency",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate("Technical data CPX-E-16DI\nWeight 670 g")])
    decision = analyze_evidence_decision_scope(
        "Does CPX-E-16DI have a weight of 670 g?", result, [], "hybrid",
    )
    assert decision.decision == "ANSWER"
    assert decision.action == "PRESERVE"


def test_missing_or_unrecognized_relation_never_vetoes(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.evidence_decision_scope_v345.v342.analyze_evidence_sufficiency",
        lambda *args, **kwargs: _baseline(),
    )
    result = RetrievalResult([_candidate("General overview without the requested specification")])
    decision = analyze_evidence_decision_scope(
        "Does CPX-E-16DI support a proprietary diagnostic profile?", result, [], "hybrid",
    )
    assert decision.decision == "ANSWER"
    assert decision.action == "PRESERVE"
    assert decision.reason_code == "NO_EXPLICIT_CONFLICT_PRESERVED"


def test_v342_abstention_and_safe_upgrade_are_preserved(monkeypatch):
    result = RetrievalResult([_candidate("Weight 670 g")])
    monkeypatch.setattr(
        "backend.retrieval.evidence_decision_scope_v345.v342.analyze_evidence_sufficiency",
        lambda *args, **kwargs: _baseline(decision="ABSTAIN"),
    )
    abstain = analyze_evidence_decision_scope("Is the weight 670 g?", result, [], "hybrid")
    assert abstain.decision == "ABSTAIN"
    assert abstain.action == "PRESERVE"

    monkeypatch.setattr(
        "backend.retrieval.evidence_decision_scope_v345.v342.analyze_evidence_sufficiency",
        lambda *args, **kwargs: _baseline(relaxed=True),
    )
    upgrade = analyze_evidence_decision_scope("Is the weight 670 g?", result, [], "hybrid")
    assert upgrade.decision == "ANSWER"
    assert upgrade.action == "UPGRADE"
