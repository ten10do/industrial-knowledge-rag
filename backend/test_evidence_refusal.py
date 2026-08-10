from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend import main
from fastapi.testclient import TestClient
from backend.evaluation.evidence_benchmark import (
    CALIBRATION_PATH,
    EVALUATION_PATH,
    calibrate_vector_policy,
    load_query_set,
)
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence import (
    DecisionReason,
    RetrievalEvidence,
    analyze_retrieval_evidence,
)
from backend.retrieval.filters import analyze_query


def _document(*, error_code="", equipment_model="G120", content="G120 F0002 DC link overvoltage"):
    return SimpleNamespace(
        page_content=content,
        metadata={"chunk_id": "chunk-1", "error_code": error_code, "equipment_model": equipment_model, "source": "fixture.txt", "page": 0},
    )


def _result(document, *, vector_score=8.0):
    candidate = RetrievalCandidate(document=document, retrieval_source="vector", vector_score=vector_score, vector_rank=1, final_rank=1)
    return RetrievalResult([candidate], corpus_documents=[document], retrieval_mode="vector")


def test_unknown_and_known_industrial_identifiers_have_opposite_decisions():
    document = _document(error_code="F0002")
    unknown = analyze_retrieval_evidence("F9999 应如何处理？", _result(document), [document], "vector")
    known = analyze_retrieval_evidence("F0002 应如何处理？", _result(document), [document], "vector")
    assert unknown.reason == DecisionReason.UNKNOWN_IDENTIFIER.value
    assert unknown.decision == "ABSTAIN"
    assert known.reason == DecisionReason.EXACT_IDENTIFIER_EVIDENCE.value
    assert known.decision == "ANSWER"


def test_unknown_model_and_unsupported_detail_abstain_without_query_specific_rules():
    document = _document(content="G120 maintenance isolates power before service.")
    unknown_model = analyze_retrieval_evidence("ABB ACS880 的参数是多少？", _result(document), [document], "vector")
    unsupported = analyze_retrieval_evidence("G120 端子需要多少 N·m 扭矩？", _result(document), [document], "vector")
    assert unknown_model.reason == DecisionReason.MODEL_MISMATCH.value
    assert unsupported.reason == DecisionReason.INSUFFICIENT_EVIDENCE.value


def test_ordinary_versions_are_not_industrial_identifiers():
    document = _document()
    evidence = analyze_retrieval_evidence(
        "V1、V2、2026 和 Python 3.11 的说明",
        _result(document),
        [document],
        "vector",
    )
    assert evidence.reason == DecisionReason.STRONG_VECTOR_EVIDENCE.value


def test_calibration_fixture_is_separate_and_policy_is_deterministic():
    calibration = load_query_set(CALIBRATION_PATH)
    evaluation = load_query_set(EVALUATION_PATH)
    assert len(calibration["queries"]) == 18
    assert len(evaluation["queries"]) == 32
    rows = [
        {"vector_distance": 4.0, "exact_identifier_match": False, "reason": "STRONG_VECTOR_EVIDENCE", "answerable": True},
        {"vector_distance": 10.0, "exact_identifier_match": False, "reason": "STRONG_VECTOR_EVIDENCE", "answerable": False},
    ]
    assert calibrate_vector_policy(rows) == calibrate_vector_policy(rows)


def test_api_abstain_skips_llm_and_omits_citations():
    document = _document()
    result = _result(document)
    evidence = RetrievalEvidence(
        has_candidates=True, exact_identifier_match=False, exact_model_match=True,
        lexical_score=None, lexical_margin=None, vector_distance=15.0,
        vector_margin=None, top1_top2_margin=None, metadata_consistency=True,
        retrieval_mode="vector", effective_mode="vector", decision="ABSTAIN",
        reason="INSUFFICIENT_EVIDENCE",
    )
    with patch("backend.main.retrieve_docs", return_value=result):
        with patch("backend.main.filter_relevant_docs", return_value=result):
            with patch("backend.main.analyze_evidence", return_value=evidence):
                with patch("backend.main.generate_answer") as generate:
                    response = TestClient(
                        main.app,
                        headers={"X-Knowledge-Base-ID": "kb-evidence-test-00000001"},
                    ).post("/ask", json={"question": "G120 端子扭矩是多少？"})
    assert response.status_code == 200
    assert response.json()["is_refused"] is True
    assert response.json()["sources"] == []
    assert response.json()["evidence"]["reason"] == "INSUFFICIENT_EVIDENCE"
    generate.assert_not_called()
