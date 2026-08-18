from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

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
from backend.retrieval.product_identity import (
    IdentityRelation,
    identity_from_metadata,
    identity_relation,
    normalize_identity_text,
)


def _document(*, error_code="", equipment_model="G120", content="G120 F0002 DC link overvoltage"):
    return SimpleNamespace(
        page_content=content,
        metadata={"chunk_id": "chunk-1", "error_code": error_code, "equipment_model": equipment_model, "source": "fixture.txt", "page": 0},
    )


def _result(document, *, vector_score=8.0, lexical_score=None):
    candidate = RetrievalCandidate(
        document=document, retrieval_source="vector", vector_score=vector_score,
        lexical_score=lexical_score, vector_rank=1, final_rank=1,
    )
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
    assert unsupported.reason == DecisionReason.MISSING_VALUE_EVIDENCE.value


def test_product_identity_normalizes_case_spacing_hyphens_and_series_suffix():
    assert normalize_identity_text("  PowerFlex\u00a0520‑Series ") == "powerflex 520 series"
    assert normalize_identity_text("powerflex 520-series") == "powerflex 520 series"


def test_product_identity_distinguishes_exact_series_family_and_mismatch():
    series = identity_from_metadata({"manufacturer": "Rockwell", "equipment_model": "PowerFlex 520-series (523/525)"})
    exact_523 = identity_from_metadata({"manufacturer": "Rockwell", "equipment_model": "PowerFlex 523"})
    exact_527 = identity_from_metadata({"manufacturer": "Rockwell", "equipment_model": "PowerFlex 527"})
    exact_5370 = identity_from_metadata({"manufacturer": "Rockwell", "equipment_model": "CompactLogix 5370"})
    exact_5380 = identity_from_metadata({"manufacturer": "Rockwell", "equipment_model": "CompactLogix 5380"})
    unknown = identity_from_metadata({})

    assert identity_relation(exact_523, series) == IdentityRelation.EXACT_MODEL
    assert identity_relation(series, exact_523) == IdentityRelation.SAME_SERIES
    assert identity_relation(exact_527, series) == IdentityRelation.SAME_FAMILY
    assert identity_relation(exact_5370, exact_5380) == IdentityRelation.SAME_FAMILY
    assert identity_relation(exact_5370, identity_from_metadata({"equipment_model": "S7-1200"})) == IdentityRelation.MISMATCH
    assert identity_relation(unknown, series) == IdentityRelation.UNKNOWN


def test_query_analysis_separates_family_series_and_exact_model():
    documents = [
        _document(equipment_model="PowerFlex 520-series (523/525)"),
        _document(equipment_model="PowerFlex 527"),
    ]
    family = analyze_query("PowerFlex 520 系列如何安全断电？", documents)
    exact = analyze_query("PowerFlex 527 如何配置网络？", documents)

    assert family.product_family == "PowerFlex"
    assert family.product_series == "PowerFlex 520"
    assert family.equipment_model == ""
    assert exact.product_family == "PowerFlex"
    assert exact.product_series == ""
    assert exact.equipment_model == "PowerFlex 527"


def test_query_analysis_uses_first_product_as_target_in_cross_model_question():
    documents = [
        _document(equipment_model="PowerFlex 520-series (523/525)"),
        _document(equipment_model="PowerFlex 527"),
    ]
    analysis = analyze_query("PowerFlex 527 能直接使用 PowerFlex 520 的 C121 吗？", documents)
    assert analysis.equipment_model == "PowerFlex 527"


def test_series_evidence_answers_but_nearby_and_unknown_exact_models_abstain():
    series_document = _document(
        equipment_model="PowerFlex 520-series (523/525)",
        content="PowerFlex 520-series disconnect power and verify DC bus voltage before service.",
    )
    series = analyze_retrieval_evidence(
        "PowerFlex 520 断电后如何确认直流母线无电？",
        _result(series_document),
        [series_document],
        "vector",
    )
    nearby = analyze_retrieval_evidence(
        "PowerFlex 527 断电后如何确认直流母线无电？",
        _result(series_document),
        [series_document],
        "vector",
    )
    unknown = analyze_retrieval_evidence(
        "PowerFlex 755 的参数是什么？",
        _result(series_document),
        [series_document],
        "vector",
    )

    assert series.decision == "ANSWER"
    assert series.identity_relation == IdentityRelation.SAME_SERIES.value
    assert nearby.decision == "ABSTAIN"
    assert nearby.reason == DecisionReason.MODEL_MISMATCH.value
    assert unknown.decision == "ABSTAIN"
    assert unknown.reason == DecisionReason.MODEL_MISMATCH.value


def test_identifier_in_candidate_text_is_known_while_unknown_s_code_abstains():
    document = _document(
        equipment_model="PowerFlex 527",
        content="PowerFlex 527 fault table: FLT S03 indicates motor overspeed.",
    )
    known = analyze_retrieval_evidence("PowerFlex 527 的 S03 是什么？", _result(document), [document], "vector")
    unknown = analyze_retrieval_evidence("PowerFlex 527 的 S98 是什么？", _result(document), [document], "vector")
    assert known.decision == "ANSWER"
    assert known.reason == DecisionReason.EXACT_IDENTIFIER_EVIDENCE.value
    assert unknown.decision == "ABSTAIN"
    assert unknown.reason == DecisionReason.UNKNOWN_IDENTIFIER.value


def test_identifier_expansion_is_shared_with_retrieval_query_analysis():
    document = _document(
        equipment_model="PowerFlex 527",
        content="PowerFlex 527 fault table: FLT S03 indicates motor overspeed.",
    )
    assert analyze_query("PowerFlex 527 的 S03 是什么？", [document]).error_code == "S03"
    assert analyze_retrieval_evidence(
        "PowerFlex 527 的 S03 是什么？", _result(document), [document], "vector",
    ).reason == DecisionReason.EXACT_IDENTIFIER_EVIDENCE.value


def test_identifier_known_only_on_another_model_does_not_authorize_answer():
    target = _document(
        equipment_model="PowerFlex 527",
        content="PowerFlex 527 EtherNet/IP configuration.",
    )
    other = _document(
        equipment_model="PowerFlex 520-series (523/525)",
        content="C121 configures communication writes for PowerFlex 520-series.",
    )
    evidence = analyze_retrieval_evidence(
        "PowerFlex 527 可以使用 C121 吗？",
        _result(target),
        [target, other],
        "vector",
    )
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.UNKNOWN_IDENTIFIER.value


def test_unsupported_protocol_and_replacement_details_remain_protected():
    document = _document(
        equipment_model="PowerFlex 527",
        content="PowerFlex 527 supports EtherNet/IP commissioning and ordinary preventive inspection.",
    )
    station = analyze_retrieval_evidence(
        "PowerFlex 527 的 PROFINET station name 怎么设置？", _result(document), [document], "vector",
    )
    replacement = analyze_retrieval_evidence(
        "PowerFlex 527 主板电容的预防性更换周期是什么？", _result(document), [document], "vector",
    )
    assert station.decision == "ABSTAIN"
    assert station.reason == DecisionReason.PROTOCOL_MISMATCH.value
    assert replacement.decision == "ABSTAIN"
    assert replacement.reason == DecisionReason.MISSING_ACTION_EVIDENCE.value


def test_unique_bare_model_does_not_bypass_incomplete_action_support():
    document = _document(equipment_model="CompactLogix 5380", content="Duplicate IP recovery guidance.")
    evidence = analyze_retrieval_evidence(
        "两台 5380 地址冲突后怎样恢复？",
        _result(document, vector_score=None, lexical_score=4.0),
        [document],
        "hybrid",
    )
    assert evidence.identity_relation == IdentityRelation.EXACT_MODEL.value
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.MISSING_ACTION_EVIDENCE.value


def test_multi_identity_query_accepts_evidence_from_either_requested_model():
    first = _document(equipment_model="Drive 100", content="Drive 100 shutdown requirement.")
    second = _document(equipment_model="Drive 200", content="Drive 200 shutdown requirement.")
    evidence = analyze_retrieval_evidence(
        "Drive 100 vs Drive 200 shutdown requirements",
        _result(second),
        [first, second],
        "vector",
    )
    assert evidence.identity_relation == IdentityRelation.EXACT_MODEL.value
    assert evidence.decision == "ANSWER"


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
    service = Mock()
    service.requested = True
    service.config.model_name = "test-reranker"
    service.retrieval_k.side_effect = lambda value: value
    with patch("backend.main.reranker", service):
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
    assert response.json()["reranker"]["reranker_effective"] is False
    service.rerank.assert_not_called()
    generate.assert_not_called()
