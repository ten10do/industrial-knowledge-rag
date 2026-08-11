from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import main
from backend.retrieval import (
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalResult,
    SupportReason,
    SupportStatus,
    analyze_query,
    support_gate_enabled,
    validate_evidence_support,
)


def _doc(chunk_id, content, model="PowerFlex 527", family="PowerFlex"):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "manufacturer": "Rockwell Automation",
            "equipment_model": model,
            "product_family": family,
        },
    )


def _result(query, candidates, corpus=None):
    corpus = corpus or candidates
    analysis = analyze_query(query, corpus)
    return RetrievalResult(
        [
            RetrievalCandidate(
                document=document,
                retrieval_source="hybrid",
                final_rank=rank,
                identity_relation="EXACT_MODEL",
                scope_match="primary",
            )
            for rank, document in enumerate(candidates, start=1)
        ],
        query_analysis=analysis,
        corpus_documents=corpus,
        retrieval_mode="hybrid",
    )


def test_supported_procedure_uses_action_and_concept_coverage():
    query = "PowerFlex 527 如何配置静态 IP 地址？"
    result = _result(query, [_doc("ip", "Configure the drive IP address. Select Static and assign the IP address.")])
    support = validate_evidence_support(query, result)
    assert support.status == SupportStatus.SUPPORTED.value
    assert support.coverage["requested_actions"]["configure"] is True
    assert support.coverage["technical_concepts"]["ip_address"] is True


def test_unsupported_procedure_reports_missing_protocol_not_topic_similarity():
    query = "PowerFlex 527 如何配置 PROFINET 设备名称并投入运行？"
    result = _result(query, [_doc("enet", "Configure the EtherNet/IP drive. Type the drive Name and select an Ethernet address.")])
    support = validate_evidence_support(query, result)
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_REQUIRED_CONCEPT.value
    assert "protocol:profinet" in support.missing_requirements


def test_supported_and_unsupported_parameter_values():
    supported_query = "PowerFlex 520 的 A438 动态制动阈值如何定义？"
    supported_doc = _doc(
        "a438",
        "A438 DB Threshold sets the DC bus voltage threshold for Dynamic Brake operation.",
        "PowerFlex 520-series (523/525)",
    )
    assert validate_evidence_support(
        supported_query, _result(supported_query, [supported_doc]),
    ).status == SupportStatus.SUPPORTED.value

    unsupported_query = "PowerFlex 527 的模拟输出精度是多少？"
    unsupported = validate_evidence_support(
        unsupported_query,
        _result(unsupported_query, [_doc("motion", "Configure integrated motion over EtherNet/IP.")]),
    )
    assert unsupported.status == SupportStatus.INSUFFICIENT.value
    assert {"concept:analog_output", "concept:accuracy"}.issubset(unsupported.missing_requirements)


def test_exact_identifier_support_and_model_mismatch_are_independent_signals():
    identifier_query = "PowerFlex 527 的 FLT S03 表示什么？"
    identifier = validate_evidence_support(
        identifier_query,
        _result(identifier_query, [_doc("s03", "FLT S03 – MTR OVERSPEED FL Motor Overspeed Factory Limit Fault")]),
    )
    assert identifier.status == SupportStatus.SUPPORTED.value
    assert identifier.coverage["identifiers"]["S03"] is True

    query = "Drive 100 如何配置 IP 地址？"
    corpus = [
        _doc("right", "Drive 100 IP address configuration", "Drive 100", "Drive"),
        _doc("wrong", "Drive 200 IP address configuration", "Drive 200", "Drive"),
    ]
    mismatch = validate_evidence_support(query, _result(query, [corpus[1]], corpus))
    assert mismatch.status == SupportStatus.INSUFFICIENT.value
    assert mismatch.reason == SupportReason.MODEL_MISMATCH.value


def test_semantic_hard_positive_is_not_reduced_to_literal_query_matching():
    query = "PowerFlex 527 维修前是否必须断开主电源？"
    document = _doc("safety", "Remove input power before servicing the drive. Verify that voltage is zero.")
    support = validate_evidence_support(query, _result(query, [document]))
    assert support.status == SupportStatus.SUPPORTED.value
    assert support.coverage["technical_concepts"]["input_power"] is True
    assert support.coverage["requested_actions"]["remove_power"] is True


def test_lexical_hard_negative_rejects_neighboring_protocol():
    query = "PowerFlex 527 如何设置 PROFIBUS 节点地址？"
    document = _doc("neighbor", "Set the EtherNet/IP node address and configure the IP address.")
    support = validate_evidence_support(query, _result(query, [document]))
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert "protocol:profibus" in support.missing_requirements


def test_multi_chunk_evidence_is_aggregated():
    query = "PowerFlex 527 如何配置静态 IP 地址并投入运行？"
    documents = [
        _doc("configure", "Configure the drive and assign a Static IP address."),
        _doc("commission", "Power cycle the drive, test it, and authorize the system for use."),
    ]
    support = validate_evidence_support(query, _result(query, documents))
    assert support.status == SupportStatus.SUPPORTED.value
    assert set(support.supporting_chunks) == {"configure", "commission"}


def test_support_output_is_deterministic_and_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SUPPORT_GATE_ENABLED", raising=False)
    assert support_gate_enabled() is False
    query = "PowerFlex 527 如何配置静态 IP 地址？"
    result = _result(query, [_doc("ip", "Configure the Static IP address.")])
    assert validate_evidence_support(query, result).as_dict() == validate_evidence_support(query, result).as_dict()


def _base_evidence(decision="ANSWER"):
    return RetrievalEvidence(
        has_candidates=True,
        exact_identifier_match=False,
        exact_model_match=True,
        lexical_score=1.0,
        lexical_margin=None,
        vector_distance=1.0,
        vector_margin=None,
        top1_top2_margin=None,
        metadata_consistency=True,
        retrieval_mode="hybrid",
        effective_mode="hybrid",
        decision=decision,
        reason="EXACT_MODEL_EVIDENCE" if decision == "ANSWER" else "INSUFFICIENT_EVIDENCE",
    )


def _api_reranker():
    service = Mock()
    service.requested = False
    service.retrieval_k.side_effect = lambda value: value
    return service


def test_base_abstain_skips_support_validator():
    query = "PowerFlex 527 如何配置 PROFINET？"
    result = _result(query, [_doc("candidate", "EtherNet/IP configuration")])
    service = _api_reranker()
    with patch("backend.main.reranker", service), patch("backend.main.retrieve_docs", return_value=result), patch(
        "backend.main.filter_relevant_docs", return_value=result,
    ), patch("backend.main.analyze_evidence", return_value=_base_evidence("ABSTAIN")), patch(
        "backend.main.support_gate_enabled", return_value=True,
    ), patch("backend.main.validate_evidence_support") as validator:
        response = TestClient(
            main.app,
            headers={"X-Knowledge-Base-ID": "kb-support-test-00000001"},
        ).post("/ask", json={"question": query})
    assert response.status_code == 200
    assert response.json()["is_refused"] is True
    validator.assert_not_called()


def test_insufficient_support_skips_llm_and_omits_citations():
    query = "PowerFlex 527 如何配置 PROFINET 设备名称？"
    result = _result(query, [_doc("candidate", "EtherNet/IP network configuration")])
    support = validate_evidence_support(query, result)
    service = _api_reranker()
    with patch("backend.main.reranker", service), patch("backend.main.retrieve_docs", return_value=result), patch(
        "backend.main.filter_relevant_docs", return_value=result,
    ), patch("backend.main.has_relevant_docs", return_value=True), patch(
        "backend.main.analyze_evidence", return_value=_base_evidence(),
    ), patch(
        "backend.main.support_gate_enabled", return_value=True,
    ), patch("backend.main.validate_evidence_support", return_value=support), patch(
        "backend.main.generate_answer",
    ) as generate:
        response = TestClient(
            main.app,
            headers={"X-Knowledge-Base-ID": "kb-support-test-00000002"},
        ).post("/ask", json={"question": query})
    payload = response.json()
    assert response.status_code == 200
    assert payload["is_refused"] is True
    assert payload["sources"] == []
    assert payload["support"]["status"] == "INSUFFICIENT"
    generate.assert_not_called()
