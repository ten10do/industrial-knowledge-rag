"""Focused regressions for the V3.18 cross-corpus Evidence contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.retrieval import (
    DecisionReason,
    RetrievalCandidate,
    RetrievalResult,
    SupportStatus,
    analyze_query,
    analyze_retrieval_evidence,
    validate_evidence_support,
)
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


def _doc(chunk_id: str, content: str, *, model: str = "ACS580", family: str = "ACS"):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "manufacturer": "ABB",
            "equipment_model": model,
            "product_family": family,
            "source": "v318-fixture.txt",
            "page": 1,
        },
    )


def _result(query: str, candidates: list, corpus: list | None = None) -> RetrievalResult:
    corpus = corpus or candidates
    return RetrievalResult(
        [
            RetrievalCandidate(
                document=document,
                retrieval_source="frozen_fixture",
                final_rank=rank,
                vector_rank=rank,
                vector_score=8.0 + rank,
                lexical_score=2.0,
            )
            for rank, document in enumerate(candidates, start=1)
        ],
        query_analysis=analyze_query(query, corpus),
        corpus_documents=corpus,
        retrieval_mode="frozen_fixture",
    )


def _evidence(query: str, candidates: list, corpus: list | None = None):
    corpus = corpus or candidates
    return analyze_retrieval_evidence(
        query,
        _result(query, candidates, corpus),
        corpus,
        "frozen_fixture",
    )


@pytest.mark.parametrize(
    ("query", "document", "reason"),
    [
        (
            "What MQTT broker port is assigned to the ACS580?",
            _doc("model", "ACS580 supports drive commissioning over fieldbus."),
            DecisionReason.PROTOCOL_MISMATCH.value,
        ),
        (
            "What does ACS580 parameter 99.99 control?",
            _doc("identifier", "ACS580 parameter 58.18 reports fieldbus status."),
            DecisionReason.UNKNOWN_PARAMETER.value,
        ),
        (
            "Which BACnet baud rate must be selected on the ACS580?",
            _doc("protocol", "ACS580 Ethernet setup uses EtherNet/IP addressing."),
            DecisionReason.PROTOCOL_MISMATCH.value,
        ),
        (
            "What is the operating-temperature range for ACS580 parameter 58.18?",
            _doc("attribute", "ACS580 parameter 58.18 reports fieldbus status."),
            DecisionReason.MISSING_ATTRIBUTE_EVIDENCE.value,
        ),
        (
            "What exact numeric voltage is required for the ACS580 safe supply?",
            _doc("value", "The ACS580 safety supply must use protective extra-low voltage."),
            DecisionReason.MISSING_VALUE_EVIDENCE.value,
        ),
        (
            "Which firmware must be installed before ACS580 parameter 58.18 is available?",
            _doc(
                "requirement",
                "ACS580 parameter 58.18 reports fieldbus status. Firmware settings are documented separately.",
            ),
            DecisionReason.MISSING_REQUIREMENT_EVIDENCE.value,
        ),
        (
            "Which exact configuration-data bytes are needed by an EtherNet/IP adapter that cannot exchange process data without configuration on the ACS580?",
            _doc("partial", "ACS580 supports EtherNet/IP process data exchange without configuration data."),
            DecisionReason.PARTIAL_EVIDENCE_ONLY.value,
        ),
    ],
)
def test_exact_model_does_not_bypass_missing_requirements(query, document, reason):
    evidence = _evidence(query, [document])
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == reason


def test_identifier_existing_elsewhere_is_not_current_evidence_support():
    current = _doc("current", "ACS580 parameter overview. Related Parameters: 30.20")
    elsewhere = _doc("elsewhere", "ACS580 parameter 30.20 Maximum torque 1 has a 0 to 300 percent range.")
    query = "What range is defined for ACS580 parameter 30.20?"
    evidence = _evidence(query, [current], [current, elsewhere])
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.IDENTIFIER_NOT_IN_EVIDENCE.value


def test_semantic_hard_positive_remains_answerable():
    document = _doc("semantic", "Disconnect input power before servicing the drive cabinet.")
    evidence = _evidence("Should technicians de-energize the ACS580 before maintenance?", [document])
    assert evidence.decision == "ANSWER"


def test_identifier_alternate_wording_remains_answerable():
    document = _doc("identifier-positive", "30.20 Maximum torque 1: default 300%; range 0 to 300%.")
    evidence = _evidence("Give the default and range for ACS580 item 30.20 Maximum torque 1.", [document])
    assert evidence.decision == "ANSWER"


def test_attribute_paraphrase_remains_answerable():
    document = _doc(
        "attribute-positive",
        "ACS580 parameter 30.20 above 100% can overheat the motor and damage equipment.",
    )
    evidence = _evidence(
        "Explain the equipment-hazard rationale for keeping ACS580 parameter 30.20 below 100 percent.",
        [document],
    )
    assert evidence.decision == "ANSWER"


def test_valid_multi_chunk_evidence_remains_answerable():
    documents = [
        _doc("multi-a", "Configure the ACS580 and assign a static IP address."),
        _doc("multi-b", "Power cycle the ACS580, test it, and authorize the system for use."),
    ]
    evidence = _evidence("How do I configure a static IP and then commission the ACS580?", documents)
    assert evidence.decision == "ANSWER"


def test_evidence_then_support_pipeline_keeps_unsupported_detail_out():
    query = "What ControlNet node address is required by the ACS580?"
    document = _doc("pipeline", "ACS580 supports EtherNet/IP commissioning.")
    result = _result(query, [document])
    evidence = analyze_retrieval_evidence(query, result, [document], "frozen_fixture")
    support = validate_evidence_support(query, result, [document]) if evidence.decision == "ANSWER" else None
    final_decision = "ANSWER" if support and support.status == SupportStatus.SUPPORTED.value else "ABSTAIN"
    assert evidence.decision == "ABSTAIN"
    assert final_decision == "ABSTAIN"


def test_rule_identity_changes_only_evidence_version():
    assert EVIDENCE_SUPPORT_RULE_VERSION == "evidence-v321.1"
    assert SUPPORT_RULE_VERSION == "support-v316.1"
