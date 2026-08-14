"""Cross-vendor Evidence/Support generalization tests for V3.11.

Covers protocol normalization and mismatch, cross-equipment detection, unknown
parameter references, hyphen-insensitive concept/action matching, hard positives
(paraphrased but supported) and hard negatives (related topic but unsupported).
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval import (
    DecisionReason,
    RetrievalCandidate,
    RetrievalResult,
    SupportReason,
    SupportStatus,
    analyze_query,
    analyze_retrieval_evidence,
    build_evidence_requirement,
    validate_evidence_support,
)
from backend.retrieval.technical import (
    PROTOCOL_ALIASES,
    contains_parameter_identifier,
    contains_term,
    extract_parameter_references,
    foreign_equipment_signal,
    matched_terms,
    normalize_parameter_identifier,
)


def _doc(chunk_id, content, model="PowerFlex 527", manufacturer="Rockwell Automation", family="PowerFlex"):
    return SimpleNamespace(
        page_content=content,
        metadata={"chunk_id": chunk_id, "manufacturer": manufacturer, "equipment_model": model, "product_family": family},
    )


def _result(query, candidates, corpus=None):
    corpus = corpus or candidates
    analysis = analyze_query(query, corpus)
    return RetrievalResult(
        [
            RetrievalCandidate(
                document=document, retrieval_source="hybrid", final_rank=rank,
                identity_relation="EXACT_MODEL", vector_score=8.0, lexical_score=1.0,
                vector_rank=rank,
            )
            for rank, document in enumerate(candidates, start=1)
        ],
        query_analysis=analysis,
        corpus_documents=corpus,
        retrieval_mode="hybrid",
    )


def test_protocol_normalization_distinguishes_protocols():
    assert matched_terms("set a PROFINET station name", PROTOCOL_ALIASES) == ("profinet",)
    assert matched_terms("uses EtherNet/IP", PROTOCOL_ALIASES) == ("ethernet_ip",)
    assert matched_terms("Modbus TCP configuration", PROTOCOL_ALIASES) == ("modbus",)
    assert matched_terms("no protocol here", PROTOCOL_ALIASES) == ()


def test_hyphen_and_whitespace_are_equivalent_for_terms():
    assert contains_term("first start-up assistant", ("startup",))
    assert contains_term("device-level ring topology", ("device-level ring",))
    assert contains_term("the EtherNet/IP adapter", PROTOCOL_ALIASES["ethernet_ip"])


def test_protocol_mismatch_abstains_when_scoped_evidence_lacks_protocol():
    document = _doc("pf527", "PowerFlex 527 supports EtherNet/IP commissioning only.")
    evidence = analyze_retrieval_evidence(
        "How do I set a PROFINET station name on a PowerFlex 527?", _result("q", [document]), [document], "hybrid",
    )
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.PROTOCOL_MISMATCH.value


def test_protocol_match_answers_when_evidence_supports_protocol():
    document = _doc("fpno", "The FPNO-21 PROFINET adapter accepts a station name.", model="FPNO-21", manufacturer="ABB", family="FPNO")
    evidence = analyze_retrieval_evidence(
        "What PROFINET station name does the FPNO-21 require?", _result("q", [document]), [document], "hybrid",
    )
    assert evidence.decision == "ANSWER"


def test_cross_equipment_abstains_when_query_references_foreign_vendor():
    document = _doc("cx", "CX-Programmer connects to an Omron CJ2 PLC.", model="CX-Programmer", manufacturer="Omron", family="CX")
    evidence = analyze_retrieval_evidence(
        "Can CX-Programmer program a Siemens S7-1200?", _result("q", [document]), [document], "hybrid",
    )
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.CROSS_EQUIPMENT.value


def test_cross_equipment_does_not_trigger_for_in_corpus_vendor():
    document = _doc("pf520", "PowerFlex 520 configures a static IP.", model="PowerFlex 520-series (523/525)")
    evidence = analyze_retrieval_evidence(
        "How do I configure a static IP on a PowerFlex 520?", _result("q", [document]), [document], "hybrid",
    )
    assert evidence.decision == "ANSWER"


def test_unknown_parameter_reference_abstains():
    document = _doc("acs", "ACS580 parameter 58.18 stores the fieldbus status.", model="ACS580", manufacturer="ABB", family="ACS")
    unknown = analyze_retrieval_evidence(
        "Is ACS580 parameter 99.99 a SIL safety setting?", _result("q", [document]), [document], "hybrid",
    )
    assert unknown.decision == "ABSTAIN"
    assert unknown.reason == DecisionReason.UNKNOWN_PARAMETER.value
    known = analyze_retrieval_evidence(
        "What does ACS580 parameter 58.18 store?", _result("q", [document]), [document], "hybrid",
    )
    assert known.decision == "ANSWER"


def test_parameter_identifier_concept_and_literal_are_separate():
    references = extract_parameter_references("What does parameter Pr.12 mean?")
    assert [(item.concept, item.identifier) for item in references] == [("parameter", "PR12")]
    requirement = build_evidence_requirement("What does parameter P041 mean?", [])
    assert requirement.identifiers == ("P041",)
    assert requirement.requested_concepts == ("parameter",)


def test_parameter_identifier_normalization_supports_corpus_forms():
    expected = {
        "04.16": "04.16", "p1080": "P1080", "r1234": "R1234",
        "A1-03": "A1-03", "Pr.12": "PR12", "MW20": "MW20",
    }
    assert {value: normalize_parameter_identifier(value) for value in expected} == expected
    assert contains_parameter_identifier("Pr.12 is followed by P1080.", "pr12")


def test_bare_numeric_known_and_unknown_parameters():
    document = _doc(
        "acs", "ACS580 99.13 ID run requested.",
        model="ACS580", manufacturer="ABB", family="ACS",
    )
    known_query = "Explain 99.13 on the ACS580."
    unknown_query = "What is 97.97 on the ACS580?"
    known = analyze_retrieval_evidence(
        known_query, _result(known_query, [document]), [document], "hybrid",
    )
    unknown = analyze_retrieval_evidence(
        unknown_query, _result(unknown_query, [document]), [document], "hybrid",
    )
    assert known.decision == "ANSWER"
    assert unknown.reason == DecisionReason.UNKNOWN_PARAMETER.value


def test_alphanumeric_known_and_unknown_parameters():
    document = _doc(
        "pf", "P041 Accel Time 1 controls acceleration.",
        model="PowerFlex 520-series (523/525)",
    )
    known_query = "Explain P041 on the PowerFlex 520."
    unknown_query = "What does parameter P9999 control on the PowerFlex 520?"
    known = analyze_retrieval_evidence(
        known_query, _result(known_query, [document]), [document], "hybrid",
    )
    unknown = analyze_retrieval_evidence(
        unknown_query, _result(unknown_query, [document]), [document], "hybrid",
    )
    assert known.decision == "ANSWER"
    assert unknown.reason == DecisionReason.UNKNOWN_PARAMETER.value


def test_parameter_prefix_is_optional_but_decimal_measurement_is_not_a_parameter():
    assert extract_parameter_references("What is 30.11 on the ACS580?")[0].identifier == "30.11"
    assert extract_parameter_references("The threshold is 30.11 V.") == ()


def test_parameter_identifier_must_exist_in_the_requested_product_scope():
    target = _doc(
        "target", "PowerFlex 520 parameter overview.",
        model="PowerFlex 520-series (523/525)",
    )
    other = _doc(
        "other", "P777 is defined for this controller.",
        model="PowerFlex 527",
    )
    query = "What does PowerFlex 520 parameter P777 control?"
    evidence = analyze_retrieval_evidence(
        query, _result(query, [target], [target, other]), [target, other], "hybrid",
    )
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.UNKNOWN_PARAMETER.value


def test_input_power_concept_matches_power_removal_wording():
    query = "Do you need to disconnect input power before servicing the controller?"
    result = _result(query, [_doc("p", "WARNING: be sure that power is removed before servicing the controller.")])
    support = validate_evidence_support(query, result)
    assert support.status == SupportStatus.SUPPORTED.value
    assert "concept:input_power" not in support.missing_requirements


def test_hard_positive_semantic_support_paraphrase():
    query = "Should I cut power before opening the drive for maintenance?"
    result = _result(query, [_doc("p", "Disconnect input power before servicing this equipment.")])
    support = validate_evidence_support(query, result)
    assert support.status == SupportStatus.SUPPORTED.value


def test_hard_negative_related_topic_but_unsupported_detail():
    query = "What is the MQTT broker port for the FPNO-21?"
    document = _doc("p", "The FPNO-21 supports PROFINET station naming.", model="FPNO-21", manufacturer="ABB", family="FPNO")
    evidence = analyze_retrieval_evidence(query, _result(query, [document]), [document], "hybrid")
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.INSUFFICIENT_EVIDENCE.value


def test_partial_support_is_rejected_not_accepted():
    query = "How do I reset the admin password if I lost it and cannot reach the module?"
    document = _doc("p", "On the Password page you can change your password and username.", model="FPNO-21", manufacturer="ABB", family="FPNO")
    evidence = analyze_retrieval_evidence(query, _result(query, [document]), [document], "hybrid")
    assert evidence.decision == "ABSTAIN"
    assert evidence.reason == DecisionReason.UNSUPPORTED_PROCEDURE.value


def test_foreign_equipment_signal_requires_corpus_vendor_metadata():
    assert foreign_equipment_signal("config a Siemens S7", [SimpleNamespace(metadata={"manufacturer": "Omron"})]) == ("siemens", "siemens")
    assert foreign_equipment_signal("config a CX-Programmer", [SimpleNamespace(metadata={"manufacturer": "Omron"})]) is None
    assert foreign_equipment_signal("config anything", [SimpleNamespace(metadata={})]) is None
