"""Focused synthetic checks for the layered evidence-v320.1 contract."""

from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval import RetrievalCandidate, RetrievalResult, analyze_query, analyze_retrieval_evidence
from backend.retrieval.evidence_contract import (
    AggregationLevel,
    RequirementCriticality,
    RequirementMatchMode,
    build_typed_requirement,
    evaluate_evidence_contract,
)


def _doc(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "manual-a",
    model: str = "ACS580",
    section: str = "Parameters",
    subsection: str = "",
    page: int = 10,
):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "manufacturer": "ABB",
            "equipment_model": model,
            "product_family": "ACS",
            "section": section,
            "subsection": subsection,
            "page": page,
        },
    )


def _candidate(document, rank: int = 1) -> RetrievalCandidate:
    return RetrievalCandidate(
        document=document,
        retrieval_source="fixed_fixture",
        final_rank=rank,
        vector_rank=rank,
        vector_score=8.0 + rank,
        lexical_score=2.0,
    )


def _contract(query: str, selected: list, corpus: list | None = None):
    corpus = corpus or selected
    return evaluate_evidence_contract(
        query,
        [_candidate(document, rank) for rank, document in enumerate(selected, 1)],
        corpus,
        analyze_query(query, corpus),
    )


def _evidence(query: str, selected: list, corpus: list | None = None):
    corpus = corpus or selected
    candidates = [_candidate(document, rank) for rank, document in enumerate(selected, 1)]
    result = RetrievalResult(
        candidates,
        query_analysis=analyze_query(query, corpus),
        corpus_documents=corpus,
        retrieval_mode="fixed_fixture",
    )
    return analyze_retrieval_evidence(query, result, corpus, "fixed_fixture")


def test_typed_requirement_separates_critical_from_optional_items():
    document = _doc("typed", "The safety signature verifies integrity because it covers the safety application.")
    requirement = build_typed_requirement(
        "Why is the ACS580 safety signature used?",
        [document],
        analyze_query("Why is the ACS580 safety signature used?", [document]),
    )
    by_kind = {(item.kind, item.value): item for item in requirement.items}
    assert by_kind[("attribute", "cause")].criticality == RequirementCriticality.CRITICAL.value
    assert by_kind[("concept", "safety_signature")].criticality == RequirementCriticality.OPTIONAL.value
    assert requirement.location is False


def test_semantic_requirement_uses_explicit_match_mode():
    document = _doc("semantic", "The setting defines the time between restart attempts.")
    result = _contract("What is the waiting time for the ACS580 restart?", [document])
    item = next(item for item in result.requirement.items if item.value == "waiting_time")
    assert item.match_mode == RequirementMatchMode.SEMANTIC_EQUIVALENT.value
    assert result.sufficient


def test_identifier_must_be_supported_by_current_evidence_not_only_global_corpus():
    related = _doc("related", "ACS580 overview. Related Parameters: 30.20")
    current = _doc("current", "30.20 Maximum torque: default 100%; range 0 to 300%.", subsection="30.20 Maximum torque")
    query = "What range is defined for ACS580 parameter 30.20?"
    assert _evidence(query, [related], [related, current]).decision == "ABSTAIN"
    assert _evidence(query, [current], [related, current]).decision == "ANSWER"


def test_attribute_equivalent_is_supported_but_related_attribute_is_not():
    default = _doc("default", "30.20 Maximum torque: factory default 100%.", subsection="30.20 Maximum torque")
    assert _contract("What default is listed for ACS580 parameter 30.20?", [default]).sufficient
    assert not _contract("What range is listed for ACS580 parameter 30.20?", [default]).sufficient


def test_value_must_be_associated_with_target_identifier():
    target = _doc("target", "30.20 Restart delay. Default: 1.0 seconds.", subsection="30.20 Restart delay")
    other = _doc("other", "30.21 Stop delay. Default: 5.0 seconds.", subsection="30.21 Stop delay")
    corpus = [target, other]
    assert _contract("Is the ACS580 parameter 30.20 default 1.0 seconds?", corpus).sufficient
    assert not _contract("Is the ACS580 parameter 30.20 default 5.0 seconds?", corpus).sufficient


def test_same_candidate_and_same_parameter_block_aggregation_are_explicit():
    combined = _doc("combined", "30.20 Maximum torque. Default: 100%. Range: 0 to 300%.", subsection="30.20 Maximum torque")
    same_candidate = _contract("Give the default and range for ACS580 parameter 30.20.", [combined])
    assert same_candidate.sufficient
    assert same_candidate.aggregation_level == AggregationLevel.SAME_CANDIDATE.value

    default = _doc("default", "30.20 Maximum torque. Default: 100%.", subsection="30.20 Maximum torque")
    value_range = _doc("range", "30.20 Maximum torque. Range: 0 to 300%.", subsection="30.20 Maximum torque")
    split = _contract("Give the default and range for ACS580 parameter 30.20.", [default, value_range])
    assert split.sufficient
    assert split.aggregation_level == AggregationLevel.SAME_PARAMETER_BLOCK.value


def test_safe_multi_chunk_aggregation_and_cross_scope_rejection():
    attempts = _doc("attempts", "Auto restart tries sets the number of restart attempts.", subsection="Restart tries")
    delay = _doc("delay", "Auto restart delay sets the time between restart attempts.", subsection="Restart delay")
    query = "Which parameters define automatic restart attempts and the delay between attempts?"
    safe = _contract(query, [attempts, delay])
    assert safe.sufficient
    assert safe.aggregation_level == AggregationLevel.SAME_SECTION.value

    foreign_delay = _doc(
        "foreign-delay",
        "Auto restart delay sets the time between restart attempts.",
        document_id="manual-b",
        model="ACS880",
        subsection="Restart delay",
    )
    assert not _contract(query, [attempts, foreign_delay]).sufficient


def test_protocol_topic_only_is_negative_and_specific_claim_is_positive():
    topic = _doc("topic", "MQTT communication overview and architecture.")
    specific = _doc("specific", "MQTT clients connect to the broker port before publishing messages.")
    query = "What MQTT broker port is used by the ACS580?"
    assert not _contract(query, [topic]).sufficient
    assert _contract(query, [specific]).sufficient


def test_action_topic_only_is_negative_and_semantic_action_is_positive():
    topic = _doc("topic", "Static IP configuration overview for the ACS580.")
    procedure = _doc("procedure", "Assign a static IP address, then confirm the new address.")
    query = "How do I configure a static IP address on the ACS580?"
    assert not _contract(query, [topic]).sufficient
    assert _contract(query, [procedure]).sufficient


def test_partial_requirement_is_negative_and_complete_requirement_is_positive():
    partial = _doc(
        "partial",
        "ACS580 parameter 30.20 reports torque. Firmware settings are documented separately.",
        subsection="30.20 Maximum torque",
    )
    complete = _doc(
        "complete",
        "Firmware version 2 or later must be installed before ACS580 parameter 30.20 is available.",
        subsection="30.20 Maximum torque",
    )
    query = "Which firmware must be installed before ACS580 parameter 30.20 is available?"
    assert not _contract(query, [partial]).sufficient
    assert _contract(query, [complete]).sufficient
