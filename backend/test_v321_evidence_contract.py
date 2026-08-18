"""Focused public checks for evidence-v321.1 matching policy."""

from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval import RetrievalCandidate, analyze_query
from backend.retrieval.evidence_contract import (
    ClaimType,
    RequirementCriticality,
    build_typed_requirement,
    evaluate_evidence_contract,
)


def _doc(chunk_id: str, text: str, *, document_id: str = "manual-a", subsection: str = ""):
    return SimpleNamespace(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "manufacturer": "Example Automation",
            "product_family": "DriveX",
            "equipment_model": "DriveX 100",
            "section": "Parameters",
            "subsection": subsection,
            "page": 10,
        },
    )


def _contract(query: str, selected: list, corpus: list | None = None):
    corpus = corpus or selected
    candidates = [
        RetrievalCandidate(document=document, retrieval_source="fixture", final_rank=index)
        for index, document in enumerate(selected, 1)
    ]
    return evaluate_evidence_contract(query, candidates, corpus, analyze_query(query, corpus))


def test_claim_types_distinguish_explicit_semantic_and_related_only():
    explicit = _contract(
        "What does object 0x1011 specify for DriveX 100?",
        [_doc("explicit", "Index (hex) Object name 1011 Restore default parameters")],
    )
    assert explicit.claims[0].claim_type == ClaimType.EXPLICIT.value

    semantic = _contract(
        "What is the waiting time for DriveX 100 restart?",
        [_doc("semantic", "The setting defines the time between restart attempts.")],
    )
    assert semantic.claims[0].claim_type == ClaimType.SEMANTIC_EQUIVALENT.value

    related = _contract(
        "Why is the DriveX 100 safety signature used?",
        [_doc("related", "Safety signature overview.")],
    )
    assert related.claims[0].claim_type == ClaimType.RELATED_ONLY.value
    assert not related.sufficient


def test_criticality_and_identifier_locality_and_alternate_notation():
    current = _doc(
        "current",
        "Index (hex) Object name 1011 Restore default parameters",
        subsection="Restore default parameters",
    )
    requirement = build_typed_requirement(
        "What does object 0x1011 specify for DriveX 100?",
        [current],
        analyze_query("What does object 0x1011 specify for DriveX 100?", [current]),
    )
    identifier = next(item for item in requirement.items if item.kind == "identifier")
    assert identifier.criticality == RequirementCriticality.CRITICAL.value
    assert _contract("What does object 0x1011 specify for DriveX 100?", [current]).sufficient

    global_only = _doc("global", "Related parameters: 0x1011")
    assert not _contract(
        "What does object 0x1011 specify for DriveX 100?", [global_only], [global_only, current]
    ).sufficient


def test_action_value_and_protocol_require_candidate_local_support():
    procedure = _doc("procedure", "Assign a static IP address, then confirm the new address.")
    assert _contract("How do I configure a static IP address on DriveX 100?", [procedure]).sufficient
    assert not _contract(
        "How do I configure a static IP address on DriveX 100?",
        [_doc("topic", "Static IP configuration overview.")],
    ).sufficient

    target = _doc("target", "P30 Restart delay. Default: 1 second.", subsection="P30 Restart delay")
    other = _doc("other", "P31 Stop delay. Default: 5 seconds.", subsection="P31 Stop delay")
    assert _contract("Is DriveX 100 parameter P30 default 1 second?", [target, other]).sufficient
    assert not _contract("Is DriveX 100 parameter P30 default 5 seconds?", [target, other]).sufficient

    assert _contract(
        "What MQTT broker port is used by DriveX 100?",
        [_doc("mqtt", "MQTT clients connect to the broker port before publishing.")],
    ).sufficient
    assert not _contract(
        "What MQTT broker port is used by DriveX 100?",
        [_doc("mqtt-topic", "MQTT communication overview.")],
    ).sufficient


def test_safe_multi_chunk_and_cross_document_unsafe_aggregation():
    attempts = _doc("attempts", "Restart tries sets the number of restart attempts.", subsection="Restart")
    delay = _doc("delay", "Restart delay sets the time between restart attempts.", subsection="Delay")
    query = "Which parameters define restart attempts and the delay between attempts?"
    assert _contract(query, [attempts, delay]).sufficient

    other_manual = _doc(
        "foreign-delay", "Restart delay sets the time between restart attempts.",
        document_id="manual-b", subsection="Delay",
    )
    assert not _contract(query, [attempts, other_manual]).sufficient
