from types import SimpleNamespace

import pytest

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_support import (
    SupportReason,
    SupportStatus,
    validate_evidence_support,
)
from backend.retrieval.filters import analyze_query


def _doc(
    chunk_id: str,
    content: str,
    *,
    manufacturer: str = "Rockwell Automation",
    model: str = "PowerFlex 527",
    family: str = "PowerFlex",
    section: str = "",
    page: int | None = None,
):
    metadata = {
        "chunk_id": chunk_id,
        "manufacturer": manufacturer,
        "equipment_model": model,
        "product_family": family,
    }
    if section:
        metadata["section"] = section
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=content, metadata=metadata)


def _result(query: str, documents: list):
    return RetrievalResult(
        [
            RetrievalCandidate(document=document, retrieval_source="frozen", final_rank=rank)
            for rank, document in enumerate(documents, start=1)
        ],
        query_analysis=analyze_query(query, documents),
        corpus_documents=documents,
        retrieval_mode="frozen_support_test",
    )


def _support(query: str, documents: list):
    return validate_evidence_support(query, _result(query, documents), documents)


def test_compatibility_requirement_rejects_related_identity_and_protocol_only():
    query = "What EtherNet/IP compatibility requirements apply to PowerFlex 527?"
    document = _doc("related", "PowerFlex 527 provides an EtherNet/IP network interface.")
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert {
        "attribute:compatibility", "attribute:requirements",
    }.issubset(support.missing_requirements)


def test_compatibility_requirement_accepts_explicit_conditions():
    query = "What EtherNet/IP compatibility requirements apply to PowerFlex 527?"
    document = _doc(
        "complete",
        "PowerFlex 527 is compatible with EtherNet/IP. Requirements: firmware version 3 is required.",
    )
    assert _support(query, [document]).status == SupportStatus.SUPPORTED.value


def test_location_requirement_needs_location_evidence():
    query = "Where in the manual does PowerFlex 527 describe maintenance support?"
    document = _doc("topic", "PowerFlex 527 maintenance and service support is described here.")
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_REQUESTED_LOCATION.value


def test_location_metadata_can_support_a_location_request():
    query = "Which section describes PowerFlex 527 maintenance support?"
    document = _doc(
        "located", "PowerFlex 527 maintenance and service support is described here.",
        section="Maintenance", page=47,
    )
    support = _support(query, [document])
    assert support.status == SupportStatus.SUPPORTED.value
    assert support.coverage["requested_location"] is True


def test_requested_attribute_missing_is_insufficient():
    query = "What firmware version compatibility requirements apply to PowerFlex 527?"
    document = _doc("partial", "PowerFlex 527 compatibility requirements are listed for the drive.")
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert "attribute:version" in support.missing_requirements


def test_requested_reinstall_action_missing_is_insufficient():
    query = "How do you reinstall PowerFlex 527 after transport?"
    document = _doc("transport", "Transport the installed PowerFlex 527 only after waiting 15 minutes.")
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert "action:reinstall" in support.missing_requirements


def test_partial_support_does_not_pass_two_of_three_requirements():
    query = "What PROFINET compatibility requirements apply to PowerFlex 527?"
    document = _doc("partial", "PowerFlex 527 is compatible with this network option.")
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert "protocol:profinet" in support.missing_requirements
    assert "attribute:requirements" in support.missing_requirements


def test_semantic_hard_positive_still_passes():
    query = "Should I cut power before opening PowerFlex 527 for maintenance?"
    document = _doc("safety", "Disconnect input power before servicing this equipment.")
    assert _support(query, [document]).status == SupportStatus.SUPPORTED.value


@pytest.mark.parametrize(
    ("manufacturer", "model", "family", "query", "content"),
    [
        (
            "Rockwell Automation", "PowerFlex 527", "PowerFlex",
            "What EtherNet/IP compatibility requirements apply to PowerFlex 527?",
            "PowerFlex 527 is compatible with EtherNet/IP; firmware version 3 is required.",
        ),
        (
            "ABB", "FPNO-21", "FPNO",
            "What PROFINET compatibility requirements apply to FPNO-21?",
            "FPNO-21 is compatible with PROFINET; device model version 2 is required.",
        ),
        (
            "Omron", "FQM1", "FQM1",
            "What Ethernet compatibility requirements apply to FQM1?",
            "FQM1 is compatible with Ethernet under the required installation conditions.",
        ),
    ],
)
def test_cross_vendor_full_support(
    manufacturer: str, model: str, family: str, query: str, content: str,
):
    document = _doc(
        "vendor", content, manufacturer=manufacturer, model=model, family=family,
    )
    assert _support(query, [document]).status == SupportStatus.SUPPORTED.value


def test_specific_value_unit_requires_the_requested_unit():
    query = "What DC bus voltage in V triggers PowerFlex 527 dynamic braking?"
    incomplete = _doc("percent", "The DC bus voltage threshold is 100 percent for dynamic braking.")
    complete = _doc("volts", "The DC bus voltage threshold is 720 V for dynamic braking.")
    assert _support(query, [incomplete]).status == SupportStatus.INSUFFICIENT.value
    assert _support(query, [complete]).status == SupportStatus.SUPPORTED.value


def test_location_annotation_boundary_class_uses_topic_plus_metadata():
    query = "Where does CX-Programmer describe PLC programming and maintenance support?"
    document = _doc(
        "location-boundary",
        "CX-Programmer supports PLC programming and maintenance.",
        manufacturer="Omron", model="CX-Programmer", family="CX",
        section="Technical Specifications", page=46,
    )
    assert _support(query, [document]).status == SupportStatus.SUPPORTED.value


def test_identifier_safety_class_remains_insufficient():
    query = "What does ACS580 fault F999 mean?"
    document = _doc(
        "other-fault", "ACS580 fault F101 describes a drive overtemperature.",
        manufacturer="ABB", model="ACS580", family="ACS",
    )
    support = _support(query, [document])
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_IDENTIFIER_SUPPORT.value
