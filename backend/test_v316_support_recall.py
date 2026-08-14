"""V3.16 precision-preserving Support recall regression tests."""

from types import SimpleNamespace

import pytest

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_support import SupportReason, SupportStatus, validate_evidence_support
from backend.retrieval.filters import analyze_query


def _doc(
    chunk_id: str,
    content: str,
    *,
    model: str = "Drive X",
    document_id: str = "manual-x",
    section: str = "Configuration",
    subsection: str = "Parameter block",
    page: int = 10,
):
    return SimpleNamespace(page_content=content, metadata={
        "chunk_id": chunk_id,
        "document_id": document_id,
        "manufacturer": "Example Industrial",
        "equipment_model": model,
        "product_family": "Drive",
        "section": section,
        "subsection": subsection,
        "page": page,
    })


def _support(query: str, documents: list):
    result = RetrievalResult(
        [RetrievalCandidate(document=document, retrieval_source="frozen", final_rank=index)
         for index, document in enumerate(documents, start=1)],
        query_analysis=analyze_query(query, documents),
        corpus_documents=documents,
        retrieval_mode="v316_frozen_support_test",
    )
    return validate_evidence_support(query, result, documents)


def test_semantic_attribute_equivalence_positive():
    support = _support(
        "Why must Drive X enable coordinated timing?",
        [_doc("cause", "Coordinated timing enables synchronous control; without it the application cannot run.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value
    assert support.coverage["requested_attributes"]["cause"] is True


def test_semantic_attribute_distinction_negative():
    support = _support(
        "Why must Drive X enable coordinated timing?",
        [_doc("topic", "Drive X stores the coordinated timing setting in the controller project.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_ATTRIBUTE_SUPPORT.value


def test_rated_value_positive():
    support = _support(
        "What rated supply voltage is specified for Drive X?",
        [_doc("rated", "Rated voltage VAC 380 to 480; acceptable variation 323 to 504 VAC.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


@pytest.mark.parametrize(
    ("query", "content"),
    [
        ("What default supply voltage is assigned to Drive X?", "Rated voltage VAC 380 to 480."),
        ("What nominal motor winding voltage is specified for Drive X?", "Rated supply voltage VAC 380 to 480."),
    ],
)
def test_value_kind_and_attribute_distinctions_remain_critical(query: str, content: str):
    assert _support(query, [_doc("distinct", content)]).status == SupportStatus.INSUFFICIENT.value


def test_extra_candidate_qualifier_is_optional_when_query_does_not_request_it():
    support = _support(
        "What voltage is specified for Drive X?",
        [_doc("qualified", "The rated voltage is 480 VAC.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


def test_same_block_value_association_passes():
    support = _support(
        "What default and range apply to Drive X parameter 30.20?",
        [_doc("same", "Parameter 30.20 Maximum torque. Default 300.0%. Range 0.0 to 1600.0%.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


def test_cross_identifier_value_association_is_rejected():
    support = _support(
        "What default and range apply to Drive X parameter 30.20?",
        [
            _doc("other-a", "Parameter 30.22 source. Default 0.", page=10),
            _doc("other-b", "Parameter 30.24 torque. Range 0.0 to 1600.0%.", page=11),
        ],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_IDENTIFIER_SUPPORT.value


def test_multi_chunk_support_with_consistent_scope_passes():
    support = _support(
        "Which objects hold the IP address and subnet mask, and what task period is required?",
        [
            _doc("network", "IP address object 0x8000:21 and subnet mask object 0x8000:22.", page=10),
            _doc("timing", "The task period must match the master period of 10 ms.", page=11),
        ],
    )
    assert support.status == SupportStatus.SUPPORTED.value
    assert support.coverage["local_value_association"] is True


def test_multi_chunk_cross_scope_aggregation_is_rejected():
    support = _support(
        "Which objects hold the IP address and subnet mask, and what task period is required?",
        [
            _doc("network", "IP address object 0x8000:21 and subnet mask object 0x8000:22.", subsection="Network", page=10),
            _doc("other", "The task period must be 10 ms.", subsection="Unrelated parameter", page=40),
        ],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_VALUE_SUPPORT.value


def test_multi_requirement_full_and_partial():
    query = "Which default IP address and subnet mask are required for Drive X?"
    complete = _support(query, [_doc("full", "Default IP address 192.168.1.1 and default subnet mask 255.255.255.0.")])
    partial = _support(query, [_doc("partial", "Default IP address 192.168.1.1; no subnet mask is specified.")])
    assert complete.status == SupportStatus.SUPPORTED.value
    assert partial.status == SupportStatus.INSUFFICIENT.value


def test_configuration_required_hard_negative_remains_insufficient():
    support = _support(
        "What configuration-data values are required for a slave that cannot operate without configuration?",
        [_doc("negative", "This master supports only slaves that require no configuration data.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert "concept:configuration_data_required" in support.missing_requirements


@pytest.mark.parametrize(
    ("query", "content"),
    [
        (
            "为什么要保留完整性标记？",
            "The integrity marker verifies the integrity of the safety application.",
        ),
        (
            "设备断电后，应该在哪些端子测量电压？",
            "Measure voltage across the DC+ and DC- bus terminals to confirm discharge.",
        ),
        (
            "为什么必须启用同步功能？",
            "Synchronization enables coordinated motion; without this setting the application cannot run.",
        ),
    ],
)
def test_frozen_over_constraint_failure_classes_are_recovered_without_ids(query: str, content: str):
    assert _support(query, [_doc("recovery", content)]).status == SupportStatus.SUPPORTED.value


def test_hard_positive_and_hard_negative_safety():
    positive = _support(
        "Why must Drive X power be isolated before maintenance?",
        [_doc("safe", "Disconnect input power before service to prevent electric shock.")],
    )
    negative = _support(
        "Which exact isolation voltage proves Drive X is safe for maintenance?",
        [_doc("unsafe", "Disconnect input power before service; the manual gives no isolation-voltage value.")],
    )
    assert positive.status == SupportStatus.SUPPORTED.value
    assert negative.status == SupportStatus.INSUFFICIENT.value
