from types import SimpleNamespace

import pytest

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence_support import SupportReason, SupportStatus, validate_evidence_support
from backend.retrieval.filters import analyze_query


def _doc(chunk_id: str, content: str, model: str = "Drive X"):
    return SimpleNamespace(page_content=content, metadata={
        "chunk_id": chunk_id, "manufacturer": "Example Industrial",
        "equipment_model": model, "product_family": "Drive",
    })


def _support(query: str, documents: list):
    result = RetrievalResult(
        [RetrievalCandidate(document=document, retrieval_source="frozen", final_rank=index)
         for index, document in enumerate(documents, start=1)],
        query_analysis=analyze_query(query, documents), corpus_documents=documents,
        retrieval_mode="frozen_support_precision_test",
    )
    return validate_evidence_support(query, result, documents)


def test_compatibility_partial_support_is_insufficient():
    support = _support(
        "Which firmware revision is required for Drive X PROFINET compatibility?",
        [_doc("partial", "Drive X supports PROFINET configuration through Adapter A.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason in {
        SupportReason.MISSING_ATTRIBUTE_SUPPORT.value,
        SupportReason.MISSING_COMPATIBILITY_SUPPORT.value,
    }


def test_compatibility_complete_support_passes():
    support = _support(
        "Which firmware revision is required for Drive X PROFINET compatibility?",
        [_doc("complete", "Drive X is compatible with PROFINET when firmware revision 3.2 or later is installed.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


@pytest.mark.parametrize(
    ("query", "content", "expected"),
    [
        ("What is the default subnet mask for Drive X?", "The default subnet mask is 255.255.255.0.", "SUPPORTED"),
        ("What is the default subnet mask for Drive X?", "The default IP address is 192.168.1.10.", "INSUFFICIENT"),
        ("What is the rated voltage of Drive X?", "The rated voltage is 480 V AC.", "SUPPORTED"),
        ("What is the rated voltage of Drive X?", "The voltage setting selects the operating mode.", "INSUFFICIENT"),
        ("What voltage in V is required for Drive X?", "The required voltage is 480 V AC.", "SUPPORTED"),
        ("What voltage in V is required for Drive X?", "The required voltage is 100 percent of nominal.", "INSUFFICIENT"),
    ],
)
def test_attribute_value_and_unit_precision(query: str, content: str, expected: str):
    assert _support(query, [_doc("candidate", content)]).status == expected


def test_value_must_be_associated_with_requested_identifier():
    support = _support(
        "What is the default voltage of Drive X parameter A100?",
        [_doc("mixed", "Parameter A100 controls speed. Parameter B200 has a default voltage of 480 V.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_VALUE_SUPPORT.value


def test_multi_requirement_complete_support_passes():
    support = _support(
        "Before removing Drive X, how long must power be off and what residual-energy limit must be reached?",
        [_doc("complete", "Turn off power, wait at least 15 minutes, and verify residual energy is 200 microjoules or fewer before removal.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


def test_multi_requirement_partial_support_is_insufficient():
    support = _support(
        "Before removing Drive X, how long must power be off and what residual-energy limit must be reached?",
        [_doc("partial", "Turn off power and wait at least 15 minutes before removal.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value


def test_semantic_hard_positive_remains_supported():
    assert _support(
        "Should power be isolated before Drive X maintenance?",
        [_doc("semantic", "Disconnect the input supply prior to servicing the equipment.")],
    ).status == SupportStatus.SUPPORTED.value


def test_specific_query_with_complete_evidence_passes():
    assert _support(
        "Which DIP-switch position restores Drive X factory defaults?",
        [_doc("complete", "Set DIP switch SW1 to position 4 to restore factory defaults.")],
    ).status == SupportStatus.SUPPORTED.value


def test_specific_query_with_incomplete_evidence_is_insufficient():
    support = _support(
        "Which DIP-switch position restores Drive X factory defaults?",
        [_doc("partial", "Enter reset value 1234 online to restore factory defaults.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason in {
        SupportReason.MISSING_ATTRIBUTE_SUPPORT.value,
        SupportReason.MISSING_REQUIRED_CONCEPT.value,
    }


def test_collapsed_default_marker_still_supports_default_value():
    support = _support(
        "What is the default login username?",
        [_doc("collapsed", "The defaultusername is admin.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


def test_version_requirement_takes_precedence_over_configuration_wording():
    support = _support(
        "Which software version is required for the configuration example?",
        [_doc("no-version", "Use the software for this configuration example; no build is specified.")],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value


@pytest.mark.parametrize("identifier", ["04.16", "58.18"])
def test_frozen_identifier_recovery_like_cases_remain_supported(identifier: str):
    support = _support(
        f"What does Drive X parameter {identifier} mean?",
        [_doc("parameter", f"Parameter {identifier} selects the drive operating mode. Value 1 enables it.")],
    )
    assert support.status == SupportStatus.SUPPORTED.value


@pytest.mark.parametrize(
    ("query", "documents", "expected_reason"),
    [
        (
            "What PROFINET compatibility requirements apply to Drive X?",
            [_doc("topic-only", "Drive X provides a PROFINET configuration interface.")],
            SupportReason.MISSING_COMPATIBILITY_SUPPORT.value,
        ),
        (
            "What does Drive X parameter A999 mean?",
            [_doc("other-id", "Parameter A100 controls the speed reference. Value 1 enables it.")],
            SupportReason.MISSING_IDENTIFIER_SUPPORT.value,
        ),
        (
            "Is Drive X guaranteed compatible with Acme firmware revision?",
            [_doc("no-vendor", "Drive X compatibility depends on checking the actual firmware revisions.")],
            SupportReason.MISSING_REQUIRED_CONCEPT.value,
        ),
        (
            "What maximum parameter value should always be used for Drive X performance?",
            [_doc("no-value", "Configure the Drive X parameter to suit the application performance.")],
            SupportReason.MISSING_ATTRIBUTE_SUPPORT.value,
        ),
        (
            "Which exact external temperature sensor model is mandatory for every Drive X installation?",
            [_doc("temperature-only", "Install Drive X within the permitted ambient temperature range.")],
            SupportReason.MISSING_REQUIRED_CONCEPT.value,
        ),
    ],
)
def test_frozen_false_support_like_cases_are_rejected(query, documents, expected_reason):
    support = _support(query, documents)
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == expected_reason


def test_value_from_an_unrelated_candidate_does_not_support_megger_torque():
    support = _support(
        "What torque value is required when disconnecting Drive X for a megger test?",
        [
            _doc("procedure", "Disconnect all Drive X connections before the megger test."),
            _doc("unrelated-value", "Tighten the mounting bolt to a torque of 5 N m."),
        ],
    )
    assert support.status == SupportStatus.INSUFFICIENT.value
    assert support.reason == SupportReason.MISSING_VALUE_SUPPORT.value
