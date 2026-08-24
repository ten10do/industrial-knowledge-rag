from langchain_core.documents import Document

from backend.retrieval.candidates import RetrievalCandidate
from backend.retrieval.evidence_table_ownership import (
    TableOwnershipClaim,
    TableOwnershipRelation,
    analyze_table_ownership,
)


def _candidate(text: str, *, model: str = "AX-100", aliases=None):
    return RetrievalCandidate(
        document=Document(page_content=text, metadata={
            "chunk_id": "chunk-1", "document_id": "doc-1",
            "equipment_model": model, "model_aliases": aliases or [model],
            "section": "Technical data",
        }),
        retrieval_source="hybrid",
    )


def _claim(attribute="Rated voltage", value="24 V", section=""):
    return TableOwnershipClaim("AX-100", "specifies", attribute, value, section)


def test_direct_row_requires_unique_model_owned_table_region():
    result = analyze_table_ownership(
        _claim(), [_candidate("Technical data\nRated voltage 24 V\nRated current 2 A")],
    )
    assert result.ownership_relation == TableOwnershipRelation.DIRECT_ROW.value
    assert result.reason_code == "UNIQUE_MODEL_ATTRIBUTE_VALUE_ROW"


def test_explicit_model_attribute_value_row_is_column_bound():
    result = analyze_table_ownership(
        _claim(), [_candidate("Specification table\nAX-100 Rated voltage 24 V")],
    )
    assert result.ownership_relation == TableOwnershipRelation.COLUMN_BOUND.value


def test_adjacent_value_can_inherit_parameter_header():
    result = analyze_table_ownership(
        _claim(), [_candidate("Technical data\nRated voltage\n24 V")],
    )
    assert result.ownership_relation == TableOwnershipRelation.HEADER_INHERITED.value


def test_explicit_section_can_be_inherited_without_proximity_only_binding():
    result = analyze_table_ownership(
        _claim(section="Electrical ratings"),
        [_candidate("Electrical ratings\nRated voltage\n24 V", aliases=["AX-100"])],
    )
    assert result.ownership_relation == TableOwnershipRelation.SECTION_INHERITED.value


def test_cross_reference_requires_attribute_and_reference_in_one_local_region():
    claim = TableOwnershipClaim("AX-100", "refers to", "Safety limits", "Page 42")
    result = analyze_table_ownership(
        claim, [_candidate("Specification table\nSafety limits\nsee Page 42")],
    )
    assert result.ownership_relation == TableOwnershipRelation.CROSS_REFERENCE.value


def test_same_value_under_wrong_parameter_is_unsupported():
    result = analyze_table_ownership(
        _claim(), [_candidate("Technical data\nRated current 24 V")],
    )
    assert result.ownership_relation == TableOwnershipRelation.UNSUPPORTED.value
    assert result.reason_code == "PARAMETER_SCOPE_NOT_OWNED"


def test_same_section_different_model_is_unsupported():
    result = analyze_table_ownership(
        _claim(), [_candidate("Technical data\nBX-200 Rated voltage 24 V", model="BX-200")],
    )
    assert result.ownership_relation == TableOwnershipRelation.UNSUPPORTED.value
    assert result.reason_code == "MODEL_SCOPE_NOT_OWNED"


def test_nearby_value_beyond_local_row_scope_is_unsupported():
    text = "Technical data\nRated voltage\nnotes\nlimits\nrange\n24 V"
    result = analyze_table_ownership(_claim(), [_candidate(text)])
    assert result.ownership_relation == TableOwnershipRelation.UNSUPPORTED.value
    assert result.reason_code == "CONFLICTING_OR_AMBIGUOUS_OWNERSHIP"


def test_default_claim_requires_explicit_qualifier():
    claim = TableOwnershipClaim("AX-100", "defaults to", "Filter time", "3 ms")
    assert analyze_table_ownership(
        claim, [_candidate("Parameter table\nFilter time 3 ms")],
    ).ownership_relation == TableOwnershipRelation.UNSUPPORTED.value
    assert analyze_table_ownership(
        claim, [_candidate("Parameter table\nFilter time 3 ms default")],
    ).ownership_relation == TableOwnershipRelation.DIRECT_ROW.value
