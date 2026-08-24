from dataclasses import replace

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v351 import (
    TableStructureClaim,
    TableStructureProof,
)
from backend.retrieval.table_structure_proof_contract_v354_candidate import (
    MERGED_CELL_COVERAGE_VERSION,
    MergedCellCoverage,
    validate_coverage_lineage,
    validate_merged_cell_coverage_proof,
)


def _claim(relation=TableOwnershipRelation.COLUMN_BOUND.value):
    return TableStructureClaim(
        "doc", relation, "model:p7", "parameter:flow", "50 L/min",
    )


def _proof(relation=TableOwnershipRelation.COLUMN_BOUND.value):
    return TableStructureProof(
        proof_id="proof-1", document_id="doc", table_region_id="table-1",
        relation=relation, model_scope_id="model:p7",
        parameter_scope_id="parameter:flow", value_scope_id="value:50",
        model_text="P7", parameter_text="Flow", value_text="50 L/min",
        cell_role="VALUE", chunk_ids=("chunk-1",),
        model_row_id="row-1", value_row_id="row-2", parameter_row_id="row-2",
        model_column_id="column-model", parameter_column_id="column-flow",
        value_column_id="column-flow", header_path=("parameter:flow",),
        merged_cell_span=(2, 1),
    )


def _coverage():
    return MergedCellCoverage(
        coverage_id="coverage-1", document_id="doc", table_region_id="table-1",
        coverage_owner_scope_id="model:p7", coverage_anchor_cell_id="cell-model-p7",
        covered_row_ids=("row-1", "row-2"), chunk_ids=("chunk-1",),
    )


def test_explicit_coverage_recovers_only_vertical_descendant_row():
    result = validate_merged_cell_coverage_proof(_proof(), _claim(), _coverage())
    assert result.valid
    assert result.coverage_used
    assert result.base_reason_codes == ("MODEL_ROW_OWNERSHIP_MISMATCH",)


def test_outside_incomplete_duplicate_and_conflicting_coverage_are_rejected():
    outside = replace(_coverage(), covered_row_ids=("row-1", "row-3"))
    assert "VALUE_ROW_OUTSIDE_COVERAGE" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), outside,
    ).reason_codes
    incomplete = replace(_coverage(), covered_row_ids=("row-1",))
    assert "VALUE_ROW_OUTSIDE_COVERAGE" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), incomplete,
    ).reason_codes
    duplicate = replace(_coverage(), covered_row_ids=("row-1", "row-1"))
    assert "DUPLICATE_COVERED_ROW_ID" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), duplicate,
    ).reason_codes
    conflict = replace(_coverage(), conflicting_coverage_ids=("coverage-2",))
    assert "CONFLICTING_COVERAGE" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), conflict,
    ).reason_codes


def test_wrong_model_table_and_other_base_errors_cannot_be_overridden():
    wrong_model = replace(_coverage(), coverage_owner_scope_id="model:p8")
    assert "COVERAGE_OWNER_SCOPE_MISMATCH" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), wrong_model,
    ).reason_codes
    wrong_table = replace(_coverage(), table_region_id="table-2")
    assert "COVERAGE_TABLE_REGION_MISMATCH" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), wrong_table,
    ).reason_codes
    wrong_column = replace(_proof(), value_column_id="column-current")
    result = validate_merged_cell_coverage_proof(wrong_column, _claim(), _coverage())
    assert not result.valid
    assert "COLUMN_OWNERSHIP_MISMATCH" in result.reason_codes


def test_v351_valid_proof_is_preserved_without_coverage():
    proof = replace(_proof(), model_row_id="row-2", merged_cell_span=(1, 1))
    result = validate_merged_cell_coverage_proof(proof, _claim())
    assert result.valid
    assert not result.coverage_used


def test_coverage_is_column_bound_only_and_requires_vertical_span_and_chunk_link():
    direct = replace(
        _proof(TableOwnershipRelation.DIRECT_ROW.value),
        parameter_row_id="row-2", value_row_id="row-2",
    )
    claim = _claim(TableOwnershipRelation.DIRECT_ROW.value)
    assert "COVERAGE_RELATION_NOT_COLUMN_BOUND" in validate_merged_cell_coverage_proof(
        direct, claim, _coverage(),
    ).reason_codes
    flat = replace(_proof(), merged_cell_span=(1, 1))
    assert "VERTICAL_COVERAGE_REQUIRES_ROW_SPAN" in validate_merged_cell_coverage_proof(
        flat, _claim(), _coverage(),
    ).reason_codes
    detached = replace(_coverage(), chunk_ids=("chunk-2",))
    assert "COVERAGE_CHUNK_SCOPE_MISMATCH" in validate_merged_cell_coverage_proof(
        _proof(), _claim(), detached,
    ).reason_codes


def test_coverage_lineage_allows_chunk_replication_but_rejects_membership_drift():
    coverage = _coverage()
    replicated = replace(coverage, chunk_ids=("chunk-2",))
    assert validate_coverage_lineage([coverage, replicated]) == ()
    drifted = replace(replicated, covered_row_ids=("row-1", "row-3"))
    assert validate_coverage_lineage([coverage, drifted]) == (
        "COVERAGE_LINEAGE_DRIFT:coverage-1",
    )
    assert MERGED_CELL_COVERAGE_VERSION == "merged-cell-coverage-v354-candidate"
