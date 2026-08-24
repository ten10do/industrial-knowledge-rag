from dataclasses import replace

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v351 import (
    TABLE_STRUCTURE_PROOF_VERSION,
    TableStructureClaim,
    TableStructureProof,
    validate_proof_lineage,
    validate_table_structure_proof,
)


def _claim(relation=TableOwnershipRelation.DIRECT_ROW.value):
    return TableStructureClaim("doc", relation, "model:ax100", "parameter:voltage", "24 V")


def _proof(relation=TableOwnershipRelation.DIRECT_ROW.value):
    return TableStructureProof(
        proof_id="proof-1", document_id="doc", table_region_id="table-1",
        relation=relation, model_scope_id="model:ax100",
        parameter_scope_id="parameter:voltage", value_scope_id="value:24v",
        model_text="AX-100", parameter_text="Voltage", value_text="24 V",
        cell_role="VALUE", chunk_ids=("chunk-1",), model_row_id="row-1",
        parameter_row_id="row-1", value_row_id="row-1",
        model_column_id="column-model", parameter_column_id="column-voltage",
        value_column_id="column-voltage", header_path=("parameter:voltage",),
    )


def test_direct_row_requires_same_parameter_and_value_row():
    proof = _proof()
    assert validate_table_structure_proof(proof, _claim()).valid
    invalid = replace(proof, value_row_id="row-2")
    result = validate_table_structure_proof(invalid, _claim())
    assert not result.valid
    assert "ROW_OWNERSHIP_MISMATCH" in result.reason_codes


def test_column_bound_requires_model_row_and_parameter_header_column():
    relation = TableOwnershipRelation.COLUMN_BOUND.value
    proof = _proof(relation)
    assert validate_table_structure_proof(proof, _claim(relation)).valid
    invalid = replace(proof, value_column_id="column-current")
    assert "COLUMN_OWNERSHIP_MISMATCH" in validate_table_structure_proof(
        invalid, _claim(relation),
    ).reason_codes


def test_header_section_and_reference_relations_have_distinct_invariants():
    header = replace(_proof(TableOwnershipRelation.HEADER_INHERITED.value),
                     merged_cell_span=(1, 2))
    assert validate_table_structure_proof(
        header, _claim(TableOwnershipRelation.HEADER_INHERITED.value),
    ).valid
    section_relation = TableOwnershipRelation.SECTION_INHERITED.value
    section = replace(_proof(section_relation), section_scope_id="section:ratings")
    section_claim = replace(_claim(section_relation), section_scope_id="section:ratings")
    assert validate_table_structure_proof(section, section_claim).valid
    reference_relation = TableOwnershipRelation.CROSS_REFERENCE.value
    reference = replace(
        _proof(reference_relation), cell_role="REFERENCE",
        reference_target="Page 42", value_text="Page 42",
    )
    reference_claim = replace(_claim(reference_relation), value_or_action="Page 42")
    assert validate_table_structure_proof(reference, reference_claim).valid


def test_contract_rejects_conflict_wrong_model_and_conclusion_fields():
    proof = _proof().as_dict()
    proof["conflicting_owner_ids"] = ["row-2"]
    proof["answer"] = "24 V"
    claim = replace(_claim(), model_scope_id="model:bx200")
    reasons = validate_table_structure_proof(proof, claim).reason_codes
    assert "CONFLICTING_OWNER" in reasons
    assert "MODEL_SCOPE_MISMATCH" in reasons
    assert any(reason.startswith("CONCLUSION_FIELD_FORBIDDEN") for reason in reasons)


def test_unsupported_is_not_a_proof_and_version_is_exact():
    proof = replace(
        _proof(TableOwnershipRelation.UNSUPPORTED.value), proof_version="old",
    )
    result = validate_table_structure_proof(
        proof, _claim(TableOwnershipRelation.UNSUPPORTED.value),
    )
    assert not result.valid
    assert "UNSUPPORTED_NOT_A_PROOF" in result.reason_codes
    assert "PROOF_VERSION_MISMATCH" in result.reason_codes
    assert TABLE_STRUCTURE_PROOF_VERSION == "table-structure-proof-v351-contract"


def test_lineage_allows_chunk_replication_but_rejects_structural_drift():
    proof = _proof()
    replicated = replace(proof, chunk_ids=("chunk-2",))
    assert validate_proof_lineage([proof, replicated]) == ()
    drifted = replace(replicated, value_row_id="row-2")
    assert validate_proof_lineage([proof, drifted]) == ("LINEAGE_DRIFT:proof-1",)
