import pytest

from backend.evaluation import v350_evidence_parser_table_structure_responsibility_reassessment as v


def _manifest() -> dict:
    return {
        "benchmark_version": v.BENCHMARK_VERSION,
        "source_benchmark_version": v.SOURCE_BENCHMARK_VERSION,
        "source_query_count": v.SOURCE_QUERY_COUNT,
        "source_dataset_sha256": v.SOURCE_DATASET_SHA256,
        "source_file_sha256": v.SOURCE_FILE_SHA256,
        "candidate_version": v.CANDIDATE_VERSION,
        "candidate_sha256": v.CANDIDATE_SHA256,
        "read_only_reassessment": True,
        "reran_v349": False,
        "modified_parser": False,
    }


def test_manifest_requires_read_only_frozen_v349_inputs():
    assert v.validate_source_manifest(_manifest()) == ()
    changed = _manifest()
    changed["modified_parser"] = True
    assert "PARSER_CHANGE_FORBIDDEN" in v.validate_source_manifest(changed)


def test_action_annotations_derive_action_correctness_and_owner():
    source = [{
        "query_id": "q", "expected": "ABSTAIN", "decision": "ANSWER",
        "ownership_relation": "HEADER_INHERITED", "action": "PRESERVE",
    }]
    row = {
        "query_id": "q", "action_kind": "BINDING", "correct": False,
        "failure_class": "ROW_CELL_LINEAGE_MISSING",
        "responsibility": "TABLE_STRUCTURE_PROVENANCE",
        "required_proof_fields": ["row_id", "parameter_scope_id", "value_scope_id"],
        "reason": "flattened adjacent rows lost cell ownership",
    }
    errors = v.validate_action_annotations([row], source)
    assert errors == ("SOURCE_OWNERSHIP_ACTIONS:1",)
    row["responsibility"] = "EVIDENCE_POLICY"
    assert "ANNOTATION_OWNER:0" in v.validate_action_annotations([row], source)


def test_error_precedence_is_action_then_retrieval_then_inherited_evidence():
    candidate = {"query_id": "q", "expected": "ANSWER", "decision": "ABSTAIN"}
    action = {
        "failure_class": "COLUMN_HEADER_LINEAGE_MISSING",
        "responsibility": "TABLE_STRUCTURE_PROVENANCE", "reason": "column lineage",
    }
    assert v.classify_error(candidate, {}, action).responsibility == "TABLE_STRUCTURE_PROVENANCE"
    assert v.classify_error(candidate, {"relevant_evidence_retrieved": False}, None).failure_class == "INHERITED_RETRIEVAL_MISSING"
    assert v.classify_error(candidate, {"relevant_evidence_retrieved": True}, None).failure_class == "INHERITED_EVIDENCE_REFUSAL"
    candidate["decision"] = "ANSWER"
    with pytest.raises(ValueError, match="NOT_ERROR"):
        v.classify_error(candidate, {}, None)


def test_complete_reassessment_routes_to_structure_proof_contract():
    summary = {
        "errors": 28, "false_answers": 1, "false_refusals": 27,
        "ownership_actions": 15, "bindings": 4, "vetoes": 11,
        "unsafe_bindings": 1, "unsafe_vetoes": 5,
    }
    result = v.decide(
        summary, integrity=True, reconciled=True, annotations_complete=True,
    )
    assert result["status"] == "RESPONSIBILITY_REASSESSMENT_COMPLETE"
    assert result["recommendation"] == "DEFINE_TABLE_STRUCTURE_PROOF_CONTRACT"
    assert v.decide(
        summary, integrity=False, reconciled=True, annotations_complete=True,
    )["status"] == "RUNTIME_INVALID"
