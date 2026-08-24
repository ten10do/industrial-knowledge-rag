import pytest

from backend.evaluation import v348_bounded_claim_binding_responsibility_reassessment as v


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
        "reran_v347": False,
    }


def test_manifest_requires_exact_frozen_v347_inputs():
    assert v.validate_source_manifest(_manifest()) == ()
    changed = _manifest()
    changed["reran_v347"] = True
    assert "V347_RERUN_FORBIDDEN" in v.validate_source_manifest(changed)


def test_annotation_owner_is_derived_from_failure_class():
    records = [{"query_id": f"q{i}", "expected": "ABSTAIN", "decision": "ANSWER"}
               for i in range(v.SOURCE_FALSE_ANSWER_COUNT)]
    rows = [{
        "query_id": f"q{i}", "failure_class": "TARGET_METADATA_BINDING_GAP",
        "responsibility": "EVIDENCE_CLAIM_BINDING", "reason": "owned row",
        "signal_in_selected_candidates": True,
    } for i in range(v.SOURCE_FALSE_ANSWER_COUNT)]
    assert v.validate_false_answer_annotations(rows, records) == ()
    rows[0]["responsibility"] = "TABLE_STRUCTURE_BOUNDARY"
    assert "ANNOTATION_OWNER:0" in v.validate_false_answer_annotations(rows, records)


def test_false_refusal_precedence_is_new_veto_then_retrieval_then_inherited():
    candidate = {
        "query_id": "q", "expected": "ANSWER", "decision": "ABSTAIN",
        "action": "VETO", "reason_code": "conflict",
    }
    baseline = {"relevant_evidence_retrieved": False}
    assert v.classify_false_refusal(candidate, baseline).failure_class == "CLAIM_BINDING_UNSAFE_VETO"
    candidate["action"] = "PRESERVE"
    assert v.classify_false_refusal(candidate, baseline).failure_class == "RETRIEVAL_MISSING"
    baseline["relevant_evidence_retrieved"] = True
    assert v.classify_false_refusal(candidate, baseline).failure_class == "INHERITED_EVIDENCE_REFUSAL"
    candidate["decision"] = "ANSWER"
    with pytest.raises(ValueError, match="NOT_FALSE_REFUSAL"):
        v.classify_false_refusal(candidate, baseline)


def test_structural_majority_routes_to_table_ownership_boundary_analysis():
    summary = {
        "errors": 27, "false_answers": 17, "false_refusals": 10,
        "unsafe_vetoes": 3, "structural_actionable_share": 11 / 20,
    }
    result = v.decide(summary, integrity=True, reconciled=True)
    assert result["status"] == "RESPONSIBILITY_REASSESSMENT_COMPLETE"
    assert result["recommendation"] == "ENTER_EVIDENCE_TABLE_OWNERSHIP_BOUNDARY_ANALYSIS"
    assert v.decide(summary, integrity=False, reconciled=True)["status"] == "RUNTIME_INVALID"
