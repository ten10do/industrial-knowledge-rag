import pytest

from backend.evaluation import v346_evidence_decision_claim_binding_reassessment as v


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
        "reran_v345": False,
    }


def test_source_manifest_accepts_exact_frozen_v345_inputs():
    assert v.validate_source_manifest(_manifest()) == ()
    manifest = _manifest()
    manifest["reran_v345"] = True
    assert "V345_RERUN_FORBIDDEN" in v.validate_source_manifest(manifest)


def test_false_refusal_precedence_is_retrieval_then_identity_then_routing():
    base = {
        "query_id": "q", "expected": "ANSWER", "decision": "ABSTAIN",
        "relevant_evidence_retrieved": False, "identity_result": "INCOMPATIBLE",
        "query_path": "FALLBACK", "reason": "reason",
    }
    assert v.classify_false_refusal(base).failure_class == "RETRIEVAL_MISSING"
    base["relevant_evidence_retrieved"] = True
    assert v.classify_false_refusal(base).failure_class == "IDENTITY_FALSE_REJECTION"
    base["identity_result"] = "COMPATIBLE"
    assert v.classify_false_refusal(base).failure_class == "ROUTING_OUTSIDE_VERIFICATION"
    base["query_path"] = "VERIFICATION"
    with pytest.raises(ValueError, match="UNATTRIBUTED_FALSE_REFUSAL"):
        v.classify_false_refusal(base)


def test_annotation_owner_must_match_failure_class():
    sources = [
        {"query_id": f"q{i}", "expected": "ABSTAIN", "decision": "ANSWER"}
        for i in range(v.SOURCE_FALSE_ANSWER_COUNT)
    ]
    rows = [{
        "query_id": f"q{i}", "failure_class": "ATTRIBUTE_VOCABULARY_GAP",
        "responsibility": "EVIDENCE_CLAIM_REPRESENTATION", "reason": "bounded gap",
        "signal_in_selected_candidates": True,
    } for i in range(v.SOURCE_FALSE_ANSWER_COUNT)]
    assert v.validate_false_answer_annotations(rows, sources) == ()
    rows[0]["responsibility"] = "RETRIEVAL_TO_EVIDENCE_SELECTION"
    assert any(
        error.startswith("ANNOTATION_OWNER_MISMATCH")
        for error in v.validate_false_answer_annotations(rows, sources)
    )


def test_decision_routes_representation_majority_to_bounded_design():
    summary = {
        "errors": 19, "false_answers": 9, "false_refusals": 10,
        "representation_fa_share": 6 / 9, "selection_fa_share": 3 / 9,
    }
    result = v.decide(summary, integrity=True, reconciled=True)
    assert result["status"] == "CLAIM_BINDING_REASSESSMENT_COMPLETE"
    assert result["recommendation"] == "ENTER_BOUNDED_EVIDENCE_CLAIM_BINDING_DESIGN"
    assert v.decide(summary, integrity=False, reconciled=True)["status"] == "RUNTIME_INVALID"
