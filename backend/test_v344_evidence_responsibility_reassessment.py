from backend.evaluation import v344_evidence_responsibility_reassessment as v


def _manifest():
    return {
        "benchmark_version": v.BENCHMARK_VERSION,
        "source_benchmark_version": v.SOURCE_BENCHMARK_VERSION,
        "source_dataset_sha256": v.SOURCE_DATASET_SHA256,
        "source_file_sha256": v.SOURCE_FILE_SHA256,
        "candidate_version": v.CANDIDATE_VERSION,
        "candidate_sha256": v.CANDIDATE_SHA256,
        "source_query_count": v.SOURCE_QUERY_COUNT,
        "read_only_reassessment": True,
        "reran_v343": False,
    }


def test_source_manifest_requires_exact_frozen_artifacts():
    assert v.validate_source_manifest(_manifest()) == ()
    changed = _manifest()
    changed["candidate_sha256"] = "0" * 64
    changed["reran_v343"] = True
    assert set(v.validate_source_manifest(changed)) >= {"CANDIDATE_SHA256", "V343_RERUN_FORBIDDEN"}


def test_false_refusal_precedence_is_retrieval_identity_routing_then_evidence():
    base = {
        "expected": "ANSWER", "baseline_decision": "ABSTAIN", "decision": "ABSTAIN",
        "parser_recoverable": True, "relevant_evidence_retrieved": False,
        "identity_result": "INCOMPATIBLE", "query_path": "FALLBACK",
        "reason": "ATTRIBUTE_RELATION_MISSING",
    }
    assert v.classify_record(base).failure_class == "RETRIEVAL_MISSING"
    assert v.classify_record({**base, "relevant_evidence_retrieved": True}).failure_class == "IDENTITY_FALSE_REJECTION"
    routed = {**base, "relevant_evidence_retrieved": True, "identity_result": "COMPATIBLE"}
    assert v.classify_record(routed).failure_class == "ROUTING_OUTSIDE_VERIFICATION"
    evidence = {**routed, "query_path": "VERIFICATION"}
    assert v.classify_record(evidence).failure_class == "EVIDENCE_RELATION_BINDING_GAP"


def test_inherited_false_answer_is_architectural_evidence_but_outside_v342_scope():
    result = v.classify_record({
        "expected": "ABSTAIN", "baseline_decision": "ANSWER", "decision": "ANSWER",
    })
    assert result.failure_class == "INHERITED_FALSE_ANSWER"
    assert result.candidate_owner == "OUTSIDE_V342_UPGRADE_ONLY_SCOPE"
    assert result.architectural_owner == "EVIDENCE_DECISION_BASELINE"


def test_unsafe_relax_is_not_counted_as_a_false_refusal():
    summary = v.summarize([{
        "failure_class": "CANDIDATE_UNSAFE_RELAX",
        "expected": "ABSTAIN", "decision": "ANSWER",
    }])
    assert summary["evidence_candidate_false_refusals"] == 0
    assert summary["candidate_unsafe_relax"] == 1
    assert summary["architectural_evidence_errors"] == 1


def test_summary_and_decision_reject_more_local_relaxation_route():
    records = []
    records.extend({"failure_class": "EVIDENCE_RELATION_BINDING_GAP", "expected": "ANSWER", "decision": "ABSTAIN"} for _ in range(2))
    records.extend({"failure_class": "RETRIEVAL_MISSING", "expected": "ANSWER", "decision": "ABSTAIN"} for _ in range(2))
    records.extend({"failure_class": "IDENTITY_FALSE_REJECTION", "expected": "ANSWER", "decision": "ABSTAIN"} for _ in range(2))
    records.append({"failure_class": "ROUTING_OUTSIDE_VERIFICATION", "expected": "ANSWER", "decision": "ABSTAIN"})
    records.extend({"failure_class": "INHERITED_FALSE_ANSWER", "expected": "ABSTAIN", "decision": "ANSWER"} for _ in range(18))
    records.extend({"failure_class": "CORRECT_ANSWER", "expected": "ANSWER", "decision": "ANSWER"} for _ in range(23))
    records.extend({"failure_class": "CORRECT_ABSTAIN", "expected": "ABSTAIN", "decision": "ABSTAIN"} for _ in range(12))
    summary = v.summarize(records)
    decision = v.decide(summary, integrity=True, reconciled=True)
    assert summary["evidence_candidate_fr_share"] == 2 / 7
    assert decision["status"] == "RESPONSIBILITY_REASSESSMENT_COMPLETE"
    assert not decision["local_sufficiency_triggered"]
    assert decision["recommendation"] == "ENTER_EVIDENCE_DECISION_SCOPE_REDESIGN"
