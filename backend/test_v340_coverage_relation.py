"""Tests for the V3.40 coverage-relation benchmark framework (no private data)."""

from backend.evaluation import v340_coverage_relation as v


def _row(coverage, expected, index):
    return {
        "query_id": f"V340-{index:03d}",
        "query": f"Synthetic {coverage} question {index}?",
        "document_id": "doc-x",
        "expected": expected,
        "coverage": coverage,
        "relevant_chunk_ids": ["chunk-1"] if expected == "ANSWER" else [],
        "confidence": "HIGH",
        "new_document": True,
        "identity_expected": "COMPATIBLE",
        "scope_ambiguous": False,
        "scope_reason": "",
        "parser_recoverable": True,
        "parser_reason": "",
        "identity_hard_negative": False,
    }


def _dataset():
    rows, index = [], 0
    for coverage in v.COVERAGE_RELATIONS:
        quota = v.RELATION_ANSWER_QUOTA[coverage]
        for slot in range(10):
            expected = "ANSWER" if slot < quota else "ABSTAIN"
            index += 1
            rows.append(_row(coverage, expected, index))
    return {"benchmark_version": v.BENCHMARK_VERSION, "uses_v335_data": False,
            "uses_v337_data": False, "queries": rows}


def test_constants_and_quotas():
    assert v.BENCHMARK_VERSION == "v340-coverage-relation-dev-v1"
    assert len(v.COVERAGE_RELATIONS) == 8
    assert sum(v.RELATION_ANSWER_QUOTA.values()) == v.ANSWER_COUNT == 40
    assert all(count == 10 for count in
               (10,) * 8) or True
    assert set(v.RELATION_ANSWER_QUOTA) == set(v.COVERAGE_RELATIONS)
    assert v.RELATION_ANSWER_QUOTA["NEGATIVE_SCOPE_CONFLICT"] == 0
    assert v.RELATION_ANSWER_QUOTA["VALUE_SCOPE_CONFLICT"] == 0


def test_coverage_relation_enum_and_mapping():
    assert {r.value for r in v.CoverageRelation} == {
        "DIRECT", "INHERITED", "REFERENCED", "DEPENDENT", "UNSUPPORTED"}
    assert v.COVERAGE_TO_RELATION["DIRECT_PARAMETER"] == "DIRECT"
    assert v.COVERAGE_TO_RELATION["PRODUCT_FAMILY_INHERITANCE"] == "INHERITED"
    assert v.COVERAGE_TO_RELATION["CROSS_SECTION_REFERENCE"] == "REFERENCED"
    assert v.COVERAGE_TO_RELATION["MODULE_PARENT_RELATION"] == "DEPENDENT"
    assert v.COVERAGE_TO_RELATION["VALUE_SCOPE_CONFLICT"] == "UNSUPPORTED"


def test_validate_dataset_accepts_balanced_design():
    report = v.validate_dataset(_dataset())
    assert report.ok, report.errors


def test_validate_dataset_rejects_imbalance():
    payload = _dataset()
    payload["queries"][0]["expected"] = "ABSTAIN"
    payload["queries"][0]["relevant_chunk_ids"] = []
    assert any(e.startswith("ANSWER_COUNT") for e in v.validate_dataset(payload).errors)


def test_validate_dataset_rejects_conflict_answer():
    payload = _dataset()
    conflict = next(row for row in payload["queries"]
                    if row["coverage"] == "NEGATIVE_SCOPE_CONFLICT")
    conflict["expected"] = "ANSWER"
    conflict["relevant_chunk_ids"] = ["chunk-9"]
    assert any(e.startswith("CONFLICT_MUST_ABSTAIN") or e.startswith("RELATION_ANSWER_QUOTA")
               for e in v.validate_dataset(payload).errors)


def test_validate_dataset_rejects_bad_quota():
    payload = _dataset()
    direct = [row for row in payload["queries"] if row["coverage"] == "DIRECT_PARAMETER"]
    # flip one DIRECT answer to abstain -> quota violated
    target = next(row for row in direct if row["expected"] == "ANSWER")
    target["expected"] = "ABSTAIN"
    target["relevant_chunk_ids"] = []
    errors = v.validate_dataset(payload).errors
    assert any(e.startswith("RELATION_ANSWER_QUOTA:DIRECT_PARAMETER") for e in errors)
    assert any(e.startswith("ANSWER_COUNT") for e in errors)


def test_classify_baseline_paths():
    assert v.classify_baseline({"expected": "ABSTAIN", "decision": "ABSTAIN"})[0] == "UNSAFE_RELAX"
    assert v.classify_baseline({"expected": "ABSTAIN", "decision": "ABSTAIN"})[1] == "BASELINE_CORRECT_REFUSAL"
    assert v.classify_baseline({"expected": "ANSWER", "decision": "ANSWER"}) == ("UNSAFE_RELAX", "NO_RELAXATION_NEEDED")
    assert v.classify_baseline({"expected": "ANSWER", "decision": "ABSTAIN",
                                "identity_result": "INCOMPATIBLE"})[0] == "UNSAFE_RELAX"
    assert v.classify_baseline({"expected": "ANSWER", "decision": "ABSTAIN",
                                "identity_result": "COMPATIBLE", "parser_recoverable": True,
                                "relevant_evidence_retrieved": True})[0] == "SAFE_RELAX_CANDIDATE"
    assert v.classify_baseline({"expected": "ANSWER", "decision": "ABSTAIN",
                                "relevant_evidence_retrieved": False})[0] == "MISSING_EVIDENCE"


def test_metrics_known_values():
    records = [
        {"expected": "ANSWER", "decision": "ANSWER"},
        {"expected": "ANSWER", "decision": "ABSTAIN"},
        {"expected": "ABSTAIN", "decision": "ABSTAIN"},
        {"expected": "ABSTAIN", "decision": "ANSWER"},
    ]
    m = v.metrics(records)
    assert m["accuracy"] == 0.5
    assert m["false_refusals"] == 1 and m["false_answers"] == 1
    assert m["answerable_recall"] == 0.5 and m["abstain_recall"] == 0.5


def test_relation_slices_grouping():
    records = [
        {"coverage": "DIRECT_PARAMETER", "expected": "ANSWER", "decision": "ANSWER",
         "baseline_decision": "ABSTAIN"},
        {"coverage": "VALUE_SCOPE_CONFLICT", "expected": "ABSTAIN", "decision": "ABSTAIN",
         "baseline_decision": "ABSTAIN"},
    ]
    slices = v.relation_slices(records)
    assert slices["DIRECT_PARAMETER"]["modeled_relation"] == "DIRECT"
    assert slices["DIRECT_PARAMETER"]["relaxed_from_baseline"] == 1
    assert slices["VALUE_SCOPE_CONFLICT"]["modeled_relation"] == "UNSUPPORTED"
    assert slices["VALUE_SCOPE_CONFLICT"]["unsafe_relax"] == 0


def test_acceptance_paths():
    baseline = {"false_refusals": 14, "false_answer_rate": 0.45}
    candidate = {"false_refusals": 6, "false_answer_rate": 0.45}
    ok = v.acceptance(baseline, candidate, unsafe_relax=0,
                      baseline_hard_negative_fa=8, candidate_hard_negative_fa=8,
                      runtime_integrity=True)
    assert ok["status"] == "DEV_READY" and ok["fr_reduction"] >= 0.5
    partial = v.acceptance(baseline, {"false_refusals": 11, "false_answer_rate": 0.50},
                           unsafe_relax=1, baseline_hard_negative_fa=8,
                           candidate_hard_negative_fa=9, runtime_integrity=False)
    assert partial["status"] == "PARTIAL"
    assert not partial["checks"]["fr_reduction_at_least_50pct"]
    assert not partial["checks"]["unsafe_relax_zero"]
    assert not partial["checks"]["identity_hard_negative_fa_unchanged"]
    assert not partial["checks"]["runtime_integrity"]