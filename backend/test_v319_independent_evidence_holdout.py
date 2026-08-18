"""Public synthetic checks for the private Corpus E validation harness."""
from copy import deepcopy

from backend.evaluation.private_benchmark import annotation_hash
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v319_independent_evidence_holdout import (
    EXPECTED_RETRIEVAL_CONFIG_HASH, failure_attribution, generalization_range,
    holdout_distribution, query_hash, validate_holdout_manifest, validate_source_manifest,
)


def _documents():
    return [{"document_id": f"doc-{index}", "file": f"doc-{index}.pdf", "source_name": "official",
             "source_type": "official_vendor_publication", "official_url": "https://vendor.example/manual.pdf",
             "manufacturer": f"Vendor {index % 2}", "equipment_type": "servo_drive", "equipment_model": f"Model {index}",
             "document_type": "manual", "language": "English", "version": "1", "publish_date": "2026-01-01", "commit_allowed": False}
            for index in range(3)]


def _manifest():
    rows = []
    for index in range(42):
        answerable = index < 22
        row = {"query_id": f"e{index:02}", "query": f"independent query {index}", "category": "semantic" if index < 8 else ("ood" if index >= 33 else "mixed"),
               "answerable": answerable, "relevant_chunk_ids": [f"chunk-{index}"] if answerable else [],
               "relevant_document_ids": [f"doc-{index % 3}"] if answerable else [], "expected_model": f"Model {index % 3}",
               "expected_error_code": "", "expected_section": "", "difficulty": "hard", "manufacturer": f"Vendor {index % 2}",
               "confidence": "HIGH" if index < 36 else ("MEDIUM" if index < 40 else "AMBIGUOUS"),
               "requested": {key: "" for key in ("identity", "identifier", "protocol", "action", "attribute", "value", "unit", "value_kind", "requirement_type", "qualifier")},
               "expected_evidence": "reviewed", "annotation_rationale": "reviewed", "failure_class": "OTHER",
               "support_gate_truth": "SUPPORTED" if answerable else "INSUFFICIENT", "semantic_hard_positive": index < 8,
               "multi_chunk_positive": index < 4, "cross_chunk_negative": 22 <= index < 25,
               "hard_negative": not answerable, "ood": index >= 33}
        rows.append(row)
    manifest = {"corpus_id": "E", "documents": _documents(), "queries": rows}
    manifest["freeze"] = {"query_sha256": query_hash(manifest), "annotation_sha256": annotation_hash(manifest)}
    manifest["freeze"]["manifest_sha256"] = hash_json({key: value for key, value in manifest.items() if key != "freeze"})
    return manifest


def test_manifest_freeze_distribution_and_coverage_are_enforced(monkeypatch):
    # This is a historical V3.19 manifest contract. Validate it against the
    # rule identity it froze, independently of the current production rule.
    monkeypatch.setattr(
        "backend.evaluation.v319_independent_evidence_holdout.EVIDENCE_SUPPORT_RULE_VERSION",
        "evidence-v318.1",
    )
    manifest = _manifest()
    distribution = validate_holdout_manifest(manifest)
    assert distribution["answerable"] == 22
    assert distribution["abstain"] == 20
    assert distribution["semantic_hard_positive"] == 8
    assert distribution["multi_chunk_positive"] == 4
    assert distribution["cross_chunk_negative"] == 3
    assert distribution["ood"] == 9
    changed = deepcopy(manifest); changed["queries"][0]["query"] = "changed after freeze"
    try:
        validate_holdout_manifest(changed)
    except ValueError as error:
        assert "QUERY_HASH" in str(error)
    else:
        raise AssertionError("freeze mutation must be rejected")


def test_source_and_generalization_contracts():
    validate_source_manifest({"documents": _documents()})
    matrix = {key: {"decision_accuracy": .8, "answerable_recall": .7, "ood_recall": .9, "false_answer_rate": .1, "false_refusal_rate": .3} for key in "ABCDE"}
    matrix["E"]["answerable_recall"] = .5
    assert generalization_range(matrix)["answerable_recall"] == {"min": .5, "max": .7}
    assert len(EXPECTED_RETRIEVAL_CONFIG_HASH) == 64


def test_evidence_failure_attribution_separates_retrieval_from_rule():
    manifest = _manifest()
    replay = {"rows": [{"query_id": "e00", "answerable": True, "base_decision": "ABSTAIN", "candidate_ids": {"evidence": []}}, {"query_id": "e22", "answerable": False, "base_decision": "ANSWER", "candidate_ids": {"evidence": []}}]}
    artifact = {"source": {"parser_audit": {"production_ingestion_audit": {}}}}
    rows = failure_attribution(replay, manifest, artifact)
    assert [row["attribution"] for row in rows] == ["RETRIEVAL_MISSING_EVIDENCE", "EVIDENCE_RULE_FAILURE"]
