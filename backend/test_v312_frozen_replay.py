from __future__ import annotations

import copy
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.evaluation.frozen_retrieval_artifact import (
    ARTIFACT_SCHEMA_VERSION, ArtifactValidationError, artifact_hash,
    candidate_from_p2_row, deserialize_candidate, load_valid_artifact,
    new_artifact_payload, seal_artifact, serialize_candidate,
    serialize_query_analysis, validate_artifact, write_immutable_artifact,
)
from backend.evaluation.resumable import (
    CheckpointCorruptionError, read_json,
)
from backend.evaluation.v311_resume import hash_json, run_query_stage
from backend.evaluation import v312_replay_runner as runner
from backend.retrieval.candidates import RetrievalCandidate
from backend.retrieval.filters import QueryAnalysis
from backend.retrieval.product_identity import ProductIdentity


def _candidate(content: str = "ACS580 parameter 99.99 value 10 Hz") -> RetrievalCandidate:
    return RetrievalCandidate(
        document=Document(page_content=content, metadata={
            "chunk_id": "chunk-1", "document_id": "doc-1",
            "manufacturer": "ABB", "equipment_model": "ACS580",
            "product_family": "ACS", "knowledge_type": "parameter",
        }),
        retrieval_source="hybrid", lexical_rank=1, vector_rank=2,
        lexical_score=8.5, vector_score=4.25, fusion_score=.03,
        final_rank=1, pre_rerank_rank=2, rerank_score=.91, rerank_rank=1,
        identity_relation="EXACT_MODEL", scope_match="primary",
        scope_level="EXACT_MODEL_SCOPE",
    )


def _analysis() -> QueryAnalysis:
    identity = ProductIdentity(
        manufacturer="ABB", product_family="ACS", equipment_model="ACS580",
        aliases=("ACS580",),
    )
    return QueryAnalysis(
        equipment_model="ACS580", manufacturer="ABB", product_family="ACS",
        identity_confidence="EXACT_MODEL", product_identities=(identity,),
    )


def _query_row(query_id: str = "q01") -> dict:
    candidate = serialize_candidate(_candidate())
    return {
        "query_id": query_id,
        "query": "What is ACS580 parameter 99.99?",
        "query_text_hash": hash_json("What is ACS580 parameter 99.99?"),
        "ground_truth": {
            "query_id": query_id, "query": "What is ACS580 parameter 99.99?",
            "answerable": True, "supported": True,
        },
        "evidence_input": {
            "retrieval_mode": "hybrid",
            "query_analysis": serialize_query_analysis(_analysis()),
            "candidate_pool": [candidate],
        },
        "final_context": [candidate],
        "retrieval_decision_inputs": {"retrieval_mode": "hybrid"},
    }


def _artifact(count: int = 1) -> dict:
    rows = [_query_row(f"q{index:02d}") for index in range(1, count + 1)]
    snapshot = [{
        "content": "99.99 F101",
        "metadata": {
            "manufacturer": "ABB", "equipment_model": "ACS580",
            "product_family": "ACS", "knowledge_type": "parameter",
        },
    }]
    return seal_artifact(new_artifact_payload(
        artifact_id="test-artifact", corpus_id="T",
        manifest_hash=hash_json({"manifest": 1}), annotation_hash="annotation",
        retrieval_config={"embedding": "frozen", "candidate_k": 7},
        queries=rows, snapshot_documents=snapshot,
        source={"p2_reused_without_retrieval": True}, rule_version="v-old",
    ))


def test_candidate_round_trip_preserves_content_metadata_scores_and_ranks():
    payload = serialize_candidate(_candidate())
    restored = deserialize_candidate(payload)
    assert restored.document.page_content == _candidate().document.page_content
    assert restored.metadata == _candidate().metadata
    assert restored.lexical_rank == 1
    assert restored.vector_score == 4.25
    assert restored.rerank_score == .91
    assert restored.identity_relation == "EXACT_MODEL"


def test_old_p2_row_is_enriched_without_fabricating_unavailable_scores():
    document = _candidate().document
    payload = candidate_from_p2_row({
        "chunk_id": "chunk-1", "rank": 2, "vector_distance": 7.0,
        "rerank_rank": 2, "section_candidate_source": "section_retrieval",
    }, document)
    assert payload["content"] == document.page_content
    assert payload["metadata"] == document.metadata
    assert payload["final_rank"] == 2
    assert payload["vector_score"] == 7.0
    assert "rerank_score" not in payload


def test_schema_hash_and_version_validate_independently_from_rule_version():
    payload = _artifact()
    payload["rule_version_at_export"] = "some-other-rule"
    payload["artifact_hash"] = artifact_hash(payload)
    report = validate_artifact(payload)
    assert report["validity"] == "VALID"
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_hash_and_expected_configuration_mismatch_are_invalid():
    payload = _artifact()
    payload["queries"][0]["query"] = "tampered"
    report = validate_artifact(
        payload, expected_manifest_hash="wrong-manifest",
        expected_annotation_hash="wrong-annotation",
        expected_retrieval_config_hash="wrong-config",
    )
    assert report["validity"] == "INVALID"
    assert "artifact hash mismatch" in report["invalid_reasons"]
    assert "retrieval_config_hash mismatch" in report["invalid_reasons"]
    assert "corpus_manifest_hash mismatch" in report["invalid_reasons"]
    assert "annotation_hash mismatch" in report["invalid_reasons"]


def test_missing_final_context_is_partial_after_resealing():
    payload = _artifact()
    del payload["queries"][0]["final_context"]
    payload["artifact_hash"] = artifact_hash(payload)
    report = validate_artifact(payload)
    assert report["validity"] == "PARTIAL"


def test_missing_query_is_partial_even_if_artifact_is_resealed():
    payload = _artifact(2)
    payload["queries"].pop()
    payload["artifact_hash"] = artifact_hash(payload)
    report = validate_artifact(payload)
    assert report["validity"] == "PARTIAL"
    assert "frozen query set is incomplete" in report["partial_reasons"]


def test_immutable_write_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "artifact.json"
    write_immutable_artifact(path, new_artifact_payload(
        artifact_id="one", corpus_id="T", manifest_hash="m",
        annotation_hash="a", retrieval_config={}, queries=[],
        snapshot_documents=[{"content": "x", "metadata": {}}],
        source={}, rule_version="v1",
    ))
    with pytest.raises(FileExistsError):
        write_immutable_artifact(path, {})


def test_corrupted_json_and_invalid_artifact_are_refused(tmp_path: Path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"broken":', encoding="utf-8")
    with pytest.raises(CheckpointCorruptionError):
        read_json(corrupt)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError):
        load_valid_artifact(invalid)


def test_replay_is_equivalent_and_does_not_call_live_dependencies(monkeypatch):
    artifact = _artifact()

    def forbidden(*args, **kwargs):
        raise AssertionError("live dependency accessed")

    from backend import rag_core
    from backend.evaluation import private_benchmark
    from backend.retrieval import reranker

    monkeypatch.setattr(rag_core, "retrieve_docs", forbidden)
    monkeypatch.setattr(rag_core, "load_vector_db", forbidden)
    monkeypatch.setattr(rag_core, "load_lexical_db", forbidden)
    monkeypatch.setattr(rag_core, "load_pdf", forbidden)
    monkeypatch.setattr(private_benchmark, "ingest_private_documents", forbidden)
    monkeypatch.setattr(reranker.CrossEncoderReranker, "rerank", forbidden)
    monkeypatch.setattr(reranker.CrossEncoderReranker, "_load_model", forbidden)
    snapshot = [Document(**{
        "page_content": item["content"], "metadata": item["metadata"],
    }) for item in artifact["corpus_snapshot"]["documents"]]
    first = runner.replay_query(artifact["queries"][0], snapshot)
    second = runner.replay_query(artifact["queries"][0], snapshot)
    assert first["base_decision"] == second["base_decision"]
    assert first["base_reason"] == second["base_reason"]
    assert first["support"] == second["support"]
    assert first["final_decision"] == second["final_decision"]


def test_resume_ten_of_thirty_and_materializes_valid_summary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", tmp_path / "runtime")
    artifact = _artifact(30)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        __import__("json").dumps(artifact, ensure_ascii=False), encoding="utf-8",
    )
    store = runner._replay_store(artifact, "resume-30")
    store.initialize()
    snapshot = [Document(page_content="99.99 F101", metadata={
        "manufacturer": "ABB", "equipment_model": "ACS580", "product_family": "ACS",
    })]
    run_query_stage(
        store, "REPLAY", "T", artifact["queries"][:10],
        runner.EVIDENCE_SUPPORT_RULE_VERSION,
        lambda row: runner.replay_query(row, snapshot),
    )
    result = runner.replay_artifact(artifact_path, "resume-30", resume=True)
    assert result["validity"] == "VALID"
    assert result["query_count"] == 30
    assert result["resume"] == {"completed_before": 10, "skipped": 10, "executed": 20}


def test_compare_reports_exact_fixed_and_introduced_failures():
    baseline = {"rows": [{
        "query_id": "q1", "base_decision": "ANSWER", "base_reason": "x",
        "support": {"status": "SUPPORTED"}, "final_decision": "ANSWER",
        "predicted_supported": True, "expected_supported": False,
    }]}
    assert runner.compare_results(baseline, copy.deepcopy(baseline))["exact_equivalence"]
    changed = copy.deepcopy(baseline)
    changed["rows"][0].update(final_decision="ABSTAIN", predicted_supported=False)
    comparison = runner.compare_results(baseline, changed)
    assert comparison["fixed_failures"] == ["q1"]
    assert not comparison["exact_equivalence"]
    assert comparison["false_support_fixed"] == ["q1"]


def test_combined_aggregation_uses_saved_rows_only():
    row_a = runner.replay_query(
        _artifact()["queries"][0],
        [Document(page_content="99.99", metadata={"equipment_model": "ACS580"})],
    )
    row_a.update(artifact_id="a", artifact_hash="ha", evidence_rule_version="v", support_rule_version="v")
    row_b = {**row_a, "query_id": "q02", "artifact_id": "b"}
    combined = runner.combine_replay_results(
        {"validity": "VALID", "artifact_id": "a", "rows": [row_a]},
        {"validity": "VALID", "artifact_id": "b", "rows": [row_b]},
    )
    assert combined["query_count"] == 2
    assert combined["source_artifact_ids"] == ["a", "b"]


@pytest.mark.parametrize(
    ("fixture_class", "answerable", "expected_supported", "base_decision", "final_decision"),
    [
        ("evidence-recovery-like", True, True, "ANSWER", "ANSWER"),
        ("support-recovery-like", True, True, "ANSWER", "ANSWER"),
        ("false-answer-caught-like", False, False, "ANSWER", "ABSTAIN"),
        ("false-support-one-like", True, False, "ANSWER", "ANSWER"),
        ("false-support-two-like", True, False, "ANSWER", "ANSWER"),
    ],
)
def test_failure_class_equivalence_fixtures(
    fixture_class, answerable, expected_supported, base_decision, final_decision,
):
    baseline = {"rows": [{
        "query_id": fixture_class, "answerable": answerable,
        "expected_supported": expected_supported, "base_decision": base_decision,
        "base_reason": "fixture", "support": {
            "status": "INSUFFICIENT" if final_decision == "ABSTAIN" else "SUPPORTED",
            "reason": "fixture",
        }, "final_decision": final_decision,
        "predicted_supported": final_decision == "ANSWER",
    }]}
    assert runner.compare_results(baseline, copy.deepcopy(baseline))["exact_equivalence"]


def test_private_path_guard_rejects_public_location(tmp_path: Path):
    with pytest.raises(ValueError):
        runner.ensure_private_path(tmp_path / "public.json")
    with pytest.raises(ValueError):
        runner._safe_identity("../outside", "run id")


def test_production_request_modules_do_not_import_replay_infrastructure():
    root = Path(__file__).resolve().parent
    production_files = [root / "main.py", root / "rag_core.py", *sorted((root / "retrieval").glob("*.py"))]
    for path in production_files:
        source = path.read_text(encoding="utf-8")
        assert "v312_replay_runner" not in source
        assert "frozen_retrieval_artifact" not in source
