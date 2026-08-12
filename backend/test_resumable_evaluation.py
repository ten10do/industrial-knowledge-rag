from __future__ import annotations

from dataclasses import replace

import pytest

from backend.evaluation.resumable import (
    CheckpointCorruptionError, CheckpointStore, EvaluationRun,
    ResumeConfigurationMismatch, atomic_write_json, read_json,
)


def _identity() -> EvaluationRun:
    return EvaluationRun(
        run_id="test-run", evaluation_version="V3.10.1", corpus_id="B", pipeline_id="P2",
        manifest_hash="manifest", annotation_hash="annotation", configuration_hash="configuration",
        started_at="2026-08-12T00:00:00+00:00", updated_at="2026-08-12T00:00:00+00:00",
    )


def test_interrupted_query_stage_resumes_without_reexecuting_completed_queries(tmp_path):
    calls: list[str] = []
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    for query_id in ("q1", "q2"):
        calls.append(query_id)
        store.save_query("RETRIEVAL", query_id, {"status": "COMPLETED", "result": {"query_id": query_id}, "latency_ms": 1}, total_queries=3)

    resumed = CheckpointStore(tmp_path, _identity())
    resumed.initialize(resume=True)
    stage = resumed.load_stage("RETRIEVAL")
    for query_id in ("q1", "q2", "q3"):
        if query_id not in stage["rows"]:
            calls.append(query_id)
            resumed.save_query("RETRIEVAL", query_id, {"status": "COMPLETED", "result": {"query_id": query_id}, "latency_ms": 1}, total_queries=3)
    assert calls == ["q1", "q2", "q3"]


def test_resume_refuses_a_configuration_mismatch(tmp_path):
    CheckpointStore(tmp_path, _identity()).initialize()
    mismatched = CheckpointStore(tmp_path, replace(_identity(), configuration_hash="other"))
    with pytest.raises(ResumeConfigurationMismatch, match="RESUME_REFUSED_CONFIGURATION_MISMATCH"):
        mismatched.initialize(resume=True)


def test_corrupted_checkpoint_is_reported_without_overwriting_it(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    path = store.stage_path("RETRIEVAL")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CheckpointCorruptionError, match="Corrupted checkpoint"):
        store.load_stage("RETRIEVAL")
    assert path.read_text(encoding="utf-8") == "{broken"


def test_atomic_json_write_replaces_a_complete_document(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"state": "first"})
    atomic_write_json(path, {"state": "second", "rows": [1, 2]})
    assert read_json(path) == {"state": "second", "rows": [1, 2]}
    assert not path.with_suffix(".json.tmp").exists()


def test_begin_stage_removes_an_ephemeral_stage_from_completed_progress(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    store.save_stage("INDEX_BUILD", {"stage": "INDEX_BUILD"})
    store.begin_stage("INDEX_BUILD", 3)
    assert "INDEX_BUILD" not in read_json(store.progress_path)["completed_stages"]
    assert read_json(store.progress_path)["current_stage"] == "INDEX_BUILD"


def test_partial_query_failure_is_retained_while_later_queries_complete(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    store.save_query("EVIDENCE", "q1", {"status": "ERROR", "result": {}, "latency_ms": 1}, total_queries=2, error={"type": "ValueError", "message": "bad row"})
    store.save_query("EVIDENCE", "q2", {"status": "COMPLETED", "result": {"query_id": "q2"}, "latency_ms": 1}, total_queries=2)
    stage = store.load_stage("EVIDENCE")
    assert stage["rows"]["q2"]["status"] == "COMPLETED"
    assert stage["errors"]["q1"] == {"type": "ValueError", "message": "bad row"}


def test_resume_may_retry_an_error_but_never_reexecutes_a_completed_query(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    store.save_query("RETRIEVAL", "q1", {"status": "COMPLETED", "result": {"query_id": "q1"}, "latency_ms": 1}, total_queries=2)
    store.save_query("RETRIEVAL", "q2", {"status": "ERROR", "result": {}, "latency_ms": 1}, total_queries=2, error={"type": "RuntimeError", "message": "temporary"})
    resumed = CheckpointStore(tmp_path, _identity())
    resumed.initialize(resume=True)
    stage = resumed.load_stage("RETRIEVAL")
    retry_ids = [query_id for query_id in ("q1", "q2") if stage["rows"].get(query_id, {}).get("status") != "COMPLETED"]
    assert retry_ids == ["q2"]
    resumed.save_query("RETRIEVAL", "q2", {"status": "COMPLETED", "result": {"query_id": "q2"}, "latency_ms": 1}, total_queries=2)
    assert "q2" not in resumed.load_stage("RETRIEVAL")["errors"]
