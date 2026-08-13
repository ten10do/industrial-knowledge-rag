"""V3.10.2 latency runner tests: markers, checkpoint, resume, watchdog, validity.

These tests exercise the evaluation infrastructure only. They never execute the
frozen retrieval/evidence/support algorithms; they use fakes so the latency
orchestration (markers, atomic checkpoint, resume, timeout handling, warmup
exclusion, combined aggregation) is verified in isolation.
"""

from __future__ import annotations

import time

import pytest

from backend.evaluation import v310_runner
from backend.evaluation.resumable import CheckpointStore, EvaluationRun
from backend.evaluation.v310_runner import (
    LATENCY_QUERY_TIMEOUT_SECONDS, _latency_needs_index, _latency_row,
    _latency_summary, _results, _run_latency_query_stage, _validity,
    aggregate_combined_latency, fixed_latency_subset,
)


def _identity() -> EvaluationRun:
    return EvaluationRun(
        run_id="test-latency", evaluation_version="V3.10.2", corpus_id="A",
        pipeline_id="P1_P2_LATENCY", manifest_hash="m", annotation_hash="a",
        configuration_hash="c", started_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
    )


def _queries(count: int = 4) -> list[dict]:
    return [{"query_id": f"q{index}", "category": "procedure", "query": f"query {index}"} for index in range(count)]


def _row(query_id: str, total: float = 100.0) -> dict:
    return {
        "query_id": query_id, "category": "procedure", "pipeline": "P2",
        "started_at": "t", "finished_at": "t",
        "stages_ms": {
            "raw_retrieval_including_query_analysis_bm25_dense_rrf_scope_section": 40.0,
            "evidence_filter": 0.1, "reranker": 50.0, "total": total,
        },
    }


def _ok_execute(query: dict, watchdog: dict) -> dict:
    watchdog["query_id"] = query["query_id"]
    watchdog["component"] = "reranker"
    watchdog["elapsed_ms"] = 1.0
    return _row(query["query_id"])


def test_first_query_markers_and_atomic_checkpoint(tmp_path, capsys):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    stage = _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(2), _ok_execute)
    assert len(_results(stage)) == 2
    saved = store.load_stage("LATENCY_P2")
    assert saved["rows"]["q0"]["status"] == "COMPLETED"
    assert saved["rows"]["q0"]["pipeline"] == "P2"
    out = capsys.readouterr().out
    assert "LATENCY_QUERY_SELECTED" in out
    assert "LATENCY_QUERY_STARTED" in out
    assert "LATENCY_QUERY_CHECKPOINTED" in out
    assert not store.stage_path("LATENCY_P2").with_suffix(".json.tmp").exists()


def test_latency_resume_does_not_reexecute_completed_samples(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(4), _ok_execute)

    calls: list[str] = []
    resumed = CheckpointStore(tmp_path, _identity())
    resumed.initialize(resume=True)

    def second_execute(query: dict, watchdog: dict) -> dict:
        calls.append(query["query_id"])
        return _ok_execute(query, watchdog)

    stage = _run_latency_query_stage(resumed, "LATENCY_P2", "A", "P2", _queries(4), second_execute)
    assert calls == []
    assert len(_results(stage)) == 4


def test_latency_resume_only_runs_missing_samples(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()
    _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(2), _ok_execute)

    calls: list[str] = []
    resumed = CheckpointStore(tmp_path, _identity())
    resumed.initialize(resume=True)
    stage = resumed.load_stage("LATENCY_P2")
    for query in _queries(4):
        if query["query_id"] not in stage["rows"]:
            calls.append(query["query_id"])
            resumed.save_query("LATENCY_P2", query["query_id"], {"status": "COMPLETED", "result": _row(query["query_id"]), "latency_ms": 1}, total_queries=4)
    assert calls == ["q2", "q3"]


def test_watchdog_timeout_saves_component_and_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(v310_runner, "LATENCY_QUERY_TIMEOUT_SECONDS", 0.3)
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()

    def stalled_execute(query: dict, watchdog: dict) -> dict:
        watchdog["query_id"] = query["query_id"]
        watchdog["component"] = "raw_retrieval"
        watchdog["elapsed_ms"] = 5000.0
        time.sleep(10)
        return _row(query["query_id"])

    with pytest.raises(v310_runner.LatencyTimeoutError):
        _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(1), stalled_execute)
    saved = store.load_stage("LATENCY_P2")
    record = saved["rows"]["q0"]
    assert record["status"] == "TIMEOUT"
    assert record["component"] == "raw_retrieval"
    assert record["error"]["type"] == "LatencyTimeoutError"
    assert saved["errors"]["q0"]["type"] == "LatencyTimeoutError"


def test_timed_out_sample_is_not_a_valid_latency_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(v310_runner, "LATENCY_QUERY_TIMEOUT_SECONDS", 0.2)
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()

    def execute(query: dict, watchdog: dict) -> dict:
        watchdog["component"] = "reranker"
        if query["query_id"] == "q0":
            time.sleep(10)
        return _row(query["query_id"])

    with pytest.raises(v310_runner.LatencyTimeoutError):
        _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(2), execute)
    stage = store.load_stage("LATENCY_P2")
    assert stage["rows"]["q0"]["status"] == "TIMEOUT"
    assert _results(stage) == []
    assert _validity(stage, 2) == "PARTIAL"


def test_error_sample_is_retained_but_excluded_from_valid_rows(tmp_path):
    store = CheckpointStore(tmp_path, _identity())
    store.initialize()

    def failing_execute(query: dict, watchdog: dict) -> dict:
        watchdog["component"] = "reranker"
        raise RuntimeError("boom")

    stage = _run_latency_query_stage(store, "LATENCY_P2", "A", "P2", _queries(2), failing_execute)
    saved = store.load_stage("LATENCY_P2")
    assert saved["rows"]["q0"]["status"] == "ERROR"
    assert saved["errors"]["q0"]["type"] == "RuntimeError"
    assert _results(stage) == []


def test_summary_counts_valid_timeout_and_error_samples():
    stage = {"rows": {"q0": {"status": "COMPLETED"}, "q1": {"status": "TIMEOUT"}, "q2": {"status": "ERROR"}}}
    summary = _latency_summary([_row("q0")], target_samples=3, stage=stage)
    assert summary["valid_samples"] == 1
    assert summary["target_samples"] == 3
    assert summary["timeout_samples"] == 1
    assert summary["error_samples"] == 1
    assert _validity(stage, 3) == "PARTIAL"
    complete = {"rows": {"q0": {"status": "COMPLETED"}, "q1": {"status": "COMPLETED"}}}
    assert _validity(complete, 2) == "VALID"


def test_empty_summary_is_safe():
    summary = _latency_summary([], target_samples=8, stage={"rows": {}})
    assert summary["sample_count"] == 0
    assert summary["median_ms"] is None
    assert summary["p95_ms"] is None


def test_warmup_is_excluded_by_construction(monkeypatch):
    class FakeReranker:
        def rerank(self, query: str, candidates, top_k: int | None = None):
            return None

    retrieved = type("R", (), {"candidates": []})()
    monkeypatch.setattr(v310_runner.rag_core, "retrieve_docs", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(v310_runner.rag_core, "filter_relevant_docs", lambda raw: retrieved)
    warmup = _latency_row(_queries(1)[0], v310_runner.P2, FakeReranker(), watchdog=None, emit_markers=False)
    assert warmup["stages_ms"]["total"] >= 0
    assert "pipeline" in warmup and warmup["pipeline"] == "P2"
    assert "started_at" in warmup and "finished_at" in warmup
    # Warmup never writes a checkpoint: it returns a row instead of calling save_query.
    assert "status" not in warmup


def test_latency_row_emits_stage_markers(tmp_path, monkeypatch, capsys):
    retrieved = type("R", (), {"candidates": []})()

    class FakeReranker:
        def rerank(self, query: str, candidates, top_k: int | None = None):
            return None

    monkeypatch.setattr(v310_runner.rag_core, "retrieve_docs", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(v310_runner.rag_core, "filter_relevant_docs", lambda raw: retrieved)
    watchdog: dict = {}
    _latency_row(_queries(1)[0], v310_runner.P2, FakeReranker(), watchdog=watchdog)
    out = capsys.readouterr().out
    assert "RAW_RETRIEVAL_STARTED" in out
    assert "RAW_RETRIEVAL_COMPLETED" in out
    assert "RERANK_STARTED" in out
    assert "RERANK_COMPLETED" in out
    assert watchdog["component"] == "reranker"


def test_needs_index_skips_complete_stages_only():
    complete = {"rows": {f"q{index}": {"status": "COMPLETED"} for index in range(8)}}
    assert _latency_needs_index(8, [complete, complete]) is False
    partial = {"rows": {f"q{index}": {"status": "COMPLETED"} for index in range(4)}}
    assert _latency_needs_index(8, [complete, partial]) is True
    timeout = {"rows": {"q0": {"status": "TIMEOUT"}, "q1": {"status": "COMPLETED"}}}
    assert _latency_needs_index(2, [timeout, timeout]) is True
    assert _latency_needs_index(2, [None, None]) is True


def test_combined_latency_uses_query_level_samples_and_passes_through_counts():
    rows = [_row(f"q{index}", total=float(100 + index)) for index in range(8)]
    corpus = {
        "validity": "VALID",
        "reports": {"p1": {"rows": rows, "target_samples": 8, "timeout_samples": 0, "error_samples": 0}, "p2": {"rows": rows, "target_samples": 8, "timeout_samples": 0, "error_samples": 0}},
    }
    combined = aggregate_combined_latency(corpus, corpus)
    assert combined["validity"] == "VALID"
    assert combined["reports"]["p1"]["sample_count"] == 16
    assert combined["reports"]["p1"]["target_samples"] == 16
    assert combined["reports"]["p1"]["median_ms"] == 103.5


def test_combined_latency_is_partial_when_either_corpus_is_partial():
    corpus = {
        "validity": "VALID",
        "reports": {"p1": {"rows": [_row("q0")], "target_samples": 8, "timeout_samples": 0, "error_samples": 0}, "p2": {"rows": [_row("q0")], "target_samples": 8, "timeout_samples": 0, "error_samples": 0}},
    }
    partial = {
        "validity": "PARTIAL",
        "reports": {"p1": {"rows": [_row("q0")], "target_samples": 8, "timeout_samples": 1, "error_samples": 0}, "p2": {"rows": [], "target_samples": 8, "timeout_samples": 1, "error_samples": 0}},
    }
    combined = aggregate_combined_latency(corpus, partial)
    assert combined["validity"] == "PARTIAL"
    assert combined["reports"]["p1"]["timeout_samples"] == 1


def test_fixed_latency_subset_is_frozen_and_balanced():
    queries = [
        {"query_id": f"{category}-{index}", "category": category}
        for category in ("identifier", "procedure", "semantic", "ood")
        for index in range(3)
    ]
    subset = fixed_latency_subset(queries)
    assert [query["query_id"] for query in subset] == [
        "identifier-0", "identifier-1", "procedure-0", "procedure-1",
        "semantic-0", "semantic-1", "ood-0", "ood-1",
    ]
    assert len(subset) == 8


def test_latency_version_is_distinct_from_correctness_version():
    assert v310_runner.LATENCY_EVALUATION_VERSION == "V3.10.2"
    assert v310_runner.EVALUATION_VERSION == "V3.10.1"
