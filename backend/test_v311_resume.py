from __future__ import annotations

import copy

import pytest

from backend.evaluation.resumable import (
    CheckpointCorruptionError,
    CheckpointStore,
    EvaluationRun,
    utc_now,
)
from backend.evaluation.v311_frozen_runner import _validate_retrieval_stage
from backend.evaluation.v311_resume import completed_results, run_query_stage


RULE_VERSION = "v311.2"


def _store(tmp_path) -> CheckpointStore:
    now = utc_now()
    return CheckpointStore(
        tmp_path,
        EvaluationRun(
            run_id="test-v311-resume",
            evaluation_version="V3.11.2",
            corpus_id="A",
            pipeline_id="TEST",
            manifest_hash="manifest",
            annotation_hash="annotation",
            configuration_hash="configuration",
            started_at=now,
            updated_at=now,
        ),
    )


def test_v311_query_checkpoints_skip_completed_rows(tmp_path):
    queries = [
        {"query_id": "q1", "query": "first"},
        {"query_id": "q2", "query": "second"},
    ]
    calls = []
    store = _store(tmp_path)
    store.initialize()
    stage, first = run_query_stage(
        store,
        "EVIDENCE",
        "a",
        queries,
        RULE_VERSION,
        lambda query: calls.append(query["query_id"]) or {
            "query_id": query["query_id"], "decision": "ANSWER",
        },
    )
    assert first == {"completed_before": 0, "skipped": 0, "executed": 2}
    assert calls == ["q1", "q2"]
    assert len(completed_results(stage, queries, RULE_VERSION)) == 2

    resumed = _store(tmp_path)
    resumed.initialize(resume=True)
    _, second = run_query_stage(
        resumed,
        "EVIDENCE",
        "a",
        queries,
        RULE_VERSION,
        lambda query: pytest.fail(f"completed query reran: {query['query_id']}"),
    )
    assert second == {"completed_before": 2, "skipped": 2, "executed": 0}


def test_v311_completed_row_rejects_result_hash_mismatch(tmp_path):
    query = {"query_id": "q1", "query": "first"}
    store = _store(tmp_path)
    store.initialize()
    stage, _ = run_query_stage(
        store,
        "EVIDENCE",
        "a",
        [query],
        RULE_VERSION,
        lambda item: {"query_id": item["query_id"], "decision": "ANSWER"},
    )
    corrupted = copy.deepcopy(stage)
    corrupted["rows"]["q1"]["result"]["decision"] = "ABSTAIN"
    with pytest.raises(CheckpointCorruptionError):
        completed_results(corrupted, [query], RULE_VERSION)


def test_frozen_retrieval_artifact_requires_exact_completed_query_set():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    stage = {
        "validity": "VALID",
        "errors": {},
        "rows": {
            query["query_id"]: {
                "status": "COMPLETED",
                "result": {"query_id": query["query_id"], "candidates": []},
            }
            for query in queries
        },
    }
    assert len(_validate_retrieval_stage(stage, queries, "P2")) == 2
    stage["rows"].pop("q2")
    with pytest.raises(CheckpointCorruptionError):
        _validate_retrieval_stage(stage, queries, "P2")
