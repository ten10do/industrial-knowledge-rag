from __future__ import annotations

from copy import deepcopy

import pytest

from backend.evaluation.benchmark_runner import CHALLENGE_PATH, run_dataset
from backend.evaluation.benchmark_schema import FAILURE_TYPES, evaluate_rows, load_manifest


def test_challenge_manifest_has_required_coverage():
    manifest = load_manifest(CHALLENGE_PATH)
    assert len(manifest["documents"]) == 13
    assert 30 <= len(manifest["queries"]) <= 50
    assert {"F0002", "F0001", "A0503", "P1080", "MW20", "0x8001", "40001"}.issubset(
        {item["expected_error_code"] for item in manifest["queries"]}
    )
    assert any(len(item["relevant_chunk_ids"]) > 1 for item in manifest["queries"])


def test_manifest_rejects_invalid_annotation(tmp_path):
    manifest = load_manifest(CHALLENGE_PATH)
    invalid = deepcopy(manifest)
    invalid["queries"][0]["relevant_chunk_ids"] = ["missing"]
    path = tmp_path / "invalid.json"
    path.write_text(__import__("json").dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown chunk"):
        load_manifest(path)


def test_manifest_allows_relative_private_file_without_embedded_content(tmp_path):
    manifest = load_manifest(CHALLENGE_PATH)
    private = deepcopy(manifest)
    private["documents"][0].pop("content")
    private["documents"][0]["commit_allowed"] = False
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(private), encoding="utf-8")
    assert load_manifest(path)["documents"][0]["file"] == "g120_faults_excerpt.txt"


def test_metrics_ranking_gap_and_failure_taxonomy():
    queries = [
        {"query_id": "rank", "answerable": True, "category": "semantic", "query_type": "semantic", "expected_error_code": "", "expected_equipment_model": "", "expected_section": "", "relevant_chunk_ids": ["good"]},
        {"query_id": "ood", "answerable": False, "category": "ood", "query_type": "ood", "expected_error_code": "", "expected_equipment_model": "", "expected_section": "", "relevant_chunk_ids": []},
    ]
    rows = [
        {"query_id": "rank", "rank": 2, "refused": False, "candidates": [{"chunk_id": "other"}, {"chunk_id": "good"}]},
        {"query_id": "ood", "rank": None, "refused": False, "candidates": [{"chunk_id": "other"}]},
    ]
    report = evaluate_rows(queries, rows)
    assert report["overall"]["recall_at_5"] == 1
    assert report["overall"]["hit_rate_at_1"] == 0
    assert report["overall"]["ranking_gap"] == 1
    assert report["failure_summary"]["RANKING_FAILURE"] == 1
    assert report["failure_summary"]["OOD_FALSE_POSITIVE"] == 1
    assert set(report["failure_summary"]) == set(FAILURE_TYPES)


def test_private_dataset_is_optional_and_ignored():
    report = run_dataset("private")
    assert report["status"] == "REAL_CORPUS_GATE_NOT_RUN"
    assert "backend/evaluation/benchmark_private/" in (
        CHALLENGE_PATH.parents[3] / ".gitignore"
    ).read_text(encoding="utf-8")


def test_synthetic_runner_is_deterministic():
    first = run_dataset("synthetic", ("bm25",))
    second = run_dataset("synthetic", ("bm25",))
    assert first["reports"]["bm25"]["metrics"] == second["reports"]["bm25"]["metrics"]
