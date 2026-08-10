from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend.evaluation import benchmark_runner
from backend.evaluation.benchmark_runner import CHALLENGE_PATH, run_dataset
from backend.evaluation.benchmark_schema import FAILURE_TYPES, classify_failure, evaluate_rows, load_manifest
from backend.evaluation.private_benchmark import (
    _candidate_rows,
    _resolve_local_file,
    _rerank_analysis,
    annotation_hash,
    load_private_manifest,
)


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


def test_specific_confusion_is_mutually_exclusive_with_generic_recall_failure():
    query = {
        "query_id": "comparison", "answerable": True, "category": "comparison",
        "query_type": "comparison", "expected_error_code": "",
        "expected_equipment_model": "G120C", "expected_section": "Commissioning",
        "relevant_chunk_ids": ["G120C-STARTUP"],
    }
    row = {
        "query_id": "comparison", "rank": None, "refused": False,
        "candidates": [{"chunk_id": "G120-STARTUP", "equipment_model": "G120", "section": "Commissioning"}],
    }
    assert classify_failure(query, row) == "MODEL_CONFUSION"


def test_private_dataset_is_optional_and_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "PRIVATE_PATH", tmp_path / "missing-manifest.json")
    report = run_dataset("private")
    assert report["status"] == "REAL_CORPUS_GATE_NOT_RUN"
    assert "backend/evaluation/benchmark_private/" in (
        CHALLENGE_PATH.parents[3] / ".gitignore"
    ).read_text(encoding="utf-8")


def _private_manifest():
    return {
        "name": "local-private-test",
        "documents": [{
            "document_id": "doc-a", "file": "documents/manual.pdf",
            "source_name": "Vendor Manual", "source_type": "official_vendor_publication",
            "manufacturer": "Vendor", "equipment_type": "drive", "equipment_model": "Drive-A",
            "document_type": "operating_manual", "language": "en", "version": "1.0",
            "publish_date": "2025-01", "commit_allowed": False,
        }],
        "queries": [{
            "query_id": "q1", "query": "How do I start Drive-A?", "category": "procedure",
            "answerable": True, "relevant_chunk_ids": ["chunk-a"],
            "relevant_document_ids": ["doc-a"], "expected_model": "Drive-A",
            "expected_error_code": "", "expected_section": "Commissioning", "difficulty": "medium",
        }],
    }


def test_private_manifest_enforces_local_only_metadata_and_freeze_hash(tmp_path):
    manifest = _private_manifest()
    manifest["freeze"] = {"annotation_sha256": annotation_hash(manifest)}
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    assert load_private_manifest(path)["freeze"]["annotation_sha256"] == annotation_hash(manifest)

    manifest["documents"][0]["commit_allowed"] = True
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="commit_allowed=false"):
        load_private_manifest(path)


def test_private_manifest_rejects_missing_files_and_stale_freeze_hash(tmp_path):
    manifest = _private_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_local_file(path, "documents/manual.pdf")

    manifest["freeze"] = {"annotation_sha256": "stale"}
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_private_manifest(path)


def test_private_rerank_analysis_handles_a_relevant_chunk_dropped_by_final_top_k():
    report = _rerank_analysis([
        {"query_id": "drop", "answerable": True, "before_rank": 4, "after_rank": None, "candidate_missing": False},
        {"query_id": "same", "answerable": True, "before_rank": 1, "after_rank": 1, "candidate_missing": False},
        {"query_id": "missing", "answerable": True, "before_rank": None, "after_rank": None, "candidate_missing": True},
        {"query_id": "ood", "answerable": False, "before_rank": None, "after_rank": None, "candidate_missing": True},
    ])
    assert report["degraded"] == 1
    assert report["same"] == 1
    assert report["candidate_missing"] == 1


def test_private_candidate_rows_include_citation_metadata():
    candidate = SimpleNamespace(
        chunk_id="chunk-a",
        metadata={"document_id": "doc-a", "source": "Vendor Manual", "page": 12, "section": "Setup"},
        vector_score=0.1,
        lexical_score=0.2,
        pre_rerank_rank=1,
        rerank_rank=1,
    )
    assert _candidate_rows([candidate])[0] == {
        "rank": 1, "chunk_id": "chunk-a", "document_id": "doc-a", "source": "Vendor Manual",
        "page": 12, "section": "Setup", "equipment_model": "", "error_code": "",
        "vector_distance": 0.1, "lexical_score": 0.2, "pre_rerank_rank": 1, "rerank_rank": 1,
    }


def test_synthetic_runner_is_deterministic():
    first = run_dataset("synthetic", ("bm25",))
    second = run_dataset("synthetic", ("bm25",))
    assert first["reports"]["bm25"]["metrics"] == second["reports"]["bm25"]["metrics"]
