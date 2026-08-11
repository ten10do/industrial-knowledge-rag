from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend.evaluation import benchmark_runner
from backend.evaluation.benchmark_runner import CHALLENGE_PATH, run_dataset
from backend.evaluation.benchmark_schema import FAILURE_TYPES, classify_failure, evaluate_rows, load_manifest
from backend.evaluation.private_benchmark import (
    _candidate_rows,
    _measure_trace_overhead,
    _resolve_local_file,
    _rerank_analysis,
    _query_for_schema,
    _support_gate_evidence_summary,
    _summary_metrics,
    _support_recovery,
    annotation_hash,
    calibration_hash,
    development_hash,
    load_support_calibration,
    load_private_calibration,
    load_private_development,
    load_private_manifest,
    support_calibration_hash,
)
from backend.retrieval import RetrievalResult


def test_private_schema_adapter_defaults_optional_identifier_label():
    adapted = _query_for_schema({"query_id": "dev", "query": "startup", "category": "procedure", "expected_model": "Drive 100"})
    assert adapted["expected_error_code"] == ""
    assert adapted["expected_equipment_model"] == "Drive 100"


def test_private_summary_accepts_section_dev_without_identifier_label():
    query = {
        "query_id": "dev", "query": "Drive 100 startup", "category": "procedure",
        "answerable": True, "relevant_chunk_ids": ["chunk-a"], "expected_model": "Drive 100",
    }
    row = {
        "query_id": "dev", "query": query["query"], "rank": 1, "refused": False,
        "candidate_ids": ["chunk-a"],
        "candidates": [{"chunk_id": "chunk-a", "equipment_model": "Drive 100", "scope_match": "primary"}],
        "retrieval_scope": {"requested_scope": "EXACT_MODEL_SCOPE"},
    }
    ood_query = {
        "query_id": "ood", "query": "Drive 100 MQTT port", "category": "ood",
        "answerable": False, "relevant_chunk_ids": [], "expected_model": "Drive 100",
    }
    ood_row = {
        "query_id": "ood", "query": ood_query["query"], "rank": None, "refused": True,
        "candidate_ids": [], "candidates": [], "retrieval_scope": {"requested_scope": "EXACT_MODEL_SCOPE"},
    }
    metrics = _summary_metrics([query, ood_query], [row, ood_row])
    assert metrics["overall"]["hit_rate_at_1"] == 1.0
    assert metrics["model_aware_metrics"]["identifier_query_count"] == 0


def test_support_recovery_separates_existing_and_introduced_false_support():
    queries = [
        {"query_id": "recover", "supported": True},
        {"query_id": "lost", "supported": True},
        {"query_id": "existing-false", "supported": False},
        {"query_id": "introduced-false", "supported": False},
    ]
    baseline = {"rows": [
        {"query_id": "recover", "status": "INSUFFICIENT"},
        {"query_id": "lost", "status": "SUPPORTED"},
        {"query_id": "existing-false", "status": "SUPPORTED"},
        {"query_id": "introduced-false", "status": "INSUFFICIENT"},
    ]}
    section = {"rows": [
        {"query_id": "recover", "status": "SUPPORTED"},
        {"query_id": "lost", "status": "INSUFFICIENT"},
        {"query_id": "existing-false", "status": "SUPPORTED"},
        {"query_id": "introduced-false", "status": "SUPPORTED"},
    ]}
    report = _support_recovery(queries, baseline, section)
    assert report["recovered_count"] == 1
    assert report["support_loss_count"] == 1
    assert report["false_support_count"] == 2
    assert report["introduced_false_support_count"] == 1


def test_support_recovery_uses_answerable_when_supported_label_is_absent():
    queries = [
        {"query_id": "q1", "answerable": True},
        {"query_id": "q2", "answerable": False},
    ]
    baseline = {"rows": [
        {"query_id": "q1", "status": "INSUFFICIENT"},
        {"query_id": "q2", "status": "INSUFFICIENT"},
    ]}
    section = {"rows": [
        {"query_id": "q1", "support": {"status": "SUPPORTED"}},
        {"query_id": "q2", "status": "INSUFFICIENT"},
    ]}

    report = _support_recovery(queries, baseline, section)

    assert report["recoverable_count"] == 1
    assert report["recovered_count"] == 1
    assert report["support_recovery_rate"] == 1.0


def test_trace_overhead_measurement_does_not_require_ood_queries(monkeypatch):
    empty = RetrievalResult([])
    monkeypatch.setattr(
        "backend.evaluation.private_benchmark.rag_core.retrieve_docs",
        lambda *args, **kwargs: empty,
    )
    monkeypatch.setattr(
        "backend.evaluation.private_benchmark.rag_core.filter_relevant_docs",
        lambda result: result,
    )
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: SimpleNamespace(result=empty))
    queries = [
        {"query_id": f"q{index}", "query": "answerable", "answerable": True}
        for index in range(3)
    ]

    report = _measure_trace_overhead(queries, None, None, reranker)

    assert report["sample_count"] == 3
    assert report["off_median_ms"] >= 0
    assert report["on_median_ms"] >= 0


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
        "lexical_rank": None, "vector_rank": None, "fusion_rank": 1,
        "identity_relation": "UNKNOWN", "scope_match": "none", "scope_level": "GLOBAL_SCOPE",
        "section_expanded": False, "section_rank": None, "neighbor_distance": None,
        "pre_section_rank": None, "section_candidate_source": "",
    }


def test_private_calibration_is_independent_sized_and_frozen(tmp_path):
    manifest = _private_manifest()
    manifest["freeze"] = {"annotation_sha256": annotation_hash(manifest)}
    calibration = {
        "name": "v3.2-test",
        "corpus_annotation_sha256": annotation_hash(manifest),
        "freeze": {},
        "queries": [
            {
                "query_id": f"c{index:02d}",
                "query": f"Independent calibration question {index}",
                "category": "procedure" if index < 12 else "ood",
                "answerable": index < 12,
                "relevant_chunk_ids": ["chunk-a"] if index < 12 else [],
                "expected_model": "Drive-A",
                "expected_error_code": "",
                "ood_type": "" if index < 12 else "unknown_model",
            }
            for index in range(20)
        ],
    }
    calibration["freeze"]["calibration_sha256"] = calibration_hash(calibration)
    path = tmp_path / "calibration.json"
    path.write_text(__import__("json").dumps(calibration), encoding="utf-8")
    loaded = load_private_calibration(path, manifest, {"chunk-a"})
    assert len(loaded["queries"]) == 20
    assert calibration_hash(loaded) == calibration["freeze"]["calibration_sha256"]


def test_private_v33_development_set_is_independent_sized_and_frozen(tmp_path):
    manifest = _private_manifest()
    manifest["freeze"] = {"annotation_sha256": annotation_hash(manifest)}
    development = {
        "name": "v3.3-test",
        "corpus_annotation_sha256": annotation_hash(manifest),
        "freeze": {},
        "queries": [
            {
                "query_id": f"d{index:02d}",
                "query": f"Independent model-aware development question {index}",
                "category": "exact_model",
                "answerable": True,
                "relevant_chunk_ids": ["chunk-a"],
                "expected_model": "Drive-A",
                "expected_error_code": "",
                "expected_scope": "EXACT_MODEL_SCOPE",
            }
            for index in range(15)
        ],
    }
    development["freeze"]["development_sha256"] = development_hash(development)
    path = tmp_path / "development.json"
    path.write_text(__import__("json").dumps(development), encoding="utf-8")
    loaded = load_private_development(path, manifest, {"chunk-a"})
    assert len(loaded["queries"]) == 15
    assert development_hash(loaded) == development["freeze"]["development_sha256"]


def test_private_v34_support_calibration_is_independent_sized_and_frozen(tmp_path):
    manifest = _private_manifest()
    manifest["freeze"] = {"annotation_sha256": annotation_hash(manifest)}
    calibration = {
        "name": "v3.4-test",
        "corpus_annotation_sha256": annotation_hash(manifest),
        "freeze": {},
        "queries": [
            {
                "query_id": f"s{index:02d}",
                "query": f"Independent evidence support question {index}",
                "category": "supported_parameter" if index < 12 else "hard_negative",
                "supported": index < 12,
                "relevant_chunk_ids": ["chunk-a"] if index < 12 else [],
                "expected_base_decision": "ANSWER",
            }
            for index in range(20)
        ],
    }
    calibration["freeze"]["support_calibration_sha256"] = support_calibration_hash(calibration)
    path = tmp_path / "support-calibration.json"
    path.write_text(__import__("json").dumps(calibration), encoding="utf-8")
    loaded = load_support_calibration(path, manifest, {"chunk-a"})
    assert len(loaded["queries"]) == 20
    assert support_calibration_hash(loaded) == calibration["freeze"]["support_calibration_sha256"]


def test_v34_support_summary_classifies_new_false_refusals():
    base = {"rows": [{
        "query_id": "q1", "answerable": True, "decision": "ANSWER",
        "category": "procedure", "ood_type": "",
    }]}
    support = {"rows": [{
        "query_id": "q1", "final_decision": "ABSTAIN",
        "support": {"status": "INSUFFICIENT"},
    }]}
    report = _support_gate_evidence_summary(base, support)
    assert report["false_refusal_taxonomy"] == {"SUPPORT_INSUFFICIENT": 1}


def test_synthetic_runner_is_deterministic():
    first = run_dataset("synthetic", ("bm25",))
    second = run_dataset("synthetic", ("bm25",))
    assert first["reports"]["bm25"]["metrics"] == second["reports"]["bm25"]["metrics"]
