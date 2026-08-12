"""Resumable V3.10.1 private cross-corpus correctness and latency evaluation.

The runner deliberately calls the frozen retrieval/evidence/support functions
unchanged. It only changes orchestration: per-query checkpoints, atomic stage
results, resume validation, and A/B-to-Combined aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base
from backend.evaluation.multi_corpus_validation import (
    combined_metrics, cross_corpus_failure_matrix, generalization_gaps,
)
from backend.evaluation.private_benchmark import (
    _evidence_report, _frozen_support_report, _run_mode, _section_metrics,
    _section_mode, _summarize_evidence, _summary_metrics, ingest_private_documents,
    load_private_manifest,
)
from backend.evaluation.resumable import (
    CheckpointStore, EvaluationRun, ResumeConfigurationMismatch, atomic_write_json,
    read_json, utc_now,
)
from backend.evaluation.retrieval_observability import overlay_relevance, query_trace_summary
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend import rag_core


EVALUATION_VERSION = "V3.10.1"
PRIVATE_ROOT = Path("backend/evaluation/benchmark_private")
RUNTIME_ROOT = PRIVATE_ROOT / "v310_runtime"
CORPUS_PATHS = {"a": Path("."), "b": Path("corpus_b")}
P1 = {"id": "P1", "mode": "hybrid_rerank", "candidate_k": 5, "section_strategy": None}
P2 = {"id": "P2", "mode": "hybrid_section_rerank", "candidate_k": 7, "section_strategy": "append_then_rerank"}
LATENCY_CATEGORIES = ("identifier", "procedure", "semantic", "ood")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _annotation_hash(manifest: dict[str, Any]) -> str:
    return str(manifest.get("annotation_enrichment", {}).get("enriched_annotation_sha256") or manifest["freeze"]["annotation_sha256"])


def evaluation_configuration() -> dict[str, Any]:
    """Capture every runtime setting that can change frozen evaluation output."""
    names = ("FULL_MAX_RELEVANT_DISTANCE", "LEXICAL_TOP_K", "VECTOR_TOP_K", "HYBRID_TOP_K", "RRF_K")
    environment = {
        key: value for key, value in os.environ.items()
        if key in names or key.startswith(("SECTION_", "EVIDENCE_", "SUPPORT_"))
    }
    reranker = RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu")
    return {
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "pipelines": {"p1": P1, "p2": P2},
        "reranker": {"enabled": reranker.enabled, "model": reranker.model_name, "candidate_k": reranker.candidate_k, "top_k": reranker.top_k, "device": reranker.device},
        "top_k": 3,
        "trace_enabled": False,
        "environment": environment,
    }


def timeout_root_cause_audit() -> list[dict[str, str]]:
    """Static audit of the timed-out V3.10 script, based on its call graph."""
    return [
        {"stage": "INDEX_BUILD", "runtime_evidence": "run_v310.py enters full_vector_knowledge_base once for A, B, and Combined.", "repeated_work": "Combined rebuilds a 11,048-chunk Chroma index after A+B already built 11,048 chunks.", "avoidable": "yes; Combined correctness aggregates saved A/B rows."},
        {"stage": "RETRIEVAL", "runtime_evidence": "_run_mode processes every frozen query for P1 and P2 in one uncheckpointed call.", "repeated_work": "an interruption loses every completed row in the current corpus/pipeline.", "avoidable": "yes; persist each completed query."},
        {"stage": "EVIDENCE", "runtime_evidence": "_evidence_report performs a fresh hybrid retrieval for every query.", "repeated_work": "separate from P1/P2 retrieval by frozen evidence-gate semantics.", "avoidable": "no semantic reuse; resumable per-query execution is possible."},
        {"stage": "SUPPORT", "runtime_evidence": "_frozen_support_report consumes P2/evidence rows after their full stages finish.", "repeated_work": "none when P2/evidence rows are retained.", "avoidable": "yes; resume directly from saved P2 and evidence rows."},
        {"stage": "MODEL_LOAD", "runtime_evidence": "one CrossEncoderReranker instance is passed to all old _run calls; rag_core embedding factory is lru_cached.", "repeated_work": "not repeated within the old process.", "avoidable": "already reused per process; no model behavior change."},
        {"stage": "INGESTION_BM25", "runtime_evidence": "the old main ingests A and B once; retrieval builds scoped lexical candidates per query.", "repeated_work": "per-query BM25 is normal retrieval work, not per-pipeline ingestion.", "avoidable": "not changed because it is retrieval behavior."},
    ]


def _store_for(corpus_id: str, manifest: dict[str, Any]) -> CheckpointStore:
    configuration = evaluation_configuration()
    now = utc_now()
    identity = EvaluationRun(
        run_id=f"v3101-corpus-{corpus_id}", evaluation_version=EVALUATION_VERSION,
        corpus_id=corpus_id.upper(), pipeline_id="P1_P2_CORRECTNESS",
        manifest_hash=_hash(manifest), annotation_hash=_annotation_hash(manifest),
        configuration_hash=_hash(configuration), started_at=now, updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def _latency_store_for(corpus_id: str, manifest: dict[str, Any]) -> CheckpointStore:
    configuration = {**evaluation_configuration(), "latency_subset_policy": "first_two_per_required_category"}
    now = utc_now()
    identity = EvaluationRun(
        run_id=f"v3101-latency-{corpus_id}", evaluation_version=EVALUATION_VERSION,
        corpus_id=corpus_id.upper(), pipeline_id="P1_P2_LATENCY",
        manifest_hash=_hash(manifest), annotation_hash=_annotation_hash(manifest),
        configuration_hash=_hash(configuration), started_at=now, updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def _records(stage: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in stage.get("rows", {}).values() if record.get("status") == "COMPLETED"]


def _results(stage: dict[str, Any]) -> list[dict[str, Any]]:
    return [record["result"] for record in _records(stage)]


def _validity(stage: dict[str, Any], total: int) -> str:
    return "VALID" if len(_records(stage)) == total and not stage.get("errors") else "PARTIAL"


def _print_progress(corpus: str, pipeline: str, stage: str, completed: int, total: int, query_id: str, started: float) -> None:
    print(
        f"[{EVALUATION_VERSION}][Corpus {corpus.upper()}][{pipeline}] {stage} {completed}/{total} "
        f"elapsed: {time.perf_counter() - started:.1f}s last query: {query_id} checkpoint: saved",
        flush=True,
    )


def _run_query_stage(
    store: CheckpointStore, stage_name: str, corpus: str, pipeline: str,
    queries: list[dict[str, Any]], execute,
) -> dict[str, Any]:
    store.begin_stage(stage_name, len(queries))
    stage = store.load_stage(stage_name) or {"stage": stage_name, "rows": {}, "errors": {}}
    started = time.perf_counter()
    for query in queries:
        if stage["rows"].get(query["query_id"], {}).get("status") == "COMPLETED":
            continue
        query_started = time.perf_counter()
        try:
            result = execute(query)
            record = {"status": "COMPLETED", "result": result, "latency_ms": (time.perf_counter() - query_started) * 1000}
            stage = store.save_query(stage_name, query["query_id"], record, total_queries=len(queries))
        except Exception as exc:  # Query failures are retained without losing completed work.
            error = {"type": type(exc).__name__, "message": str(exc)}
            record = {"status": "ERROR", "result": {}, "latency_ms": (time.perf_counter() - query_started) * 1000, "error": error}
            stage = store.save_query(stage_name, query["query_id"], record, total_queries=len(queries), error=error)
        _print_progress(corpus, pipeline, stage_name, len(stage["rows"]), len(queries), query["query_id"], started)
    return stage


def _retrieval_row(query: dict[str, Any], pipeline: dict[str, Any], reranker) -> dict[str, Any]:
    report = _run_mode(
        pipeline["mode"], [query], None, None, reranker, candidate_k=pipeline["candidate_k"],
        section_strategy=pipeline["section_strategy"] or "current", summarize=False,
    )
    return report["rows"][0]


def _run_corpus(corpus_id: str, *, resume: bool, restart: bool) -> dict[str, Any]:
    relative = CORPUS_PATHS[corpus_id]
    manifest_path = PRIVATE_ROOT / relative / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    queries = manifest["queries"]
    store = _store_for(corpus_id, manifest)
    store.initialize(resume=resume, restart=restart)
    store.begin_stage("STAGE_01_CORPUS_LOAD", len(queries))
    print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Load] started documents: {len(manifest['documents'])}", flush=True)
    documents, parser_audit = ingest_private_documents(manifest_path, manifest)
    store.save_stage("STAGE_01_CORPUS_LOAD", {
        "stage": "STAGE_01_CORPUS_LOAD", "documents": len(manifest["documents"]),
        "chunks": len(documents), "queries": len(queries), "parser_audit": parser_audit,
        "validity": "VALID",
    })
    print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Load] complete chunks: {len(documents)}", flush=True)
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))

    p2_stage = store.load_stage("STAGE_03_P2_RETRIEVAL")
    p1_stage = store.load_stage("STAGE_04_P1_RETRIEVAL")
    evidence_stage = store.load_stage("STAGE_06_EVIDENCE")
    needs_index = any(
        stage is None or any(record.get("status") != "COMPLETED" for record in stage.get("rows", {}).values())
        or len(stage.get("rows", {})) < len(queries)
        for stage in (p2_stage, p1_stage, evidence_stage)
    )
    context = full_vector_knowledge_base(documents) if needs_index else nullcontext()
    if needs_index:
        index_started = time.perf_counter()
        store.begin_stage("STAGE_02_INDEX_BUILD", len(queries))
        print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Index] build started chunks: {len(documents)}", flush=True)
    with context:
        if needs_index:
            store.save_stage("STAGE_02_INDEX_BUILD", {
                "stage": "STAGE_02_INDEX_BUILD", "chunks": len(documents), "ephemeral": True,
                "elapsed_seconds": time.perf_counter() - index_started, "validity": "VALID",
            })
            print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Index] build complete", flush=True)
        p2_stage = _run_query_stage(store, "STAGE_03_P2_RETRIEVAL", corpus_id, "P2", queries, lambda query: _retrieval_row(query, P2, reranker))
        p2_rows = _results(p2_stage)
        p2_report = _summary_metrics([query for query in queries if query["query_id"] in {row["query_id"] for row in p2_rows}], p2_rows) if p2_rows else {"rows": []}
        store.save_stage("STAGE_03_P2_RETRIEVAL", {**p2_stage, "report": p2_report, "validity": _validity(p2_stage, len(queries))})
        p1_stage = _run_query_stage(store, "STAGE_04_P1_RETRIEVAL", corpus_id, "P1", queries, lambda query: _retrieval_row(query, P1, reranker))
        p1_rows = _results(p1_stage)
        p1_report = _summary_metrics([query for query in queries if query["query_id"] in {row["query_id"] for row in p1_rows}], p1_rows) if p1_rows else {"rows": []}
        store.save_stage("STAGE_04_P1_RETRIEVAL", {**p1_stage, "report": p1_report, "validity": _validity(p1_stage, len(queries))})
        evidence_stage = _run_query_stage(
            store, "STAGE_06_EVIDENCE", corpus_id, "Evidence", queries,
            lambda query: _evidence_report([query], documents, summarize=False)["rows"][0],
        )
    evidence_rows = _results(evidence_stage)
    section = _section_metrics(queries, p1_report, p2_report) if len(p1_rows) == len(queries) == len(p2_rows) else {"validity": "PARTIAL"}
    store.save_stage("STAGE_05_SECTION_ANALYSIS", {"stage": "STAGE_05_SECTION_ANALYSIS", "section": section, "validity": "VALID" if "rows" in section else "PARTIAL"})
    evidence = _summarize_evidence(evidence_rows) if evidence_rows else {"rows": []}
    store.save_stage("STAGE_06_EVIDENCE", {**evidence_stage, "report": evidence, "validity": _validity(evidence_stage, len(queries))})

    def support_row(query: dict[str, Any]) -> dict[str, Any]:
        p2_row = next(row for row in p2_rows if row["query_id"] == query["query_id"])
        evidence_row = next(row for row in evidence_rows if row["query_id"] == query["query_id"])
        return _frozen_support_report([query], {"rows": [p2_row]}, {"rows": [evidence_row]}, documents)["rows"][0]

    support_stage = _run_query_stage(store, "STAGE_07_SUPPORT", corpus_id, "Support", queries, support_row)
    support_rows = _results(support_stage)
    if support_rows:
        supported = [row for row in support_rows if row["expected_supported"]]
        unsupported = [row for row in support_rows if not row["expected_supported"]]
        support = {
            "support_accuracy": sum(row["predicted_supported"] == row["expected_supported"] for row in support_rows) / len(support_rows),
            "supported_recall": sum(row["predicted_supported"] for row in supported) / len(supported) if supported else None,
            "unsupported_recall": sum(not row["predicted_supported"] for row in unsupported) / len(unsupported) if unsupported else None,
            "false_support_rate": sum(row["predicted_supported"] for row in unsupported) / len(unsupported) if unsupported else None,
            "false_insufficient_rate": sum(not row["predicted_supported"] for row in supported) / len(supported) if supported else None,
            "rows": support_rows,
        }
    else:
        support = {"rows": []}
    store.save_stage("STAGE_07_SUPPORT", {**support_stage, "report": support, "validity": _validity(support_stage, len(queries))})
    failure = {"p1": p1_report.get("failure_summary", {}), "p2": p2_report.get("failure_summary", {})}
    store.save_stage("STAGE_08_FAILURE_TAXONOMY", {"stage": "STAGE_08_FAILURE_TAXONOMY", "report": failure, "validity": "VALID" if len(p2_rows) == len(queries) else "PARTIAL"})
    validity = "VALID" if all(_validity(stage, len(queries)) == "VALID" for stage in (p1_stage, p2_stage, evidence_stage, support_stage)) else "PARTIAL"
    summary = {
        "evaluation_version": EVALUATION_VERSION, "corpus_id": corpus_id.upper(), "validity": validity,
        "corpus": {"documents": len(manifest["documents"]), "chunks": len(documents), "queries": len(queries)},
        "retrieval": {"p1": p1_report, "p2": p2_report}, "section": section,
        "evidence": evidence, "support": support, "failure": failure,
        "parser_audit": parser_audit, "timeout_root_cause_audit": timeout_root_cause_audit(),
    }
    store.save_stage("STAGE_10_FINAL_AGGREGATION", {"stage": "STAGE_10_FINAL_AGGREGATION", "summary": summary, "validity": validity})
    store.finalize("COMPLETED" if validity == "VALID" else "PARTIAL")
    atomic_write_json(store.root / "summary.json", summary)
    return summary


def _load_summary(corpus_id: str) -> dict[str, Any]:
    path = RUNTIME_ROOT / f"v3101-corpus-{corpus_id}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Corpus {corpus_id.upper()} has no completed correctness summary: {path}")
    return read_json(path)


def aggregate_combined(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Aggregate A/B rows only; never execute a third Combined retrieval run."""
    valid = a["validity"] == b["validity"] == "VALID"
    p1 = combined_metrics({"A": a["retrieval"]["p1"], "B": b["retrieval"]["p1"]})
    p2 = combined_metrics({"A": a["retrieval"]["p2"], "B": b["retrieval"]["p2"]})
    gaps = generalization_gaps(
        a["retrieval"]["p2"], b["retrieval"]["p2"], a["section"], b["section"],
        a["evidence"], b["evidence"], a["support"], b["support"],
    )
    matrix = cross_corpus_failure_matrix({
        "A": {"retrieval": a["retrieval"]["p2"], "evidence": a["evidence"], "support": a["support"]},
        "B": {"retrieval": b["retrieval"]["p2"], "evidence": b["evidence"], "support": b["support"]},
    })
    return {"evaluation_version": EVALUATION_VERSION, "validity": "VALID" if valid else "PARTIAL", "retrieval": {"p1": p1, "p2": p2}, "generalization_gaps": gaps, "failure_matrix": matrix}


def fixed_latency_subset(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a frozen, category-balanced 5--10 query subset before timing."""
    selected = [
        query for category in LATENCY_CATEGORIES
        for query in [item for item in queries if item["category"] == category][:2]
    ]
    if len(selected) < 5:
        raise ValueError("Latency subset requires at least five identifier/procedure/semantic/OOD frozen queries.")
    return selected[:10]


def _latency_row(query: dict[str, Any], pipeline: dict[str, Any], reranker) -> dict[str, Any]:
    """Measure only separable operations; raw retrieval contains its overlapping sub-stages."""
    with _section_mode(pipeline["id"] == "P2"):
        total_started = time.perf_counter()
        raw_started = time.perf_counter()
        raw = rag_core.retrieve_docs(
            query["query"], k=pipeline["candidate_k"], knowledge_base_id="v25-full-vector-benchmark",
            retrieval_mode="hybrid", trace_enabled=False,
            section_merge_strategy=pipeline["section_strategy"],
        )
        raw_ms = (time.perf_counter() - raw_started) * 1000
        filter_started = time.perf_counter()
        candidates = rag_core.filter_relevant_docs(raw)
        evidence_filter_ms = (time.perf_counter() - filter_started) * 1000
        rerank_started = time.perf_counter()
        reranker.rerank(query["query"], candidates, top_k=3)
        reranker_ms = (time.perf_counter() - rerank_started) * 1000
    return {
        "query_id": query["query_id"], "category": query["category"],
        "stages_ms": {
            "raw_retrieval_including_query_analysis_bm25_dense_rrf_scope_section": raw_ms,
            "evidence_filter": evidence_filter_ms, "reranker": reranker_ms,
            "total": (time.perf_counter() - total_started) * 1000,
        },
    }


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [row["stages_ms"]["total"] for row in rows]
    p95_index = max(0, int(len(totals) * .95 + .999999) - 1)
    stages = {key: statistics.median([row["stages_ms"][key] for row in rows]) for key in rows[0]["stages_ms"]} if rows else {}
    return {
        "sample_count": len(rows), "median_ms": statistics.median(totals) if totals else None,
        "p95_ms": sorted(totals)[p95_index] if totals else None, "stage_median_ms": stages,
        "stage_note": "raw_retrieval is intentionally not split because query analysis/BM25/Dense/RRF/scope/section overlap inside the frozen retrieval call.",
        "rows": rows,
    }


def _run_latency_corpus(corpus_id: str, *, resume: bool, restart: bool) -> dict[str, Any]:
    manifest_path = PRIVATE_ROOT / CORPUS_PATHS[corpus_id] / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    subset = fixed_latency_subset(manifest["queries"])
    store = _latency_store_for(corpus_id, manifest)
    store.initialize(resume=resume, restart=restart)
    store.begin_stage("STAGE_01_CORPUS_LOAD", len(subset))
    print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Latency Load] started", flush=True)
    documents, _ = ingest_private_documents(manifest_path, manifest)
    store.save_stage("STAGE_01_CORPUS_LOAD", {"stage": "STAGE_01_CORPUS_LOAD", "chunks": len(documents), "validity": "VALID"})
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))
    index_started = time.perf_counter()
    store.begin_stage("STAGE_02_INDEX_BUILD", len(subset))
    print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Latency Index] build started chunks: {len(documents)}", flush=True)
    with full_vector_knowledge_base(documents):
        store.save_stage("STAGE_02_INDEX_BUILD", {
            "stage": "STAGE_02_INDEX_BUILD", "chunks": len(documents), "ephemeral": True,
            "elapsed_seconds": time.perf_counter() - index_started, "validity": "VALID",
        })
        print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Latency Index] build complete", flush=True)
        # One warm-up per pipeline; it is explicitly excluded from query samples.
        for pipeline in (P2, P1):
            _latency_row(subset[0], pipeline, reranker)
            _run_query_stage(store, f"LATENCY_{pipeline['id']}", corpus_id, pipeline["id"], subset, lambda query, item=pipeline: _latency_row(query, item, reranker))
    reports = {}
    for pipeline in (P1, P2):
        stage = store.load_stage(f"LATENCY_{pipeline['id']}") or {"rows": {}}
        rows = _results(stage)
        reports[pipeline["id"].lower()] = {**_latency_summary(rows), "validity": _validity(stage, len(subset))}
        store.save_stage(f"LATENCY_{pipeline['id']}", {**stage, "report": reports[pipeline["id"].lower()], "validity": _validity(stage, len(subset))})
    validity = "VALID" if all(item["validity"] == "VALID" for item in reports.values()) else "PARTIAL"
    summary = {
        "evaluation_version": EVALUATION_VERSION, "corpus_id": corpus_id.upper(), "validity": validity,
        "policy": {"warmup_runs": 1, "measured_runs": len(subset), "tracing_enabled": False, "latency_subset_hash": _hash([item["query_id"] for item in subset])},
        "reports": reports,
    }
    atomic_write_json(store.root / "latency_results.json", summary)
    store.finalize("COMPLETED" if validity == "VALID" else "PARTIAL")
    return summary


def _run_failure_traces(corpus_id: str, *, resume: bool, restart: bool) -> dict[str, Any]:
    """Trace only frozen safety/semantic cases after correctness identifies them."""
    manifest_path = PRIVATE_ROOT / CORPUS_PATHS[corpus_id] / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    queries = [query for query in manifest["queries"] if query["category"] in {"safety", "semantic"}]
    store = _store_for(corpus_id, manifest)
    store.initialize(resume=resume, restart=restart)
    store.begin_stage("STAGE_01_CORPUS_LOAD", len(queries))
    documents, _ = ingest_private_documents(manifest_path, manifest)
    store.save_stage("STAGE_01_CORPUS_LOAD", {"stage": "STAGE_01_CORPUS_LOAD", "chunks": len(documents), "validity": "VALID"})
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))
    store.begin_stage("STAGE_09_FAILURE_TRACE", len(queries))
    print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Failure Trace] index build started cases: {len(queries)}", flush=True)
    with full_vector_knowledge_base(documents):
        print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}][Failure Trace] index build complete", flush=True)
        stage = _run_query_stage(
            store, "STAGE_09_FAILURE_TRACE", corpus_id, "P2 trace", queries,
            lambda query: _run_mode(
                P2["mode"], [query], None, None, reranker, candidate_k=P2["candidate_k"],
                section_strategy=P2["section_strategy"], trace_enabled=True, summarize=False,
            )["rows"][0],
        )
    rows = []
    queries_by_id = {query["query_id"]: query for query in queries}
    for row in _results(stage):
        query = queries_by_id[row["query_id"]]
        trace = overlay_relevance(row.get("trace", {}), query)
        rows.append({
            "query_id": query["query_id"], "query": query["query"], "category": query["category"],
            "answerable": query["answerable"], "expected_section": query.get("expected_section", ""),
            "retrieval_rank": row.get("rank"), "reranker_candidates": row.get("candidates", []),
            **query_trace_summary(query, trace),
        })
    report = {"corpus_id": corpus_id.upper(), "validity": _validity(stage, len(queries)), "rows": rows}
    store.save_stage("STAGE_09_FAILURE_TRACE", {**stage, "report": report, "validity": report["validity"]})
    atomic_write_json(store.root / "failure_trace_results.json", report)
    return report


def aggregate_combined_latency(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    reports = {}
    for pipeline in ("p1", "p2"):
        rows = [*a["reports"][pipeline]["rows"], *b["reports"][pipeline]["rows"]]
        reports[pipeline] = _latency_summary(rows) if rows else {"sample_count": 0, "median_ms": None, "p95_ms": None}
    return {
        "evaluation_version": EVALUATION_VERSION,
        "validity": "VALID" if a["validity"] == b["validity"] == "VALID" else "PARTIAL",
        "policy": {"source": "A/B query-level latency samples; no Combined retrieval rerun"}, "reports": reports,
    }


def _load_latency(corpus_id: str) -> dict[str, Any]:
    path = RUNTIME_ROOT / f"v3101-latency-{corpus_id}" / "latency_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Corpus {corpus_id.upper()} has no completed latency result: {path}")
    return read_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("a", "b", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--latency", action="store_true", help="Run the independent fixed-subset latency benchmark.")
    parser.add_argument("--failure-trace", action="store_true", help="Trace only frozen safety/semantic cases using P2.")
    args = parser.parse_args(argv)
    if args.resume and args.restart:
        parser.error("--resume and --restart cannot be used together")
    selected = ("b", "a") if args.corpus == "all" else (args.corpus,)
    for corpus_id in selected:
        summary = _run_failure_traces(corpus_id, resume=args.resume, restart=args.restart) if args.failure_trace else (
            _run_latency_corpus(corpus_id, resume=args.resume, restart=args.restart)
            if args.latency else _run_corpus(corpus_id, resume=args.resume, restart=args.restart)
        )
        kind = "failure-trace" if args.failure_trace else "latency" if args.latency else "correctness"
        print(f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}] {kind} {summary['validity']}", flush=True)
    if args.corpus == "all" and args.failure_trace:
        return 0
    if args.corpus == "all" and args.latency:
        combined = aggregate_combined_latency(_load_latency("a"), _load_latency("b"))
        atomic_write_json(RUNTIME_ROOT / "combined_latency.json", combined)
        print(f"[{EVALUATION_VERSION}][Combined] latency aggregation {combined['validity']}", flush=True)
    elif args.corpus == "all":
        combined = aggregate_combined(_load_summary("a"), _load_summary("b"))
        atomic_write_json(RUNTIME_ROOT / "combined_summary.json", combined)
        print(f"[{EVALUATION_VERSION}][Combined] aggregation {combined['validity']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
