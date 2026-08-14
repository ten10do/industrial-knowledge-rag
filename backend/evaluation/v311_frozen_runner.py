"""Resume-safe V3.11.2 Evidence/Support replay over frozen A/B retrieval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base  # noqa: E402
from backend.evaluation.private_benchmark import (  # noqa: E402
    _evidence_report,
    _frozen_support_report,
    _summarize_evidence,
    ingest_private_documents,
    load_private_manifest,
)
from backend.evaluation.resumable import (  # noqa: E402
    CheckpointCorruptionError,
    CheckpointStore,
    EvaluationRun,
    ResumeConfigurationMismatch,
    atomic_write_json,
    read_json,
    utc_now,
)
from backend.evaluation.v310_runner import evaluation_configuration  # noqa: E402
from backend.evaluation.v311_resume import (  # noqa: E402
    completed_results,
    hash_json,
    run_query_stage,
)
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION  # noqa: E402


PRIVATE_ROOT = Path("backend/evaluation/benchmark_private")
V310_RUNTIME_ROOT = PRIVATE_ROOT / "v310_runtime"
RUNTIME_ROOT = PRIVATE_ROOT / "v311_runtime"
CORPUS_PATHS = {"a": Path("."), "b": Path("corpus_b")}
EVALUATION_VERSION = "V3.11.2"


def _annotation_hash(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get("annotation_enrichment", {}).get("enriched_annotation_sha256")
        or manifest["freeze"]["annotation_sha256"]
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_retrieval_stage(
    stage: dict[str, Any], queries: list[dict[str, Any]], stage_name: str,
) -> list[dict[str, Any]]:
    query_ids = [query["query_id"] for query in queries]
    rows = stage.get("rows", {})
    if set(rows) != set(query_ids) or stage.get("validity") != "VALID" or stage.get("errors"):
        raise CheckpointCorruptionError(f"Invalid frozen retrieval artifact: {stage_name}")
    results = []
    for query_id in query_ids:
        record = rows[query_id]
        result = record.get("result")
        if (
            record.get("status") != "COMPLETED"
            or not isinstance(result, dict)
            or result.get("query_id") != query_id
        ):
            raise CheckpointCorruptionError(
                f"Invalid frozen retrieval row: {stage_name}/{query_id}"
            )
        results.append(result)
    return results


def _load_retrieval_artifact(corpus_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    source_root = V310_RUNTIME_ROOT / f"v3101-corpus-{corpus_id}"
    run_path = source_root / "run_manifest.json"
    p1_path = source_root / "stages" / "stage_04_p1_retrieval.json"
    p2_path = source_root / "stages" / "stage_03_p2_retrieval.json"
    summary_path = source_root / "summary.json"
    run = read_json(run_path)
    queries = manifest["queries"]
    expected = {
        "run_id": f"v3101-corpus-{corpus_id}",
        "evaluation_version": "V3.10.1",
        "corpus_id": corpus_id.upper(),
        "pipeline_id": "P1_P2_CORRECTNESS",
        "manifest_hash": hash_json(manifest),
        "annotation_hash": _annotation_hash(manifest),
        "configuration_hash": hash_json(evaluation_configuration()),
        "status": "COMPLETED",
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ResumeConfigurationMismatch("FROZEN_RETRIEVAL_ARTIFACT_IDENTITY_MISMATCH")
    p1_stage = read_json(p1_path)
    p2_stage = read_json(p2_path)
    p1_rows = _validate_retrieval_stage(p1_stage, queries, "P1")
    p2_rows = _validate_retrieval_stage(p2_stage, queries, "P2")
    summary = read_json(summary_path)
    if summary.get("validity") != "VALID":
        raise CheckpointCorruptionError("Frozen retrieval summary is not VALID")
    return {
        "source_run": run,
        "source_run_manifest_sha256": _file_hash(run_path),
        "p1_stage_sha256": _file_hash(p1_path),
        "p2_stage_sha256": _file_hash(p2_path),
        "p1_rows": p1_rows,
        "p2_rows": p2_rows,
        "retrieval": summary["retrieval"],
        "section": summary["section"],
    }


def _store_for(
    corpus_id: str,
    manifest: dict[str, Any],
    artifact: dict[str, Any],
) -> CheckpointStore:
    configuration = {
        "runner": "v311_frozen_replay_v1",
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "evidence": {"retrieval_mode": "hybrid", "k": 5},
        "support": {
            "candidate_source": artifact["source_run"]["run_id"],
            "p2_stage_sha256": artifact["p2_stage_sha256"],
        },
        "retrieval_configuration_hash": artifact["source_run"]["configuration_hash"],
    }
    now = utc_now()
    identity = EvaluationRun(
        run_id=f"v3112-frozen-{corpus_id}",
        evaluation_version=EVALUATION_VERSION,
        corpus_id=corpus_id.upper(),
        pipeline_id="FROZEN_EVIDENCE_SUPPORT_REPLAY",
        manifest_hash=hash_json(manifest),
        annotation_hash=_annotation_hash(manifest),
        configuration_hash=hash_json(configuration),
        started_at=now,
        updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def _support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    false_support = [row for row in unsupported if row["predicted_supported"]]
    false_insufficient = [row for row in supported if not row["predicted_supported"]]
    return {
        "support_accuracy": sum(
            row["predicted_supported"] == row["expected_supported"] for row in rows
        ) / len(rows),
        "supported_recall": sum(row["predicted_supported"] for row in supported) / len(supported),
        "unsupported_recall": sum(not row["predicted_supported"] for row in unsupported) / len(unsupported),
        "false_support_rate": len(false_support) / len(unsupported),
        "false_insufficient_rate": len(false_insufficient) / len(supported),
        "false_support_ids": [row["query_id"] for row in false_support],
        "false_insufficient_ids": [row["query_id"] for row in false_insufficient],
        "rows": rows,
    }


def _run_corpus(corpus_id: str, *, resume: bool) -> dict[str, Any]:
    manifest_path = PRIVATE_ROOT / CORPUS_PATHS[corpus_id] / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    queries = manifest["queries"]
    artifact = _load_retrieval_artifact(corpus_id, manifest)
    store = _store_for(corpus_id, manifest, artifact)
    store.initialize(resume=resume)
    evidence_stage = store.load_stage("EVIDENCE")
    support_stage = store.load_stage("SUPPORT")
    evidence_complete = len(completed_results(
        evidence_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
    )) == len(queries)
    support_complete = len(completed_results(
        support_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
    )) == len(queries)
    existing_summary_path = store.root / "summary.json"
    existing_summary = (
        read_json(existing_summary_path) if existing_summary_path.exists() else None
    )
    documents = []
    parser_audit = existing_summary.get("parser_audit", {}) if existing_summary else {}
    if not (evidence_complete and support_complete):
        store.begin_stage("CORPUS_LOAD", len(queries))
        documents, parser_audit = ingest_private_documents(manifest_path, manifest)
        store.save_stage("CORPUS_LOAD", {
            "stage": "CORPUS_LOAD",
            "documents": len(manifest["documents"]),
            "chunks": len(documents),
            "parser_audit": parser_audit,
            "validity": "VALID",
        })
    context = full_vector_knowledge_base(documents) if not evidence_complete else nullcontext()
    if not evidence_complete:
        store.begin_stage("INDEX_BUILD", len(queries))
    with context:
        if not evidence_complete:
            store.save_stage("INDEX_BUILD", {
                "stage": "INDEX_BUILD",
                "chunks": len(documents),
                "ephemeral": True,
                "validity": "VALID",
            })
        evidence_stage, evidence_stats = run_query_stage(
            store,
            "EVIDENCE",
            corpus_id,
            queries,
            EVIDENCE_SUPPORT_RULE_VERSION,
            lambda query: _evidence_report(
                [query], documents, summarize=False,
            )["rows"][0],
        )
    evidence_rows = completed_results(
        evidence_stage,
        queries,
        EVIDENCE_SUPPORT_RULE_VERSION,
        require_complete=True,
    )
    evidence_by_id = {row["query_id"]: row for row in evidence_rows}
    p2_by_id = {row["query_id"]: row for row in artifact["p2_rows"]}

    def support_row(query: dict[str, Any]) -> dict[str, Any]:
        query_id = query["query_id"]
        return _frozen_support_report(
            [query],
            {"rows": [p2_by_id[query_id]]},
            {"rows": [evidence_by_id[query_id]]},
            documents,
        )["rows"][0]

    support_stage, support_stats = run_query_stage(
        store,
        "SUPPORT",
        corpus_id,
        queries,
        EVIDENCE_SUPPORT_RULE_VERSION,
        support_row,
    )
    support_rows = completed_results(
        support_stage,
        queries,
        EVIDENCE_SUPPORT_RULE_VERSION,
        require_complete=True,
    )
    evidence = _summarize_evidence(evidence_rows)
    support = _support_summary(support_rows)
    store.save_stage("EVIDENCE", {
        **evidence_stage,
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "report": evidence,
        "validity": "VALID",
    })
    store.save_stage("SUPPORT", {
        **support_stage,
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "report": support,
        "validity": "VALID",
    })
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "corpus_id": corpus_id.upper(),
        "validity": "VALID",
        "corpus": (
            existing_summary["corpus"]
            if not documents and existing_summary
            else {
                "documents": len(manifest["documents"]),
                "chunks": len(documents),
                "queries": len(queries),
            }
        ),
        "retrieval": artifact["retrieval"],
        "section": artifact["section"],
        "evidence": evidence,
        "support": support,
        "parser_audit": parser_audit,
        "retrieval_artifact": {
            "source_run_id": artifact["source_run"]["run_id"],
            "source_run_manifest_sha256": artifact["source_run_manifest_sha256"],
            "p1_stage_sha256": artifact["p1_stage_sha256"],
            "p2_stage_sha256": artifact["p2_stage_sha256"],
            "p1_completed_queries": len(artifact["p1_rows"]),
            "p2_completed_queries": len(artifact["p2_rows"]),
        },
        "resume": {"evidence": evidence_stats, "support": support_stats},
    }
    store.save_stage("FINAL", {
        "stage": "FINAL", "summary": summary, "validity": "VALID",
    })
    store.finalize("COMPLETED")
    atomic_write_json(store.root / "summary.json", summary)
    return summary


def _aggregate_if_complete() -> dict[str, Any] | None:
    paths = {
        corpus: RUNTIME_ROOT / f"v3112-frozen-{corpus}" / "summary.json"
        for corpus in ("a", "b")
    }
    if not all(path.exists() for path in paths.values()):
        return None
    corpora = {corpus: read_json(path) for corpus, path in paths.items()}
    if any(
        item.get("validity") != "VALID"
        or item.get("rule_version") != EVIDENCE_SUPPORT_RULE_VERSION
        for item in corpora.values()
    ):
        raise CheckpointCorruptionError("Cannot aggregate invalid V3.11.2 frozen summaries")
    evidence_rows = [
        row for item in corpora.values() for row in item["evidence"]["rows"]
    ]
    support_rows = [
        row for item in corpora.values() for row in item["support"]["rows"]
    ]
    combined = {
        "evaluation_version": EVALUATION_VERSION,
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "validity": "VALID",
        "corpora": {key: str(path) for key, path in paths.items()},
        "evidence": _summarize_evidence(evidence_rows),
        "support": _support_summary(support_rows),
    }
    atomic_write_json(RUNTIME_ROOT / "v3112_combined_summary.json", combined)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("a", "b", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    selected = ("b", "a") if args.corpus == "all" else (args.corpus,)
    for corpus_id in selected:
        summary = _run_corpus(corpus_id, resume=args.resume)
        print(
            f"[{EVALUATION_VERSION}][Corpus {corpus_id.upper()}] frozen {summary['validity']}",
            flush=True,
        )
        for stage, stats in summary["resume"].items():
            print(
                f"[{stage}] completed before={stats['completed_before']} "
                f"skipped={stats['skipped']} executed={stats['executed']}",
                flush=True,
            )
    combined = _aggregate_if_complete()
    if combined:
        print(f"[{EVALUATION_VERSION}][Combined] frozen VALID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
