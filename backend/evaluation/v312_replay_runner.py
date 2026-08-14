"""V3.12 frozen retrieval artifact export, validation, replay, and comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document

from backend.evaluation.frozen_retrieval_artifact import (
    ARTIFACT_SCHEMA_VERSION, artifact_inspection, candidate_from_p2_row,
    deserialize_document, file_sha256, load_valid_artifact, make_retrieval_result,
    new_artifact_payload, serialize_candidate,
    serialize_query_analysis, validate_artifact, write_immutable_artifact,
)
from backend.evaluation.resumable import (
    CheckpointStore, EvaluationRun, atomic_write_json, read_json, utc_now,
)
from backend.evaluation.v311_resume import completed_results, hash_json, run_query_stage
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.evidence_support import skipped_support, validate_evidence_support
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
RUNTIME_ROOT = PRIVATE_ROOT / "v312_runtime"
ARTIFACT_ROOT = PRIVATE_ROOT / "v312_artifacts"
CORPUS_PATHS = {"a": Path("."), "b": Path("corpus_b")}
EVALUATION_VERSION = "V3.12"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def ensure_private_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError(f"Private evaluation output must stay under {PRIVATE_ROOT}")
    return resolved


def _safe_identity(value: str, field: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}; use 1-128 letters, digits, dots, dashes, or underscores")
    return value


def _annotation_hash(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get("annotation_enrichment", {}).get("enriched_annotation_sha256")
        or manifest["freeze"]["annotation_sha256"]
    )


def retrieval_configuration() -> dict[str, Any]:
    """Bind artifacts to every implementation/config surface that produced them."""
    from backend import rag_core
    from backend.evaluation.v310_runner import evaluation_configuration
    from backend.retrieval.section import load_section_config

    source_files = (
        "backend/retrieval/tokenizer.py", "backend/retrieval/bm25.py",
        "backend/retrieval/fusion.py", "backend/retrieval/filters.py",
        "backend/retrieval/product_identity.py", "backend/retrieval/scope.py",
        "backend/retrieval/section.py", "backend/retrieval/reranker.py",
    )
    section_config, section_fallback = load_section_config()
    effective = {
        "lexical_top_k": int(os.getenv("LEXICAL_TOP_K", str(rag_core.DEFAULT_LEXICAL_TOP_K))),
        "vector_top_k": int(os.getenv("VECTOR_TOP_K", str(rag_core.DEFAULT_VECTOR_TOP_K))),
        "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", str(rag_core.DEFAULT_HYBRID_TOP_K))),
        "rrf_k": int(os.getenv("RRF_K", str(rag_core.DEFAULT_RRF_K))),
        "max_relevant_distance": rag_core.get_relevance_threshold(),
    }
    return {
        "source_evaluation_configuration": evaluation_configuration(),
        "effective_retrieval": effective,
        "tokenization": {"implementation": "backend.retrieval.tokenizer.tokenize"},
        "product_identity": {"implementation": "backend.retrieval.product_identity"},
        "section": {**asdict(section_config), "fallback_reason": section_fallback},
        "source_code_sha256": {
            name: file_sha256(PROJECT_ROOT / name) for name in source_files
        },
        "evidence_input": {"mode": "hybrid", "k": 5},
        "final_context": {"pipeline": "P2", "top_k": 3},
    }


def _snapshot_documents(documents: list[Document]) -> list[dict[str, Any]]:
    """Project only the corpus-global terms and identities read by the gates."""
    from backend.retrieval.evidence import EVIDENCE_IDENTIFIER_PATTERN
    from backend.retrieval.technical import _PARAMETER_IDENTIFIER_PATTERN

    metadata_fields = (
        "manufacturer", "equipment_model", "product_family", "product_series",
        "model_aliases", "aliases", "equipment_type", "document_type",
        "knowledge_type", "error_code",
    )
    grouped: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = {
            field: document.metadata[field]
            for field in metadata_fields if field in document.metadata
        }
        key = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
        entry = grouped.setdefault(key, {"metadata": metadata, "terms": set()})
        content = str(document.page_content or "")
        entry["terms"].update(match.group(0) for match in EVIDENCE_IDENTIFIER_PATTERN.finditer(content))
        entry["terms"].update(match.group("identifier") for match in _PARAMETER_IDENTIFIER_PATTERN.finditer(content))
    return [
        {"content": " ".join(sorted(entry["terms"], key=str.casefold)), "metadata": entry["metadata"]}
        for _, entry in sorted(grouped.items())
    ]


def _export_store(
    artifact_id: str, corpus_id: str, manifest: dict[str, Any], config: dict[str, Any],
) -> CheckpointStore:
    now = utc_now()
    identity = EvaluationRun(
        run_id=f"export-{artifact_id}", evaluation_version=EVALUATION_VERSION,
        corpus_id=corpus_id.upper(), pipeline_id="FROZEN_RETRIEVAL_EXPORT",
        manifest_hash=hash_json(manifest), annotation_hash=_annotation_hash(manifest),
        configuration_hash=hash_json(config), started_at=now, updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def export_artifact(
    corpus_id: str, output_path: Path, artifact_id: str, *, resume: bool = False,
) -> dict[str, Any]:
    """One-time export. P2 is reused; only its missing Evidence input is retrieved."""
    from backend import rag_core
    from backend.evaluation.full_vector_benchmark import (
        FULL_BENCHMARK_KNOWLEDGE_BASE_ID, full_vector_knowledge_base,
    )
    from backend.evaluation.private_benchmark import ingest_private_documents, load_private_manifest
    from backend.evaluation.v311_frozen_runner import _load_retrieval_artifact

    artifact_id = _safe_identity(artifact_id, "artifact id")
    output_path = ensure_private_path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {output_path}")
    manifest_path = PRIVATE_ROOT / CORPUS_PATHS[corpus_id] / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    queries = manifest["queries"]
    frozen = _load_retrieval_artifact(corpus_id, manifest)
    documents, parser_audit = ingest_private_documents(manifest_path, manifest)
    documents_by_chunk = {str(item.metadata.get("chunk_id", "")): item for item in documents}
    config = retrieval_configuration()
    store = _export_store(artifact_id, corpus_id, manifest, config)
    store.initialize(resume=resume)

    existing = store.load_stage("CAPTURE_EVIDENCE_INPUT")
    complete = len(completed_results(existing, queries, ARTIFACT_SCHEMA_VERSION)) == len(queries)
    if not complete:
        with full_vector_knowledge_base(documents):
            stage, stats = run_query_stage(
                store, "CAPTURE_EVIDENCE_INPUT", corpus_id, queries,
                ARTIFACT_SCHEMA_VERSION,
                lambda query: _capture_evidence_input(
                    query, rag_core.retrieve_docs(
                        query["query"], k=5,
                        knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
                        retrieval_mode="hybrid",
                    ),
                ),
            )
    else:
        stage, stats = existing, {"completed_before": len(queries), "skipped": len(queries), "executed": 0}
    captured = completed_results(stage, queries, ARTIFACT_SCHEMA_VERSION, require_complete=True)
    captured_by_id = {row["query_id"]: row for row in captured}
    p2_by_id = {row["query_id"]: row for row in frozen["p2_rows"]}

    artifact_queries = []
    for query in queries:
        query_id = query["query_id"]
        final_context = []
        for candidate in p2_by_id[query_id].get("candidates", []):
            chunk_id = str(candidate["chunk_id"])
            if chunk_id not in documents_by_chunk:
                raise KeyError(f"P2 chunk missing from frozen corpus: {query_id}/{chunk_id}")
            final_context.append(candidate_from_p2_row(candidate, documents_by_chunk[chunk_id]))
        artifact_queries.append({
            "query_id": query_id,
            "query": query["query"],
            "query_text_hash": hash_json(query["query"]),
            "ground_truth": query,
            "evidence_input": captured_by_id[query_id]["evidence_input"],
            "final_context": final_context,
            "retrieval_decision_inputs": {
                "retrieval_mode": "hybrid",
                "source_p2_retrieval_scope": p2_by_id[query_id].get("retrieval_scope", {}),
                "source_p2_section_retrieval": p2_by_id[query_id].get("section_retrieval", {}),
            },
        })

    payload = new_artifact_payload(
        artifact_id=artifact_id, corpus_id=corpus_id.upper(),
        manifest_hash=hash_json(manifest), annotation_hash=_annotation_hash(manifest),
        retrieval_config=config, queries=artifact_queries,
        snapshot_documents=_snapshot_documents(documents),
        source={
            "retrieval_run_id": frozen["source_run"]["run_id"],
            "retrieval_run_manifest_sha256": frozen["source_run_manifest_sha256"],
            "p2_stage_sha256": frozen["p2_stage_sha256"],
            "p2_reused_without_retrieval": True,
            "evidence_input_retrieved": True,
            "parser_audit_hash": hash_json(parser_audit),
            "resume": stats,
        },
        rule_version=EVIDENCE_SUPPORT_RULE_VERSION,
    )
    write_immutable_artifact(output_path, payload)
    report = validate_artifact(read_json(output_path))
    if report["validity"] != "VALID":
        raise RuntimeError(f"Exported artifact failed validation: {report}")
    store.finalize("COMPLETED")
    return {**report, "path": str(output_path), "bytes": output_path.stat().st_size}


def _capture_evidence_input(query: dict[str, Any], result: Any) -> dict[str, Any]:
    analysis = getattr(result, "query_analysis", None)
    if analysis is None:
        raise ValueError(f"Missing query analysis: {query['query_id']}")
    return {
        "query_id": query["query_id"],
        "evidence_input": {
            "retrieval_mode": getattr(result, "retrieval_mode", "hybrid") or "hybrid",
            "query_analysis": serialize_query_analysis(analysis),
            "candidate_pool": [serialize_candidate(item) for item in result.candidates],
        },
    }


def _expected_supported(query: dict[str, Any]) -> bool:
    if query.get("support_gate_truth") in {"SUPPORTED", "INSUFFICIENT"}:
        return query["support_gate_truth"] == "SUPPORTED"
    return bool(query.get("supported", query.get("answerable", False)))


def replay_query(row: dict[str, Any], snapshot: list[Document]) -> dict[str, Any]:
    evidence_input = row["evidence_input"]
    evidence_result = make_retrieval_result(
        evidence_input["candidate_pool"], evidence_input["query_analysis"], snapshot,
        evidence_input.get("retrieval_mode", "hybrid"),
    )
    started = time.perf_counter()
    evidence = analyze_retrieval_evidence(
        row["query"], evidence_result, snapshot,
        evidence_input.get("retrieval_mode", "hybrid"),
    )
    evidence_ms = (time.perf_counter() - started) * 1000
    if evidence.decision == "ANSWER":
        support_result = make_retrieval_result(
            row["final_context"], evidence_input["query_analysis"], snapshot,
        )
        started = time.perf_counter()
        support = validate_evidence_support(row["query"], support_result, snapshot)
        support_ms = (time.perf_counter() - started) * 1000
    else:
        support = skipped_support()
        support_ms = 0.0
    truth = row["ground_truth"]
    expected_supported = _expected_supported(truth)
    final_decision = "ABSTAIN" if evidence.decision == "ABSTAIN" or support.status == "INSUFFICIENT" else "ANSWER"
    return {
        "query_id": row["query_id"], "query": row["query"],
        "answerable": bool(truth["answerable"]),
        "expected_supported": expected_supported,
        "evidence": evidence.as_dict(), "support": support.as_dict(),
        "base_decision": evidence.decision, "base_reason": evidence.reason,
        "final_decision": final_decision,
        "predicted_supported": final_decision == "ANSWER",
        "latency_ms": {"evidence": evidence_ms, "support": support_ms, "total": evidence_ms + support_ms},
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    ood = [row for row in rows if not row["answerable"]]
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    false_answers = [row["query_id"] for row in ood if row["base_decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in answerable if row["base_decision"] == "ABSTAIN"]
    false_support = [row["query_id"] for row in unsupported if row["predicted_supported"]]
    false_insufficient = [row["query_id"] for row in supported if not row["predicted_supported"]]
    return {
        "evidence": {
            "decision_accuracy": sum((row["base_decision"] == "ANSWER") == row["answerable"] for row in rows) / len(rows),
            "ood_recall": sum(row["base_decision"] == "ABSTAIN" for row in ood) / len(ood) if ood else None,
            "answerable_recall": sum(row["base_decision"] == "ANSWER" for row in answerable) / len(answerable) if answerable else None,
            "false_answer_rate": len(false_answers) / len(ood) if ood else None,
            "false_refusal_rate": len(false_refusals) / len(answerable) if answerable else None,
            "false_answer_ids": false_answers, "false_refusal_ids": false_refusals,
        },
        "support": {
            "support_accuracy": sum(row["predicted_supported"] == row["expected_supported"] for row in rows) / len(rows),
            "supported_recall": sum(row["predicted_supported"] for row in supported) / len(supported) if supported else None,
            "unsupported_recall": sum(not row["predicted_supported"] for row in unsupported) / len(unsupported) if unsupported else None,
            "false_support_rate": len(false_support) / len(unsupported) if unsupported else None,
            "false_insufficient_rate": len(false_insufficient) / len(supported) if supported else None,
            "false_support_ids": false_support, "false_insufficient_ids": false_insufficient,
        },
        "latency_ms": {
            "median": statistics.median(row["latency_ms"]["total"] for row in rows),
            "total": sum(row["latency_ms"]["total"] for row in rows),
        },
    }


def _failure_mapping(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = []
    for row in rows:
        failure_types = []
        if not row["answerable"] and row["base_decision"] == "ANSWER":
            failure_types.append("FALSE_ANSWER")
        if row["answerable"] and row["base_decision"] == "ABSTAIN":
            failure_types.append("FALSE_REFUSAL")
        if not row["expected_supported"] and row["predicted_supported"]:
            failure_types.append("FALSE_SUPPORT")
        if row["expected_supported"] and not row["predicted_supported"]:
            failure_types.append("FALSE_INSUFFICIENT")
        if failure_types:
            mapping.append({
                "query_id": row["query_id"], "failure_types": failure_types,
                "candidate_ids": row.get("candidate_ids", {}),
                "artifact_id": row.get("artifact_id", ""),
                "artifact_hash": row.get("artifact_hash", ""),
                "evidence_rule_version": row.get("evidence_rule_version", ""),
                "support_rule_version": row.get("support_rule_version", ""),
            })
    return mapping


def _replay_store(artifact: dict[str, Any], run_id: str) -> CheckpointStore:
    now = utc_now()
    identity = EvaluationRun(
        run_id=run_id, evaluation_version=EVALUATION_VERSION,
        corpus_id=artifact["corpus_id"], pipeline_id="EVIDENCE_SUPPORT_ARTIFACT_REPLAY",
        manifest_hash=artifact["corpus_manifest_hash"], annotation_hash=artifact["annotation_hash"],
        configuration_hash=hash_json({
            "artifact_hash": artifact["artifact_hash"],
            "artifact_schema_version": artifact["schema_version"],
            "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        }), started_at=now, updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def replay_artifact(
    path: Path, run_id: str, *, resume: bool = False,
    expected_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = _safe_identity(run_id, "run id")
    path = ensure_private_path(path)
    expected = {}
    if expected_manifest is not None:
        expected = {
            "expected_manifest_hash": hash_json(expected_manifest),
            "expected_annotation_hash": _annotation_hash(expected_manifest),
        }
    artifact = load_valid_artifact(path, **expected)
    snapshot = [deserialize_document(item) for item in artifact["corpus_snapshot"]["documents"]]
    queries = artifact["queries"]
    store = _replay_store(artifact, run_id)
    store.initialize(resume=resume)
    started = time.perf_counter()
    stage, resume_stats = run_query_stage(
        store, "REPLAY", artifact["corpus_id"], queries,
        EVIDENCE_SUPPORT_RULE_VERSION,
        lambda row: {
            **replay_query(row, snapshot),
            "artifact_id": artifact["artifact_id"],
            "artifact_hash": artifact["artifact_hash"],
            "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
            "support_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
            "ground_truth": row["ground_truth"],
            "candidate_ids": {
                "evidence": [item["chunk_id"] for item in row["evidence_input"]["candidate_pool"]],
                "final_context": [item["chunk_id"] for item in row["final_context"]],
            },
        },
    )
    rows = completed_results(stage, queries, EVIDENCE_SUPPORT_RULE_VERSION, require_complete=True)
    result = {
        "evaluation_version": EVALUATION_VERSION,
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "artifact_schema_version": artifact["schema_version"],
        "artifact_id": artifact["artifact_id"], "artifact_hash": artifact["artifact_hash"],
        "retrieval_artifact_id": artifact["artifact_id"],
        "retrieval_artifact_hash": artifact["artifact_hash"],
        "evaluation_annotation_hash": artifact["annotation_hash"],
        "corpus_id": artifact["corpus_id"], "validity": "VALID",
        "query_count": len(rows), "metrics": _metrics(rows), "rows": rows,
        "failure_mapping": _failure_mapping(rows),
        "replay_elapsed_seconds": time.perf_counter() - started,
        "artifact_bytes": path.stat().st_size, "resume": resume_stats,
    }
    output = store.root / "summary.json"
    atomic_write_json(output, result)
    store.finalize("COMPLETED")
    return {**result, "output_path": str(output)}


def _row_view(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("rows") or result.get("evidence", {}).get("rows", [])
    support_rows = {row["query_id"]: row for row in result.get("support", {}).get("rows", [])}
    view = {}
    for row in rows:
        query_id = row["query_id"]
        support = support_rows.get(query_id, row)
        view[query_id] = {
            "base_decision": row.get("base_decision", row.get("decision", row.get("evidence", {}).get("decision"))),
            "base_reason": row.get("base_reason", row.get("reason", row.get("evidence", {}).get("reason"))),
            "support_status": (support.get("support") or {}).get("status"),
            "support_reason": (support.get("support") or {}).get("reason"),
            "final_decision": support.get("final_decision", row.get("final_decision")),
            "predicted_supported": support.get("predicted_supported", row.get("predicted_supported")),
            "answerable": row.get("answerable"),
            "expected_supported": support.get("expected_supported", row.get("expected_supported")),
        }
    return view


def _metric_view(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    evidence = metrics.get("evidence", result.get("evidence", {}))
    support = metrics.get("support", result.get("support", {}))
    evidence_names = (
        "decision_accuracy", "ood_recall", "answerable_recall",
        "false_answer_rate", "false_refusal_rate",
    )
    support_names = (
        "support_accuracy", "supported_recall", "unsupported_recall",
        "false_support_rate", "false_insufficient_rate",
    )
    return {
        **{f"evidence.{name}": evidence.get(name) for name in evidence_names},
        **{f"support.{name}": support.get(name) for name in support_names},
    }


def compare_results(baseline: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    before, after = _row_view(baseline), _row_view(replay)
    missing = sorted(set(before) - set(after))
    introduced = []
    fixed = []
    changed = []
    fields = (
        "base_decision", "base_reason", "support_status", "support_reason",
        "final_decision", "predicted_supported",
    )
    for query_id in sorted(set(before) & set(after)):
        delta = {field: [before[query_id].get(field), after[query_id].get(field)] for field in fields if before[query_id].get(field) != after[query_id].get(field)}
        if delta:
            changed.append({"query_id": query_id, "changes": delta})
        before_bad = before[query_id].get("final_decision") != ("ANSWER" if before[query_id].get("expected_supported") else "ABSTAIN")
        after_bad = after[query_id].get("final_decision") != ("ANSWER" if after[query_id].get("expected_supported") else "ABSTAIN")
        if before_bad and not after_bad:
            fixed.append(query_id)
        elif not before_bad and after_bad:
            introduced.append(query_id)
    failure_changes = {}
    predicates = {
        "false_answer": lambda row: not row.get("answerable") and row.get("base_decision") == "ANSWER",
        "false_refusal": lambda row: bool(row.get("answerable")) and row.get("base_decision") == "ABSTAIN",
        "false_support": lambda row: not row.get("expected_supported") and bool(row.get("predicted_supported")),
        "false_insufficient": lambda row: bool(row.get("expected_supported")) and not row.get("predicted_supported"),
    }
    for name, predicate in predicates.items():
        before_ids = {query_id for query_id, row in before.items() if predicate(row)}
        after_ids = {query_id for query_id, row in after.items() if predicate(row)}
        failure_changes[f"{name}_fixed"] = sorted(before_ids - after_ids)
        failure_changes[f"{name}_introduced"] = sorted(after_ids - before_ids)
    before_metrics, after_metrics = _metric_view(baseline), _metric_view(replay)
    metric_changes = {
        name: [before_metrics[name], after_metrics[name]]
        for name in before_metrics if before_metrics[name] != after_metrics[name]
    }
    exact = not missing and not changed and not metric_changes
    return {
        "validity": "VALID" if exact else "INVALID",
        "exact_equivalence": exact,
        "metric_equivalence": not metric_changes,
        "metric_changes": metric_changes,
        "missing_query_ids": missing,
        "changed_rows": changed,
        "fixed_failures": fixed,
        "introduced_failures": introduced,
        **failure_changes,
        "baseline_count": len(before), "replay_count": len(after),
    }


def combine_replay_results(
    corpus_a: dict[str, Any], corpus_b: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate saved A/B replay rows; no Combined retrieval artifact is built."""
    if corpus_a.get("validity") != "VALID" or corpus_b.get("validity") != "VALID":
        raise ValueError("Combined replay requires two VALID corpus results")
    rows = list(corpus_a["rows"]) + list(corpus_b["rows"])
    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Combined replay contains duplicate query ids")
    return {
        "evaluation_version": EVALUATION_VERSION,
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "corpus_id": "COMBINED", "validity": "VALID",
        "source_artifact_ids": [corpus_a["artifact_id"], corpus_b["artifact_id"]],
        "query_count": len(rows), "metrics": _metrics(rows),
        "failure_mapping": _failure_mapping(rows), "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--corpus", choices=("a", "b"), required=True)
    export.add_argument("--artifact-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--resume", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--artifact", type=Path, required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--artifact", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--resume", action="store_true")
    replay.add_argument("--manifest", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--replay", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "export":
        result = export_artifact(args.corpus, args.output, args.artifact_id, resume=args.resume)
    elif args.command == "validate":
        path = ensure_private_path(args.artifact)
        result = validate_artifact(read_json(path))
    elif args.command == "inspect":
        path = ensure_private_path(args.artifact)
        result = artifact_inspection(read_json(path), path)
    elif args.command == "replay":
        manifest = read_json(args.manifest) if args.manifest else None
        result = replay_artifact(
            args.artifact, args.run_id, resume=args.resume,
            expected_manifest=manifest,
        )
    else:
        result = compare_results(read_json(args.baseline), read_json(args.replay))
        if args.output:
            atomic_write_json(ensure_private_path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("validity") == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
