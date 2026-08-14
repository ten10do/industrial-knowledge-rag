"""V3.11 cross-corpus Evidence/Support calibration runner.

Runs the frozen evidence and support gates over the independent V3.11
calibration set, grouped by corpus (A=Rockwell, B=ABB+Omron), and reports
Evidence/Support metrics per manufacturer and combined. It reuses the frozen
retrieval/evidence/support functions unchanged; this is evaluation framework
only, never a change to the RAG algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document  # noqa: E402

from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base  # noqa: E402
from backend.evaluation.private_benchmark import (  # noqa: E402
    _evidence_report, _support_candidate_rows, _support_report,
    ingest_private_documents, load_private_manifest,
)
from backend.evaluation.resumable import (  # noqa: E402
    CheckpointStore, EvaluationRun, atomic_write_json, utc_now,
)
from backend.evaluation.v311_resume import (  # noqa: E402
    completed_results, hash_json, run_query_stage,
)
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig  # noqa: E402
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION  # noqa: E402

PRIVATE_ROOT = Path("backend/evaluation/benchmark_private")
CORPUS_PATHS = {"a": Path("."), "b": Path("corpus_b")}
CALIBRATION_PATH = PRIVATE_ROOT / "annotations" / "v311_calibration.json"
RUNTIME_ROOT = PRIVATE_ROOT / "v311_runtime"
EVALUATION_VERSION = "V3.11.2"


def calibration_hash(calibration: dict) -> str:
    payload = json.dumps(calibration["queries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_text_hash(calibration: dict) -> str:
    payload = [
        {"query_id": query["query_id"], "corpus": query["corpus"], "query": query["query"]}
        for query in calibration["queries"]
    ]
    frozen = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(frozen.encode("utf-8")).hexdigest()


def _manufacturer_of(model: str, documents: list[Document]) -> str:
    for doc in documents:
        if doc.metadata.get("equipment_model", "") == model or model in (doc.metadata.get("model_aliases", "") or "").split("|"):
            return doc.metadata.get("manufacturer", "")
    # Fallback mapping for the known corpus.
    mapping = {
        "CompactLogix 5370": "Rockwell Automation", "CompactLogix 5380": "Rockwell Automation",
        "PowerFlex 520-series (523/525)": "Rockwell Automation", "PowerFlex 527": "Rockwell Automation",
        "ACS580": "ABB", "FPNO-21": "ABB", "FPNO-99": "ABB",
        "CX-Programmer": "Omron",
    }
    return mapping.get(model, model or "unknown")


def _evidence_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    ood = [r for r in rows if not r["answerable"]]
    false_answer = [r for r in ood if r["decision"] == "ANSWER"]
    false_refusal = [r for r in answerable if r["decision"] == "ABSTAIN"]
    return {
        "count": len(rows),
        "decision_accuracy": round(sum((r["decision"] == "ANSWER") == r["answerable"] for r in rows) / len(rows), 4) if rows else None,
        "ood_recall": round(sum(r["decision"] == "ABSTAIN" for r in ood) / len(ood), 4) if ood else None,
        "answerable_recall": round(sum(r["decision"] == "ANSWER" for r in answerable) / len(answerable), 4) if answerable else None,
        "false_answer": round(len(false_answer) / len(ood), 4) if ood else None,
        "false_refusal": round(len(false_refusal) / len(answerable), 4) if answerable else None,
        "false_answer_ids": [r["query_id"] for r in false_answer],
        "false_refusal_ids": [r["query_id"] for r in false_refusal],
    }


def _support_metrics(rows: list[dict]) -> dict:
    supported = [r for r in rows if r["expected_supported"]]
    unsupported = [r for r in rows if not r["expected_supported"]]
    false_support = [r for r in unsupported if r["predicted_supported"]]
    false_insufficient = [r for r in supported if not r["predicted_supported"]]
    return {
        "count": len(rows),
        "support_accuracy": round(sum(r["predicted_supported"] == r["expected_supported"] for r in rows) / len(rows), 4) if rows else None,
        "supported_recall": round(sum(r["predicted_supported"] for r in supported) / len(supported), 4) if supported else None,
        "unsupported_recall": round(sum(not r["predicted_supported"] for r in unsupported) / len(unsupported), 4) if unsupported else None,
        "false_support": round(len(false_support) / len(unsupported), 4) if unsupported else None,
        "false_insufficient": round(len(false_insufficient) / len(supported), 4) if supported else None,
        "false_support_ids": [r["query_id"] for r in false_support],
        "false_insufficient_ids": [r["query_id"] for r in false_insufficient],
    }


def _grouped_metrics(rows: list[dict], metric_fn) -> dict:
    by_manufacturer = {}
    for row in rows:
        by_manufacturer.setdefault(row["manufacturer"], []).append(row)
    out = {}
    for manufacturer, group in by_manufacturer.items():
        out[manufacturer] = metric_fn(group)
    return out


def _store_for(corpus_id: str, calibration: dict, manifest: dict) -> CheckpointStore:
    frozen_hash = calibration_hash(calibration)
    configuration = {
        "runner": "v311_calibration_runner_resume_v1",
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "retrieval_mode": "hybrid",
        "retrieval_k": 5,
        "reranker": {"candidate_k": 7, "top_k": 3, "device": "cpu"},
    }
    now = utc_now()
    identity = EvaluationRun(
        run_id=f"v3112-calibration-{frozen_hash[:12]}-{corpus_id}",
        evaluation_version=EVALUATION_VERSION,
        corpus_id=corpus_id.upper(),
        pipeline_id="EVIDENCE_SUPPORT_CALIBRATION",
        manifest_hash=hash_json(manifest),
        annotation_hash=frozen_hash,
        configuration_hash=hash_json(configuration),
        started_at=now,
        updated_at=now,
    )
    return CheckpointStore(RUNTIME_ROOT, identity)


def _run_corpus(
    corpus_id: str,
    queries: list[dict],
    calibration: dict,
    *,
    resume: bool,
) -> tuple[dict, dict]:
    manifest_path = PRIVATE_ROOT / CORPUS_PATHS[corpus_id] / "manifest.json"
    manifest = load_private_manifest(manifest_path)
    store = _store_for(corpus_id, calibration, manifest)
    store.initialize(resume=resume)
    evidence_stage = store.load_stage("EVIDENCE")
    support_stage = store.load_stage("SUPPORT")
    evidence_complete = len(completed_results(
        evidence_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
    )) == len(queries)
    support_complete = len(completed_results(
        support_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
    )) == len(queries)
    documents: list[Document] = []
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
    reranker = CrossEncoderReranker(
        RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu")
    ) if not support_complete else None
    context = full_vector_knowledge_base(documents) if documents else nullcontext()
    if documents:
        store.begin_stage("INDEX_BUILD", len(queries))
    with context:
        if documents:
            store.save_stage("INDEX_BUILD", {
                "stage": "INDEX_BUILD",
                "chunks": len(documents),
                "ephemeral": True,
                "validity": "VALID",
            })
        def evidence_row(query: dict) -> dict:
            row = _evidence_report([query], documents, summarize=False)["rows"][0]
            expected_model = queries_by_id[corpus_id][row["query_id"]]["expected_model"]
            return {
                **row,
                "expected_model": expected_model,
                "manufacturer": _manufacturer_of(expected_model, documents),
            }

        evidence_stage, evidence_stats = run_query_stage(
            store, "EVIDENCE", corpus_id, queries,
            EVIDENCE_SUPPORT_RULE_VERSION, evidence_row,
        )
        evidence_rows = completed_results(
            evidence_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
            require_complete=True,
        )

        def support_row(query: dict) -> dict:
            candidate_rows, base_rows = _support_candidate_rows(
                [query], documents, reranker,
            )
            row = _support_report(
                [query], candidate_rows, base_rows, documents,
            )["rows"][0]
            return {
                **row,
                "manufacturer": _manufacturer_of(queries_by_id[corpus_id][row["query_id"]]["expected_model"], documents),
            }

        support_stage, support_stats = run_query_stage(
            store, "SUPPORT", corpus_id, queries,
            EVIDENCE_SUPPORT_RULE_VERSION, support_row,
        )
    support_rows = completed_results(
        support_stage, queries, EVIDENCE_SUPPORT_RULE_VERSION,
        require_complete=True,
    )
    store.save_stage("EVIDENCE", {
        **evidence_stage,
        "validity": "VALID",
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
    })
    store.save_stage("SUPPORT", {
        **support_stage,
        "validity": "VALID",
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
    })
    store.finalize("COMPLETED")
    payload = {
        "corpus": corpus_id.upper(),
        "evidence": _evidence_metrics(evidence_rows),
        "support": _support_metrics(support_rows),
        "evidence_by_manufacturer": _grouped_metrics(evidence_rows, _evidence_metrics),
        "support_by_manufacturer": _grouped_metrics(support_rows, _support_metrics),
        "evidence_rows": evidence_rows,
        "support_rows": support_rows,
    }
    return payload, {"evidence": evidence_stats, "support": support_stats}


queries_by_id: dict[str, dict] = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("a", "b", "all"), default="all")
    parser.add_argument("--calibration", default=str(CALIBRATION_PATH))
    parser.add_argument("--output", default=str(PRIVATE_ROOT / "annotations" / "v311_calibration_result.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-result",
        help="Reuse already completed corpora from a matching v311.2 result JSON.",
    )
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    queries = calibration["queries"]
    queries_by_id.clear()
    for query in queries:
        queries_by_id.setdefault(query["corpus"], {})[query["query_id"]] = query

    result = {
        "name": calibration["name"],
        "rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "calibration_hash": calibration_hash(calibration),
        "query_text_hash": query_text_hash(calibration),
        "query_count": len(queries),
        "answerable": sum(q["answerable"] for q in queries),
        "ood": sum(not q["answerable"] for q in queries),
        "corpora": {},
        "resume": {},
    }
    if args.reuse_result:
        reused = json.loads(Path(args.reuse_result).read_text(encoding="utf-8"))
        identity_keys = ("name", "rule_version", "calibration_hash", "query_text_hash")
        if any(reused.get(key) != result.get(key) for key in identity_keys):
            raise ValueError("REUSE_REFUSED_CALIBRATION_IDENTITY_MISMATCH")
        result["corpora"].update(reused.get("corpora", {}))
        result["resume"]["reused_result"] = {
            "path": str(Path(args.reuse_result)),
            "corpora": sorted(reused.get("corpora", {})),
        }
    selected = ("b", "a") if args.corpus == "all" else (args.corpus,)
    for corpus_id in selected:
        corpus_queries = [q for q in queries if q["corpus"] == corpus_id]
        if not corpus_queries:
            continue
        print(f"[V3.11][calibration] running corpus {corpus_id.upper()} with {len(corpus_queries)} queries", flush=True)
        result["corpora"][corpus_id], result["resume"][corpus_id] = _run_corpus(
            corpus_id, corpus_queries, calibration, resume=args.resume,
        )

    # Combined metrics across corpora.
    all_evidence = [row for c in result["corpora"].values() for row in c["evidence_rows"]]
    all_support = [row for c in result["corpora"].values() for row in c["support_rows"]]
    result["combined"] = {
        "evidence": _evidence_metrics(all_evidence),
        "support": _support_metrics(all_support),
        "evidence_by_manufacturer": _grouped_metrics(all_evidence, _evidence_metrics),
        "support_by_manufacturer": _grouped_metrics(all_support, _support_metrics),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, result)

    print("\n===== V3.11 CALIBRATION RESULT =====")
    print(f"hash: {result['calibration_hash']}")
    print(f"queries: {result['query_count']} (answerable {result['answerable']}, ood {result['ood']})")
    for scope, payload in (("combined", result["combined"]),):
        print(f"  [Combined Evidence] acc={payload['evidence']['decision_accuracy']} ood_recall={payload['evidence']['ood_recall']} ans_recall={payload['evidence']['answerable_recall']} false_ans={payload['evidence']['false_answer']} false_ref={payload['evidence']['false_refusal']}")
        print(f"  [Combined Support ] acc={payload['support']['support_accuracy']} sup_recall={payload['support']['supported_recall']} unsup_recall={payload['support']['unsupported_recall']} false_sup={payload['support']['false_support']} false_insuf={payload['support']['false_insufficient']}")
    for corpus_id, payload in result["corpora"].items():
        e, s = payload["evidence"], payload["support"]
        print(f"  [{corpus_id.upper()}] Evidence acc={e['decision_accuracy']} ood={e['ood_recall']} ans={e['answerable_recall']} fa={e['false_answer']} fr={e['false_refusal']}")
        print(f"  [{corpus_id.upper()}] Support  acc={s['support_accuracy']} sup={s['supported_recall']} unsup={s['unsupported_recall']} fs={s['false_support']} fi={s['false_insufficient']}")
        for mfr, m in payload["evidence_by_manufacturer"].items():
            print(f"    [{mfr}] Evidence acc={m['decision_accuracy']} ood={m['ood_recall']} fa={m['false_answer']} fr={m['false_refusal']} ids_fa={m['false_answer_ids']} ids_fr={m['false_refusal_ids']}")
        for mfr, m in payload["support_by_manufacturer"].items():
            print(f"    [{mfr}] Support  acc={m['support_accuracy']} fs={m['false_support']} fi={m['false_insufficient']} ids_fs={m['false_support_ids']} ids_fi={m['false_insufficient_ids']}")
        resume_stats = result["resume"].get(corpus_id, {})
        for stage in ("evidence", "support"):
            stats = resume_stats.get(stage, {})
            if stats:
                print(
                    f"    [{stage}] completed before={stats['completed_before']} "
                    f"skipped={stats['skipped']} executed={stats['executed']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
