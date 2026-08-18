"""Candidate-fixed calibration and reporting for the V3.20 Evidence contract.

The calibration reuses immutable candidate payloads from A/B/C artifacts but
contains newly authored paired queries.  It performs no retrieval and never
opens D/E until the final frozen replay phase.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import deserialize_candidate, deserialize_document
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.retrieval import RetrievalResult, analyze_query
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CALIBRATION_PATH = PRIVATE_ROOT / "v320_calibration" / "evidence_contract_calibration.json"
RESULT_ROOT = PRIVATE_ROOT / "v320_results"
REQUIRED_FAILURE_CLASSES = frozenset({
    "identifier", "protocol", "attribute", "value", "action", "requirement",
    "semantic", "multi_chunk", "cross_scope", "qualifier",
})


def _payload_without_freeze(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "freeze"}


def validate_calibration(manifest: dict[str, Any]) -> dict[str, Any]:
    queries = manifest.get("queries", [])
    if not 48 <= len(queries) <= 60:
        raise ValueError("V320_CALIBRATION_REQUIRES_48_TO_60_QUERIES")
    answerable = [row for row in queries if row.get("answerable")]
    abstain = [row for row in queries if not row.get("answerable")]
    if not 24 <= len(answerable) <= 30 or not 24 <= len(abstain) <= 30:
        raise ValueError("V320_CALIBRATION_BALANCE_INVALID")
    ids = [row.get("query_id") for row in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("V320_DUPLICATE_QUERY_ID")
    pairs = Counter(row.get("pair_id") for row in queries)
    if not pairs or any(count != 2 for count in pairs.values()):
        raise ValueError("V320_CALIBRATION_MUST_BE_PAIRED")
    for pair_id in pairs:
        labels = {bool(row["answerable"]) for row in queries if row["pair_id"] == pair_id}
        if labels != {False, True}:
            raise ValueError(f"V320_PAIR_LABEL_INVALID:{pair_id}")
    classes = {row.get("failure_class") for row in queries}
    if not REQUIRED_FAILURE_CLASSES.issubset(classes):
        raise ValueError(f"V320_FAILURE_CLASS_MISSING:{sorted(REQUIRED_FAILURE_CLASSES-classes)}")
    manufacturers = {row.get("manufacturer") for row in queries}
    if len(manufacturers - {None, ""}) < 4:
        raise ValueError("V320_REQUIRES_FOUR_MANUFACTURERS")
    if sum(bool(row.get("semantic_positive")) and row["answerable"] for row in queries) < 10:
        raise ValueError("V320_SEMANTIC_POSITIVE_MINIMUM")
    if sum(bool(row.get("multi_chunk_positive")) and row["answerable"] for row in queries) < 5:
        raise ValueError("V320_MULTI_CHUNK_POSITIVE_MINIMUM")
    if sum(bool(row.get("unsafe_multi_chunk_negative")) and not row["answerable"] for row in queries) < 5:
        raise ValueError("V320_MULTI_CHUNK_NEGATIVE_MINIMUM")
    candidates = manifest.get("candidates", {})
    if any(chunk_id not in candidates for row in queries for chunk_id in row.get("candidate_chunk_ids", [])):
        raise ValueError("V320_UNKNOWN_CANDIDATE")
    freeze = manifest.get("freeze", {})
    if freeze.get("query_sha256") != hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in queries]):
        raise ValueError("V320_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != hash_json(queries):
        raise ValueError("V320_ANNOTATION_HASH_MISMATCH")
    if freeze.get("manifest_sha256") != hash_json(_payload_without_freeze(manifest)):
        raise ValueError("V320_MANIFEST_HASH_MISMATCH")
    return {
        "queries": len(queries), "answerable": len(answerable), "abstain": len(abstain),
        "manufacturers": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in queries).items())),
        "pairs": len(pairs),
        "semantic_positive": sum(bool(row.get("semantic_positive")) for row in answerable),
        "multi_chunk_positive": sum(bool(row.get("multi_chunk_positive")) for row in answerable),
        "unsafe_multi_chunk_negative": sum(bool(row.get("unsafe_multi_chunk_negative")) for row in abstain),
    }


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    manifest = read_json(ensure_private_path(path or CALIBRATION_PATH))
    validate_calibration(manifest)
    return manifest


def calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    abstain = [row for row in rows if not row["answerable"]]
    false_answers = [row["query_id"] for row in abstain if row["decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in answerable if row["decision"] == "ABSTAIN"]
    return {
        "decision_accuracy": sum((row["decision"] == "ANSWER") == row["answerable"] for row in rows) / len(rows),
        "answerable_recall": 1 - len(false_refusals) / len(answerable),
        "abstain_recall": 1 - len(false_answers) / len(abstain),
        "false_answer_rate": len(false_answers) / len(abstain),
        "false_refusal_rate": len(false_refusals) / len(answerable),
        "false_answer_ids": false_answers, "false_refusal_ids": false_refusals,
    }


def evaluate_calibration(manifest: dict[str, Any]) -> dict[str, Any]:
    distribution = validate_calibration(manifest)
    snapshot = [deserialize_document(row) for row in manifest["corpus_snapshot"]]
    candidates = {chunk_id: deserialize_candidate(payload) for chunk_id, payload in manifest["candidates"].items()}
    rows = []
    started = time.perf_counter()
    for query in manifest["queries"]:
        selected = [candidates[chunk_id] for chunk_id in query["candidate_chunk_ids"]]
        result = RetrievalResult(selected, query_analysis=analyze_query(query["query"], snapshot), corpus_documents=snapshot, retrieval_mode="calibration_fixed")
        evidence = analyze_retrieval_evidence(query["query"], result, snapshot, "calibration_fixed")
        rows.append({
            "query_id": query["query_id"], "pair_id": query["pair_id"], "query": query["query"],
            "answerable": query["answerable"], "manufacturer": query["manufacturer"],
            "failure_class": query["failure_class"], "candidate_chunk_ids": query["candidate_chunk_ids"],
            "decision": evidence.decision, "reason": evidence.reason, "evidence": evidence.as_dict(),
        })
    by_class = {
        name: calibration_metrics([row for row in rows if row["failure_class"] == name])
        for name in sorted({row["failure_class"] for row in rows})
    }
    return {
        "calibration_id": manifest["calibration_id"], "validity": "VALID",
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION, "freeze": manifest["freeze"],
        "distribution": distribution, "metrics": calibration_metrics(rows),
        "by_failure_class": by_class, "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "live_retrieval": False, "candidate_source": manifest["candidate_source"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = load_calibration()
    if args.command == "validate":
        report = {"validity": "VALID", "distribution": validate_calibration(manifest), "freeze": manifest["freeze"]}
    else:
        report = evaluate_calibration(manifest)
        if args.output:
            atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
