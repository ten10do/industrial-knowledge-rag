"""Run the frozen V3.13 Support calibration without live retrieval."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import (
    deserialize_candidate, deserialize_document, load_valid_artifact,
)
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.retrieval.candidates import RetrievalResult
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION, validate_evidence_support
from backend.retrieval.filters import analyze_query


def _annotation_identity(query: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in query.items() if key != "rationale"}


def validate_calibration(calibration: dict[str, Any]) -> None:
    queries = calibration.get("queries", [])
    if calibration.get("query_count") != len(queries) or not queries:
        raise ValueError("CALIBRATION_QUERY_COUNT_MISMATCH")
    query_hash = hash_json([
        {"calibration_id": row["calibration_id"], "query": row["query"]}
        for row in queries
    ])
    annotation_hash = hash_json([_annotation_identity(row) for row in queries])
    freeze = calibration.get("freeze", {})
    if freeze.get("query_hash") != query_hash:
        raise ValueError("CALIBRATION_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_hash") != annotation_hash:
        raise ValueError("CALIBRATION_ANNOTATION_HASH_MISMATCH")
    identifiers = [row["calibration_id"] for row in queries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("CALIBRATION_DUPLICATE_ID")
    if any(row.get("expected_support") not in {"SUPPORTED", "INSUFFICIENT"} for row in queries):
        raise ValueError("CALIBRATION_INVALID_SUPPORT_LABEL")


def _candidate_index(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for query in artifact["queries"]:
        for candidate in query["final_context"]:
            result.setdefault(candidate["chunk_id"], candidate)
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["expected_support"] == "SUPPORTED"]
    unsupported = [row for row in rows if row["expected_support"] == "INSUFFICIENT"]
    false_support = [row for row in unsupported if row["predicted_support"] == "SUPPORTED"]
    false_insufficient = [row for row in supported if row["predicted_support"] != "SUPPORTED"]
    return {
        "query_count": len(rows),
        "support_accuracy": sum(row["predicted_support"] == row["expected_support"] for row in rows) / len(rows),
        "supported_recall": sum(row["predicted_support"] == "SUPPORTED" for row in supported) / len(supported),
        "unsupported_recall": sum(row["predicted_support"] != "SUPPORTED" for row in unsupported) / len(unsupported),
        "false_support_rate": len(false_support) / len(unsupported),
        "false_insufficient_rate": len(false_insufficient) / len(supported),
        "false_support_ids": [row["calibration_id"] for row in false_support],
        "false_insufficient_ids": [row["calibration_id"] for row in false_insufficient],
        "status_distribution": dict(Counter(row["predicted_support"] for row in rows)),
    }


def run_calibration(
    calibration_path: Path,
    artifact_paths: dict[str, Path],
    *,
    rule_version: str,
) -> dict[str, Any]:
    if rule_version != SUPPORT_RULE_VERSION:
        raise ValueError(
            f"SUPPORT_RULE_VERSION_MISMATCH:{rule_version}:{SUPPORT_RULE_VERSION}"
        )
    calibration = read_json(calibration_path)
    validate_calibration(calibration)
    artifacts = {
        corpus_id: load_valid_artifact(path)
        for corpus_id, path in artifact_paths.items()
    }
    for corpus_id, expected in calibration["source_artifacts"].items():
        artifact = artifacts[corpus_id]
        if (
            artifact["artifact_id"] != expected["artifact_id"]
            or artifact["artifact_hash"] != expected["artifact_hash"]
            or artifact["corpus_manifest_hash"] != expected["corpus_manifest_hash"]
        ):
            raise ValueError(f"CALIBRATION_ARTIFACT_MISMATCH:{corpus_id}")

    indexes = {key: _candidate_index(value) for key, value in artifacts.items()}
    snapshots = {
        key: [deserialize_document(item) for item in value["corpus_snapshot"]["documents"]]
        for key, value in artifacts.items()
    }
    rows = []
    started = time.perf_counter()
    for query in calibration["queries"]:
        corpus_id = query["corpus_id"]
        try:
            candidate_payloads = [indexes[corpus_id][chunk_id] for chunk_id in query["candidate_chunk_ids"]]
        except KeyError as exc:
            raise ValueError(
                f"CALIBRATION_CANDIDATE_MISSING:{query['calibration_id']}:{exc.args[0]}"
            ) from exc
        candidates = [deserialize_candidate(item) for item in candidate_payloads]
        documents = snapshots[corpus_id]
        result = RetrievalResult(
            candidates,
            query_analysis=analyze_query(query["query"], documents),
            corpus_documents=documents,
            retrieval_mode="frozen_support_calibration",
        )
        query_started = time.perf_counter()
        support = validate_evidence_support(query["query"], result, documents)
        rows.append({
            **query,
            "predicted_support": support.status,
            "support": support.as_dict(),
            "latency_ms": (time.perf_counter() - query_started) * 1000,
        })

    by_manufacturer = {
        manufacturer: _metrics([row for row in rows if row["manufacturer"] == manufacturer])
        for manufacturer in sorted({row["manufacturer"] for row in rows})
    }
    return {
        "calibration_name": calibration["name"],
        "calibration_version": calibration["version"],
        "query_hash": calibration["freeze"]["query_hash"],
        "annotation_hash": calibration["freeze"]["annotation_hash"],
        "support_rule_version": rule_version,
        "source_artifacts": calibration["source_artifacts"],
        "validity": "VALID",
        "metrics": {"combined": _metrics(rows), "by_manufacturer": by_manufacturer},
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--rule-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_private_path(args.output)
    report = run_calibration(
        ensure_private_path(args.calibration),
        {
            "A": ensure_private_path(args.artifact_a),
            "B": ensure_private_path(args.artifact_b),
        },
        rule_version=args.rule_version,
    )
    atomic_write_json(output, report)
    print(json.dumps({
        "validity": report["validity"],
        "support_rule_version": report["support_rule_version"],
        "metrics": report["metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
