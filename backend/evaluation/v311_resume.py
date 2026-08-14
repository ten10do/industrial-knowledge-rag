"""Rule-bound per-query checkpoints for V3.11 evaluations."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from backend.evaluation.resumable import CheckpointCorruptionError, CheckpointStore


def hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_completed_record(
    query_id: str,
    query: dict[str, Any],
    record: dict[str, Any],
    rule_version: str,
) -> None:
    result = record.get("result")
    valid = (
        record.get("status") == "COMPLETED"
        and record.get("rule_version") == rule_version
        and record.get("query_hash") == hash_json(query)
        and isinstance(result, dict)
        and result.get("query_id") == query_id
        and record.get("result_hash") == hash_json(result)
    )
    if not valid:
        raise CheckpointCorruptionError(
            f"Invalid completed V3.11 checkpoint row: {query_id}"
        )


def completed_results(
    stage: dict[str, Any] | None,
    queries: list[dict[str, Any]],
    rule_version: str,
    *,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    if stage is None:
        if require_complete:
            raise CheckpointCorruptionError("Missing V3.11 checkpoint stage")
        return []
    query_by_id = {query["query_id"]: query for query in queries}
    unknown = set(stage.get("rows", {})) - set(query_by_id)
    if unknown:
        raise CheckpointCorruptionError(
            f"Checkpoint contains unknown query ids: {sorted(unknown)}"
        )
    results = []
    for query in queries:
        record = stage.get("rows", {}).get(query["query_id"])
        if record is None or record.get("status") != "COMPLETED":
            if require_complete:
                raise CheckpointCorruptionError(
                    f"Missing completed V3.11 checkpoint row: {query['query_id']}"
                )
            continue
        _validate_completed_record(query["query_id"], query, record, rule_version)
        results.append(record["result"])
    return results


def run_query_stage(
    store: CheckpointStore,
    stage_name: str,
    corpus: str,
    queries: list[dict[str, Any]],
    rule_version: str,
    execute: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Execute only missing/failed queries and atomically persist every result."""
    existing = store.load_stage(stage_name)
    completed_before = len(completed_results(existing, queries, rule_version))
    store.begin_stage(stage_name, len(queries))
    stage = existing or {"stage": stage_name, "rows": {}, "errors": {}}
    skipped = 0
    executed = 0
    started = time.perf_counter()
    for query in queries:
        query_id = query["query_id"]
        record = stage["rows"].get(query_id)
        if record is not None and record.get("status") == "COMPLETED":
            _validate_completed_record(query_id, query, record, rule_version)
            skipped += 1
            continue
        query_started = time.perf_counter()
        executed += 1
        try:
            result = execute(query)
            if result.get("query_id") != query_id:
                raise ValueError(
                    f"Result query id mismatch: expected {query_id}, got {result.get('query_id')}"
                )
            record = {
                "status": "COMPLETED",
                "rule_version": rule_version,
                "query_hash": hash_json(query),
                "result_hash": hash_json(result),
                "result": result,
                "latency_ms": (time.perf_counter() - query_started) * 1000,
            }
            stage = store.save_query(
                stage_name, query_id, record, total_queries=len(queries)
            )
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            record = {
                "status": "ERROR",
                "rule_version": rule_version,
                "query_hash": hash_json(query),
                "result": {},
                "latency_ms": (time.perf_counter() - query_started) * 1000,
                "error": error,
            }
            stage = store.save_query(
                stage_name,
                query_id,
                record,
                total_queries=len(queries),
                error=error,
            )
        completed = len(completed_results(stage, queries, rule_version))
        print(
            f"[V3.11][Corpus {corpus.upper()}][{stage_name}] "
            f"{completed}/{len(queries)} elapsed: {time.perf_counter() - started:.1f}s "
            f"last query: {query_id} checkpoint: saved",
            flush=True,
        )
    return stage, {
        "completed_before": completed_before,
        "skipped": skipped,
        "executed": executed,
    }
