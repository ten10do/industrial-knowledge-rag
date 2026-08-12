"""Shared manifest validation and deterministic retrieval metrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


QUERY_TYPES = {
    "identifier", "semantic", "parameter", "procedure", "fault",
    "maintenance", "comparison", "mixed", "ood",
}
FAILURE_TYPES = (
    "RECALL_FAILURE", "RANKING_FAILURE", "IDENTIFIER_CONFUSION",
    "MODEL_CONFUSION", "SECTION_CONFUSION", "SEMANTIC_CONFUSION",
    "OOD_FALSE_POSITIVE", "METADATA_FAILURE", "OVER_FILTER_FAILURE",
    "AMBIGUOUS_MODEL", "SCOPE_FALLBACK",
)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    queries = manifest.get("queries", [])
    if not documents or not queries:
        raise ValueError("Benchmark manifest requires documents and queries.")
    chunk_ids = set()
    for document in documents:
        required = {
            "document_id", "chunk_id", "file", "document_type",
            "manufacturer", "equipment_type", "equipment_model", "language",
            "source_type", "commit_allowed",
        }
        if not required.issubset(document) or document["chunk_id"] in chunk_ids:
            raise ValueError("Benchmark document metadata is invalid or duplicated.")
        if not isinstance(document["commit_allowed"], bool):
            raise ValueError("Benchmark document commit_allowed must be boolean.")
        if Path(document["file"]).is_absolute():
            raise ValueError("Benchmark document file must be relative to its manifest.")
        if not document.get("content") and not document["file"]:
            raise ValueError("Benchmark document requires content or a relative file.")
        chunk_ids.add(document["chunk_id"])
    query_ids = set()
    for query in queries:
        required = {
            "query_id", "query", "category", "query_type", "difficulty",
            "expected_document_id", "expected_chunk_id", "expected_section",
            "expected_error_code", "expected_equipment_model", "answerable",
            "relevant_chunk_ids",
        }
        if not required.issubset(query) or query["query_id"] in query_ids:
            raise ValueError("Benchmark query annotation is invalid or duplicated.")
        if query["query_type"] not in QUERY_TYPES or query["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("Benchmark query type or difficulty is invalid.")
        if not set(query["relevant_chunk_ids"]).issubset(chunk_ids):
            raise ValueError("Benchmark query references an unknown chunk.")
        if query["answerable"] != bool(query["relevant_chunk_ids"]):
            raise ValueError("Answerable flag must match relevant chunks.")
        if query["answerable"] and query["expected_chunk_id"] not in query["relevant_chunk_ids"]:
            raise ValueError("Expected chunk must be one of the relevant chunks.")
        query_ids.add(query["query_id"])
    return manifest


def rank_of(candidates: list[dict], relevant_chunk_ids: list[str]) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["chunk_id"] in relevant_chunk_ids:
            return rank
    return None


def classify_failure(query: dict, row: dict) -> str | None:
    if not query["answerable"]:
        return "OOD_FALSE_POSITIVE" if not row["refused"] else None
    rank = row["rank"]
    top = row["candidates"][0] if row["candidates"] else {}
    if rank and rank > 1:
        return "RANKING_FAILURE"
    if rank:
        return None
    scope = row.get("retrieval_scope", {})
    if scope.get("requested_scope") == "UNKNOWN_SCOPE":
        return "AMBIGUOUS_MODEL"
    if query["expected_error_code"] and top.get("error_code"):
        return "IDENTIFIER_CONFUSION"
    if (
        query["expected_equipment_model"]
        and top.get("equipment_model")
        and top.get("equipment_model") != query["expected_equipment_model"]
    ):
        return "MODEL_CONFUSION"
    if query["expected_section"] and top.get("section"):
        return "SECTION_CONFUSION"
    if query["query_type"] == "semantic":
        return "SEMANTIC_CONFUSION"
    return "RECALL_FAILURE"


def evaluate_rows(queries: list[dict], rows: list[dict]) -> dict:
    rows_by_id = {row["query_id"]: row for row in rows}
    answerable = [query for query in queries if query["answerable"]]
    category = defaultdict(list)
    failures = Counter()
    for query in queries:
        row = rows_by_id[query["query_id"]]
        failure = classify_failure(query, row)
        row["failure_type"] = failure
        if failure:
            failures[failure] += 1
        if row.get("retrieval_scope", {}).get("fallback_used"):
            failures["SCOPE_FALLBACK"] += 1
        if query["answerable"]:
            category[query["category"]].append(row)

    def metrics(items: list[dict]) -> dict:
        ranks = [item["rank"] for item in items]
        return {
            "count": len(items),
            "hit_rate_at_1": sum(rank == 1 for rank in ranks) / len(items),
            "hit_rate_at_3": sum((rank or 99) <= 3 for rank in ranks) / len(items),
            "recall_at_5": sum((rank or 99) <= 5 for rank in ranks) / len(items),
            "mrr": sum(1 / rank for rank in ranks if rank) / len(items),
        }

    overall = metrics([rows_by_id[query["query_id"]] for query in answerable])
    ood = [query for query in queries if not query["answerable"]]
    overall["ood_refusal_accuracy"] = (
        sum(rows_by_id[query["query_id"]]["refused"] for query in ood) / len(ood)
        if ood else None
    )
    overall["ranking_gap"] = overall["recall_at_5"] - overall["hit_rate_at_1"]
    return {
        "overall": overall,
        "category_metrics": {name: metrics(items) for name, items in sorted(category.items())},
        "failure_summary": {name: failures.get(name, 0) for name in FAILURE_TYPES},
        "rows": rows,
    }
