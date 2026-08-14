"""Immutable, self-contained retrieval artifacts for offline Evidence/Support replay.

The artifact contains rule inputs, not rule outputs.  Loading and replaying this
module never imports the live retrieval stack, a vector database, a PDF parser,
or the reranker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.evaluation.resumable import atomic_write_json, read_json, utc_now
from backend.evaluation.v311_resume import hash_json
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.filters import QueryAnalysis
from backend.retrieval.product_identity import ProductIdentity


ARTIFACT_SCHEMA_VERSION = "retrieval-artifact-v1"
VALIDITY_VALUES = {"VALID", "PARTIAL", "INVALID"}
CANDIDATE_FIELDS = (
    "retrieval_source", "lexical_rank", "vector_rank", "lexical_score",
    "vector_score", "fusion_score", "final_rank", "pre_rerank_rank",
    "rerank_score", "rerank_rank", "evidence_score", "exact_metadata_match",
    "identity_relation", "scope_match", "scope_level", "section_expanded",
    "section_rank", "neighbor_distance", "pre_section_rank",
    "section_candidate_source", "candidate_source", "preservation_class",
)


class ArtifactValidationError(RuntimeError):
    """The artifact cannot be trusted for replay."""


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_document(document: Document) -> dict[str, Any]:
    return {
        "content": str(document.page_content),
        "metadata": _json_value(dict(document.metadata or {})),
    }


def deserialize_document(payload: dict[str, Any]) -> Document:
    return Document(
        page_content=str(payload["content"]),
        metadata=dict(payload["metadata"]),
    )


def serialize_candidate(candidate: RetrievalCandidate) -> dict[str, Any]:
    payload = serialize_document(candidate.document)
    payload.update({
        field: _json_value(value)
        for field in CANDIDATE_FIELDS
        if (value := getattr(candidate, field)) is not None
    })
    payload["chunk_id"] = candidate.chunk_id
    payload["document_id"] = str(candidate.metadata.get("document_id", ""))
    for field in (
        "manufacturer", "product_family", "product_series", "equipment_model",
        "section", "subsection", "page",
    ):
        if field in candidate.metadata:
            payload[field] = _json_value(candidate.metadata[field])
    return payload


def candidate_from_p2_row(row: dict[str, Any], document: Document) -> dict[str, Any]:
    """Attach the original P2 rank/provenance to its exact parsed chunk."""
    payload = serialize_document(document)
    candidate_fields = {
        "chunk_id": str(row["chunk_id"]),
        "document_id": str(document.metadata.get("document_id", "")),
        "retrieval_source": row.get("section_candidate_source") or "hybrid",
        "final_rank": row.get("rank"),
        "lexical_rank": row.get("lexical_rank"),
        "vector_rank": row.get("vector_rank"),
        "lexical_score": row.get("lexical_score"),
        "vector_score": row.get("vector_distance"),
        "pre_rerank_rank": row.get("pre_rerank_rank"),
        "fusion_rank": row.get("fusion_rank"),
        "rerank_rank": row.get("rerank_rank"),
        "identity_relation": row.get("identity_relation", "UNKNOWN"),
        "scope_match": row.get("scope_match", "none"),
        "scope_level": row.get("scope_level", "GLOBAL_SCOPE"),
        "section_expanded": row.get("section_expanded", False),
        "section_rank": row.get("section_rank"),
        "neighbor_distance": row.get("neighbor_distance"),
        "pre_section_rank": row.get("pre_section_rank"),
        "section_candidate_source": row.get("section_candidate_source", ""),
    }
    payload.update({key: value for key, value in candidate_fields.items() if value is not None})
    for field in (
        "manufacturer", "product_family", "product_series", "equipment_model",
        "section", "subsection", "page",
    ):
        if field in document.metadata:
            payload[field] = _json_value(document.metadata[field])
    return payload


def deserialize_candidate(payload: dict[str, Any]) -> RetrievalCandidate:
    kwargs = {field: payload[field] for field in CANDIDATE_FIELDS if field in payload}
    return RetrievalCandidate(document=deserialize_document(payload), **kwargs)


def serialize_query_analysis(analysis: QueryAnalysis) -> dict[str, Any]:
    return _json_value(asdict(analysis))


def deserialize_query_analysis(payload: dict[str, Any]) -> QueryAnalysis:
    values = dict(payload)
    values["identifiers"] = tuple(values.get("identifiers", ()))
    values["product_identities"] = tuple(
        ProductIdentity(
            manufacturer=item.get("manufacturer", ""),
            product_family=item.get("product_family", ""),
            product_series=item.get("product_series", ""),
            equipment_model=item.get("equipment_model", ""),
            aliases=tuple(item.get("aliases", ())),
        )
        for item in values.get("product_identities", ())
    )
    return QueryAnalysis(**values)


def make_retrieval_result(
    candidates: list[dict[str, Any]],
    query_analysis: dict[str, Any],
    corpus_snapshot: list[Document],
    retrieval_mode: str = "hybrid",
) -> RetrievalResult:
    return RetrievalResult(
        [deserialize_candidate(item) for item in candidates],
        query_analysis=deserialize_query_analysis(query_analysis),
        corpus_documents=corpus_snapshot,
        retrieval_mode=retrieval_mode,
    )


def artifact_hash(payload: dict[str, Any]) -> str:
    return hash_json({key: value for key, value in payload.items() if key != "artifact_hash"})


def seal_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = {**payload, "schema_version": ARTIFACT_SCHEMA_VERSION}
    sealed["artifact_hash"] = artifact_hash(sealed)
    return sealed


def write_immutable_artifact(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    atomic_write_json(path, seal_artifact(payload))


def validate_artifact(
    payload: dict[str, Any],
    *,
    expected_corpus_id: str | None = None,
    expected_manifest_hash: str | None = None,
    expected_annotation_hash: str | None = None,
    expected_retrieval_config_hash: str | None = None,
) -> dict[str, Any]:
    invalid: list[str] = []
    partial: list[str] = []
    required = {
        "schema_version", "artifact_version", "artifact_id", "artifact_hash",
        "corpus_id", "corpus_manifest_hash", "annotation_hash",
        "retrieval_config_hash", "retrieval_config", "embedding_config",
        "reranker_config", "queries", "query_count", "corpus_snapshot",
        "expected_query_count", "frozen_query_set_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        invalid.append(f"missing top-level fields: {missing}")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        invalid.append("schema version mismatch")
    if payload.get("artifact_version") != ARTIFACT_SCHEMA_VERSION:
        invalid.append("artifact version mismatch")
    if payload.get("artifact_hash") != artifact_hash(payload):
        invalid.append("artifact hash mismatch")
    if payload.get("manifest_hash") != payload.get("corpus_manifest_hash"):
        invalid.append("corpus manifest hash aliases disagree")
    checks = (
        ("corpus_id", expected_corpus_id),
        ("corpus_manifest_hash", expected_manifest_hash),
        ("annotation_hash", expected_annotation_hash),
        ("retrieval_config_hash", expected_retrieval_config_hash),
    )
    for field, expected in checks:
        if expected is not None and payload.get(field) != expected:
            invalid.append(f"{field} mismatch")
    if payload.get("retrieval_config_hash") != hash_json(payload.get("retrieval_config", {})):
        invalid.append("retrieval config hash mismatch")
    source_config = payload.get("retrieval_config", {}).get("source_evaluation_configuration", {})
    if payload.get("embedding_config") != {"model": source_config.get("embedding_model", "")}:
        invalid.append("embedding config mismatch")
    if payload.get("reranker_config") != source_config.get("reranker", {}):
        invalid.append("reranker config mismatch")

    queries = payload.get("queries")
    if not isinstance(queries, list):
        invalid.append("queries must be a list")
        queries = []
    if not queries:
        invalid.append("artifact contains no queries")
    if payload.get("query_count") != len(queries):
        partial.append("query count mismatch")
    if payload.get("expected_query_count") != len(queries):
        partial.append("frozen query set is incomplete")
    seen: set[str] = set()
    for index, row in enumerate(queries):
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in seen:
            invalid.append(f"invalid or duplicate query id at index {index}")
        seen.add(query_id)
        if row.get("query_text_hash") != hash_json(row.get("query", "")):
            invalid.append(f"query text hash mismatch: {query_id}")
        truth = row.get("ground_truth", {})
        if truth.get("query_id") != query_id or truth.get("query") != row.get("query"):
            invalid.append(f"ground truth identity mismatch: {query_id}")
        if (
            "evidence_input" not in row or "final_context" not in row
            or "retrieval_decision_inputs" not in row
        ):
            partial.append(f"missing replay input: {query_id}")
            continue
        evidence_input = row["evidence_input"]
        if "candidate_pool" not in evidence_input or "query_analysis" not in evidence_input:
            partial.append(f"missing Evidence input: {query_id}")
        for collection_name, candidates in (
            ("candidate_pool", evidence_input.get("candidate_pool", [])),
            ("final_context", row.get("final_context", [])),
        ):
            if not isinstance(candidates, list):
                invalid.append(f"{collection_name} must be a list: {query_id}")
                continue
            for candidate in candidates:
                if not all(field in candidate for field in ("chunk_id", "document_id", "content", "metadata")):
                    partial.append(f"incomplete {collection_name} candidate: {query_id}")
                    continue
                metadata = candidate["metadata"]
                if (
                    str(metadata.get("chunk_id", "")) != str(candidate["chunk_id"])
                    or str(metadata.get("document_id", "")) != str(candidate["document_id"])
                ):
                    invalid.append(f"candidate identity mismatch: {query_id}/{candidate['chunk_id']}")

    snapshot = payload.get("corpus_snapshot", {})
    documents = snapshot.get("documents", []) if isinstance(snapshot, dict) else []
    if not isinstance(documents, list) or not documents:
        partial.append("missing corpus snapshot documents")
    elif snapshot.get("snapshot_hash") != hash_json(documents):
        invalid.append("corpus snapshot hash mismatch")

    if not partial:
        query_set = [
            {"query_id": row.get("query_id"), "query": row.get("query")}
            for row in queries
        ]
        if payload.get("frozen_query_set_hash") != hash_json(query_set):
            invalid.append("frozen query set hash mismatch")

    validity = "INVALID" if invalid else "PARTIAL" if partial else "VALID"
    return {
        "artifact_id": payload.get("artifact_id", ""),
        "validity": validity,
        "invalid_reasons": invalid,
        "partial_reasons": partial,
        "query_count": len(queries),
        "expected_query_count": payload.get("expected_query_count"),
    }


def load_valid_artifact(path: Path, **expected: str) -> dict[str, Any]:
    payload = read_json(path)
    report = validate_artifact(payload, **expected)
    if report["validity"] != "VALID":
        raise ArtifactValidationError(json.dumps(report, ensure_ascii=False))
    return payload


def artifact_inspection(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    report = validate_artifact(payload)
    return {
        **report,
        "schema_version": payload.get("schema_version"),
        "corpus_id": payload.get("corpus_id"),
        "pipeline_id": payload.get("pipeline_id"),
        "retrieval_config_hash": payload.get("retrieval_config_hash"),
        "created_at": payload.get("created_at"),
        "rule_version_at_export": payload.get("rule_version_at_export"),
        "artifact_bytes": path.stat().st_size if path and path.exists() else None,
        "candidate_counts": {
            row.get("query_id", ""): {
                "evidence": len(row.get("evidence_input", {}).get("candidate_pool", [])),
                "final_context": len(row.get("final_context", [])),
            }
            for row in payload.get("queries", [])
        },
    }


def new_artifact_payload(
    *, artifact_id: str, corpus_id: str, manifest_hash: str,
    annotation_hash: str, retrieval_config: dict[str, Any],
    queries: list[dict[str, Any]], snapshot_documents: list[dict[str, Any]],
    source: dict[str, Any], rule_version: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "corpus_id": corpus_id,
        "pipeline_id": "P2_FROZEN_WITH_EVIDENCE_INPUT",
        "manifest_hash": manifest_hash,
        "corpus_manifest_hash": manifest_hash,
        "annotation_hash": annotation_hash,
        "retrieval_config": retrieval_config,
        "retrieval_config_hash": hash_json(retrieval_config),
        "embedding_config": {
            "model": retrieval_config.get("source_evaluation_configuration", {}).get("embedding_model", ""),
        },
        "reranker_config": retrieval_config.get("source_evaluation_configuration", {}).get("reranker", {}),
        "query_count": len(queries),
        "expected_query_count": len(queries),
        "frozen_query_set_hash": hash_json([
            {"query_id": row["query_id"], "query": row["query"]}
            for row in queries
        ]),
        "queries": queries,
        "corpus_snapshot": {
            "kind": "evidence-gate-term-index-v1",
            "documents": snapshot_documents,
            "snapshot_hash": hash_json(snapshot_documents),
        },
        "source": source,
        "rule_version_at_export": rule_version,
    }
