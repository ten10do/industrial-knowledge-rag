"""V3.17 independent Corpus D support holdout validation.

The module deliberately has one live boundary: ``export_holdout_artifact``.
Every later result is derived from the immutable retrieval artifact by offline
Evidence/Support replay; this module never changes retrieval or rule behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import (
    candidate_from_p2_row,
    file_sha256,
    new_artifact_payload,
    validate_artifact,
    write_immutable_artifact,
)
from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base
from backend.evaluation.private_benchmark import (
    _run_mode,
    annotation_hash,
    ingest_private_documents,
    load_private_manifest,
)
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v310_runner import P2
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import (
    _capture_evidence_input,
    _snapshot_documents,
    ensure_private_path,
    replay_artifact,
    retrieval_configuration,
)
from backend.evaluation.v314_holdout_validation import (
    LOCATION_GROUND_TRUTH_POLICY_V1,
    generalization_range,
)
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_D_ROOT = PRIVATE_ROOT / "corpus_d"
SOURCE_MANIFEST_PATH = CORPUS_D_ROOT / "source_manifest.json"
HOLDOUT_MANIFEST_PATH = CORPUS_D_ROOT / "manifest.json"
RUNTIME_ROOT = PRIVATE_ROOT / "v317_runtime"
ARTIFACT_ROOT = PRIVATE_ROOT / "v317_artifacts"
RESULT_ROOT = PRIVATE_ROOT / "v317_results"
FROZEN_RULE_IDENTITY = {"evidence": "v311.2", "support": "support-v316.1"}
CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
REQUIRED_CONTRACT_TAGS = frozenset({
    "attribute", "value", "unit", "value_kind", "compatibility",
    "requirement", "version", "protocol", "action_procedure",
    "installation", "configuration", "safety", "maintenance", "location",
    "semantic_equivalence", "multi_requirement", "multi_chunk_support",
})
REQUIRED_VALUE_KINDS = frozenset({"default", "range", "maximum", "minimum", "rated_nominal"})
SUPPORT_FAILURE_TAXONOMY = frozenset({
    "PARTIAL_SUPPORT_ACCEPTED", "OVER_CONSTRAINED_ATTRIBUTE",
    "OVER_CONSTRAINED_VALUE", "OVER_CONSTRAINED_REQUIREMENT",
    "OVER_CONSTRAINED_QUALIFIER", "SEMANTIC_EQUIVALENCE_MISSED",
    "MULTI_CHUNK_AGGREGATION_MISSED", "LOCAL_ASSOCIATION_TOO_STRICT",
    "VALUE_KIND_FAILURE", "UNIT_FAILURE", "ACTION_FAILURE", "LOCATION_FAILURE",
    "OTHER",
})


def _assert_rule_identity() -> None:
    actual = {"evidence": EVIDENCE_SUPPORT_RULE_VERSION, "support": SUPPORT_RULE_VERSION}
    if actual != FROZEN_RULE_IDENTITY:
        raise RuntimeError(f"FROZEN_RULE_IDENTITY_MISMATCH:{actual}")


def query_hash(manifest: dict[str, Any]) -> str:
    return hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in manifest["queries"]])


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hash_json({key: value for key, value in manifest.items() if key != "freeze"})


def _validate_documents(documents: list[dict[str, Any]]) -> None:
    if len(documents) < 3:
        raise ValueError("CORPUS_D_REQUIRES_THREE_DOCUMENTS")
    manufacturers = {str(item.get("manufacturer", "")) for item in documents}
    if len(manufacturers - {""}) < 2:
        raise ValueError("CORPUS_D_REQUIRES_TWO_MANUFACTURERS")
    for item in documents:
        if item.get("source_type") != "official_vendor_publication":
            raise ValueError(f"CORPUS_D_NON_OFFICIAL_SOURCE:{item.get('document_id', '')}")
        if not str(item.get("official_url", "")).startswith("https://"):
            raise ValueError(f"CORPUS_D_OFFICIAL_URL_REQUIRED:{item.get('document_id', '')}")
        if item.get("commit_allowed") is not False:
            raise ValueError(f"CORPUS_D_PRIVATE_DOCUMENT_REQUIRED:{item.get('document_id', '')}")


def validate_source_manifest(source: dict[str, Any]) -> None:
    documents = source.get("documents")
    if not isinstance(documents, list):
        raise ValueError("CORPUS_D_SOURCE_DOCUMENTS_REQUIRED")
    _validate_documents(documents)


def _validate_location_annotation(query: dict[str, Any]) -> None:
    expectation = query.get("location_expectation", "none")
    if expectation not in LOCATION_GROUND_TRUTH_POLICY_V1:
        raise ValueError(f"INVALID_LOCATION_EXPECTATION:{query['query_id']}")
    required = {
        "specific_page": "expected_page",
        "specific_section": "expected_section",
        "exact_subsection": "expected_subsection",
    }
    field = required.get(expectation)
    if field and not str(query.get(field, "")):
        raise ValueError(f"LOCATION_METADATA_REQUIRED:{query['query_id']}:{field}")


def _requested_complete(query: dict[str, Any]) -> bool:
    requested = query.get("requested")
    fields = {
        "identity", "identifier", "protocol", "action", "attribute", "value",
        "unit", "value_kind", "requirement_type", "location", "qualifier",
    }
    return isinstance(requested, dict) and fields.issubset(requested)


def validate_holdout_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate and freeze Corpus D before the first live retrieval."""
    _assert_rule_identity()
    _validate_documents(manifest.get("documents", []))
    queries = manifest.get("queries", [])
    if not 36 <= len(queries) <= 48:
        raise ValueError("HOLDOUT_D_QUERY_COUNT_MUST_BE_36_TO_48")
    supported = [row for row in queries if row.get("support_gate_truth") == "SUPPORTED"]
    unsupported = [row for row in queries if row.get("support_gate_truth") == "INSUFFICIENT"]
    if len(supported) < 18 or len(unsupported) < 15 or len(supported) + len(unsupported) != len(queries):
        raise ValueError("HOLDOUT_D_SUPPORT_DISTRIBUTION_INVALID")
    manufacturers = {item["manufacturer"] for item in manifest["documents"]}
    contract_tags: set[str] = set()
    value_kinds: set[str] = set()
    for row in queries:
        required = {
            "support_gate_truth", "manufacturer", "requirement_category",
            "confidence", "contract_tags", "expected_evidence", "annotation_rationale",
            "failure_class", "location_expectation",
        }
        if not required.issubset(row) or not _requested_complete(row):
            raise ValueError(f"HOLDOUT_D_ANNOTATION_INCOMPLETE:{row.get('query_id', '')}")
        if row["support_gate_truth"] not in {"SUPPORTED", "INSUFFICIENT"}:
            raise ValueError(f"INVALID_SUPPORT_TRUTH:{row['query_id']}")
        if row["manufacturer"] not in manufacturers:
            raise ValueError(f"UNKNOWN_QUERY_MANUFACTURER:{row['query_id']}")
        if row["confidence"] not in CONFIDENCE_LEVELS:
            raise ValueError(f"INVALID_ANNOTATION_CONFIDENCE:{row['query_id']}")
        if row["failure_class"] not in SUPPORT_FAILURE_TAXONOMY:
            raise ValueError(f"INVALID_SUPPORT_FAILURE_CLASS:{row['query_id']}")
        if not isinstance(row["contract_tags"], list) or not row["contract_tags"]:
            raise ValueError(f"MISSING_CONTRACT_TAGS:{row['query_id']}")
        contract_tags.update(row["contract_tags"])
        value_kinds.update(row["requested"]["value_kind"] if isinstance(row["requested"]["value_kind"], list) else [row["requested"]["value_kind"]])
        _validate_location_annotation(row)
    if not REQUIRED_CONTRACT_TAGS.issubset(contract_tags):
        raise ValueError(f"HOLDOUT_D_CONTRACT_COVERAGE_MISSING:{sorted(REQUIRED_CONTRACT_TAGS - contract_tags)}")
    if not REQUIRED_VALUE_KINDS.issubset(value_kinds):
        raise ValueError(f"HOLDOUT_D_VALUE_KIND_COVERAGE_MISSING:{sorted(REQUIRED_VALUE_KINDS - value_kinds)}")
    if sum("action_procedure" in row["contract_tags"] for row in queries) < 3:
        raise ValueError("HOLDOUT_D_ACTION_COVERAGE_MINIMUM")
    if sum(bool(row.get("semantic_hard_positive")) and row["support_gate_truth"] == "SUPPORTED" for row in queries) < 4:
        raise ValueError("HOLDOUT_D_SEMANTIC_POSITIVE_MINIMUM")
    if sum(bool(row.get("partial_support_negative")) and row["support_gate_truth"] == "INSUFFICIENT" for row in queries) < 5:
        raise ValueError("HOLDOUT_D_PARTIAL_NEGATIVE_MINIMUM")
    if sum(bool(row.get("multi_chunk_positive")) and row["support_gate_truth"] == "SUPPORTED" for row in queries) < 3:
        raise ValueError("HOLDOUT_D_MULTI_CHUNK_POSITIVE_MINIMUM")
    if sum(bool(row.get("cross_scope_negative")) and row["support_gate_truth"] == "INSUFFICIENT" for row in queries) < 2:
        raise ValueError("HOLDOUT_D_CROSS_SCOPE_NEGATIVE_MINIMUM")
    freeze = manifest.get("freeze", {})
    if freeze.get("query_sha256") != query_hash(manifest):
        raise ValueError("HOLDOUT_D_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != annotation_hash(manifest):
        raise ValueError("HOLDOUT_D_ANNOTATION_HASH_MISMATCH")
    if freeze.get("manifest_sha256") != manifest_hash(manifest):
        raise ValueError("HOLDOUT_D_MANIFEST_HASH_MISMATCH")
    return holdout_distribution(manifest)


def load_holdout_manifest(path: Path | None = None) -> dict[str, Any]:
    path = ensure_private_path(path or HOLDOUT_MANIFEST_PATH)
    manifest = load_private_manifest(path)
    validate_holdout_manifest(manifest)
    return manifest


def freeze_holdout_manifest(path: Path | None = None) -> dict[str, Any]:
    """Seal a reviewed draft once; a sealed manifest is never rewritten."""
    path = ensure_private_path(path or HOLDOUT_MANIFEST_PATH)
    manifest = read_json(path)
    existing = manifest.get("freeze", {})
    if any(existing.get(field) for field in ("query_sha256", "annotation_sha256", "manifest_sha256")):
        raise RuntimeError("HOLDOUT_D_ALREADY_FROZEN")
    manifest["freeze"] = {
        "query_sha256": query_hash(manifest),
        "annotation_sha256": annotation_hash(manifest),
        "manifest_sha256": manifest_hash(manifest),
    }
    atomic_write_json(path, manifest)
    validate_holdout_manifest(manifest)
    return manifest["freeze"]


def holdout_distribution(manifest: dict[str, Any]) -> dict[str, Any]:
    queries = manifest["queries"]
    return {
        "queries": len(queries),
        "support": dict(sorted(Counter(row["support_gate_truth"] for row in queries).items())),
        "manufacturer": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "category": dict(sorted(Counter(row["requirement_category"] for row in queries).items())),
        "confidence": dict(sorted(Counter(row["confidence"] for row in queries).items())),
        "action_count": sum("action_procedure" in row["contract_tags"] for row in queries),
        "semantic_hard_positive_count": sum(bool(row.get("semantic_hard_positive")) for row in queries),
        "multi_chunk_positive_count": sum(bool(row.get("multi_chunk_positive")) for row in queries),
        "cross_scope_negative_count": sum(bool(row.get("cross_scope_negative")) for row in queries),
    }


def parser_audit_samples(documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for document in documents:
        grouped[str(document.metadata["document_id"])].append(document)
    report: dict[str, list[dict[str, Any]]] = {}
    for document_id, rows in grouped.items():
        indexes = sorted({0, len(rows) // 7, (len(rows) * 2) // 7, len(rows) // 2, (len(rows) * 5) // 7, len(rows) - 1})
        report[document_id] = [{
            "chunk_id": str(rows[index].metadata.get("chunk_id", "")),
            "page": rows[index].metadata.get("page"),
            "section": rows[index].metadata.get("section", ""),
            "subsection": rows[index].metadata.get("subsection", ""),
            "manufacturer": rows[index].metadata.get("manufacturer", ""),
            "equipment_model": rows[index].metadata.get("equipment_model", ""),
            "has_parameter_like_text": any(token in rows[index].page_content.lower() for token in ("parameter", "ip", "index", "voltage", "address")),
            "has_procedure_like_text": any(token in rows[index].page_content.lower() for token in ("must", "install", "connect", "configure", "wire")),
            "has_protocol_like_text": any(token in rows[index].page_content.lower() for token in ("ethernet", "profinet", "tcp", "ip", "protocol")),
            "has_safety_like_text": any(token in rows[index].page_content.lower() for token in ("warning", "caution", "safety", "danger")),
            "has_value_or_unit": any(token in rows[index].page_content.lower() for token in ("v", "mm", "ms", "maximum", "minimum", "rated")),
        } for index in indexes]
    return report


def audit_corpus_source(path: Path | None = None) -> dict[str, Any]:
    """Use the production PDF loader and industrial chunker before annotation."""
    _assert_rule_identity()
    path = ensure_private_path(path or SOURCE_MANIFEST_PATH)
    source = read_json(path)
    validate_source_manifest(source)
    documents, production_audit = ingest_private_documents(path, source)
    report = {
        "corpus_id": "D",
        "production_ingestion": "YES",
        "documents": len(source["documents"]),
        "manufacturers": sorted({row["manufacturer"] for row in source["documents"]}),
        "production_ingestion_audit": production_audit,
        "representative_samples": parser_audit_samples(documents),
    }
    atomic_write_json(RUNTIME_ROOT / "corpus_d_parser_audit.json", report)
    return report


def _p2_rows_live(manifest: dict[str, Any], documents: list[Any]) -> list[dict[str, Any]]:
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))
    with full_vector_knowledge_base(documents):
        return _run_mode(
            P2["mode"], manifest["queries"], None, None, reranker,
            candidate_k=P2["candidate_k"], section_strategy=P2["section_strategy"], summarize=False,
        )["rows"]


def export_holdout_artifact(output_path: Path, artifact_id: str = "v317-frozen-d-v1") -> dict[str, Any]:
    """The only permitted Corpus D live retrieval; seals immutable artifact v1."""
    _assert_rule_identity()
    output_path = ensure_private_path(output_path)
    if output_path.exists():
        raise FileExistsError(f"IMMUTABLE_ARTIFACT_EXISTS:{output_path}")
    manifest = load_holdout_manifest()
    documents, production_audit = ingest_private_documents(HOLDOUT_MANIFEST_PATH, manifest)
    parser_audit = {"production_ingestion_audit": production_audit, "representative_samples": parser_audit_samples(documents)}
    by_chunk = {str(item.metadata["chunk_id"]): item for item in documents}
    started = time.perf_counter()
    p2_rows = _p2_rows_live(manifest, documents)
    p2_by_id = {row["query_id"]: row for row in p2_rows}
    from backend import rag_core
    from backend.evaluation.full_vector_benchmark import FULL_BENCHMARK_KNOWLEDGE_BASE_ID

    evidence_by_id: dict[str, dict[str, Any]] = {}
    with full_vector_knowledge_base(documents):
        for query in manifest["queries"]:
            result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode="hybrid")
            evidence_by_id[query["query_id"]] = _capture_evidence_input(query, result)["evidence_input"]
    artifact_queries = []
    for query in manifest["queries"]:
        p2 = p2_by_id[query["query_id"]]
        final_context = [candidate_from_p2_row(candidate, by_chunk[str(candidate["chunk_id"])]) for candidate in p2["candidates"]]
        artifact_queries.append({
            "query_id": query["query_id"], "query": query["query"], "query_text_hash": hash_json(query["query"]),
            "ground_truth": query, "evidence_input": evidence_by_id[query["query_id"]], "final_context": final_context,
            "retrieval_decision_inputs": {"retrieval_mode": "hybrid", "source_p2_retrieval_scope": p2.get("retrieval_scope", {}), "source_p2_section_retrieval": p2.get("section_retrieval")},
        })
    payload = new_artifact_payload(
        artifact_id=artifact_id, corpus_id="D", manifest_hash=hash_json(manifest), annotation_hash=annotation_hash(manifest),
        retrieval_config=retrieval_configuration(), queries=artifact_queries, snapshot_documents=_snapshot_documents(documents),
        source={
            "p2_retrieved_live_once": True,
            "evidence_input_retrieved_live_once": True,
            "live_retrieval_elapsed_seconds": time.perf_counter() - started,
            "parser_audit": parser_audit,
            "document_sha256": {item["document_id"]: file_sha256(CORPUS_D_ROOT / item["file"]) for item in manifest["documents"]},
        },
        rule_version=EVIDENCE_SUPPORT_RULE_VERSION,
    )
    write_immutable_artifact(output_path, payload)
    report = validate_artifact(read_json(output_path))
    if report["validity"] != "VALID":
        raise RuntimeError(f"ARTIFACT_NOT_VALID:{report}")
    result = {**report, "path": str(output_path), "bytes": output_path.stat().st_size, "parser_audit": parser_audit, "p2_rows": p2_rows}
    atomic_write_json(RUNTIME_ROOT / "corpus_d_export.json", result)
    return result


def replay_holdout_artifact(path: Path, run_id: str = "v317-replay-d") -> dict[str, Any]:
    _assert_rule_identity()
    return replay_artifact(ensure_private_path(path), run_id, expected_manifest=load_holdout_manifest())


def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    false_support = [row["query_id"] for row in unsupported if row["predicted_supported"]]
    false_insufficient = [row["query_id"] for row in supported if not row["predicted_supported"]]
    return {
        "query_count": len(rows), "supported": len(supported), "unsupported": len(unsupported),
        "false_support": false_support, "false_insufficient": false_insufficient,
        "false_support_rate": len(false_support) / len(unsupported) if unsupported else None,
        "false_insufficient_rate": len(false_insufficient) / len(supported) if supported else None,
    }


def confidence_metrics(replay: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    rows = [{**row, "annotation": annotations[row["query_id"]]} for row in replay["rows"]]
    return {
        "ALL": {"support": replay["metrics"]["support"], "counts": _slice(rows)},
        "HIGH_ONLY": {"counts": _slice([row for row in rows if row["annotation"]["confidence"] == "HIGH"])},
        "HIGH_MEDIUM": {"counts": _slice([row for row in rows if row["annotation"]["confidence"] in {"HIGH", "MEDIUM"}])},
        "AMBIGUOUS": [row["query_id"] for row in rows if row["annotation"]["confidence"] == "AMBIGUOUS"],
    }


def _support_failure_class(row: dict[str, Any]) -> str:
    annotation = row["annotation"]
    if row["expected_supported"] is False and row["predicted_supported"]:
        return "PARTIAL_SUPPORT_ACCEPTED" if annotation.get("partial_support_negative") else "OTHER"
    reason = str(row.get("support", {}).get("reason", ""))
    return {
        "MISSING_ATTRIBUTE_SUPPORT": "OVER_CONSTRAINED_ATTRIBUTE",
        "MISSING_VALUE_SUPPORT": "OVER_CONSTRAINED_VALUE",
        "MISSING_REQUIRED_CONCEPT": "OVER_CONSTRAINED_REQUIREMENT",
        "MISSING_QUALIFIER_SUPPORT": "OVER_CONSTRAINED_QUALIFIER",
        "MISSING_UNIT_SUPPORT": "UNIT_FAILURE",
        "MISSING_ACTION_SUPPORT": "ACTION_FAILURE",
        "MISSING_LOCATION_SUPPORT": "LOCATION_FAILURE",
    }.get(reason, annotation.get("failure_class", "OTHER"))


def failure_attribution(
    replay: dict[str, Any], manifest: dict[str, Any], artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    artifact_rows = {row["query_id"]: row for row in replay["rows"]}
    parser_issues = {
        document_id: item.get("issues", [])
        for document_id, item in artifact.get("source", {}).get("parser_audit", {}).get("production_ingestion_audit", {}).items()
    }
    failures = []
    for query_id, row in artifact_rows.items():
        if row["predicted_supported"] == row["expected_supported"]:
            continue
        annotation = annotations[query_id]
        final_context = set(row.get("candidate_ids", {}).get("final_context", []))
        relevant_document_issues = [
            issue
            for document_id in annotation.get("relevant_document_ids", [])
            for issue in parser_issues.get(document_id, [])
        ]
        if relevant_document_issues:
            attribution = "PARSER_METADATA_FAILURE"
        elif annotation["confidence"] == "AMBIGUOUS":
            attribution = "ANNOTATION_AMBIGUITY"
        elif row["expected_supported"] and not final_context.intersection(annotation["relevant_chunk_ids"]):
            attribution = "RETRIEVAL_MISSING_EVIDENCE"
        elif row["expected_supported"] and row["base_decision"] == "ABSTAIN":
            attribution = "EVIDENCE_RULE_FAILURE"
        else:
            attribution = "SUPPORT_RULE_FAILURE"
        failures.append({
            "query_id": query_id, "attribution": attribution,
            "support_failure_class": _support_failure_class({**row, "annotation": annotation}),
            "expected": annotation["support_gate_truth"],
            "predicted": "SUPPORTED" if row["predicted_supported"] else "INSUFFICIENT",
            "confidence": annotation["confidence"],
            "post_freeze_status": "POST_FREEZE_DISCOVERED_ISSUE",
        })
    return failures


def holdout_analysis(replay: dict[str, Any], manifest: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    rows = [{**row, "annotation": annotations[row["query_id"]]} for row in replay["rows"]]
    by_manufacturer = {
        manufacturer: _slice([row for row in rows if row["annotation"]["manufacturer"] == manufacturer])
        for manufacturer in sorted({row["annotation"]["manufacturer"] for row in rows})
    }
    by_category = {
        category: _slice([row for row in rows if row["annotation"]["requirement_category"] == category])
        for category in sorted({row["annotation"]["requirement_category"] for row in rows})
    }
    query_rows = {row["query_id"]: row for row in artifact["queries"]}
    positives = [row for row in manifest["queries"] if row["answerable"]]
    ranks = []
    model_confusion = []
    identifier_hits = []
    for query in positives:
        context = query_rows[query["query_id"]]["final_context"]
        rank = next((index for index, item in enumerate(context, 1) if item["chunk_id"] in query["relevant_chunk_ids"]), None)
        ranks.append(rank)
        if query["requested"]["identity"] and context:
            model_confusion.append(context[0]["metadata"].get("equipment_model") != query["expected_model"])
        if query["requested"]["identifier"]:
            identifier_hits.append(rank is not None and rank <= 5)
    selected = lambda marker: [row for row in rows if row["annotation"].get(marker)]
    failures = failure_attribution(replay, manifest, artifact)
    attribution_categories = (
        "SUPPORT_RULE_FAILURE", "EVIDENCE_RULE_FAILURE", "RETRIEVAL_MISSING_EVIDENCE",
        "PARSER_METADATA_FAILURE", "ANNOTATION_AMBIGUITY",
    )
    return {
        "status": "READY" if not any(item["attribution"] == "SUPPORT_RULE_FAILURE" for item in failures) else "PARTIAL",
        "holdout": holdout_distribution(manifest),
        "artifact": validate_artifact(artifact),
        "retrieval": {
            "hit_at_1": sum(rank == 1 for rank in ranks) / len(positives),
            "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(positives),
            "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(positives),
            "model_confusion_rate": sum(model_confusion) / len(model_confusion) if model_confusion else None,
            "identifier_recall_at_5": sum(identifier_hits) / len(identifier_hits) if identifier_hits else None,
        },
        "evidence": replay["metrics"]["evidence"],
        "support": replay["metrics"]["support"],
        "confidence_metrics": confidence_metrics(replay, manifest),
        "failure_attribution": failures,
        "failure_attribution_counts": {
            category: sum(item["attribution"] == category for item in failures)
            for category in attribution_categories
        },
        "support_failure_taxonomy": dict(sorted(Counter(item["support_failure_class"] for item in failures).items())),
        "by_manufacturer": by_manufacturer,
        "by_category": by_category,
        "semantic_cases": [{"query_id": row["query_id"], "expected": row["expected_supported"], "predicted": row["predicted_supported"]} for row in selected("semantic_hard_positive")],
        "action_cases": [{"query_id": row["query_id"], "expected": row["expected_supported"], "predicted": row["predicted_supported"]} for row in rows if "action_procedure" in row["annotation"]["contract_tags"]],
        "multi_chunk_cases": [{"query_id": row["query_id"], "expected": row["expected_supported"], "predicted": row["predicted_supported"]} for row in selected("multi_chunk_positive") + selected("cross_scope_negative")],
    }


def abc_d_matrix(d_replay: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "A": PRIVATE_ROOT / "v312_runtime" / "v316-final-a" / "summary.json",
        "B": PRIVATE_ROOT / "v312_runtime" / "v316-final-b" / "summary.json",
        "C": PRIVATE_ROOT / "v312_runtime" / "v316-final-c" / "summary.json",
    }
    results = {corpus: read_json(path) for corpus, path in paths.items()}
    results["D"] = d_replay
    fields = ("support_accuracy", "supported_recall", "unsupported_recall", "false_support_rate", "false_insufficient_rate")
    matrix = {corpus: {field: result["metrics"]["support"][field] for field in fields} for corpus, result in results.items()}
    return {"matrix": matrix, "generalization_range": generalization_range(matrix)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "freeze", "validate", "export", "replay", "analyze"))
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_ROOT / "v317-frozen-d-v1.json")
    parser.add_argument("--run-id", default="v317-replay-d")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "v317_final_analysis.json")
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit_corpus_source()
    elif args.command == "freeze":
        report = {"freeze": freeze_holdout_manifest()}
    elif args.command == "validate":
        manifest = load_holdout_manifest()
        report = {"validity": "VALID", "distribution": holdout_distribution(manifest), "freeze": manifest["freeze"]}
    elif args.command == "export":
        report = export_holdout_artifact(args.artifact)
    elif args.command == "replay":
        report = replay_holdout_artifact(args.artifact, args.run_id)
    else:
        manifest = load_holdout_manifest()
        artifact = read_json(ensure_private_path(args.artifact))
        replay = read_json(PRIVATE_ROOT / "v312_runtime" / args.run_id / "summary.json")
        report = {**holdout_analysis(replay, manifest, artifact), **abc_d_matrix(replay)}
        atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
