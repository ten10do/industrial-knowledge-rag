"""V3.14 independent Corpus C holdout export, replay, and support analysis.

Only ``export_holdout_artifact`` can access the production ingestion/retrieval
pipeline.  Every result after that boundary is derived from the immutable
artifact through the V3.12 offline replay implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import (
    candidate_from_p2_row, file_sha256, new_artifact_payload, validate_artifact,
    write_immutable_artifact,
)
from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base
from backend.evaluation.private_benchmark import (
    _run_mode, annotation_hash, ingest_private_documents, load_private_manifest,
)
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v310_runner import P2
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import (
    _capture_evidence_input, _snapshot_documents, ensure_private_path,
    replay_artifact, retrieval_configuration,
)
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_C_ROOT = PRIVATE_ROOT / "corpus_c"
RUNTIME_ROOT = PRIVATE_ROOT / "v314_runtime"
ARTIFACT_ROOT = PRIVATE_ROOT / "v314_artifacts"
FROZEN_RULE_IDENTITY = {"evidence": "v311.2", "support": "support-v313.1"}
LOCATION_GROUND_TRUTH_POLICY_V1 = {
    "general_where": "Correct content plus any non-empty section, subsection, or page metadata is sufficient.",
    "specific_section": "A non-empty section or chapter metadata value is required.",
    "specific_page": "A page metadata value is required.",
    "exact_subsection": "Subsection metadata is required only when the query explicitly asks for a subsection.",
    "none": "The query has no location-support requirement.",
}
REQUIREMENT_CATEGORIES = {
    "compatibility", "configuration", "installation", "safety", "maintenance",
    "protocol", "action", "attribute", "value_unit", "location", "semantic",
}
FAILURE_TAXONOMY = {
    "PARTIAL_SUPPORT_ACCEPTED", "OVER_CONSTRAINED_REQUIREMENT", "ACTION_FALSE_MATCH",
    "ATTRIBUTE_FALSE_MATCH", "LOCATION_REQUIREMENT_FAILURE", "VALUE_UNIT_FAILURE",
    "PROTOCOL_FALSE_MATCH", "RETRIEVAL_MISSING_EVIDENCE", "ANNOTATION_AMBIGUITY", "OTHER",
}


def _assert_rule_identity() -> None:
    actual = {"evidence": EVIDENCE_SUPPORT_RULE_VERSION, "support": SUPPORT_RULE_VERSION}
    if actual != FROZEN_RULE_IDENTITY:
        raise RuntimeError(f"FROZEN_RULE_IDENTITY_MISMATCH:{actual}")


def query_hash(manifest: dict[str, Any]) -> str:
    return hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in manifest["queries"]])


def validate_location_annotation(query: dict[str, Any]) -> None:
    expectation = query.get("location_expectation", "none")
    if expectation not in LOCATION_GROUND_TRUTH_POLICY_V1:
        raise ValueError(f"INVALID_LOCATION_EXPECTATION:{query['query_id']}")
    if expectation == "specific_page" and not str(query.get("expected_page", "")):
        raise ValueError(f"LOCATION_PAGE_REQUIRED:{query['query_id']}")
    if expectation == "specific_section" and not str(query.get("expected_section", "")):
        raise ValueError(f"LOCATION_SECTION_REQUIRED:{query['query_id']}")
    if expectation == "exact_subsection" and not str(query.get("expected_subsection", "")):
        raise ValueError(f"LOCATION_SUBSECTION_REQUIRED:{query['query_id']}")


def validate_holdout_manifest(manifest: dict[str, Any]) -> None:
    """Validate C before its first benchmark, without changing any rule."""
    _assert_rule_identity()
    count = len(manifest["queries"])
    if not 30 <= count <= 40:
        raise ValueError("HOLDOUT_QUERY_COUNT_MUST_BE_30_TO_40")
    if len(manifest["documents"]) < 3:
        raise ValueError("HOLDOUT_REQUIRES_THREE_DOCUMENTS")
    manufacturers = {item["manufacturer"] for item in manifest["documents"]}
    if len(manufacturers) < 2:
        raise ValueError("HOLDOUT_REQUIRES_TWO_MANUFACTURERS")
    supported = [row for row in manifest["queries"] if row.get("support_gate_truth") == "SUPPORTED"]
    unsupported = [row for row in manifest["queries"] if row.get("support_gate_truth") == "INSUFFICIENT"]
    if len(supported) < 15 or len(unsupported) < 12 or len(supported) + len(unsupported) != count:
        raise ValueError("HOLDOUT_SUPPORT_DISTRIBUTION_INVALID")
    for row in manifest["queries"]:
        required = {"support_gate_truth", "requirement_category", "failure_class", "manufacturer", "location_expectation"}
        if not required.issubset(row):
            raise ValueError(f"HOLDOUT_ANNOTATION_INCOMPLETE:{row.get('query_id', '')}")
        if row["support_gate_truth"] not in {"SUPPORTED", "INSUFFICIENT"}:
            raise ValueError(f"INVALID_SUPPORT_TRUTH:{row['query_id']}")
        if row["requirement_category"] not in REQUIREMENT_CATEGORIES:
            raise ValueError(f"INVALID_REQUIREMENT_CATEGORY:{row['query_id']}")
        if row["failure_class"] not in FAILURE_TAXONOMY:
            raise ValueError(f"INVALID_FAILURE_CLASS:{row['query_id']}")
        if row["manufacturer"] not in manufacturers:
            raise ValueError(f"UNKNOWN_QUERY_MANUFACTURER:{row['query_id']}")
        validate_location_annotation(row)
    freeze = manifest.get("freeze", {})
    if freeze.get("query_sha256") != query_hash(manifest):
        raise ValueError("HOLDOUT_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != annotation_hash(manifest):
        raise ValueError("HOLDOUT_ANNOTATION_HASH_MISMATCH")


def load_holdout_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or CORPUS_C_ROOT / "manifest.json"
    manifest = load_private_manifest(ensure_private_path(path))
    validate_holdout_manifest(manifest)
    return manifest


def parser_audit_samples(documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
    """Record representative parsed chunks; this observes but never repairs parsing."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    for document in documents:
        grouped[str(document.metadata["document_id"])].append(document)
    report: dict[str, list[dict[str, Any]]] = {}
    for document_id, rows in grouped.items():
        indexes = sorted({0, len(rows) // 6, len(rows) // 3, len(rows) // 2, (len(rows) * 2) // 3, (len(rows) * 5) // 6, len(rows) - 1})
        report[document_id] = [{
            "chunk_id": str(rows[index].metadata.get("chunk_id", "")),
            "page": rows[index].metadata.get("page"),
            "section": rows[index].metadata.get("section", ""),
            "subsection": rows[index].metadata.get("subsection", ""),
            "equipment_model": rows[index].metadata.get("equipment_model", ""),
            "has_parameter_like_text": any(token in rows[index].page_content.lower() for token in ("parameter", "ip", "index", "voltage")),
            "has_procedure_like_text": any(token in rows[index].page_content.lower() for token in ("must", "install", "connect", "configure", "\u8bbe\u7f6e", "\u5fc5\u987b")),
            "has_safety_like_text": any(token in rows[index].page_content.lower() for token in ("warning", "caution", "safety", "\u5371\u9669", "\u8b66\u544a")),
        } for index in indexes]
    return report


def _p2_rows_live(manifest: dict[str, Any], documents: list[Any]) -> list[dict[str, Any]]:
    """The sole C P2 retrieval pass, using its frozen A/B configuration exactly."""
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))
    with full_vector_knowledge_base(documents):
        return _run_mode(
            P2["mode"], manifest["queries"], None, None, reranker,
            candidate_k=P2["candidate_k"], section_strategy=P2["section_strategy"], summarize=False,
        )["rows"]


def export_holdout_artifact(output_path: Path, artifact_id: str = "v314-frozen-c") -> dict[str, Any]:
    """Perform the one permitted live C retrieval then immediately seal it."""
    _assert_rule_identity()
    output_path = ensure_private_path(output_path)
    if output_path.exists():
        raise FileExistsError(f"IMMUTABLE_ARTIFACT_EXISTS:{output_path}")
    manifest = load_holdout_manifest()
    documents, production_audit = ingest_private_documents(CORPUS_C_ROOT / "manifest.json", manifest)
    audit = {"production_ingestion_audit": production_audit, "representative_samples": parser_audit_samples(documents)}
    by_chunk = {str(item.metadata["chunk_id"]): item for item in documents}
    started = time.perf_counter()
    p2_rows = _p2_rows_live(manifest, documents)
    p2_by_id = {row["query_id"]: row for row in p2_rows}
    from backend import rag_core
    from backend.evaluation.full_vector_benchmark import FULL_BENCHMARK_KNOWLEDGE_BASE_ID
    evidence_by_id: dict[str, dict[str, Any]] = {}
    # This is still the pre-artifact live capture; replay never reaches this code path.
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
    config = retrieval_configuration()
    payload = new_artifact_payload(
        artifact_id=artifact_id, corpus_id="C", manifest_hash=hash_json(manifest), annotation_hash=annotation_hash(manifest),
        retrieval_config=config, queries=artifact_queries, snapshot_documents=_snapshot_documents(documents),
        source={"p2_retrieved_live_once": True, "evidence_input_retrieved_live_once": True,
                "live_retrieval_elapsed_seconds": time.perf_counter() - started, "parser_audit": audit,
                "document_sha256": {item["document_id"]: file_sha256(CORPUS_C_ROOT / item["file"]) for item in manifest["documents"]}},
        rule_version=EVIDENCE_SUPPORT_RULE_VERSION,
    )
    write_immutable_artifact(output_path, payload)
    report = validate_artifact(read_json(output_path))
    if report["validity"] != "VALID":
        raise RuntimeError(f"ARTIFACT_NOT_VALID:{report}")
    atomic_write_json(RUNTIME_ROOT / "corpus_c_export.json", {**report, "artifact": str(output_path), "parser_audit": audit, "p2_rows": p2_rows})
    return {**report, "path": str(output_path), "bytes": output_path.stat().st_size, "p2_rows": p2_rows, "parser_audit": audit}


def replay_holdout_artifact(path: Path, run_id: str = "v314-replay-c") -> dict[str, Any]:
    """Offline C replay: no BM25, Chroma, CrossEncoder, or PDF parser imports here."""
    _assert_rule_identity()
    manifest = load_holdout_manifest()
    return replay_artifact(ensure_private_path(path), run_id, expected_manifest=manifest)


def retrieval_metrics(queries: list[dict[str, Any]], p2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["query_id"]: row for row in p2_rows}
    answerable = [row for row in queries if row["answerable"]]
    ranks = [by_id[row["query_id"]].get("rank") for row in answerable]
    model_queries = [row for row in answerable if row.get("expected_model")]
    confusions = []
    for query in model_queries:
        top = (by_id[query["query_id"]].get("candidates") or [{}])[0]
        if top.get("equipment_model") and top.get("equipment_model") != query["expected_model"]:
            confusions.append(query["query_id"])
    return {
        "hit_at_1": sum(rank == 1 for rank in ranks) / len(answerable),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(answerable),
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(answerable),
        "model_confusion": {"count": len(confusions), "rate": len(confusions) / len(model_queries), "query_ids": confusions},
        "identifier_recall_at_5": sum(by_id[row["query_id"]].get("rank") is not None for row in model_queries) / len(model_queries),
    }


def _support_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    false_support = [row for row in unsupported if row["predicted_supported"]]
    false_insufficient = [row for row in supported if not row["predicted_supported"]]
    return {"count": len(rows), "supported_count": len(supported), "unsupported_count": len(unsupported),
            "false_support": [row["query_id"] for row in false_support], "false_insufficient": [row["query_id"] for row in false_insufficient]}


def support_analysis(replay: dict[str, Any], manifest: dict[str, Any], p2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = {row["query_id"]: row for row in manifest["queries"]}
    p2 = {row["query_id"]: row for row in p2_rows}
    rows = [{**row, **{"annotation": truth[row["query_id"]]}} for row in replay["rows"]]
    categories = {category: _support_slice([row for row in rows if row["annotation"]["requirement_category"] == category]) for category in sorted(REQUIREMENT_CATEGORIES)}
    manufacturers = {manufacturer: _support_slice([row for row in rows if row["annotation"]["manufacturer"] == manufacturer]) for manufacturer in sorted({row["annotation"]["manufacturer"] for row in rows})}
    failures = []
    for row in rows:
        wrong = row["predicted_supported"] != row["expected_supported"]
        if not wrong:
            continue
        annotation = row["annotation"]
        final_ids = set(p2[row["query_id"]].get("candidate_ids", []))
        failure_class = "RETRIEVAL_MISSING_EVIDENCE" if annotation["support_gate_truth"] == "SUPPORTED" and not final_ids.intersection(annotation["relevant_chunk_ids"]) else annotation["failure_class"]
        failures.append({"query_id": row["query_id"], "failure_class": failure_class, "expected": annotation["support_gate_truth"], "predicted": "SUPPORTED" if row["predicted_supported"] else "INSUFFICIENT", "manufacturer": annotation["manufacturer"], "requirement_category": annotation["requirement_category"], "location_expectation": annotation["location_expectation"]})
    return {"precision_safety_view": {"unsupported_recall": replay["metrics"]["support"]["unsupported_recall"], "false_support_rate": replay["metrics"]["support"]["false_support_rate"], "supported_recall": replay["metrics"]["support"]["supported_recall"], "false_insufficient_rate": replay["metrics"]["support"]["false_insufficient_rate"]}, "by_requirement_category": categories, "by_manufacturer": manufacturers, "failures": failures, "over_constrained_failures": [row for row in failures if row["failure_class"] == "OVER_CONSTRAINED_REQUIREMENT"], "partial_support_failures": [row for row in failures if row["failure_class"] == "PARTIAL_SUPPORT_ACCEPTED"], "location_cases": [{"query_id": row["query_id"], "expectation": row["annotation"]["location_expectation"], "expected": row["annotation"]["support_gate_truth"], "predicted": "SUPPORTED" if row["predicted_supported"] else "INSUFFICIENT"} for row in rows if row["annotation"]["location_expectation"] != "none"]}


def support_matrix(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = ("support_accuracy", "supported_recall", "unsupported_recall", "false_support_rate", "false_insufficient_rate")
    return {corpus: {field: result["metrics"]["support"][field] for field in fields} for corpus, result in results.items()}


def generalization_range(matrix: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {field: {"min": min(values), "max": max(values), "range": max(values) - min(values)} for field in next(iter(matrix.values())) for values in [[row[field] for row in matrix.values()]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "export", "replay"))
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_ROOT / "v314-frozen-c-20260814.json")
    parser.add_argument("--run-id", default="v314-replay-c")
    args = parser.parse_args(argv)
    if args.command == "validate":
        manifest = load_holdout_manifest()
        print(json.dumps({"validity": "VALID", "queries": len(manifest["queries"]), "freeze": manifest["freeze"]}, ensure_ascii=False))
    elif args.command == "export":
        report = export_holdout_artifact(args.artifact)
        print(json.dumps({key: report[key] for key in ("validity", "query_count", "path", "bytes")}, ensure_ascii=False))
    else:
        report = replay_holdout_artifact(args.artifact, args.run_id)
        print(json.dumps({"validity": report["validity"], "output": report["output_path"], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
