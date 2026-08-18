"""V3.19 independent Corpus E Evidence validation.

This is an evaluation harness only.  It deliberately has two separate
boundaries: one production parser audit before the manifest is frozen and one
live retrieval export after it is frozen.  All subsequent work replays the
immutable artifact and therefore cannot touch retrieval, a parser, or rules.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.evaluation.frozen_retrieval_artifact import (
    candidate_from_p2_row, file_sha256, new_artifact_payload,
    validate_artifact, write_immutable_artifact,
)
from backend.evaluation.full_vector_benchmark import full_vector_knowledge_base
from backend.evaluation.private_benchmark import (
    _run_mode, annotation_hash, ingest_private_documents, load_private_manifest,
)
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v310_runner import P2
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import (
    _capture_evidence_input, _snapshot_documents, compare_results,
    ensure_private_path, replay_artifact, retrieval_configuration,
)
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_ROOT = PRIVATE_ROOT / "corpus_e"
SOURCE_MANIFEST_PATH = CORPUS_ROOT / "source_manifest.json"
HOLDOUT_MANIFEST_PATH = CORPUS_ROOT / "holdout_manifest_v3.json"
PARSED_CHUNKS_PATH = CORPUS_ROOT / "production_chunks.json"
RUNTIME_ROOT = PRIVATE_ROOT / "v319_runtime"
ARTIFACT_ROOT = PRIVATE_ROOT / "v319_artifacts"
RESULT_ROOT = PRIVATE_ROOT / "v319_results"
FROZEN_RULE_IDENTITY = {"evidence": "evidence-v318.1", "support": "support-v316.1"}
EXPECTED_RETRIEVAL_CONFIG_HASH = "d2980ebbb7e40c5b7fdc8df9a50f140f032d00bcaf98251c570ae05f3e94590c"
CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
REQUESTED_FIELDS = frozenset({
    "identity", "identifier", "protocol", "action", "attribute", "value",
    "unit", "value_kind", "requirement_type", "qualifier",
})
EVIDENCE_FAILURE_TAXONOMY = frozenset({
    "IDENTITY_ONLY_FALSE_ANSWER", "IDENTIFIER_EXISTENCE_TOO_BROAD",
    "PROTOCOL_TOPIC_OVERMATCH", "ATTRIBUTE_NOT_SUPPORTED", "VALUE_NOT_SUPPORTED",
    "REQUIREMENT_NOT_SUPPORTED", "ACTION_NOT_SUPPORTED", "PARTIAL_EVIDENCE_ACCEPTED",
    "SEMANTIC_TOPIC_ONLY_MATCH", "CROSS_CHUNK_SCOPE_LEAK",
    "OVER_CONSTRAINED_ATTRIBUTE", "OVER_CONSTRAINED_VALUE",
    "OVER_CONSTRAINED_REQUIREMENT", "OVER_CONSTRAINED_QUALIFIER",
    "SEMANTIC_EQUIVALENCE_MISSED", "MULTI_CHUNK_AGGREGATION_MISSED",
    "LOCAL_ASSOCIATION_TOO_STRICT", "OTHER",
})


def _assert_rule_identity() -> None:
    actual = {"evidence": EVIDENCE_SUPPORT_RULE_VERSION, "support": SUPPORT_RULE_VERSION}
    if actual != FROZEN_RULE_IDENTITY:
        raise RuntimeError(f"FROZEN_RULE_IDENTITY_MISMATCH:{actual}")


def query_hash(manifest: dict[str, Any]) -> str:
    return hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in manifest["queries"]])


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hash_json({key: value for key, value in manifest.items() if key != "freeze"})


def validate_source_manifest(source: dict[str, Any]) -> None:
    documents = source.get("documents")
    if not isinstance(documents, list) or len(documents) < 3:
        raise ValueError("CORPUS_E_REQUIRES_THREE_DOCUMENTS")
    manufacturers = {str(item.get("manufacturer", "")) for item in documents}
    if len(manufacturers - {""}) < 2:
        raise ValueError("CORPUS_E_REQUIRES_TWO_MANUFACTURERS")
    for item in documents:
        if item.get("source_type") != "official_vendor_publication":
            raise ValueError(f"CORPUS_E_NON_OFFICIAL_SOURCE:{item.get('document_id', '')}")
        if not str(item.get("official_url", "")).startswith("https://"):
            raise ValueError(f"CORPUS_E_OFFICIAL_URL_REQUIRED:{item.get('document_id', '')}")
        if item.get("commit_allowed") is not False:
            raise ValueError(f"CORPUS_E_PRIVATE_DOCUMENT_REQUIRED:{item.get('document_id', '')}")


def _requested_complete(row: dict[str, Any]) -> bool:
    return isinstance(row.get("requested"), dict) and REQUESTED_FIELDS.issubset(row["requested"])


def holdout_distribution(manifest: dict[str, Any]) -> dict[str, Any]:
    queries = manifest["queries"]
    return {
        "queries": len(queries),
        "answerable": sum(bool(row["answerable"]) for row in queries),
        "abstain": sum(not bool(row["answerable"]) for row in queries),
        "manufacturer": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "category": dict(sorted(Counter(row["category"] for row in queries).items())),
        "confidence": dict(sorted(Counter(row["confidence"] for row in queries).items())),
        "semantic_hard_positive": sum(bool(row.get("semantic_hard_positive")) for row in queries),
        "multi_chunk_positive": sum(bool(row.get("multi_chunk_positive")) for row in queries),
        "cross_chunk_negative": sum(bool(row.get("cross_chunk_negative")) for row in queries),
        "hard_negative": sum(bool(row.get("hard_negative")) for row in queries),
        "ood": sum(bool(row.get("ood")) for row in queries),
    }


def validate_holdout_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    _assert_rule_identity()
    documents, queries = manifest.get("documents", []), manifest.get("queries", [])
    validate_source_manifest({"documents": documents})
    if not 40 <= len(queries) <= 50:
        raise ValueError("HOLDOUT_E_QUERY_COUNT_MUST_BE_40_TO_50")
    answerable = [row for row in queries if row.get("answerable")]
    abstain = [row for row in queries if not row.get("answerable")]
    if not 20 <= len(answerable) <= 25 or not 18 <= len(abstain) <= 22:
        raise ValueError("HOLDOUT_E_ANSWER_ABSTAIN_DISTRIBUTION_INVALID")
    manufacturers = {row["manufacturer"] for row in documents}
    ids: set[str] = set()
    for row in queries:
        required = {"query_id", "query", "answerable", "manufacturer", "confidence", "requested",
                    "expected_evidence", "annotation_rationale", "failure_class", "support_gate_truth"}
        if not required.issubset(row) or not _requested_complete(row):
            raise ValueError(f"HOLDOUT_E_ANNOTATION_INCOMPLETE:{row.get('query_id', '')}")
        if row["query_id"] in ids or row["manufacturer"] not in manufacturers:
            raise ValueError(f"HOLDOUT_E_IDENTITY_INVALID:{row.get('query_id', '')}")
        if row["confidence"] not in CONFIDENCE_LEVELS or row["failure_class"] not in EVIDENCE_FAILURE_TAXONOMY:
            raise ValueError(f"HOLDOUT_E_ANNOTATION_VALUE_INVALID:{row['query_id']}")
        if row["answerable"] != bool(row.get("relevant_chunk_ids")):
            raise ValueError(f"HOLDOUT_E_RELEVANT_CHUNK_CONTRACT:{row['query_id']}")
        if row["support_gate_truth"] != ("SUPPORTED" if row["answerable"] else "INSUFFICIENT"):
            raise ValueError(f"HOLDOUT_E_SUPPORT_TRUTH_INVALID:{row['query_id']}")
        ids.add(row["query_id"])
    distribution = holdout_distribution(manifest)
    if distribution["confidence"].get("HIGH", 0) / len(queries) < .80:
        raise ValueError("HOLDOUT_E_HIGH_CONFIDENCE_MINIMUM")
    if distribution["semantic_hard_positive"] < 6 or distribution["multi_chunk_positive"] < 4:
        raise ValueError("HOLDOUT_E_UTILITY_COVERAGE_MINIMUM")
    if distribution["cross_chunk_negative"] < 3 or distribution["ood"] not in {8, 9, 10}:
        raise ValueError("HOLDOUT_E_SAFETY_COVERAGE_MINIMUM")
    freeze = manifest.get("freeze", {})
    if freeze.get("query_sha256") != query_hash(manifest):
        raise ValueError("HOLDOUT_E_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != annotation_hash(manifest):
        raise ValueError("HOLDOUT_E_ANNOTATION_HASH_MISMATCH")
    if freeze.get("manifest_sha256") != manifest_hash(manifest):
        raise ValueError("HOLDOUT_E_MANIFEST_HASH_MISMATCH")
    return distribution


def load_holdout_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest = load_private_manifest(ensure_private_path(path or HOLDOUT_MANIFEST_PATH))
    validate_holdout_manifest(manifest)
    return manifest


def freeze_holdout_manifest(path: Path | None = None) -> dict[str, str]:
    path = ensure_private_path(path or HOLDOUT_MANIFEST_PATH)
    manifest = read_json(path)
    if any(manifest.get("freeze", {}).get(key) for key in ("query_sha256", "annotation_sha256", "manifest_sha256")):
        raise RuntimeError("HOLDOUT_E_ALREADY_FROZEN")
    manifest["freeze"] = {"query_sha256": query_hash(manifest), "annotation_sha256": annotation_hash(manifest)}
    manifest["freeze"]["manifest_sha256"] = manifest_hash(manifest)
    atomic_write_json(path, manifest)
    validate_holdout_manifest(manifest)
    return manifest["freeze"]


def parser_audit_samples(documents: list[Document]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        groups[str(document.metadata["document_id"])].append(document)
    return {document_id: [
        {"chunk_id": str(rows[index].metadata["chunk_id"]), "page": rows[index].metadata.get("page"),
         "section": rows[index].metadata.get("section", ""), "manufacturer": rows[index].metadata.get("manufacturer", ""),
         "equipment_model": rows[index].metadata.get("equipment_model", ""),
         "sample_text": " ".join(rows[index].page_content.split())[:500]}
        for index in sorted({0, len(rows)//9, (len(rows)*2)//9, (len(rows)*3)//9, len(rows)//2, (len(rows)*6)//9, (len(rows)*7)//9, (len(rows)*8)//9, len(rows)-1})]
        for document_id, rows in groups.items()}


def audit_corpus_source() -> dict[str, Any]:
    """The single pre-freeze production PDF ingestion, persisted for annotation/export."""
    _assert_rule_identity()
    if PARSED_CHUNKS_PATH.exists():
        raise FileExistsError(f"PRODUCTION_INGESTION_ALREADY_RECORDED:{PARSED_CHUNKS_PATH}")
    source = read_json(ensure_private_path(SOURCE_MANIFEST_PATH))
    validate_source_manifest(source)
    documents, ingestion_audit = ingest_private_documents(SOURCE_MANIFEST_PATH, source)
    serialized = [{"content": document.page_content, "metadata": dict(document.metadata)} for document in documents]
    atomic_write_json(PARSED_CHUNKS_PATH, {"documents": serialized, "source_manifest_hash": hash_json(source)})
    report = {"corpus_id": "E", "production_ingestion": "YES", "documents": len(source["documents"]),
              "manufacturers": sorted({row["manufacturer"] for row in source["documents"]}),
              "chunks": len(documents), "production_ingestion_audit": ingestion_audit,
              "representative_samples": parser_audit_samples(documents)}
    atomic_write_json(RUNTIME_ROOT / "corpus_e_parser_audit.json", report)
    return report


def _parsed_documents() -> list[Document]:
    payload = read_json(ensure_private_path(PARSED_CHUNKS_PATH))
    return [Document(page_content=row["content"], metadata=row["metadata"]) for row in payload["documents"]]


def _p2_rows_live(manifest: dict[str, Any], documents: list[Document]) -> list[dict[str, Any]]:
    reranker = CrossEncoderReranker(RerankerConfig(enabled=True, candidate_k=7, top_k=3, device="cpu"))
    with full_vector_knowledge_base(documents):
        return _run_mode(P2["mode"], manifest["queries"], None, None, reranker, candidate_k=P2["candidate_k"], section_strategy=P2["section_strategy"], summarize=False)["rows"]


def export_holdout_artifact(output_path: Path, artifact_id: str = "v319-frozen-e-v1") -> dict[str, Any]:
    """The only permitted E live retrieval.  It never parses PDFs."""
    _assert_rule_identity()
    output_path = ensure_private_path(output_path)
    if output_path.exists():
        raise FileExistsError(f"IMMUTABLE_ARTIFACT_EXISTS:{output_path}")
    config = retrieval_configuration()
    if hash_json(config) != EXPECTED_RETRIEVAL_CONFIG_HASH:
        raise RuntimeError(f"RETRIEVAL_CONFIG_HASH_MISMATCH:{hash_json(config)}")
    manifest, documents = load_holdout_manifest(), _parsed_documents()
    parser_audit = read_json(ensure_private_path(RUNTIME_ROOT / "corpus_e_parser_audit.json"))
    by_chunk = {str(row.metadata["chunk_id"]): row for row in documents}
    started = time.perf_counter()
    p2_rows = _p2_rows_live(manifest, documents)
    p2_by_id = {row["query_id"]: row for row in p2_rows}
    from backend import rag_core
    from backend.evaluation.full_vector_benchmark import FULL_BENCHMARK_KNOWLEDGE_BASE_ID
    evidence_inputs: dict[str, dict[str, Any]] = {}
    with full_vector_knowledge_base(documents):
        for query in manifest["queries"]:
            result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode="hybrid")
            evidence_inputs[query["query_id"]] = _capture_evidence_input(query, result)["evidence_input"]
    rows = []
    for query in manifest["queries"]:
        p2 = p2_by_id[query["query_id"]]
        rows.append({"query_id": query["query_id"], "query": query["query"], "query_text_hash": hash_json(query["query"]),
                     "ground_truth": query, "evidence_input": evidence_inputs[query["query_id"]],
                     "final_context": [candidate_from_p2_row(candidate, by_chunk[str(candidate["chunk_id"])]) for candidate in p2["candidates"]],
                     "retrieval_decision_inputs": {"retrieval_mode": "hybrid", "source_p2_retrieval_scope": p2.get("retrieval_scope", {}), "source_p2_section_retrieval": p2.get("section_retrieval")}})
    payload = new_artifact_payload(artifact_id=artifact_id, corpus_id="E", manifest_hash=hash_json(manifest), annotation_hash=annotation_hash(manifest), retrieval_config=config, queries=rows, snapshot_documents=_snapshot_documents(documents), source={"p2_retrieved_live_once": True, "evidence_input_retrieved_live_once": True, "live_retrieval_elapsed_seconds": time.perf_counter()-started, "parser_audit": parser_audit, "document_sha256": {row["document_id"]: file_sha256(CORPUS_ROOT / row["file"]) for row in manifest["documents"]}}, rule_version=EVIDENCE_SUPPORT_RULE_VERSION)
    write_immutable_artifact(output_path, payload)
    inspection = validate_artifact(read_json(output_path), expected_corpus_id="E", expected_manifest_hash=hash_json(manifest), expected_annotation_hash=annotation_hash(manifest), expected_retrieval_config_hash=EXPECTED_RETRIEVAL_CONFIG_HASH)
    if inspection["validity"] != "VALID":
        raise RuntimeError(f"ARTIFACT_NOT_VALID:{inspection}")
    result = {**inspection, "path": str(output_path), "bytes": output_path.stat().st_size, "capture_seconds": time.perf_counter()-started, "p2_rows": p2_rows}
    atomic_write_json(RUNTIME_ROOT / "corpus_e_export.json", result)
    return result


def replay_holdout_artifact(path: Path, run_id: str = "v319-replay-e") -> dict[str, Any]:
    _assert_rule_identity()
    return replay_artifact(ensure_private_path(path), run_id, expected_manifest=load_holdout_manifest())


def _evidence_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers, abstains = [row for row in rows if row["answerable"]], [row for row in rows if not row["answerable"]]
    false_answers = [row["query_id"] for row in abstains if row["base_decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in answers if row["base_decision"] == "ABSTAIN"]
    return {"queries": len(rows), "answerable": len(answers), "abstain": len(abstains), "false_answer": false_answers, "false_refusal": false_refusals,
            "accuracy": sum((row["base_decision"] == "ANSWER") == row["answerable"] for row in rows) / len(rows) if rows else None,
            "answerable_recall": 1-len(false_refusals)/len(answers) if answers else None, "abstain_recall": 1-len(false_answers)/len(abstains) if abstains else None,
            "false_answer_rate": len(false_answers)/len(abstains) if abstains else None, "false_refusal_rate": len(false_refusals)/len(answers) if answers else None}


def confidence_metrics(replay: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    rows = [{**row, **annotations[row["query_id"]]} for row in replay["rows"]]
    return {"ALL": _evidence_slice(rows), "HIGH_ONLY": _evidence_slice([row for row in rows if row["confidence"] == "HIGH"]), "HIGH_MEDIUM": _evidence_slice([row for row in rows if row["confidence"] in {"HIGH", "MEDIUM"}]), "AMBIGUOUS": [row["query_id"] for row in rows if row["confidence"] == "AMBIGUOUS"]}


def failure_attribution(replay: dict[str, Any], manifest: dict[str, Any], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    parser_audit = artifact.get("source", {}).get("parser_audit", {}).get("production_ingestion_audit", {})
    failures = []
    for row in replay["rows"]:
        if (row["base_decision"] == "ANSWER") == row["answerable"]:
            continue
        annotation = annotations[row["query_id"]]
        candidate_ids = set(row.get("candidate_ids", {}).get("evidence", []))
        parser_issues = [issue for doc_id in annotation["relevant_document_ids"] for issue in parser_audit.get(doc_id, {}).get("issues", [])]
        if annotation["confidence"] == "AMBIGUOUS": attribution = "ANNOTATION_AMBIGUITY"
        elif parser_issues: attribution = "PARSER_METADATA_FAILURE"
        elif annotation["answerable"] and not candidate_ids.intersection(annotation["relevant_chunk_ids"]): attribution = "RETRIEVAL_MISSING_EVIDENCE"
        else: attribution = "EVIDENCE_RULE_FAILURE"
        failures.append({"query_id": row["query_id"], "attribution": attribution, "failure_class": annotation["failure_class"], "expected": "ANSWER" if annotation["answerable"] else "ABSTAIN", "actual": row["base_decision"], "confidence": annotation["confidence"], "post_freeze_status": "POST_FREEZE_DISCOVERED_ISSUE"})
    return failures


def _retrieval_metrics(artifact: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = {row["query_id"]: row for row in artifact["queries"]}
    positives = [row for row in manifest["queries"] if row["answerable"]]
    ranks = []
    model_confusions, identifier_hits = [], []
    for query in positives:
        context = rows[query["query_id"]]["final_context"]
        rank = next((index for index, candidate in enumerate(context, 1) if candidate["chunk_id"] in query["relevant_chunk_ids"]), None)
        ranks.append(rank)
        if query["requested"]["identity"] and context:
            model_confusions.append(context[0]["metadata"].get("equipment_model") != query["expected_model"])
        if query["requested"]["identifier"]:
            identifier_hits.append(rank is not None and rank <= 5)
    return {"hit_at_1": sum(rank == 1 for rank in ranks)/len(positives), "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks)/len(positives), "mrr": sum(1/rank if rank else 0 for rank in ranks)/len(positives), "model_confusion_rate": sum(model_confusions)/len(model_confusions) if model_confusions else None, "identifier_recall_at_5": sum(identifier_hits)/len(identifier_hits) if identifier_hits else None}


def generalization_range(matrix: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = ("decision_accuracy", "answerable_recall", "ood_recall", "false_answer_rate", "false_refusal_rate")
    return {field: {"min": min(row[field] for row in matrix.values() if row[field] is not None), "max": max(row[field] for row in matrix.values() if row[field] is not None)} for field in fields}


def matrix_and_range(e_replay: dict[str, Any]) -> dict[str, Any]:
    summaries = {key: read_json(PRIVATE_ROOT / "v312_runtime" / f"v318-final-{key.lower()}" / "summary.json") for key in "ABCD"}
    summaries["E"] = e_replay
    fields = ("decision_accuracy", "answerable_recall", "ood_recall", "false_answer_rate", "false_refusal_rate")
    matrix = {corpus: {field: row["metrics"]["evidence"].get(field) for field in fields} for corpus, row in summaries.items()}
    return {"matrix": matrix, "generalization_range": generalization_range(matrix)}


def holdout_analysis(replay: dict[str, Any], manifest: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    annotations = {row["query_id"]: row for row in manifest["queries"]}
    rows = [{**row, **annotations[row["query_id"]]} for row in replay["rows"]]
    failures = failure_attribution(replay, manifest, artifact)
    by = lambda field: {value: _evidence_slice([row for row in rows if row[field] == value]) for value in sorted({row[field] for row in rows})}
    return {"status": "READY", "holdout": holdout_distribution(manifest), "artifact": validate_artifact(artifact), "retrieval": _retrieval_metrics(artifact, manifest), "evidence": replay["metrics"]["evidence"], "support": replay["metrics"]["support"], "final_pipeline": {"accuracy": sum((row["final_decision"] == "ANSWER") == row["answerable"] for row in rows)/len(rows), "false_final_answer": [row["query_id"] for row in rows if not row["answerable"] and row["final_decision"] == "ANSWER"], "false_final_refusal": [row["query_id"] for row in rows if row["answerable"] and row["final_decision"] == "ABSTAIN"]}, "confidence_metrics": confidence_metrics(replay, manifest), "failure_attribution": failures, "failure_attribution_counts": dict(Counter(row["attribution"] for row in failures)), "evidence_failure_taxonomy": dict(Counter(row["failure_class"] for row in failures)), "by_manufacturer": by("manufacturer"), "by_category": by("category"), "semantic_cases": [row["query_id"] for row in rows if row.get("semantic_hard_positive")], "multi_chunk_cases": [row["query_id"] for row in rows if row.get("multi_chunk_positive")], "hard_negative_cases": [row["query_id"] for row in rows if row.get("hard_negative")], **matrix_and_range(replay)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "freeze", "validate", "export", "replay", "analyze"))
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_ROOT / "v319-frozen-e-v1.json")
    parser.add_argument("--run-id", default="v319-replay-e")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "v319_final_analysis.json")
    args = parser.parse_args(argv)
    if args.command == "audit": report = audit_corpus_source()
    elif args.command == "freeze": report = {"freeze": freeze_holdout_manifest()}
    elif args.command == "validate":
        manifest = load_holdout_manifest(); report = {"validity": "VALID", "distribution": holdout_distribution(manifest), "freeze": manifest["freeze"]}
    elif args.command == "export": report = export_holdout_artifact(args.artifact)
    elif args.command == "replay": report = replay_holdout_artifact(args.artifact, args.run_id)
    else:
        manifest, artifact = load_holdout_manifest(), read_json(ensure_private_path(args.artifact))
        replay = read_json(PRIVATE_ROOT / "v312_runtime" / args.run_id / "summary.json")
        report = holdout_analysis(replay, manifest, artifact); atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=True, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
