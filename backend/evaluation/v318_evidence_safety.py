"""V3.18 cross-corpus Evidence safety calibration and frozen replay analysis."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import (
    deserialize_candidate,
    load_valid_artifact,
)
from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.retrieval.candidates import RetrievalResult
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.evidence_support import (
    SUPPORT_RULE_VERSION,
    build_evidence_requirement,
    skipped_support,
    validate_evidence_support,
)
from backend.retrieval.filters import analyze_query
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CALIBRATION_PATH = PRIVATE_ROOT / "v318_calibration" / "evidence_safety_calibration.json"
RESULT_ROOT = PRIVATE_ROOT / "v318_results"
ALLOWED_CORPORA = frozenset({"A", "B", "C"})
REQUIRED_MANUFACTURERS = frozenset({
    "Rockwell Automation", "ABB", "Omron", "Beckhoff",
})
FAILURE_CLASSES = frozenset({
    "IDENTITY_ONLY_FALSE_ANSWER",
    "IDENTIFIER_EXISTENCE_TOO_BROAD",
    "PROTOCOL_TOPIC_OVERMATCH",
    "ATTRIBUTE_NOT_SUPPORTED",
    "VALUE_NOT_SUPPORTED",
    "REQUIREMENT_NOT_SUPPORTED",
    "ACTION_NOT_SUPPORTED",
    "PARTIAL_EVIDENCE_ACCEPTED",
    "SEMANTIC_TOPIC_ONLY_MATCH",
    "CROSS_CHUNK_SCOPE_LEAK",
    "OVER_CONSTRAINED_FALSE_REFUSAL",
})


def query_hash(calibration: dict[str, Any]) -> str:
    return hash_json([
        {"calibration_id": row["calibration_id"], "query": row["query"]}
        for row in calibration["queries"]
    ])


def annotation_hash(calibration: dict[str, Any]) -> str:
    return hash_json([
        {key: value for key, value in row.items() if key != "query"}
        for row in calibration["queries"]
    ])


def _normalize_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _artifact_queries(artifacts: dict[str, dict[str, Any]]) -> set[str]:
    return {
        _normalize_query(row["query"])
        for artifact in artifacts.values()
        for row in artifact["queries"]
    }


def distribution(calibration: dict[str, Any]) -> dict[str, Any]:
    queries = calibration["queries"]
    return {
        "queries": len(queries),
        "ground_truth": dict(sorted(Counter(
            "ANSWER" if row["answerable"] else "ABSTAIN" for row in queries
        ).items())),
        "manufacturer": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "category": dict(sorted(Counter(row["category"] for row in queries).items())),
        "failure_class": dict(sorted(Counter(row["failure_class"] for row in queries).items())),
    }


def validate_calibration(
    calibration: dict[str, Any], artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    queries = calibration.get("queries", [])
    if not 30 <= len(queries) <= 40:
        raise ValueError("V318_CALIBRATION_SIZE")
    if sum(bool(row.get("answerable")) for row in queries) < 16:
        raise ValueError("V318_ANSWERABLE_MINIMUM")
    if sum(not bool(row.get("answerable")) for row in queries) < 14:
        raise ValueError("V318_ABSTAIN_MINIMUM")
    if set(calibration.get("source_artifacts", {})) != ALLOWED_CORPORA:
        raise ValueError("V318_SOURCE_CORPORA_MUST_BE_A_B_C")
    if set(artifacts) != ALLOWED_CORPORA:
        raise ValueError("V318_ARTIFACTS_MUST_BE_A_B_C")

    frozen_queries = _artifact_queries(artifacts)
    identifiers = []
    for row in queries:
        required = {
            "calibration_id", "query", "answerable", "corpus_id", "manufacturer",
            "category", "failure_class", "candidate_chunk_ids", "rationale",
        }
        if not required.issubset(row):
            raise ValueError(f"V318_ANNOTATION_INCOMPLETE:{row.get('calibration_id', '')}")
        if row["corpus_id"] not in ALLOWED_CORPORA:
            raise ValueError(f"V318_INVALID_CORPUS:{row['calibration_id']}")
        if row["manufacturer"] not in REQUIRED_MANUFACTURERS:
            raise ValueError(f"V318_INVALID_MANUFACTURER:{row['calibration_id']}")
        if row["failure_class"] not in FAILURE_CLASSES:
            raise ValueError(f"V318_INVALID_FAILURE_CLASS:{row['calibration_id']}")
        if not row["candidate_chunk_ids"]:
            raise ValueError(f"V318_CANDIDATES_REQUIRED:{row['calibration_id']}")
        if _normalize_query(row["query"]) in frozen_queries:
            raise ValueError(f"V318_FROZEN_QUERY_REUSED:{row['calibration_id']}")
        identifiers.append(row["calibration_id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("V318_DUPLICATE_ID")
    if set(row["manufacturer"] for row in queries) != REQUIRED_MANUFACTURERS:
        raise ValueError("V318_MANUFACTURER_COVERAGE")

    freeze = calibration.get("freeze", {})
    if freeze.get("query_sha256") != query_hash(calibration):
        raise ValueError("V318_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != annotation_hash(calibration):
        raise ValueError("V318_ANNOTATION_HASH_MISMATCH")
    return distribution(calibration)


def freeze_calibration(path: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, str]:
    path = ensure_private_path(path)
    calibration = read_json(path)
    if any(calibration.get("freeze", {}).values()):
        raise RuntimeError("V318_CALIBRATION_ALREADY_FROZEN")
    calibration["freeze"] = {
        "query_sha256": query_hash(calibration),
        "annotation_sha256": annotation_hash(calibration),
    }
    atomic_write_json(path, calibration)
    validate_calibration(calibration, artifacts)
    return calibration["freeze"]


def _candidate_index(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in artifact["queries"]:
        for candidate in row["evidence_input"]["candidate_pool"] + row["final_context"]:
            result.setdefault(candidate["chunk_id"], candidate)
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    abstain = [row for row in rows if not row["answerable"]]
    false_answers = [row["calibration_id"] for row in abstain if row["decision"] == "ANSWER"]
    false_refusals = [row["calibration_id"] for row in answerable if row["decision"] == "ABSTAIN"]
    return {
        "decision_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "answerable_recall": (
            sum(row["decision"] == "ANSWER" for row in answerable) / len(answerable)
            if answerable else None
        ),
        "abstain_recall": (
            sum(row["decision"] == "ABSTAIN" for row in abstain) / len(abstain)
            if abstain else None
        ),
        "false_answer_rate": len(false_answers) / len(abstain) if abstain else None,
        "false_refusal_rate": len(false_refusals) / len(answerable) if answerable else None,
        "false_answer_ids": false_answers,
        "false_refusal_ids": false_refusals,
    }


def run_calibration(
    calibration: dict[str, Any], artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validate_calibration(calibration, artifacts)
    indexes = {corpus: _candidate_index(artifact) for corpus, artifact in artifacts.items()}
    documents = {
        corpus: [deserialize_candidate(item).document for item in index.values()]
        for corpus, index in indexes.items()
    }
    rows = []
    started = time.perf_counter()
    for annotation in calibration["queries"]:
        corpus = annotation["corpus_id"]
        try:
            candidates = [
                deserialize_candidate(indexes[corpus][chunk_id])
                for chunk_id in annotation["candidate_chunk_ids"]
            ]
        except KeyError as exc:
            raise ValueError(
                f"V318_CANDIDATE_MISSING:{annotation['calibration_id']}:{exc.args[0]}"
            ) from exc
        corpus_documents = documents[corpus]
        analysis = analyze_query(annotation["query"], corpus_documents)
        result = RetrievalResult(
            candidates,
            query_analysis=analysis,
            corpus_documents=corpus_documents,
            retrieval_mode="frozen_evidence_calibration",
        )
        evidence = analyze_retrieval_evidence(
            annotation["query"], result, corpus_documents, "frozen_evidence_calibration",
        )
        if evidence.decision == "ANSWER":
            support = validate_evidence_support(annotation["query"], result, corpus_documents)
        else:
            support = skipped_support()
        expected = "ANSWER" if annotation["answerable"] else "ABSTAIN"
        rows.append({
            **annotation,
            "expected_decision": expected,
            "decision": evidence.decision,
            "reason": evidence.reason,
            "correct": evidence.decision == expected,
            "evidence": evidence.as_dict(),
            "requirement": build_evidence_requirement(
                annotation["query"], corpus_documents, analysis,
            ).as_dict(),
            "support": support.as_dict(),
            "final_decision": (
                "ANSWER"
                if evidence.decision == "ANSWER" and support.status == "SUPPORTED"
                else "ABSTAIN"
            ),
        })
    return {
        "validity": "VALID",
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "nli_used": False,
        "freeze": calibration["freeze"],
        "distribution": distribution(calibration),
        "metrics": _metrics(rows),
        "by_manufacturer": {
            manufacturer: _metrics([row for row in rows if row["manufacturer"] == manufacturer])
            for manufacturer in sorted(REQUIRED_MANUFACTURERS)
        },
        "by_failure_class": {
            failure_class: _metrics([row for row in rows if row["failure_class"] == failure_class])
            for failure_class in sorted({row["failure_class"] for row in rows})
        },
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def load_artifacts(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for corpus, source in calibration["source_artifacts"].items():
        artifact = load_valid_artifact(ensure_private_path(Path(source["path"])))
        if artifact["artifact_id"] != source["artifact_id"] or artifact["artifact_hash"] != source["artifact_hash"]:
            raise ValueError(f"V318_ARTIFACT_IDENTITY_MISMATCH:{corpus}")
        artifacts[corpus] = artifact
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run"))
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = ensure_private_path(args.calibration)
    calibration = read_json(path)
    artifacts = load_artifacts(calibration)
    if args.command == "freeze":
        report: dict[str, Any] = {"freeze": freeze_calibration(path, artifacts)}
    elif args.command == "validate":
        report = {"validity": "VALID", "distribution": validate_calibration(calibration, artifacts)}
    else:
        report = run_calibration(calibration, artifacts)
        if args.output:
            atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
