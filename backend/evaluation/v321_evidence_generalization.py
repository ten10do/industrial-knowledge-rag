"""Offline development-set gate for the V3.21 Evidence contract.

Both splits consume immutable Corpus F candidate payloads. DEV-CHECK is guarded
by a private one-shot ledger: one V3.20 baseline and one frozen V3.21 candidate.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.frozen_retrieval_artifact import (
    deserialize_candidate, deserialize_document, file_sha256,
)
from backend.evaluation.resumable import atomic_write_json, read_json, utc_now
from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.retrieval import RetrievalResult, analyze_query
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_ROOT = PRIVATE_ROOT / "corpus_f"
TRAIN_PATH = CORPUS_ROOT / "dev_train.json"
CHECK_PATH = CORPUS_ROOT / "dev_check.json"
RESULT_ROOT = PRIVATE_ROOT / "v321_results"
CHECK_LEDGER = RESULT_ROOT / "dev_check_one_shot.json"
CANDIDATE_FREEZE = RESULT_ROOT / "candidate_freeze.json"
EVIDENCE_CONTRACT_PATH = PROJECT_ROOT / "backend" / "retrieval" / "evidence_contract.py"
EVIDENCE_PATH = PROJECT_ROOT / "backend" / "retrieval" / "evidence.py"
TECHNICAL_PATH = PROJECT_ROOT / "backend" / "retrieval" / "technical.py"

BASELINE_VERSION = "evidence-v320.1"
CANDIDATE_VERSION = "evidence-v321.1"
REQUIRED_FAILURE_CLASSES = frozenset({
    "identifier", "protocol", "attribute", "value", "action", "requirement",
    "semantic", "multi_chunk", "cross_scope", "qualifier",
})
CLAIM_TYPES = frozenset({
    "EXPLICIT", "NORMALIZED_EQUIVALENT", "SEMANTIC_EQUIVALENT",
    "RELATED_ONLY", "ABSENT",
})
CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
FORBIDDEN_PRODUCT_TOKENS = (
    "modicon m221", "fx5-enet", "s7-1200", "sigma-7", "sgd7s", "frenic",
)


def _without_freeze(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "freeze"}


def _document_ids(manifest: dict[str, Any]) -> set[str]:
    return {str(row["document_id"]) for row in manifest.get("documents", [])}


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    split = manifest.get("split")
    if split not in {"DEV-TRAIN", "DEV-CHECK"}:
        raise ValueError("V321_SPLIT_INVALID")
    queries = manifest.get("queries", [])
    lower, upper = ((48, 60) if split == "DEV-TRAIN" else (24, 30))
    if not lower <= len(queries) <= upper:
        raise ValueError(f"V321_{split}_QUERY_COUNT_INVALID")
    answers = [row for row in queries if row.get("answerable")]
    abstains = [row for row in queries if not row.get("answerable")]
    if len(answers) != len(abstains):
        raise ValueError(f"V321_{split}_LABEL_BALANCE_INVALID")

    ids = [str(row.get("query_id", "")) for row in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"V321_{split}_QUERY_ID_INVALID")
    pairs = Counter(row.get("pair_id") for row in queries)
    if not pairs or any(count != 2 for count in pairs.values()):
        raise ValueError(f"V321_{split}_PAIRS_REQUIRED")
    for pair_id in pairs:
        labels = {bool(row["answerable"]) for row in queries if row["pair_id"] == pair_id}
        if labels != {False, True}:
            raise ValueError(f"V321_PAIR_LABEL_INVALID:{pair_id}")

    classes = {str(row.get("failure_class", "")) for row in queries}
    if not REQUIRED_FAILURE_CLASSES.issubset(classes):
        raise ValueError(f"V321_FAILURE_CLASS_MISSING:{sorted(REQUIRED_FAILURE_CLASSES - classes)}")
    if any(row.get("claim_type") not in CLAIM_TYPES for row in queries):
        raise ValueError(f"V321_{split}_CLAIM_TYPE_INVALID")
    if any(row.get("confidence") not in CONFIDENCE_LEVELS for row in queries):
        raise ValueError(f"V321_{split}_CONFIDENCE_INVALID")
    if split == "DEV-CHECK" and sum(row["confidence"] == "HIGH" for row in queries) / len(queries) < .85:
        raise ValueError("V321_DEV_CHECK_HIGH_CONFIDENCE_MINIMUM")

    semantic_minimum = 12 if split == "DEV-TRAIN" else 6
    multi_minimum = 6 if split == "DEV-TRAIN" else 3
    semantic = sum(bool(row.get("semantic_positive")) for row in answers)
    safe_multi = sum(bool(row.get("multi_chunk_positive")) for row in answers)
    unsafe_multi = sum(bool(row.get("unsafe_multi_chunk_negative")) for row in abstains)
    if semantic < semantic_minimum:
        raise ValueError(f"V321_{split}_SEMANTIC_POSITIVE_MINIMUM")
    if safe_multi < multi_minimum or unsafe_multi < multi_minimum:
        raise ValueError(f"V321_{split}_MULTI_CHUNK_MINIMUM")
    if not any(row.get("cross_document_negative") for row in abstains):
        raise ValueError(f"V321_{split}_CROSS_DOCUMENT_NEGATIVE_REQUIRED")

    documents = manifest.get("documents", [])
    document_ids = _document_ids(manifest)
    if len(document_ids) != len(documents) or len(document_ids) < 3:
        raise ValueError(f"V321_{split}_DOCUMENTS_INVALID")
    if len({row.get("manufacturer") for row in documents}) < 3:
        raise ValueError(f"V321_{split}_MANUFACTURER_DIVERSITY_INVALID")
    inventory_text = json.dumps(documents, ensure_ascii=False).casefold()
    if any(token in inventory_text for token in FORBIDDEN_PRODUCT_TOKENS):
        raise ValueError(f"V321_{split}_FROZEN_PRODUCT_LEAK")
    candidates = manifest.get("candidates", {})
    for row in queries:
        selected = row.get("candidate_chunk_ids", [])
        if not selected or any(chunk_id not in candidates for chunk_id in selected):
            raise ValueError(f"V321_UNKNOWN_CANDIDATE:{row['query_id']}")
        if any(str(candidates[chunk_id].get("document_id", "")) not in document_ids for chunk_id in selected):
            raise ValueError(f"V321_CROSS_SPLIT_CANDIDATE:{row['query_id']}")

    freeze = manifest.get("freeze", {})
    query_hash = hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in queries])
    if freeze.get("query_sha256") != query_hash:
        raise ValueError(f"V321_{split}_QUERY_HASH_MISMATCH")
    if freeze.get("annotation_sha256") != hash_json(queries):
        raise ValueError(f"V321_{split}_ANNOTATION_HASH_MISMATCH")
    if freeze.get("manifest_sha256") != hash_json(_without_freeze(manifest)):
        raise ValueError(f"V321_{split}_MANIFEST_HASH_MISMATCH")
    return {
        "split": split,
        "queries": len(queries),
        "answerable": len(answers),
        "abstain": len(abstains),
        "pairs": len(pairs),
        "documents": len(document_ids),
        "manufacturers": dict(sorted(Counter(row["manufacturer"] for row in documents).items())),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in queries).items())),
        "focus": dict(sorted(Counter(row["focus"] for row in queries).items())),
        "confidence": dict(sorted(Counter(row["confidence"] for row in queries).items())),
        "claim_types": dict(sorted(Counter(row["claim_type"] for row in queries).items())),
        "semantic_positive": semantic,
        "safe_multi_chunk_positive": safe_multi,
        "unsafe_multi_chunk_negative": unsafe_multi,
        "cross_document_negative": sum(bool(row.get("cross_document_negative")) for row in abstains),
        "nli_used": "NO",
    }


def validate_independence(train: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    train_documents, check_documents = _document_ids(train), _document_ids(check)
    overlap = sorted(train_documents & check_documents)
    if overlap:
        raise ValueError(f"V321_TRAIN_CHECK_DOCUMENT_OVERLAP:{overlap}")
    train_queries = {row["query"].strip().casefold() for row in train["queries"]}
    check_queries = {row["query"].strip().casefold() for row in check["queries"]}
    if train_queries & check_queries:
        raise ValueError("V321_TRAIN_CHECK_QUERY_OVERLAP")
    train_candidates = {row["metadata"].get("document_id") for row in train["candidates"].values()}
    check_candidates = {row["metadata"].get("document_id") for row in check["candidates"].values()}
    if train_candidates & check_candidates:
        raise ValueError("V321_TRAIN_CHECK_CANDIDATE_DOCUMENT_OVERLAP")
    return {
        "document_disjoint": True,
        "manual_disjoint": True,
        "query_disjoint": True,
        "candidate_document_disjoint": True,
        "train_documents": sorted(train_documents),
        "check_documents": sorted(check_documents),
    }


def load_sets() -> tuple[dict[str, Any], dict[str, Any]]:
    train = read_json(ensure_private_path(TRAIN_PATH))
    check = read_json(ensure_private_path(CHECK_PATH))
    validate_manifest(train)
    validate_manifest(check)
    validate_independence(train, check)
    return train, check


def decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = [row for row in rows if row["answerable"]]
    abstains = [row for row in rows if not row["answerable"]]
    false_answers = [row["query_id"] for row in abstains if row["decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in answers if row["decision"] == "ABSTAIN"]
    return {
        "decision_accuracy": sum((row["decision"] == "ANSWER") == row["answerable"] for row in rows) / len(rows),
        "answerable_recall": 1 - len(false_refusals) / len(answers),
        "abstain_recall": 1 - len(false_answers) / len(abstains),
        "false_answer_rate": len(false_answers) / len(abstains),
        "false_refusal_rate": len(false_refusals) / len(answers),
        "false_answer_ids": false_answers,
        "false_refusal_ids": false_refusals,
    }


def evaluate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    distribution = validate_manifest(manifest)
    snapshot = [deserialize_document(row) for row in manifest["corpus_snapshot"]]
    candidates = {chunk_id: deserialize_candidate(payload) for chunk_id, payload in manifest["candidates"].items()}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query in manifest["queries"]:
        selected = [candidates[chunk_id] for chunk_id in query["candidate_chunk_ids"]]
        result = RetrievalResult(
            selected,
            query_analysis=analyze_query(query["query"], snapshot),
            corpus_documents=snapshot,
            retrieval_mode="corpus_f_candidate_fixed",
        )
        evidence = analyze_retrieval_evidence(query["query"], result, snapshot, "corpus_f_candidate_fixed")
        rows.append({
            "query_id": query["query_id"],
            "pair_id": query["pair_id"],
            "query": query["query"],
            "answerable": query["answerable"],
            "document_id": query["document_id"],
            "manufacturer": query["manufacturer"],
            "failure_class": query["failure_class"],
            "focus": query["focus"],
            "confidence": query["confidence"],
            "claim_type": query["claim_type"],
            "candidate_chunk_ids": query["candidate_chunk_ids"],
            "decision": evidence.decision,
            "reason": evidence.reason,
            "evidence": evidence.as_dict(),
        })
    return {
        "development_set_id": manifest["development_set_id"],
        "split": manifest["split"],
        "validity": "VALID",
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "freeze": manifest["freeze"],
        "distribution": distribution,
        "metrics": decision_metrics(rows),
        "by_failure_class": {
            name: decision_metrics([row for row in rows if row["failure_class"] == name])
            for name in sorted({row["failure_class"] for row in rows})
        },
        "by_confidence": {
            name: decision_metrics([row for row in rows if row["confidence"] == name])
            for name in sorted({row["confidence"] for row in rows})
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "live_retrieval": False,
        "pdf_parser": False,
        "nli_used": "NO",
    }


def candidate_rule_hashes() -> dict[str, str]:
    return {
        "evidence_sha256": file_sha256(EVIDENCE_PATH),
        "evidence_contract_sha256": file_sha256(EVIDENCE_CONTRACT_PATH),
        "technical_sha256": file_sha256(TECHNICAL_PATH),
    }


def freeze_candidate() -> dict[str, Any]:
    if EVIDENCE_SUPPORT_RULE_VERSION != CANDIDATE_VERSION:
        raise RuntimeError(f"V321_CANDIDATE_VERSION_REQUIRED:{EVIDENCE_SUPPORT_RULE_VERSION}")
    if CANDIDATE_FREEZE.exists():
        raise FileExistsError("V321_CANDIDATE_ALREADY_FROZEN")
    train, check = load_sets()
    payload = {
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "support_integrity": SUPPORT_RULE_VERSION == "support-v316.1",
        "train_manifest_sha256": train["freeze"]["manifest_sha256"],
        "check_manifest_sha256": check["freeze"]["manifest_sha256"],
        "rule_hashes": candidate_rule_hashes(),
        "frozen_at": utc_now(),
    }
    atomic_write_json(ensure_private_path(CANDIDATE_FREEZE), payload)
    return payload


def _check_phase_allowed(phase: str, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = BASELINE_VERSION if phase == "baseline" else CANDIDATE_VERSION
    if EVIDENCE_SUPPORT_RULE_VERSION != expected:
        raise RuntimeError(f"V321_DEV_CHECK_{phase.upper()}_VERSION_MISMATCH:{EVIDENCE_SUPPORT_RULE_VERSION}")
    ledger = read_json(CHECK_LEDGER) if CHECK_LEDGER.exists() else {"phases": {}}
    if phase in ledger.get("phases", {}):
        raise RuntimeError(f"V321_DEV_CHECK_{phase.upper()}_ALREADY_CONSUMED")
    if phase == "candidate":
        freeze = read_json(ensure_private_path(CANDIDATE_FREEZE))
        if freeze.get("rule_hashes") != candidate_rule_hashes():
            raise RuntimeError("V321_CANDIDATE_RULE_HASH_MISMATCH")
        if freeze.get("check_manifest_sha256") != manifest["freeze"]["manifest_sha256"]:
            raise RuntimeError("V321_CANDIDATE_CHECK_HASH_MISMATCH")
    return ledger


def evaluate_check_once(phase: str, output: Path) -> dict[str, Any]:
    if phase not in {"baseline", "candidate"}:
        raise ValueError("V321_DEV_CHECK_PHASE_INVALID")
    _, check = load_sets()
    ledger = _check_phase_allowed(phase, check)
    output = ensure_private_path(output)
    if output.exists():
        raise FileExistsError(f"V321_RESULT_ALREADY_EXISTS:{output}")
    report = evaluate_manifest(check)
    atomic_write_json(output, report)
    phases = dict(ledger.get("phases", {}))
    phases[phase] = {
        "consumed_at": utc_now(),
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "manifest_sha256": check["freeze"]["manifest_sha256"],
        "result_path": str(output.relative_to(PRIVATE_ROOT)),
        "result_sha256": file_sha256(output),
    }
    atomic_write_json(ensure_private_path(CHECK_LEDGER), {"phases": phases})
    return report


def public_report(report: Any) -> Any:
    """Remove private query rows and failure identifiers from console output."""
    if isinstance(report, dict):
        return {
            key: public_report(value)
            for key, value in report.items()
            if key != "rows" and not key.endswith("_ids")
        }
    if isinstance(report, list):
        return [public_report(value) for value in report]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    train_parser = subparsers.add_parser("evaluate-train")
    train_parser.add_argument("--output", type=Path, required=True)
    check_parser = subparsers.add_parser("evaluate-check")
    check_parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    check_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("freeze-candidate")
    args = parser.parse_args(argv)

    if args.command == "validate":
        train, check = load_sets()
        report = {
            "validity": "VALID",
            "train": validate_manifest(train),
            "check": validate_manifest(check),
            "independence": validate_independence(train, check),
            "freeze": {"train": train["freeze"], "check": check["freeze"]},
        }
    elif args.command == "evaluate-train":
        train, _ = load_sets()
        output = ensure_private_path(args.output)
        if output.exists():
            raise FileExistsError(f"V321_RESULT_ALREADY_EXISTS:{output}")
        report = evaluate_manifest(train)
        atomic_write_json(output, report)
    elif args.command == "evaluate-check":
        report = evaluate_check_once(args.phase, args.output)
    else:
        report = freeze_candidate()
    print(json.dumps(public_report(report), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
