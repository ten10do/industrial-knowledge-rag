"""V3.22 development-only Evidence benchmark validation and evaluation.

This module deliberately contains no one-shot or sealed holdout mechanism.  It
validates DEV-TRAIN-V2 and DEV-TUNE-V2, both of which are development assets.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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
CORPUS_ROOT = PRIVATE_ROOT / "corpus_g"
TRAIN_PATH = CORPUS_ROOT / "dev_train_v2.json"
TUNE_PATH = CORPUS_ROOT / "dev_tune_v2.json"
DEFAULT_HISTORY_PATHS = (
    PRIVATE_ROOT / "manifest.json",
    PRIVATE_ROOT / "corpus_b" / "manifest.json",
    PRIVATE_ROOT / "corpus_c" / "manifest.json",
    PRIVATE_ROOT / "corpus_d" / "manifest.json",
    PRIVATE_ROOT / "corpus_e" / "holdout_manifest_v3.json",
    PRIVATE_ROOT / "v320_calibration" / "evidence_contract_calibration.json",
    PRIVATE_ROOT / "corpus_f" / "dev_train.json",
    PRIVATE_ROOT / "corpus_f" / "dev_check.json",
)
DEFAULT_HOLDOUT_PATHS = (
    PRIVATE_ROOT / "corpus_d" / "source_manifest.json",
    PRIVATE_ROOT / "corpus_e" / "source_manifest.json",
)

EVIDENCE_DEV_BENCHMARK_VERSION = "v322-dev-v1"
BENCHMARK_VERSION = EVIDENCE_DEV_BENCHMARK_VERSION
SPLIT_LIMITS = {"DEV-TRAIN-V2": (64, 90), "DEV-TUNE-V2": (36, 50)}
SEMANTIC_MINIMUM = {"DEV-TRAIN-V2": 16, "DEV-TUNE-V2": 10}
MULTI_MINIMUM = {"DEV-TRAIN-V2": 8, "DEV-TUNE-V2": 5}
HIGH_CONFIDENCE_MINIMUM = {"DEV-TRAIN-V2": .85, "DEV-TUNE-V2": .90}
FAILURE_CLASSES = frozenset({
    "identifier", "protocol", "attribute", "value", "action", "requirement",
    "semantic", "multi_chunk", "cross_scope", "qualifier",
})
DOCUMENT_STYLES = frozenset({
    "PARAMETER_TABLE", "PROCEDURE_STEPS", "NARRATIVE_SPEC", "SAFETY_BLOCK",
    "PROTOCOL_REFERENCE", "CONFIGURATION_TABLE", "CROSS_REFERENCE_HEAVY",
    "TROUBLESHOOTING", "MIXED",
})
DIFFICULTIES = frozenset({
    "L1_EXPLICIT", "L2_NORMALIZED", "L3_SEMANTIC",
    "L4_SCOPE_COMPOSITION", "L5_HARD_NEAR_MISS",
})
SURFACE_FORMS = frozenset({
    "literal", "alias", "abbreviation", "table", "sentence", "bullet",
    "cross_reference", "multi_chunk", "semantic_paraphrase",
})
NEGATIVE_HARDNESS = frozenset({"N1", "N2", "N3", "N4", "N5"})
EXPECTED_SCOPES = frozenset({
    "SAME_CANDIDATE", "SAME_PARAMETER_BLOCK", "SAME_SECTION", "ADJACENT_SECTION",
})
CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
CLAIM_TYPES = frozenset({
    "EXPLICIT", "NORMALIZED_EQUIVALENT", "SEMANTIC_EQUIVALENT",
    "RELATED_ONLY", "ABSENT",
})
FORBIDDEN_PRODUCT_TOKENS = (
    "modicon m221", "fx5-enet", "s7-1200", "sigma-7", "sgd7s", "frenic",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fail(code: str, detail: Any = None) -> None:
    suffix = "" if detail is None else f":{detail}"
    raise ValueError(f"V322_{code}{suffix}")


def _without_freeze(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "freeze"}


def _normalize_query(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _query_tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.casefold()))


def _documents(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("document_id", "")): row for row in manifest.get("documents", [])}


def _ratio(rows: list[dict[str, Any]], predicate: Any) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    split = str(manifest.get("split", ""))
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        _fail("BENCHMARK_VERSION_INVALID")
    if split not in SPLIT_LIMITS:
        _fail("SPLIT_INVALID", split)
    queries = manifest.get("queries", [])
    lower, upper = SPLIT_LIMITS[split]
    if not lower <= len(queries) <= upper:
        _fail("QUERY_COUNT_INVALID", split)

    ids = [str(row.get("query_id", "")) for row in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        _fail("QUERY_ID_INVALID", split)
    answers = [row for row in queries if row.get("answerable") is True]
    abstains = [row for row in queries if row.get("answerable") is False]
    answer_ratio = len(answers) / len(queries)
    if not .45 <= answer_ratio <= .55 or len(answers) + len(abstains) != len(queries):
        _fail("LABEL_BALANCE_INVALID", split)

    pairs = Counter(str(row.get("pair_id", "")) for row in queries)
    if "" in pairs or any(count != 2 for count in pairs.values()):
        _fail("PAIR_CARDINALITY_INVALID", split)
    for pair_id in pairs:
        rows = [row for row in queries if row["pair_id"] == pair_id]
        if {row["answerable"] for row in rows} != {False, True}:
            _fail("PAIR_LABEL_INVALID", pair_id)
        if len({row.get("failure_class") for row in rows}) != 1:
            _fail("PAIR_FAMILY_INVALID", pair_id)

    classes = {str(row.get("failure_class", "")) for row in queries}
    if not FAILURE_CLASSES.issubset(classes):
        _fail("FAILURE_CLASS_MISSING", sorted(FAILURE_CLASSES - classes))
    if any(row.get("document_style") not in DOCUMENT_STYLES for row in queries):
        _fail("DOCUMENT_STYLE_INVALID", split)
    if len({row["document_style"] for row in queries}) < 4:
        _fail("DOCUMENT_STYLE_DIVERSITY_INVALID", split)
    if len({str(row.get("focus", "")) for row in queries} - {""}) < 4:
        _fail("FOCUS_DIVERSITY_INVALID", split)
    if any(row.get("difficulty") not in DIFFICULTIES for row in queries):
        _fail("DIFFICULTY_INVALID", split)
    if _ratio(queries, lambda row: row["difficulty"] in {"L3_SEMANTIC", "L4_SCOPE_COMPOSITION", "L5_HARD_NEAR_MISS"}) < .60:
        _fail("HARD_DIFFICULTY_SHARE_INVALID", split)
    if any(row.get("surface_form_type") not in SURFACE_FORMS for row in queries):
        _fail("SURFACE_FORM_INVALID", split)
    if any(row.get("claim_type") not in CLAIM_TYPES for row in queries):
        _fail("CLAIM_TYPE_INVALID", split)
    if any(row.get("confidence") not in CONFIDENCE_LEVELS for row in queries):
        _fail("CONFIDENCE_INVALID", split)
    if _ratio(queries, lambda row: row["confidence"] == "HIGH") < HIGH_CONFIDENCE_MINIMUM[split]:
        _fail("HIGH_CONFIDENCE_MINIMUM", split)
    if any(row["confidence"] == "AMBIGUOUS" and row.get("core", True) for row in queries):
        _fail("AMBIGUOUS_CORE_CASE", split)

    for row in queries:
        if not isinstance(row.get("critical_requirements"), list) or not row["critical_requirements"]:
            _fail("CRITICAL_REQUIREMENTS_REQUIRED", row.get("query_id"))
        if not isinstance(row.get("non_critical_cues"), list):
            _fail("NON_CRITICAL_CUES_REQUIRED", row.get("query_id"))
        if not str(row.get("annotation_rationale", "")).strip():
            _fail("ANNOTATION_RATIONALE_REQUIRED", row.get("query_id"))
        if row.get("expected_scope") not in EXPECTED_SCOPES:
            _fail("EXPECTED_SCOPE_INVALID", row.get("query_id"))
        if row["answerable"] is False:
            if row.get("negative_hardness") not in NEGATIVE_HARDNESS:
                _fail("NEGATIVE_HARDNESS_INVALID", row.get("query_id"))
            if not str(row.get("forbidden_scope_reason", "")).strip():
                _fail("FORBIDDEN_SCOPE_REASON_REQUIRED", row.get("query_id"))
    if _ratio(abstains, lambda row: row["negative_hardness"] in {"N3", "N4", "N5"}) < .60:
        _fail("HARD_NEGATIVE_SHARE_INVALID", split)

    semantic = sum(bool(row.get("semantic_positive")) for row in answers)
    safe_multi = sum(bool(row.get("multi_chunk_positive")) for row in answers)
    unsafe_multi = sum(bool(row.get("unsafe_multi_chunk_negative")) for row in abstains)
    if any(row.get("semantic_positive") and (not row["answerable"] or row["claim_type"] != "SEMANTIC_EQUIVALENT") for row in queries):
        _fail("SEMANTIC_POSITIVE_LABEL_INVALID", split)
    if any(row.get("multi_chunk_positive") and (not row["answerable"] or len(set(row.get("candidate_chunk_ids", []))) < 2) for row in queries):
        _fail("SAFE_MULTI_CHUNK_LABEL_INVALID", split)
    if any(row.get("unsafe_multi_chunk_negative") and (row["answerable"] or len(set(row.get("candidate_chunk_ids", []))) < 2) for row in queries):
        _fail("UNSAFE_MULTI_CHUNK_LABEL_INVALID", split)
    if semantic < SEMANTIC_MINIMUM[split]:
        _fail("SEMANTIC_POSITIVE_MINIMUM", split)
    if safe_multi < MULTI_MINIMUM[split] or unsafe_multi < MULTI_MINIMUM[split]:
        _fail("MULTI_CHUNK_MINIMUM", split)

    document_rows = manifest.get("documents", [])
    documents = _documents(manifest)
    if "" in documents or len(documents) != len(document_rows) or len(documents) < 4:
        _fail("DOCUMENT_COUNT_INVALID", split)
    if len({row.get("manufacturer") for row in documents.values()}) < 3:
        _fail("MANUFACTURER_DIVERSITY_INVALID", split)
    if split == "DEV-TUNE-V2" and not any(row.get("manufacturer_slice") == "UNSEEN" for row in documents.values()):
        _fail("UNSEEN_MANUFACTURER_REQUIRED")
    if any(sum(row.get("document_id") == document_id for row in queries) > 18 for document_id in documents):
        _fail("PER_DOCUMENT_CASE_LIMIT", split)
    inventory = json.dumps(list(documents.values()), ensure_ascii=False).casefold()
    if any(token in inventory for token in FORBIDDEN_PRODUCT_TOKENS):
        _fail("D_E_PRODUCT_LEAK", split)

    candidates = manifest.get("candidates", {})
    for row in queries:
        document_id = str(row.get("document_id", ""))
        selected = row.get("candidate_chunk_ids", [])
        if document_id not in documents:
            _fail("QUERY_DOCUMENT_UNKNOWN", row["query_id"])
        document = documents[document_id]
        if row.get("manufacturer") != document.get("manufacturer") or row.get("manufacturer_slice") != document.get("manufacturer_slice"):
            _fail("QUERY_DOCUMENT_METADATA_MISMATCH", row["query_id"])
        if not selected or any(chunk_id not in candidates for chunk_id in selected):
            _fail("CANDIDATE_UNKNOWN", row["query_id"])
        candidate_documents = {
            str(candidates[chunk_id].get("metadata", {}).get("document_id", candidates[chunk_id].get("document_id", "")))
            for chunk_id in selected
        }
        if not candidate_documents.issubset(documents):
            _fail("CROSS_SPLIT_CANDIDATE", row["query_id"])

    freeze = manifest.get("freeze", {})
    query_rows = [{"query_id": row["query_id"], "query": row["query"]} for row in queries]
    if freeze.get("query_sha256") != hash_json(query_rows):
        _fail("QUERY_HASH_MISMATCH", split)
    if freeze.get("annotation_sha256") != hash_json(queries):
        _fail("ANNOTATION_HASH_MISMATCH", split)
    if freeze.get("manifest_sha256") != hash_json(_without_freeze(manifest)):
        _fail("MANIFEST_HASH_MISMATCH", split)

    return {
        "split": split,
        "queries": len(queries),
        "answerable": len(answers),
        "abstain": len(abstains),
        "pairs": len(pairs),
        "documents": len(documents),
        "manufacturers": dict(sorted(Counter(row["manufacturer"] for row in documents.values()).items())),
        "manufacturer_slices": dict(sorted(Counter(row["manufacturer_slice"] for row in documents.values()).items())),
        "equipment_types": dict(sorted(Counter(row["equipment_type"] for row in documents.values()).items())),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in queries).items())),
        "focus": dict(sorted(Counter(row["focus"] for row in queries).items())),
        "difficulty": dict(sorted(Counter(row["difficulty"] for row in queries).items())),
        "document_style": dict(sorted(Counter(row["document_style"] for row in queries).items())),
        "negative_hardness": dict(sorted(Counter(row["negative_hardness"] for row in abstains).items())),
        "confidence": dict(sorted(Counter(row["confidence"] for row in queries).items())),
        "semantic_positive": semantic,
        "safe_multi_chunk_positive": safe_multi,
        "unsafe_multi_chunk_negative": unsafe_multi,
        "hard_difficulty_share": _ratio(queries, lambda row: row["difficulty"] in {"L3_SEMANTIC", "L4_SCOPE_COMPOSITION", "L5_HARD_NEAR_MISS"}),
        "hard_negative_share": _ratio(abstains, lambda row: row["negative_hardness"] in {"N3", "N4", "N5"}),
        "nli_used": "NO",
    }


def validate_independence(train: dict[str, Any], tune: dict[str, Any]) -> dict[str, Any]:
    train_docs, tune_docs = _documents(train), _documents(tune)
    document_overlap = sorted(set(train_docs) & set(tune_docs))
    if document_overlap:
        _fail("TRAIN_TUNE_DOCUMENT_LEAKAGE", document_overlap)
    train_paths = {str(row.get("source_path", row.get("file", ""))).casefold() for row in train_docs.values()}
    tune_paths = {str(row.get("source_path", row.get("file", ""))).casefold() for row in tune_docs.values()}
    path_overlap = sorted((train_paths & tune_paths) - {""})
    if path_overlap:
        _fail("TRAIN_TUNE_SOURCE_PATH_LEAKAGE", path_overlap)
    train_lines = {str(row.get("product_family", "")).strip().casefold() for row in train_docs.values()}
    tune_lines = {str(row.get("product_family", "")).strip().casefold() for row in tune_docs.values()}
    product_line_overlap = sorted((train_lines & tune_lines) - {""})
    return {
        "document_disjoint": True,
        "source_path_disjoint": True,
        "train_tune_document_leakage": 0,
        "product_line_overlap": product_line_overlap,
        "product_line_disjoint": not product_line_overlap,
        "train_documents": sorted(train_docs),
        "tune_documents": sorted(tune_docs),
    }


def query_similarity_audit(
    train: dict[str, Any], tune: dict[str, Any], historical_manifests: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    historical = []
    for manifest in historical_manifests:
        historical.extend(row for row in manifest.get("queries", []) if isinstance(row.get("query"), str))
    current = train.get("queries", []) + tune.get("queries", [])
    exact: list[dict[str, str]] = []
    normalized: list[dict[str, str]] = []
    high_overlap: list[dict[str, Any]] = []
    for row in current:
        query = str(row["query"])
        query_norm, query_tokens = _normalize_query(query), _query_tokens(query)
        for old in historical:
            old_query = str(old["query"])
            old_id = str(old.get("query_id", "historical-unknown"))
            if query == old_query:
                exact.append({"query_id": row["query_id"], "historical_query_id": old_id})
                continue
            if query_norm == _normalize_query(old_query):
                normalized.append({"query_id": row["query_id"], "historical_query_id": old_id})
                continue
            old_tokens = _query_tokens(old_query)
            union = query_tokens | old_tokens
            score = len(query_tokens & old_tokens) / len(union) if union else 0.0
            if min(len(query_tokens), len(old_tokens)) >= 5 and score >= .80:
                high_overlap.append({
                    "query_id": row["query_id"], "historical_query_id": old_id,
                    "token_jaccard": round(score, 6),
                })
    if exact or normalized or high_overlap:
        _fail("HISTORICAL_QUERY_LEAKAGE", {"exact": exact, "normalized": normalized, "high_overlap": high_overlap})
    return {
        "historical_queries_audited": len(historical),
        "current_queries_audited": len(current),
        "exact_matches": 0,
        "normalized_matches": 0,
        "high_token_overlap_matches": 0,
        "embedding_similarity_used": "NO",
    }


def holdout_document_audit(
    train: dict[str, Any], tune: dict[str, Any], holdout_manifests: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    current = list(_documents(train).values()) + list(_documents(tune).values())
    holdouts = [document for manifest in holdout_manifests for document in manifest.get("documents", [])]
    current_ids = {str(row.get("document_id", "")).casefold() for row in current}
    holdout_ids = {str(row.get("document_id", "")).casefold() for row in holdouts}
    current_urls = {str(row.get("official_url", "")).casefold() for row in current} - {""}
    holdout_urls = {str(row.get("official_url", "")).casefold() for row in holdouts} - {""}
    current_paths = {
        str(value).casefold()
        for row in current for value in (row.get("source_path", ""), row.get("file", ""))
    } - {""}
    holdout_paths = {
        str(value).casefold()
        for row in holdouts for value in (row.get("source_path", ""), row.get("file", ""))
    } - {""}
    leaks = {
        "document_ids": sorted((current_ids & holdout_ids) - {""}),
        "official_urls": sorted(current_urls & holdout_urls),
        "exact_source_paths": sorted(current_paths & holdout_paths),
    }
    if any(leaks.values()):
        _fail("D_E_DOCUMENT_LEAKAGE", leaks)
    return {
        "holdout_documents_audited": len(holdouts),
        "d_e_document_leakage": 0,
        "document_id_matches": 0,
        "official_url_matches": 0,
        "exact_source_path_matches": 0,
    }


def validate_benchmark(
    train: dict[str, Any], tune: dict[str, Any], historical_manifests: Iterable[dict[str, Any]] = (),
    holdout_manifests: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    train_report = validate_manifest(train)
    tune_report = validate_manifest(tune)
    independence = validate_independence(train, tune)
    documents = list(_documents(train).values()) + list(_documents(tune).values())
    if len(documents) < 8:
        _fail("CORPUS_DOCUMENT_MINIMUM")
    if len({row["manufacturer"] for row in documents}) < 4:
        _fail("CORPUS_MANUFACTURER_MINIMUM")
    if len({row["equipment_type"] for row in documents}) < 3:
        _fail("CORPUS_CATEGORY_MINIMUM")
    similarity = query_similarity_audit(train, tune, historical_manifests)
    holdout_audit = holdout_document_audit(train, tune, holdout_manifests)
    return {
        "validity": "VALID",
        "benchmark_version": BENCHMARK_VERSION,
        "sealed_gate": "NO",
        "train": train_report,
        "tune": tune_report,
        "independence": independence,
        "query_similarity": similarity,
        "holdout_document_audit": holdout_audit,
        "freeze": {"train": train["freeze"], "tune": tune["freeze"]},
    }


def decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = [row for row in rows if row["answerable"]]
    abstains = [row for row in rows if not row["answerable"]]
    false_answers = [row["query_id"] for row in abstains if row["decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in answers if row["decision"] == "ABSTAIN"]
    return {
        "decision_accuracy": sum((row["decision"] == "ANSWER") == row["answerable"] for row in rows) / len(rows),
        "answerable_recall": 1 - len(false_refusals) / len(answers) if answers else 0.0,
        "abstain_recall": 1 - len(false_answers) / len(abstains) if abstains else 0.0,
        "false_answer_rate": len(false_answers) / len(abstains) if abstains else 0.0,
        "false_refusal_rate": len(false_refusals) / len(answers) if answers else 0.0,
        "false_answer_ids": false_answers,
        "false_refusal_ids": false_refusals,
    }


def evaluate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    distribution = validate_manifest(manifest)
    snapshot = [deserialize_document(row) for row in manifest["corpus_snapshot"]]
    candidates = {key: deserialize_candidate(value) for key, value in manifest["candidates"].items()}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query in manifest["queries"]:
        selected = [candidates[chunk_id] for chunk_id in query["candidate_chunk_ids"]]
        result = RetrievalResult(
            selected,
            query_analysis=analyze_query(query["query"], snapshot),
            corpus_documents=snapshot,
            retrieval_mode="corpus_g_candidate_fixed",
        )
        evidence = analyze_retrieval_evidence(query["query"], result, snapshot, "corpus_g_candidate_fixed")
        rows.append({
            **{key: query[key] for key in (
                "query_id", "pair_id", "query", "answerable", "document_id", "manufacturer",
                "manufacturer_slice", "failure_class", "focus", "difficulty", "document_style",
            )},
            "decision": evidence.decision,
            "reason": evidence.reason,
            "evidence": evidence.as_dict(),
        })

    def by(field: str) -> dict[str, Any]:
        return {
            value: decision_metrics([row for row in rows if row[field] == value])
            for value in sorted({row[field] for row in rows})
        }

    return {
        "development_set_id": manifest["development_set_id"],
        "benchmark_version": BENCHMARK_VERSION,
        "split": manifest["split"],
        "validity": "VALID",
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "freeze": manifest["freeze"],
        "distribution": distribution,
        "metrics": decision_metrics(rows),
        "by_difficulty": by("difficulty"),
        "by_document_style": by("document_style"),
        "by_manufacturer_slice": by("manufacturer_slice"),
        "by_failure_class": by("failure_class"),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "live_retrieval": False,
        "pdf_parser": False,
        "nli_used": "NO",
    }


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("split") != candidate.get("split"):
        _fail("COMPARISON_SPLIT_MISMATCH")
    if baseline.get("freeze", {}).get("manifest_sha256") != candidate.get("freeze", {}).get("manifest_sha256"):
        _fail("COMPARISON_MANIFEST_MISMATCH")

    metric_names = (
        "decision_accuracy", "answerable_recall", "abstain_recall",
        "false_answer_rate", "false_refusal_rate",
    )

    def deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
        return {name: after[name] - before[name] for name in metric_names}

    def slice_deltas(field: str) -> dict[str, Any]:
        before, after = baseline[field], candidate[field]
        if set(before) != set(after):
            _fail("COMPARISON_SLICE_MISMATCH", field)
        return {name: deltas(before[name], after[name]) for name in sorted(before)}

    return {
        "benchmark_version": candidate["benchmark_version"],
        "split": candidate["split"],
        "baseline_rule_version": baseline["evidence_rule_version"],
        "candidate_rule_version": candidate["evidence_rule_version"],
        "support_rule_version": candidate["support_rule_version"],
        "manifest_sha256": candidate["freeze"]["manifest_sha256"],
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "metric_delta_candidate_minus_baseline": deltas(baseline["metrics"], candidate["metrics"]),
        "slice_deltas": {
            "difficulty": slice_deltas("by_difficulty"),
            "document_style": slice_deltas("by_document_style"),
            "manufacturer_slice": slice_deltas("by_manufacturer_slice"),
            "failure_class": slice_deltas("by_failure_class"),
        },
    }


def _public(report: Any) -> Any:
    if isinstance(report, dict):
        return {key: _public(value) for key, value in report.items() if key != "rows" and not key.endswith("_ids")}
    if isinstance(report, list):
        return [_public(value) for value in report]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate", "compare"))
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--tune", type=Path, default=TUNE_PATH)
    parser.add_argument("--split", choices=("train", "tune"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        if not args.baseline_report or not args.candidate_report or not args.output:
            parser.error("compare requires --baseline-report, --candidate-report, and --output")
        report = compare_reports(
            read_json(ensure_private_path(args.baseline_report)),
            read_json(ensure_private_path(args.candidate_report)),
        )
        output = ensure_private_path(args.output)
        if output.exists():
            raise FileExistsError(f"V322_RESULT_ALREADY_EXISTS:{output}")
        atomic_write_json(output, report)
        print(json.dumps(_public(report), ensure_ascii=True, indent=2))
        return 0
    train = read_json(ensure_private_path(args.train))
    tune = read_json(ensure_private_path(args.tune))
    if args.command == "validate":
        history_paths = args.history or list(DEFAULT_HISTORY_PATHS)
        holdout_paths = args.holdout_manifest or list(DEFAULT_HOLDOUT_PATHS)
        history = [read_json(ensure_private_path(path)) for path in history_paths]
        holdouts = [read_json(ensure_private_path(path)) for path in holdout_paths]
        report = validate_benchmark(train, tune, history, holdouts)
    else:
        if not args.split or not args.output:
            parser.error("evaluate requires --split and --output")
        report = evaluate_manifest(train if args.split == "train" else tune)
        output = ensure_private_path(args.output)
        if output.exists():
            raise FileExistsError(f"V322_RESULT_ALREADY_EXISTS:{output}")
        atomic_write_json(output, report)
    print(json.dumps(_public(report), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
