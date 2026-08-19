"""V3.26 sealed evidence-gate framework (framework only; no gate data).

Provides the frozen-candidate fingerprint, gate annotation schema validation,
document/query leakage auditing (D/E holdouts are audited hash-only, never by
reading their private query plaintext), the freeze ledger, and one-shot
execution enforcement.

All gate corpus, query, annotation and result data live under the ignored
``benchmark_private/`` tree and are never committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.evaluation.frozen_retrieval_artifact import file_sha256
from backend.evaluation.v311_resume import hash_json

SEALED_GATE_ID = "evidence-sealed-gate-v1"

# Candidate source files whose exact bytes are part of the manifest fingerprint.
CANDIDATE_SOURCE_FILES: tuple[str, ...] = (
    "backend/retrieval/evidence.py",
    "backend/retrieval/evidence_contract.py",
    "backend/retrieval/technical.py",
    "backend/retrieval/semantic_judge.py",
    "backend/retrieval/semantic_judge_localnli.py",
)

# Frozen V3.25 local-NLI judge configuration (NOT to be retuned at gate time).
FROZEN_JUDGE_CONFIG: dict[str, Any] = {
    "candidate_version": "evidence-v325-local-nli-candidate",
    "candidate_status": "EXPERIMENTAL_CANDIDATE",
    "model_name": "cross-encoder/nli-deberta-v3-xsmall",
    "entailment_threshold": 0.5,
    "contradiction_threshold": 0.5,
    "unknown_floor": 0.33,
    "unknown_policy": "A",
    "semantic_judge_default": "OFF",
}

SUPPORT_BLOB_VERSION = "support-v316.1"

ALLOWED_RELATION_TYPES = frozenset(
    {
        "ROLE", "PREDICATE", "CONDITION", "ACTION", "DEFAULT_TARGET",
        "DIRECTION_ORDER", "QUANTIFIER", "OWNERSHIP", "SUBREGISTER",
    }
)
ALLOWED_CONFIDENCE = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
ALLOWED_GROUND_TRUTH = frozenset({"ANSWER", "ABSTAIN"})


# ---------------------------------------------------------------------------
# Candidate fingerprint
# ---------------------------------------------------------------------------

MODEL_SNAPSHOT_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)


def model_snapshot_hashes(snapshot_dir: Path) -> dict[str, str]:
    """SHA256 of the canonical local NLI snapshot files (reproducibility record)."""
    snapshot = Path(snapshot_dir)
    result: dict[str, str] = {}
    for rel in MODEL_SNAPSHOT_FILES:
        path = snapshot / rel
        if path.exists():
            result[rel] = file_sha256(path)
        else:
            result[rel] = "MISSING"
    return result


def validate_model_snapshot(snapshot_dir: Path, expected: dict[str, str]) -> list[str]:
    actual = model_snapshot_hashes(snapshot_dir)
    mismatches = [rel for rel in MODEL_SNAPSHOT_FILES if actual.get(rel) != expected.get(rel)]
    return [f"MODEL_FILE_MISMATCH:{rel}" for rel in mismatches]


def candidate_fingerprint(project_root: Path) -> dict[str, Any]:
    """Deterministic fingerprint of the frozen candidate (code + config + model + support)."""
    root = project_root.resolve()
    files = {rel: file_sha256(root / rel) for rel in CANDIDATE_SOURCE_FILES}
    support_blob = file_sha256(root / "backend" / "retrieval" / "evidence_support.py")
    manifest = {
        "sealed_gate_id": SEALED_GATE_ID,
        "source_files": files,
        "support": {"version": SUPPORT_BLOB_VERSION, "evidence_support_sha256": support_blob},
        "judge": FROZEN_JUDGE_CONFIG,
    }
    manifest["candidate_manifest_hash"] = hash_json(manifest)
    return manifest


def validate_candidate_fingerprint(expected: dict[str, Any], project_root: Path) -> list[str]:
    """Return the set of fingerprint fields that changed vs the frozen manifest."""
    current = candidate_fingerprint(project_root)
    problems: list[str] = []
    for rel in CANDIDATE_SOURCE_FILES:
        if current["source_files"][rel] != expected.get("source_files", {}).get(rel):
            problems.append(f"SOURCE_CHANGED:{rel}")
    if current["support"]["evidence_support_sha256"] != expected.get("support", {}).get("evidence_support_sha256"):
        problems.append("SUPPORT_CHANGED:evidence_support.py")
    if current["judge"] != expected.get("judge"):
        problems.append("JUDGE_CONFIG_CHANGED")
    return problems


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationAnnotation:
    subject: str | None
    predicate: str | None
    object: str | None
    condition: str | None
    direction: str | None


@dataclass(frozen=True)
class GateQueryAnnotation:
    query_id: str
    query: str
    ground_truth: str
    manufacturer: str
    product: str
    document: str
    category: str
    difficulty: str
    document_style: str
    confidence: str
    expected_evidence: str
    expected_scope: str
    critical_requirements: tuple[str, ...]
    relation_type: str | None
    relation: RelationAnnotation | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        data = asdict(self)
        data["critical_requirements"] = list(self.critical_requirements)
        return data


def validate_annotation(annotation: GateQueryAnnotation) -> list[str]:
    violations: list[str] = []
    if not annotation.query_id:
        violations.append("MISSING_query_id")
    if not annotation.query.strip():
        violations.append("EMPTY_query")
    if annotation.ground_truth not in ALLOWED_GROUND_TRUTH:
        violations.append(f"INVALID_ground_truth:{annotation.ground_truth}")
    if annotation.confidence not in ALLOWED_CONFIDENCE:
        violations.append(f"INVALID_confidence:{annotation.confidence}")
    if annotation.relation_type is not None and annotation.relation_type not in ALLOWED_RELATION_TYPES:
        violations.append(f"INVALID_relation_type:{annotation.relation_type}")
    if not annotation.difficulty:
        violations.append("MISSING_difficulty")
    if not annotation.rationale.strip():
        violations.append("MISSING_rationale")
    if annotation.relation is not None:
        for field in ("subject", "predicate", "object", "condition", "direction"):
            value = getattr(annotation.relation, field)
            if value is not None and not isinstance(value, str):
                violations.append(f"NON_STR_relation.{field}")
    return violations


def validate_annotation_set(annotations: Sequence[GateQueryAnnotation]) -> dict[str, Any]:
    per_row = {a.query_id: validate_annotation(a) for a in annotations}
    high = sum(1 for a in annotations if a.confidence == "HIGH")
    return {
        "count": len(annotations),
        "violations": {qid: v for qid, v in per_row.items() if v},
        "confidence": {"HIGH": high, "MEDIUM": sum(1 for a in annotations if a.confidence == "MEDIUM"), "AMBIGUOUS": sum(1 for a in annotations if a.confidence == "AMBIGUOUS")},
        "high_ratio": round(high / len(annotations), 4) if annotations else 0.0,
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _norm_query(text: str) -> str:
    return " ".join(str(text).lower().split())


def _tokens(text: str) -> set[str]:
    return {token for token in _norm_query(text).split() if len(token) > 1 and token.isalnum()}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def document_overlap_audit(
    gate_documents: Sequence[dict[str, Any]],
    prior_documents: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare Corpus H documents against prior-corpus document metadata (no query access)."""
    def keys(documents: Iterable[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
        ids, urls, paths = set(), set(), set()
        for doc in documents:
            for field in ("document_id",):
                value = str(doc.get(field, "")).casefold()
                if value:
                    ids.add(value)
            for field in ("official_url", "source_url", "url"):
                value = str(doc.get(field, "")).casefold()
                if value:
                    urls.add(value)
            for field in ("file", "source_path"):
                value = str(doc.get(field, "")).casefold()
                if value:
                    paths.add(value)
        return ids, urls, paths

    g_ids, g_urls, g_paths = keys(gate_documents)
    p_ids, p_urls, p_paths = keys(prior_documents)
    return {
        "gate_documents": len(gate_documents),
        "prior_documents": len(prior_documents),
        "document_id_overlap": sorted((g_ids & p_ids) - {""}),
        "official_url_overlap": sorted((g_urls & p_urls) - {""}),
        "source_path_overlap": sorted((g_paths & p_paths) - {""}),
    }


def query_duplicate_audit(
    gate_queries: Sequence[str],
    prior_query_rows: Sequence[dict[str, Any]],
    *,
    hash_only: bool = False,
) -> dict[str, Any]:
    """Audit gate queries against prior queries.

    ``prior_query_rows`` have a ``query_hashes`` dict (hash_json(query_text) -> query_id)
    when hash_only is True (used for D/E holdouts whose plaintext must not be read),
    otherwise a ``text`` field carrying the plaintext.
    """
    exact = []
    normalized = []
    high_overlap = []
    if hash_only:
        prior_hashes: dict[str, str] = {}
        for row in prior_query_rows:
            for value, qid in (row.get("query_hashes") or {}).items():
                prior_hashes[value] = qid
        for query in gate_queries:
            query_hash = hash_json(query)
            if query_hash in prior_hashes:
                exact.append({"gate_query": query, "prior_hash": query_hash, "prior_query_id": prior_hashes[query_hash]})
        return {
            "mode": "hash_only",
            "exact_duplicate": exact,
            "normalized_duplicate": "NOT_AUDITED",
            "high_token_overlap": "NOT_AUDITED",
        }

    tokens_by_prior = []
    norm_prior: dict[str, str] = {}
    for row in prior_query_rows:
        text = str(row.get("text", ""))
        if text:
            tokens_by_prior.append((text, row.get("query_id", "")))
            norm_prior[_norm_query(text)] = row.get("query_id", "")
    for query in gate_queries:
        nq = _norm_query(query)
        if nq in norm_prior:
            normalized.append({"gate_query": query, "prior_query_id": norm_prior[nq]})
        for text, qid in tokens_by_prior:
            if _token_jaccard(query, text) >= 0.8 and _norm_query(query) != _norm_query(text):
                high_overlap.append({"gate_query": query, "prior_query_id": qid, "jaccard": round(_token_jaccard(query, text), 3)})
    return {
        "mode": "full",
        "exact_duplicate": [],  # exact == normalized here
        "normalized_duplicate": normalized,
        "high_token_overlap": high_overlap,
    }


# ---------------------------------------------------------------------------
# Freeze ledger + one-shot enforcement
# ---------------------------------------------------------------------------

def gate_hashes(
    manifest: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, str]:
    return {
        "corpus_manifest_hash": hash_json(manifest),
        "query_hash": hash_json(queries),
        "annotation_hash": hash_json(annotations),
        "gate_config_hash": hash_json(config),
    }


def freeze_ledger(
    candidate: dict[str, Any],
    hashes: dict[str, str],
    *,
    frozen_at: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    ledger = {
        "sealed_gate_id": SEALED_GATE_ID,
        "created_at": created_at or frozen_at,
        "frozen_at": frozen_at,
        "candidate_fingerprint": {
            "candidate_manifest_hash": candidate["candidate_manifest_hash"],
            "judge": candidate["judge"],
        },
        "corpus_manifest_hash": hashes["corpus_manifest_hash"],
        "query_hash": hashes["query_hash"],
        "annotation_hash": hashes["annotation_hash"],
        "gate_config_hash": hashes["gate_config_hash"],
        "first_execution_at": None,
    }
    return ledger


def enforce_one_shot(ledger: dict[str, Any], candidate_hash: str, hashes: dict[str, str]) -> list[str]:
    """Return integrity/one-shot violations before a gate execution.

    Order of checks matters: candidate/freeze integrity first, one-shot last.
    """
    violations: list[str] = []
    if ledger.get("sealed_gate_id") != SEALED_GATE_ID:
        violations.append("SEALED_GATE_ID_MISMATCH")
    frozen = ledger.get("candidate_fingerprint", {})
    if frozen.get("candidate_manifest_hash") != candidate_hash:
        violations.append("CANDIDATE_MANIFEST_MISMATCH")
    for field in ("corpus_manifest_hash", "query_hash", "annotation_hash", "gate_config_hash"):
        if ledger.get(field) != hashes.get(field):
            violations.append(f"GATE_{field.upper()}_MUTATED")
    if ledger.get("first_execution_at"):
        violations.append("ONE_SHOT_VIOLATION")
    return violations


def record_execution(ledger: dict[str, Any], executed_at: str, result_hashes: dict[str, str]) -> dict[str, Any]:
    updated = dict(ledger)
    updated["first_execution_at"] = executed_at
    updated["result_hashes"] = result_hashes
    return updated