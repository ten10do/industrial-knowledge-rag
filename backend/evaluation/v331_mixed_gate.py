"""V3.31 mixed sealed evidence-gate framework (framework only; no gate data).

Freezes the full V3.30 *mixed* candidate (V3.25 verification stack + V3.30 open
sufficiency + V3.29 boundary) as a deterministic fingerprint, records the local
NLI model snapshot, pre-registers the gate decision policy, and validates the
mixed-gate query/annotation schema (distribution, balance, difficulty, quota,
confidence, review).  It also provides the one-shot execution ledger and the
failure-attribution taxonomy.

All gate corpus, query, annotation and result data live under the ignored
``benchmark_private/`` tree and are never committed.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.evaluation.frozen_retrieval_artifact import file_sha256
from backend.evaluation.v311_resume import hash_json

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

SEALED_GATE_ID = "mixed-evidence-sealed-gate-v2"
CANDIDATE_VERSION = "evidence-v330-mixed-candidate"
CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

# V3.25 verification stack + V3.30 open sufficiency + V3.29 boundary, plus the
# runtime Evidence plumbing those paths import.  ``semantic_judge.py`` carries
# the V3.24 ambiguity router and the judge contract; ``evidence_contract.py``
# carries the claim parser/segments; ``evidence_querytype.py`` the query router;
# ``evidence_answerobject.py`` the V3.28 (non-gating) answer-object surface.
CANDIDATE_SOURCE_FILES: tuple[str, ...] = (
    "backend/retrieval/evidence.py",
    "backend/retrieval/evidence_contract.py",
    "backend/retrieval/technical.py",
    "backend/retrieval/semantic_judge.py",
    "backend/retrieval/semantic_judge_localnli.py",
    "backend/retrieval/evidence_answerobject.py",
    "backend/retrieval/evidence_querytype.py",
    "backend/retrieval/evidence_boundary.py",
    "backend/retrieval/evidence_openquestion.py",
)
SUPPORT_SOURCE_FILE = "backend/retrieval/evidence_support.py"
SUPPORT_BLOB_VERSION = "support-v316.1"

MODEL_REPO = "cross-encoder/nli-deberta-v3-xsmall"
LABEL_MAPPING = {0: "contradiction", 1: "entailment", 2: "neutral"}

FROZEN_JUDGE_CONFIG: dict[str, Any] = {
    "candidate_version": "evidence-v325-local-nli-candidate",
    "candidate_status": "EXPERIMENTAL_CANDIDATE",
    "model_repo": MODEL_REPO,
    "label_mapping": {"0": "contradiction", "1": "entailment", "2": "neutral"},
    "entailment_threshold": 0.5,
    "contradiction_threshold": 0.5,
    "unknown_floor": 0.33,
    "unknown_policy": "A",
    "unknown_fallback": "rule_fallback",
}

PRODUCTION_DEFAULTS: dict[str, str] = {
    "SEMANTIC_JUDGE_DEFAULT": "OFF",
    "OPEN_QUESTION_SUFFICIENCY_DEFAULT": "OFF",
    "GROUNDING_ENRICHMENT_DEFAULT": "OFF",
    "SUPPORT_GATE": "OFF",
    "RERANKER": "OFF",
}

# ---------------------------------------------------------------------------
# Candidate + model fingerprint
# ---------------------------------------------------------------------------

MODEL_SNAPSHOT_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)


def model_snapshot_hashes(snapshot_dir: Path) -> dict[str, str]:
    snapshot = Path(snapshot_dir)
    return {
        rel: file_sha256(snapshot / rel) if (snapshot / rel).exists() else "MISSING"
        for rel in MODEL_SNAPSHOT_FILES
    }


def validate_model_snapshot(snapshot_dir: Path, expected: dict[str, str]) -> list[str]:
    actual = model_snapshot_hashes(snapshot_dir)
    return [f"MODEL_FILE_MISMATCH:{rel}" for rel in MODEL_SNAPSHOT_FILES if actual.get(rel) != expected.get(rel)]


def _resolve(rel: str, project_root: Path) -> str:
    path = (project_root / rel).resolve()
    return file_sha256(path) if path.exists() else "MISSING"


def candidate_fingerprint(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    files = {rel: _resolve(rel, root) for rel in CANDIDATE_SOURCE_FILES}
    support = _resolve(SUPPORT_SOURCE_FILE, root)
    manifest: dict[str, Any] = {
        "sealed_gate_id": SEALED_GATE_ID,
        "candidate_version": CANDIDATE_VERSION,
        "candidate_status": CANDIDATE_STATUS,
        "source_files": files,
        "support": {"version": SUPPORT_BLOB_VERSION, "evidence_support_sha256": support},
        "judge": FROZEN_JUDGE_CONFIG,
        "model_repo": MODEL_REPO,
        "production_defaults": PRODUCTION_DEFAULTS,
    }
    manifest["candidate_manifest_hash"] = hash_json(manifest)
    return manifest


def validate_candidate_fingerprint(expected: dict[str, Any], project_root: Path) -> list[str]:
    current = candidate_fingerprint(project_root)
    problems: list[str] = []
    for rel in CANDIDATE_SOURCE_FILES:
        if current["source_files"][rel] != expected.get("source_files", {}).get(rel):
            problems.append(f"SOURCE_CHANGED:{rel}")
    if current["support"]["evidence_support_sha256"] != expected.get("support", {}).get("evidence_support_sha256"):
        problems.append("SUPPORT_CHANGED:evidence_support.py")
    if current["judge"] != expected.get("judge"):
        problems.append("JUDGE_CONFIG_CHANGED")
    if current["production_defaults"] != expected.get("production_defaults"):
        problems.append("PRODUCTION_DEFAULTS_CHANGED")
    return problems


def candidate_manifest(project_root: Path, *, snapshot_hashes: dict[str, str]) -> dict[str, Any]:
    manifest = candidate_fingerprint(project_root)
    manifest["model_snapshot"] = {
        "repo": MODEL_REPO,
        "label_mapping": LABEL_MAPPING,
        "files": snapshot_hashes,
    }
    manifest["candidate_manifest_hash"] = hash_json(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Query / annotation schema
# ---------------------------------------------------------------------------

ALLOWED_QUERY_TYPES = frozenset({"VERIFICATION", "OPEN"})
ALLOWED_GROUND_TRUTH = frozenset({"ANSWER", "ABSTAIN"})
ALLOWED_CONFIDENCE = frozenset({"HIGH", "MEDIUM", "AMBIGUOUS"})
ALLOWED_DIFFICULTY = frozenset({"L1", "L2", "L3", "L4", "L5"})
ALLOWED_CATEGORIES = frozenset({
    "identifier", "value", "action_procedure", "relation", "attribute",
    "protocol", "terminal_channel", "scope", "condition_mode", "target",
    "safety", "procedure",
})
HARD_NEAR_MISS_LEVELS = frozenset({"L3", "L4", "L5"})

# V3.30 open relation types that open queries may target.
ALLOWED_OPEN_RELATIONS = frozenset({
    "HAS_IDENTIFIER", "HAS_VALUE", "HAS_DEFAULT_VALUE", "HAS_RANGE",
    "HAS_SETTING", "USES_TERMINAL", "USES_CHANNEL", "HAS_ATTRIBUTE",
    "REQUIRES_ACTION", "LOCATED_AT", "USES_PROTOCOL", "HAS_PROCEDURE",
})

# V3.27/V3.26 relation-annotation predicates (verification relation reasoning).
ALLOWED_RELATION_PREDICATES = frozenset({
    "ROLE", "PREDICATE", "CONDITION", "ACTION", "DEFAULT_TARGET",
    "DIRECTION_ORDER", "QUANTIFIER", "OWNERSHIP", "SUBREGISTER",
})

FAILURE_ATTRIBUTION_TYPES = frozenset({
    "RETRIEVAL_MISSING_EVIDENCE",
    "PARSER_STRUCTURE_LOSS",
    "PRODUCT_IDENTITY_ERROR",
    "EVIDENCE_VERIFICATION_FAILURE",
    "OPEN_SUFFICIENCY_FAILURE",
    "NLI_JUDGE_FAILURE",
    "ROUTER_MISS",
    "ANNOTATION_AMBIGUITY",
    "OTHER",
})


@dataclasses.dataclass(frozen=True)
class RelationAnnotation:
    subject: str | None
    predicate: str | None
    object: str | None
    condition: str | None
    direction: str | None


@dataclasses.dataclass(frozen=True)
class MixedGateAnnotation:
    query_id: str
    query: str
    ground_truth: str
    query_type: str
    manufacturer: str
    model: str
    document: str
    category: str
    difficulty: str
    document_style: str
    confidence: str
    target: str
    relation: str | None
    requested_slot: str | None
    critical_requirements: tuple[str, ...]
    expected_evidence: str
    expected_scope: str
    relation_annotation: RelationAnnotation | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["critical_requirements"] = list(self.critical_requirements)
        return data


def validate_annotation(annotation: MixedGateAnnotation) -> list[str]:
    v: list[str] = []
    if not annotation.query_id:
        v.append("MISSING_query_id")
    if not annotation.query.strip():
        v.append("EMPTY_query")
    if annotation.query_type not in ALLOWED_QUERY_TYPES:
        v.append(f"INVALID_query_type:{annotation.query_type}")
    if annotation.ground_truth not in ALLOWED_GROUND_TRUTH:
        v.append(f"INVALID_ground_truth:{annotation.ground_truth}")
    if annotation.difficulty not in ALLOWED_DIFFICULTY:
        v.append(f"INVALID_difficulty:{annotation.difficulty}")
    if annotation.confidence not in ALLOWED_CONFIDENCE:
        v.append(f"INVALID_confidence:{annotation.confidence}")
    if annotation.category not in ALLOWED_CATEGORIES:
        v.append(f"INVALID_category:{annotation.category}")
    if annotation.relation is not None and annotation.relation not in ALLOWED_OPEN_RELATIONS:
        v.append(f"INVALID_open_relation:{annotation.relation}")
    if not annotation.rationale.strip():
        v.append("MISSING_rationale")
    if not annotation.expected_evidence.strip():
        v.append("MISSING_expected_evidence")
    if annotation.relation_annotation is not None:
        for field in ("subject", "predicate", "object", "condition", "direction"):
            value = getattr(annotation.relation_annotation, field)
            if value is not None and not isinstance(value, str):
                v.append(f"NON_STR_relation.{field}")
    return v


def validate_annotation_set(annotations: Sequence[MixedGateAnnotation]) -> dict[str, Any]:
    per_row = {a.query_id: validate_annotation(a) for a in annotations}
    high = sum(1 for a in annotations if a.confidence == "HIGH")
    return {
        "count": len(annotations),
        "violations": {qid: v for qid, v in per_row.items() if v},
        "confidence": {
            "HIGH": high,
            "MEDIUM": sum(1 for a in annotations if a.confidence == "MEDIUM"),
            "AMBIGUOUS": sum(1 for a in annotations if a.confidence == "AMBIGUOUS"),
        },
        "high_ratio": round(high / len(annotations), 4) if annotations else 0.0,
    }


# ---------------------------------------------------------------------------
# Mixed-gate distribution / quota validators
# ---------------------------------------------------------------------------

def validate_mixed_distribution(annotations: Sequence[MixedGateAnnotation]) -> list[str]:
    """Enforce query-type split (>=40% each), ANSWER/ABSTAIN balance, difficulty,
    relation coverage, identifier and action quotas, and confidence minimums."""
    problems: list[str] = []
    n = len(annotations)
    if n < 56:
        problems.append(f"QUERY_COUNT_BELOW_MIN:56:{n}")

    verification = [a for a in annotations if a.query_type == "VERIFICATION"]
    open_q = [a for a in annotations if a.query_type == "OPEN"]
    if n and len(verification) / n < 0.40:
        problems.append(f"VERIFICATION_RATIO_BELOW_40:{round(len(verification) / n, 3)}")
    if n and len(open_q) / n < 0.40:
        problems.append(f"OPEN_RATIO_BELOW_40:{round(len(open_q) / n, 3)}")

    answer = [a for a in annotations if a.ground_truth == "ANSWER"]
    abstain = [a for a in annotations if a.ground_truth == "ABSTAIN"]
    answer_ratio = len(answer) / n if n else 0.0
    if not (0.45 <= answer_ratio <= 0.55):
        problems.append(f"ANSWER_BALANCE_OUT_OF_BAND_45_55:{round(answer_ratio, 3)}")

    hard = [a for a in annotations if a.difficulty in HARD_NEAR_MISS_LEVELS]
    if n and len(hard) / n < 0.80:
        problems.append(f"HARD_NEAR_MISS_RATIO_BELOW_80:{round(len(hard) / n, 3)}")

    identifier_pos = sum(1 for a in annotations if a.category == "identifier" and a.ground_truth == "ANSWER")
    identifier_neg = sum(1 for a in annotations if a.category == "identifier" and a.ground_truth == "ABSTAIN")
    if identifier_pos < 6:
        problems.append(f"IDENTIFIER_POSITIVE_BELOW_6:{identifier_pos}")
    if identifier_neg < 6:
        problems.append(f"IDENTIFIER_NEGATIVE_BELOW_6:{identifier_neg}")

    action_pos = sum(1 for a in annotations if a.category in {"action_procedure", "procedure"} and a.ground_truth == "ANSWER")
    action_neg = sum(1 for a in annotations if a.category in {"action_procedure", "procedure"} and a.ground_truth == "ABSTAIN")
    if action_pos < 4:
        problems.append(f"ACTION_POSITIVE_BELOW_4:{action_pos}")
    if action_neg < 4:
        problems.append(f"ACTION_NEGATIVE_BELOW_4:{action_neg}")

    high = sum(1 for a in annotations if a.confidence == "HIGH")
    if n and high / n < 0.85:
        problems.append(f"HIGH_CONFIDENCE_RATIO_BELOW_85:{round(high / n, 3)}")

    return problems


# ---------------------------------------------------------------------------
# Gate decision policy (pre-registered; frozen BEFORE execution)
# ---------------------------------------------------------------------------

GATE_DECISION_POLICY: dict[str, Any] = {
    "policy_id": "GATE_DECISION_POLICY_v331_1",
    "applies_to": "END_TO_END_GATE (primary); EVIDENCE_CONDITIONAL reported, not a gate decision",
    "PASS": {
        "overall_accuracy": ">= 0.70",
        "abstain_recall": ">= 0.60",
        "answerable_recall": ">= 0.55",
        "false_answer_rate": "<= 0.20",
        "false_refusal_rate": "<= 0.45",
        "verification_accuracy": ">= 0.68",
        "open_accuracy": ">= 0.65",
    },
    "PARTIAL": {
        "overall_accuracy": ">= 0.60",
        "abstain_recall": ">= 0.45",
        "answerable_recall": ">= 0.40",
    },
    "FAIL": "does not meet PARTIAL bands",
    "order": "PASS if all PASS conditions hold; else PARTIAL if all PARTIAL conditions hold; else FAIL",
}


def gate_decision_policy_hash(policy: dict[str, Any] | None = None) -> str:
    return hash_json(policy if policy is not None else GATE_DECISION_POLICY)


def _ge(value: float | None, bound: float) -> bool:
    return value is not None and value >= bound


def _le(value: float | None, bound: float) -> bool:
    return value is not None and value <= bound


def evaluate_gate_decision(metrics: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the pre-registered policy to the measured E2E metrics.  Returns the
    verdict plus the checked conditions so the decision is accountable, not
    post-hoc."""
    policy = policy if policy is not None else GATE_DECISION_POLICY
    v = metrics.get("verification", {})
    o = metrics.get("open", {})

    pass_checks = {
        "overall_accuracy": [_ge(metrics.get("accuracy"), 0.70), metrics.get("accuracy")],
        "abstain_recall": [_ge(metrics.get("abstain_recall"), 0.60), metrics.get("abstain_recall")],
        "answerable_recall": [_ge(metrics.get("answerable_recall"), 0.55), metrics.get("answerable_recall")],
        "false_answer_rate": [_le(metrics.get("false_answer_rate"), 0.20), metrics.get("false_answer_rate")],
        "false_refusal_rate": [_le(metrics.get("false_refusal_rate"), 0.45), metrics.get("false_refusal_rate")],
        "verification_accuracy": [_ge(v.get("accuracy"), 0.68), v.get("accuracy")],
        "open_accuracy": [_ge(o.get("accuracy"), 0.65), o.get("accuracy")],
    }
    partial_checks = {
        "overall_accuracy": [_ge(metrics.get("accuracy"), 0.60), metrics.get("accuracy")],
        "abstain_recall": [_ge(metrics.get("abstain_recall"), 0.45), metrics.get("abstain_recall")],
        "answerable_recall": [_ge(metrics.get("answerable_recall"), 0.40), metrics.get("answerable_recall")],
    }
    pass_ok = all(item[0] for item in pass_checks.values())
    partial_ok = all(item[0] for item in partial_checks.values())
    verdict = "PASS" if pass_ok else "PARTIAL" if partial_ok else "FAIL"
    return {
        "verdict": verdict,
        "policy_id": policy.get("policy_id"),
        "pass_checks": {k: {"held": item[0], "value": item[1]} for k, item in pass_checks.items()},
        "partial_checks": {k: {"held": item[0], "value": item[1]} for k, item in partial_checks.items()},
    }


# ---------------------------------------------------------------------------
# One-shot freeze ledger + enforcement
# ---------------------------------------------------------------------------

def gate_hashes(
    corpus_manifest: dict[str, Any] | str,
    queries: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    decision_policy: dict[str, Any],
    config: dict[str, Any],
    retrieval_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    return {
        "corpus_manifest_hash": hash_json(corpus_manifest),
        "query_hash": hash_json(queries),
        "annotation_hash": hash_json(annotations),
        "decision_policy_hash": gate_decision_policy_hash(decision_policy),
        "gate_config_hash": hash_json(config),
        "retrieval_config_hash": hash_json(retrieval_config) if retrieval_config is not None else "NOT_APPLICABLE",
    }


def freeze_ledger(
    candidate: dict[str, Any],
    hashes: dict[str, str],
    *,
    frozen_at: str,
) -> dict[str, Any]:
    return {
        "sealed_gate_id": SEALED_GATE_ID,
        "frozen_at": frozen_at,
        "candidate_fingerprint": {
            "candidate_manifest_hash": candidate["candidate_manifest_hash"],
            "judge": candidate["judge"],
            "model_snapshot": candidate.get("model_snapshot"),
        },
        **hashes,
        "first_execution_at": None,
    }


def enforce_one_shot(ledger: dict[str, Any], candidate_hash: str, hashes: dict[str, str]) -> list[str]:
    violations: list[str] = []
    if ledger.get("sealed_gate_id") != SEALED_GATE_ID:
        violations.append("SEALED_GATE_ID_MISMATCH")
    frozen = ledger.get("candidate_fingerprint", {})
    if frozen.get("candidate_manifest_hash") != candidate_hash:
        violations.append("CANDIDATE_MANIFEST_MISMATCH")
    for field in ("corpus_manifest_hash", "query_hash", "annotation_hash", "decision_policy_hash", "gate_config_hash"):
        if ledger.get(field) != hashes.get(field):
            violations.append(f"GATE_{field.upper()}_MUTATED")
    if ledger.get("first_execution_at"):
        violations.append("ONE_SHOT_VIOLATION")
    return violations


def record_execution(ledger: dict[str, Any], executed_at: str, result_hashes: dict[str, str]) -> dict[str, Any]:
    updated = dict(ledger)
    updated["first_execution_at"] = executed_at
    updated["result_hashes"] = result_hashes
    updated["official_gate_runs"] = 1
    return updated


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------

def attribute_failure(
    *,
    answerable: bool,
    decision: str,
    retrieved_evidence_present: bool,
    parser_ok: bool,
    identity_ok: bool,
) -> str:
    """Determine the failure category for a wrong decision (ANSWER when ABSTAIN
    expected, or ABSTAIN when ANSWER expected).  Callers that can't assess a
    stage should coalesce to OTHER rather than guess."""
    if answerable and decision == "ABSTAIN":
        if not retrieved_evidence_present:
            return "RETRIEVAL_MISSING_EVIDENCE"
        if not parser_ok:
            return "PARSER_STRUCTURE_LOSS"
        if not identity_ok:
            return "PRODUCT_IDENTITY_ERROR"
        return "OPEN_SUFFICIENCY_FAILURE"
    if not answerable and decision == "ANSWER":
        if not retrieved_evidence_present:
            return "RETRIEVAL_MISSING_EVIDENCE"
        if not identity_ok:
            return "PRODUCT_IDENTITY_ERROR"
        return "EVIDENCE_VERIFICATION_FAILURE"
    return "OTHER"


def collect_failure_attribution(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count failure categories present on rows tagged with ``failure_attribution``."""
    from collections import Counter
    return dict(Counter(str(row.get("failure_attribution", "OTHER")) for row in rows if row.get("failure_attribution")))


# ---------------------------------------------------------------------------
# Annotation review gate (second-pass; prediction-blind by construction)
# ---------------------------------------------------------------------------

REVIEW_MINIMUM_HARD_CASES = 24


def validate_review_gate(reviewed_query_ids: set[str] | None = None, minimum: int = REVIEW_MINIMUM_HARD_CASES) -> list[str]:
    """Enforce the second-pass annotation review quota (hardest cases re-reviewed
    by a reviewer who has seen only the manual + annotation, never a prediction)."""
    reviewed = len(reviewed_query_ids or set())
    if reviewed < minimum:
        return [f"REVIEW_COUNT_BELOW_MINIMUM:{minimum}:{reviewed}"]
    return []