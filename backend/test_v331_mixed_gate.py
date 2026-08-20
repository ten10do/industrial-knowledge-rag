"""Tests for the V3.31 mixed sealed evidence-gate framework.

Deterministic only: no model weights, no network, no gate data.  Pins the
candidate/manifest fingerprinting, the mixed-gate quota validators, the
pre-registered decision policy, one-shot enforcement, hash-only holdout
leakage enforcement, the failure-attribution schema, and the frozen retrieval
artifact contract.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.evaluation.v331_mixed_gate import (  # noqa: E402
    FAILURE_ATTRIBUTION_TYPES,
    GATE_DECISION_POLICY,
    PRODUCTION_DEFAULTS,
    SEALED_GATE_ID,
    MixedGateAnnotation,
    RelationAnnotation,
    candidate_fingerprint,
    candidate_manifest,
    enforce_one_shot,
    evaluate_gate_decision,
    freeze_ledger,
    gate_decision_policy_hash,
    record_execution,
    validate_annotation,
    validate_annotation_set,
    validate_candidate_fingerprint,
    validate_mixed_distribution,
    validate_review_gate,
)
from backend.evaluation.v326_sealed_gate import query_duplicate_audit  # noqa: E402
from backend.evaluation.frozen_retrieval_artifact import (  # noqa: E402
    new_artifact_payload,
    seal_artifact,
)


PROJECT = Path(__file__).resolve().parent.parent


def _annotation(query_id="j001", *, query_type="VERIFICATION", answerable=True,
                category="identifier", difficulty="L5", confidence="HIGH",
                relation=None, relation_annotation=None):
    return MixedGateAnnotation(
        query_id=query_id,
        query=f"query {query_id}",
        ground_truth="ANSWER" if answerable else "ABSTAIN",
        query_type=query_type,
        manufacturer="M",
        model="M1",
        document="doc-1",
        category=category,
        difficulty=difficulty,
        document_style="CONFIGURATION_TABLE",
        confidence=confidence,
        target="target",
        relation=relation,
        requested_slot=None,
        critical_requirements=("req",),
        expected_evidence="chunk-1",
        expected_scope="SAME_CHUNK",
        relation_annotation=relation_annotation,
        rationale="grounded",
    )


def _build() -> list[MixedGateAnnotation]:
    """A valid, fully-quota-compliant 60-query mixed set."""
    anns: list[MixedGateAnnotation] = []

    def add(answerable, query_type, category, difficulty):
        i = len(anns)
        anns.append(_annotation(query_id=f"j{i:02d}", query_type=query_type,
                                answerable=answerable, category=category, difficulty=difficulty))

    for _ in range(6):
        add(True, "VERIFICATION", "identifier", "L4")
    for _ in range(6):
        add(False, "VERIFICATION", "identifier", "L5")
    for _ in range(4):
        add(True, "OPEN", "action_procedure", "L4")
    for _ in range(4):
        add(False, "OPEN", "action_procedure", "L5")
    for _ in range(10):
        add(True, "OPEN", "value", "L4")
    for _ in range(10):
        add(False, "OPEN", "value", "L5")
    for _ in range(10):
        add(True, "VERIFICATION", "attribute", "L3")
    for _ in range(10):
        add(False, "VERIFICATION", "relation", "L4")
    return anns


# ---------------------------------------------------------------------------
# Candidate / manifest fingerprint
# ---------------------------------------------------------------------------

def test_candidate_fingerprint_reproducible():
    a = candidate_fingerprint(PROJECT)
    b = candidate_fingerprint(PROJECT)
    assert a["candidate_manifest_hash"] == b["candidate_manifest_hash"]
    assert a["sealed_gate_id"] == SEALED_GATE_ID == "mixed-evidence-sealed-gate-v2"
    assert len(a["source_files"]) >= 9
    assert a["support"]["version"] == "support-v316.1"
    assert a["judge"]["unknown_fallback"] == "rule_fallback"
    assert a["production_defaults"] == PRODUCTION_DEFAULTS


def test_candidate_manifest_includes_model_snapshot():
    manifest = candidate_manifest(PROJECT, snapshot_hashes={"model.safetensors": "abc", "config.json": "def",
                                                            "tokenizer_config.json": "x", "tokenizer.json": "y",
                                                            "special_tokens_map.json": "z"})
    assert "candidate_manifest_hash" in manifest
    assert manifest["model_snapshot"]["repo"] == "cross-encoder/nli-deberta-v3-xsmall"
    assert manifest["model_snapshot"]["files"]["model.safetensors"] == "abc"
    assert manifest["model_snapshot"]["label_mapping"][1] == "entailment"


def test_fingerprint_mutation_rejection():
    fp = candidate_fingerprint(PROJECT)
    mutated = dict(fp)
    mutated["judge"] = {**fp["judge"], "entailment_threshold": 0.99}
    problems = validate_candidate_fingerprint(mutated, PROJECT)
    assert any("JUDGE_CONFIG_CHANGED" in p for p in problems)


# ---------------------------------------------------------------------------
# Annotation schema + distribution validators
# ---------------------------------------------------------------------------

def test_annotation_schema_validation():
    assert validate_annotation(_annotation()) == []
    assert validate_annotation(_annotation(query_id="")) == ["MISSING_query_id"]
    assert any("INVALID_query_type" in v for v in validate_annotation(_annotation(query_type="POLAR")))
    assert any("INVALID_confidence" in v for v in validate_annotation(_annotation(confidence="LOW")))


def test_relation_annotation_non_str_rejected():
    ann = _annotation(relation_annotation=RelationAnnotation("s", "p", 123, None, None))
    assert any("NON_STR_relation.object" in v for v in validate_annotation(ann))


def test_mixed_distribution_valid():
    assert validate_mixed_distribution(_build()) == []


def test_mixed_distribution_rejects_polar_heavy():
    anns = [dataclasses.replace(a, query_type="OPEN") for a in _build()]
    problems = validate_mixed_distribution(anns)
    assert any("VERIFICATION_RATIO" in p for p in problems)


def test_answer_abstain_balance_enforced():
    anns = [dataclasses.replace(a, ground_truth="ABSTAIN")
            if i < 20 else a for i, a in enumerate(_build())]
    problems = validate_mixed_distribution(anns)
    assert any("ANSWER_BALANCE" in p for p in problems)


def test_hard_near_miss_ratio_enforced():
    anns = [dataclasses.replace(a, difficulty="L2")
            if i < 20 else a for i, a in enumerate(_build())]
    problems = validate_mixed_distribution(anns)
    assert any("HARD_NEAR_MISS_RATIO" in p for p in problems)


def test_identifier_quota_enforced():
    anns = [dataclasses.replace(a, category="value")
            if a.category == "identifier" else a for a in _build()]
    problems = validate_mixed_distribution(anns)
    assert any("IDENTIFIER_POSITIVE" in p for p in problems)
    assert any("IDENTIFIER_NEGATIVE" in p for p in problems)


def test_action_quota_enforced():
    anns = [dataclasses.replace(a, category="value")
            if a.category == "action_procedure" else a for a in _build()]
    problems = validate_mixed_distribution(anns)
    assert any("ACTION_POSITIVE" in p for p in problems)
    assert any("ACTION_NEGATIVE" in p for p in problems)


def test_confidence_high_ratio_enforced():
    anns = [dataclasses.replace(a, confidence="MEDIUM")
            if i < 20 else a for i, a in enumerate(_build())]
    problems = validate_mixed_distribution(anns)
    assert any("HIGH_CONFIDENCE" in p for p in problems)


def test_annotation_set_validation_summary():
    anns = [_annotation(query_id=f"j{i:03d}") for i in range(10)]
    report = validate_annotation_set(anns)
    assert report["violations"] == {}
    assert report["confidence"]["HIGH"] == 10
    assert report["high_ratio"] == 1.0


def test_review_gate_minimum():
    assert validate_review_gate(set(range(24))) == []
    assert any("REVIEW_COUNT" in p for p in validate_review_gate(set(range(10))))


# ---------------------------------------------------------------------------
# Decision policy (pre-registered, deterministic)
# ---------------------------------------------------------------------------

def test_decision_policy_hash_stable():
    assert gate_decision_policy_hash() == gate_decision_policy_hash(GATE_DECISION_POLICY)


def test_decision_policy_pass_partial_fail():
    pass_metrics = {"accuracy": 0.75, "abstain_recall": 0.7, "answerable_recall": 0.65,
                    "false_answer_rate": 0.1, "false_refusal_rate": 0.35,
                    "verification": {"accuracy": 0.72}, "open": {"accuracy": 0.7}}
    assert evaluate_gate_decision(pass_metrics)["verdict"] == "PASS"

    partial_metrics = {"accuracy": 0.63, "abstain_recall": 0.5, "answerable_recall": 0.45,
                       "false_answer_rate": 0.3, "false_refusal_rate": 0.5,
                       "verification": {"accuracy": 0.5}, "open": {"accuracy": 0.5}}
    assert evaluate_gate_decision(partial_metrics)["verdict"] == "PARTIAL"

    fail_metrics = {"accuracy": 0.5, "abstain_recall": 0.4, "answerable_recall": 0.3,
                    "false_answer_rate": 0.4, "false_refusal_rate": 0.6,
                    "verification": {"accuracy": 0.4}, "open": {"accuracy": 0.4}}
    assert evaluate_gate_decision(fail_metrics)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# One-shot + mutation rejection
# ---------------------------------------------------------------------------

def _ledger_and_hashes():
    candidate = {"candidate_manifest_hash": "c-hash", "judge": {}, "model_snapshot": {}}
    hashes = {"corpus_manifest_hash": "cm", "query_hash": "q", "annotation_hash": "a",
              "decision_policy_hash": "p", "gate_config_hash": "g", "retrieval_config_hash": "rc"}
    ledger = freeze_ledger(candidate, hashes, frozen_at="2025-01-01T00:00:00Z")
    return ledger, hashes


def test_one_shot_enforcement():
    ledger, hashes = _ledger_and_hashes()
    assert enforce_one_shot(ledger, "c-hash", hashes) == []
    executed = record_execution(ledger, "2025-01-02T00:00:00Z", {"result_sha256": "r"})
    assert executed["official_gate_runs"] == 1
    assert any("ONE_SHOT_VIOLATION" in v for v in enforce_one_shot(executed, "c-hash", hashes))


def test_candidate_mutation_rejection():
    ledger, hashes = _ledger_and_hashes()
    assert any("CANDIDATE_MANIFEST_MISMATCH" in v for v in enforce_one_shot(ledger, "other-hash", hashes))


def test_query_annotation_mutation_rejection():
    ledger, hashes = _ledger_and_hashes()
    mutated = {**hashes, "query_hash": "tampered"}
    assert any("QUERY_HASH_MUTATED" in v for v in enforce_one_shot(ledger, "c-hash", mutated))
    mutated = {**hashes, "annotation_hash": "tampered"}
    assert any("ANNOTATION_HASH_MUTATED" in v for v in enforce_one_shot(ledger, "c-hash", mutated))


# ---------------------------------------------------------------------------
# Holdout hash-only leakage + frozen artifact + attribution
# ---------------------------------------------------------------------------

def test_d_e_h_hash_only_leakage_never_reads_plaintext():
    prior = [{"query_hashes": {"hashed": "j000x"}}]
    report = query_duplicate_audit(["secret plaintext query"], prior, hash_only=True)
    assert report["mode"] == "hash_only"
    assert report["exact_duplicate"] == []
    assert report["normalized_duplicate"] == "NOT_AUDITED"
    assert report["high_token_overlap"] == "NOT_AUDITED"


def test_frozen_retrieval_artifact_seal_deterministic():
    payload = new_artifact_payload(
        artifact_id="j-artifact", corpus_id="J", manifest_hash="mh",
        annotation_hash="ah",
        retrieval_config={"source_evaluation_configuration": {"embedding_model": "E", "reranker": {}}},
        queries=[{"query_id": "j001", "query": "q", "query_text_hash": "qh"}],
        snapshot_documents=[{"content": "c", "metadata": {"chunk_id": "chunk-1", "document_id": "doc-1"}}],
        source={"name": "Corpus J"}, rule_version="evidence-v323.1",
    )
    assert seal_artifact(payload)["artifact_hash"] == seal_artifact(payload)["artifact_hash"]


def test_failure_attribution_schema():
    assert FAILURE_ATTRIBUTION_TYPES == {
        "RETRIEVAL_MISSING_EVIDENCE", "PARSER_STRUCTURE_LOSS", "PRODUCT_IDENTITY_ERROR",
        "EVIDENCE_VERIFICATION_FAILURE", "OPEN_SUFFICIENCY_FAILURE", "NLI_JUDGE_FAILURE",
        "ROUTER_MISS", "ANNOTATION_AMBIGUITY", "OTHER",
    }


def test_grounding_decision_invariance_constants():
    from backend.retrieval.evidence_boundary import (
        GROUNDING_DECISION_AUTHORITY,
        NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER,
    )
    from backend.retrieval.evidence_openquestion import (
        NORMALIZATION_DECISION_AUTHORITY,
        OPEN_QUESTION_SUFFICIENCY_DEFAULT,
    )
    assert GROUNDING_DECISION_AUTHORITY == "NONE"
    assert NORMALIZATION_DECISION_AUTHORITY == "NONE"
    assert NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER == "NO"
    assert OPEN_QUESTION_SUFFICIENCY_DEFAULT == "OFF"
    assert PRODUCTION_DEFAULTS["SEMANTIC_JUDGE_DEFAULT"] == "OFF"