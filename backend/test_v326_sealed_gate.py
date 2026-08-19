"""Public tests for the V3.26 sealed-gate framework (no gate data, no weights)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.evaluation.v311_resume import hash_json
from backend.evaluation.v326_sealed_gate import (
    CANDIDATE_SOURCE_FILES,
    GateQueryAnnotation,
    RelationAnnotation,
    SEALED_GATE_ID,
    candidate_fingerprint,
    document_overlap_audit,
    enforce_one_shot,
    freeze_ledger,
    gate_hashes,
    model_snapshot_hashes,
    query_duplicate_audit,
    record_execution,
    validate_annotation,
    validate_candidate_fingerprint,
    validate_model_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _anno(**overrides):
    base = dict(
        query_id="h001",
        query="Does device A broadcast to B?",
        ground_truth="ABSTAIN",
        manufacturer="Wago",
        product="750-352",
        document="wago-750-fieldbus",
        category="protocol",
        difficulty="L5_HARD_NEAR_MISS",
        document_style="PROTOCOL_REFERENCE",
        confidence="HIGH",
        expected_evidence="chunk text",
        expected_scope="SAME_CANDIDATE",
        critical_requirements=("value:247", "value:0"),
        relation_type="ROLE",
        relation=RelationAnnotation("device A", "broadcast", "device B", None, "reversed"),
        rationale="role reversal",
    )
    base.update(overrides)
    return GateQueryAnnotation(**base)


# ---------------------------------------------------------------------------
# Candidate fingerprint
# ---------------------------------------------------------------------------

def test_candidate_fingerprint_is_deterministic_and_covers_files():
    fp = candidate_fingerprint(PROJECT_ROOT)
    again = candidate_fingerprint(PROJECT_ROOT)
    assert fp == again
    assert fp["candidate_manifest_hash"] == fp["candidate_manifest_hash"]
    assert set(fp["source_files"]) == set(CANDIDATE_SOURCE_FILES)
    assert fp["support"]["version"] == "support-v316.1"
    assert fp["judge"]["semantic_judge_default"] == "OFF"


def test_candidate_mismatch_invalidates_gate():
    fp = candidate_fingerprint(PROJECT_ROOT)
    tampered = dict(fp)
    tampered["source_files"] = dict(fp["source_files"])
    tampered["source_files"][CANDIDATE_SOURCE_FILES[0]] = "0" * 64
    tampered["candidate_manifest_hash"] = "deadbeef"
    problems = validate_candidate_fingerprint(tampered, PROJECT_ROOT)
    assert any(problem.startswith("SOURCE_CHANGED:") for problem in problems)


def test_support_change_is_detected():
    fp = candidate_fingerprint(PROJECT_ROOT)
    tampered = dict(fp)
    tampered["support"] = {**fp["support"], "evidence_support_sha256": "0" * 64}
    problems = validate_candidate_fingerprint(tampered, PROJECT_ROOT)
    assert any(p == "SUPPORT_CHANGED:evidence_support.py" for p in problems)


# ---------------------------------------------------------------------------
# Model snapshot validation
# ---------------------------------------------------------------------------

def test_model_snapshot_hashes_validate(tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"{}\n")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "tokenizer_config.json").write_bytes(b"{}\n")
    (snapshot / "tokenizer.json").write_bytes(b"{}")
    (snapshot / "special_tokens_map.json").write_bytes(b"{}")
    hashes = model_snapshot_hashes(snapshot)
    assert hashes["config.json"] == hashlib.sha256(b"{}\n").hexdigest()
    assert hashes["model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert validate_model_snapshot(snapshot, hashes) == []
    # mutate one file -> mismatch
    (snapshot / "model.safetensors").write_bytes(b"changed")
    mismatches = validate_model_snapshot(snapshot, hashes)
    assert "MODEL_FILE_MISMATCH:model.safetensors" in mismatches


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

def test_annotation_schema_valid():
    assert validate_annotation(_anno()) == []


def test_annotation_invalid_ground_truth_confidence_relation():
    violations = validate_annotation(_anno(ground_truth="MAYBE"))
    assert "INVALID_ground_truth:MAYBE" in violations


def test_annotation_invalid_confidence():
    assert "INVALID_confidence:LOW" in validate_annotation(_anno(confidence="LOW"))


def test_annotation_invalid_relation_type():
    assert "INVALID_relation_type:FOO" in validate_annotation(_anno(relation_type="FOO"))


def test_annotation_missing_rationale():
    assert "MISSING_rationale" in validate_annotation(_anno(rationale=""))


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def test_document_overlap_detects_url_and_id():
    gate = [{"document_id": "sew-mdx61", "official_url": "https://sew-eurodrive.com/a.pdf", "file": "sew-mdx61.pdf"}]
    prior = [{"document_id": "abb-acs880", "official_url": "https://sew-eurodrive.com/a.pdf", "file": "abb.pdf"}]
    report = document_overlap_audit(gate, prior)
    assert report["official_url_overlap"] == ["https://sew-eurodrive.com/a.pdf"]
    assert "sew-mdx61" not in report["document_id_overlap"]


def test_query_duplicate_full_mode():
    prior = [{"query_id": "q1", "text": "Does device A broadcast to B?"}]
    gate = ["Does device A broadcast to B?", "Is parameter X the default?"]
    report = query_duplicate_audit(gate, prior)
    assert report["mode"] == "full"
    assert len(report["normalized_duplicate"]) == 1


def test_query_duplicate_token_overlap_flagged():
    prior = [{"query_id": "q1", "text": "Does device A broadcast to all slaves while B targets one?"}]
    gate = ["Does device A broadcast to all slaves while B targets one slave?"]
    report = query_duplicate_audit(gate, prior)
    assert len(report["high_token_overlap"]) == 1


def test_query_duplicate_hash_only_never_reads_text():
    # hash-only mode receives only pre-computed hashes, never plaintext
    query_text = "secret holdout query"
    prior_hash = hash_json(query_text)
    prior = [{"query_hashes": {prior_hash: "d001"}}]
    report = query_duplicate_audit([query_text], prior, hash_only=True)
    assert report["mode"] == "hash_only"
    assert len(report["exact_duplicate"]) == 1
    assert report["normalized_duplicate"] == "NOT_AUDITED"
    assert report["high_token_overlap"] == "NOT_AUDITED"


# ---------------------------------------------------------------------------
# Freeze ledger + one-shot enforcement
# ---------------------------------------------------------------------------

def test_freeze_ledger_and_hashes():
    fp = candidate_fingerprint(PROJECT_ROOT)
    hashes = gate_hashes({"corpus": "H"}, [{"query_id": "h1"}], [{"a": 1}], {"version": 1})
    ledger = freeze_ledger(fp, hashes, frozen_at="2026-01-01T00:00:00Z")
    assert ledger["sealed_gate_id"] == SEALED_GATE_ID
    assert ledger["first_execution_at"] is None
    assert ledger["candidate_fingerprint"]["candidate_manifest_hash"] == fp["candidate_manifest_hash"]


def test_one_shot_enforcement_rejects_second_run():
    fp = candidate_fingerprint(PROJECT_ROOT)
    hashes = gate_hashes({"c": "H"}, [], [], {})
    ledger = freeze_ledger(fp, hashes, frozen_at="2026-01-01")
    assert enforce_one_shot(ledger, fp["candidate_manifest_hash"], hashes) == []
    executed = record_execution(ledger, "2026-01-02T00:00:00Z", {"result": "x"})
    assert "ONE_SHOT_VIOLATION" in enforce_one_shot(executed, fp["candidate_manifest_hash"], hashes)


def test_post_run_mutation_rejected():
    fp = candidate_fingerprint(PROJECT_ROOT)
    hashes = gate_hashes({"c": "H"}, [{"q": "1"}], [], {})
    ledger = freeze_ledger(fp, hashes, frozen_at="2026-01-01")
    mutated = {**hashes, "query_hash": "9" * 64}
    violations = enforce_one_shot(ledger, fp["candidate_manifest_hash"], mutated)
    assert "GATE_QUERY_HASH_MUTATED" in violations


def test_candidate_mismatch_blocks_execution():
    fp = candidate_fingerprint(PROJECT_ROOT)
    hashes = gate_hashes({"c": "H"}, [], [], {})
    ledger = freeze_ledger(fp, hashes, frozen_at="2026-01-01")
    violations = enforce_one_shot(ledger, "0" * 64, hashes)
    assert "CANDIDATE_MANIFEST_MISMATCH" in violations