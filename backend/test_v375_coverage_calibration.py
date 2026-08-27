"""V3.75 Evidence Coverage/Sufficiency Calibration tests.

These tests freeze the score semantics, the vector-gate threshold boundary, and
the coverage-attribution helpers used by the V3.75 audit. They do not require
the embedding model or Chroma index; decisions are exercised via synthetic
candidates against the real ``analyze_retrieval_evidence`` runtime.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.coverage_attribution import (
    blocking_mechanism,
    is_coverage_bound,
)
from backend.retrieval.evidence import (
    EvidencePolicy,
    RetrievalEvidence,
    analyze_retrieval_evidence,
    default_policy,
)

MAX_VECTOR_DISTANCE = 13.234710693359375


def _document(*, error_code="", equipment_model="G120", content="G120 drive parameter list and operation."):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": "chunk-1",
            "error_code": error_code,
            "equipment_model": equipment_model,
            "source": "fixture.txt",
            "page": 0,
        },
    )


def _result(document, *, vector_score=None, lexical_score=None):
    candidate = RetrievalCandidate(
        document=document,
        retrieval_source="vector",
        vector_score=vector_score,
        lexical_score=lexical_score,
        vector_rank=1,
        final_rank=1,
    )
    return RetrievalResult([candidate], corpus_documents=[document], retrieval_mode="vector")


def _evidence(*, reason, identity_relation, vector_distance=None, lexical_score=None, has_candidates=True):
    return RetrievalEvidence(
        has_candidates=has_candidates,
        exact_identifier_match=False,
        exact_model_match=(identity_relation == "EXACT_MODEL"),
        lexical_score=lexical_score,
        lexical_margin=None,
        vector_distance=vector_distance,
        vector_margin=None,
        top1_top2_margin=None,
        metadata_consistency=True,
        retrieval_mode="v",
        effective_mode="v",
        decision="ABSTAIN",
        reason=reason,
        identity_relation=identity_relation,
    )


# --- Score semantics ---------------------------------------------------------


def test_score_semantics_max_vector_distance():
    assert default_policy().max_vector_distance == MAX_VECTOR_DISTANCE
    # Lower distance is closer/better: the gate is "<= threshold".
    assert MAX_VECTOR_DISTANCE < 20.0  # legacy MAX_RELEVANT_DISTANCE is separate


# --- Vector-gate threshold boundary (score wired) ----------------------------


def test_vector_gate_threshold_boundary_wired():
    document = _document()
    # Within threshold → ANSWER via exact model.
    within = analyze_retrieval_evidence(
        "G120 的参数是多少？", _result(document, vector_score=8.0), [document], "vector",
    )
    assert within.decision == "ANSWER"
    # Above threshold → ABSTAIN.
    above = analyze_retrieval_evidence(
        "G120 的参数是多少？", _result(document, vector_score=20.0), [document], "vector",
    )
    assert above.decision == "ABSTAIN"


def test_vector_gate_threshold_is_lower_is_better():
    document = _document()
    close = analyze_retrieval_evidence(
        "G120 的参数是多少？", _result(document, vector_score=MAX_VECTOR_DISTANCE - 0.5), [document], "vector",
    )
    far = analyze_retrieval_evidence(
        "G120 的参数是多少？", _result(document, vector_score=MAX_VECTOR_DISTANCE + 0.5), [document], "vector",
    )
    assert close.decision == "ANSWER"
    assert far.decision == "ABSTAIN"


# --- Unwired score → threshold bypassed (low-leverage finding) ---------------


def test_unwired_vector_score_abstains_regardless_of_threshold():
    """Baseline harness does not wire vector_score, so vector_distance is None and
    the gate is bypassed. No threshold value can rescue the refusal."""
    document = _document()
    rr = _result(document, vector_score=None)  # baseline harness behaviour
    baseline = analyze_retrieval_evidence(
        "G120 的参数是多少？", rr, [document], "vector",
    )
    assert baseline.decision == "ABSTAIN"
    assert baseline.vector_distance is None

    # Even an absurdly permissive threshold cannot rescue it.
    huge = analyze_retrieval_evidence(
        "G120 的参数是多少？", rr, [document], "vector",
        policy=EvidencePolicy(max_vector_distance=1000.0),
    )
    assert huge.decision == "ABSTAIN"
    assert huge.vector_distance is None


# --- Coverage attribution helpers --------------------------------------------


def test_blocking_mechanism_classification():
    assert blocking_mechanism(_evidence(reason="NO_CANDIDATE", identity_relation="UNKNOWN")) == "RETRIEVAL_MISS"
    assert blocking_mechanism(_evidence(reason="MODEL_MISMATCH", identity_relation="MISMATCH")) == "IDENTITY"
    assert blocking_mechanism(_evidence(reason="UNKNOWN_IDENTIFIER", identity_relation="MISMATCH")) == "IDENTIFIER"
    assert blocking_mechanism(_evidence(reason="MISSING_ATTRIBUTE_EVIDENCE", identity_relation="UNKNOWN")) == "CONTRACT_COVERAGE"
    assert blocking_mechanism(_evidence(reason="PARTIAL_EVIDENCE_ONLY", identity_relation="UNKNOWN")) == "CONTRACT_COVERAGE"
    # Coverage gate with both scores None → unwired.
    assert blocking_mechanism(
        _evidence(reason="INSUFFICIENT_EVIDENCE", identity_relation="EXACT_MODEL", vector_distance=None, lexical_score=None)
    ) == "COVERAGE_SCORE_UNWIRED"
    # Coverage gate with a real vector score → the genuine threshold region.
    assert blocking_mechanism(
        _evidence(reason="INSUFFICIENT_EVIDENCE", identity_relation="EXACT_MODEL", vector_distance=20.0)
    ) == "VECTOR_LEXICAL_COVERAGE"


def test_coverage_bound_classification():
    # Identity accepted + coverage reject → coverage-bound FR.
    assert is_coverage_bound(
        _evidence(reason="INSUFFICIENT_EVIDENCE", identity_relation="EXACT_MODEL", vector_distance=None)
    )
    assert is_coverage_bound(
        _evidence(reason="WEAK_RETRIEVAL_EVIDENCE", identity_relation="SAME_SERIES", vector_distance=20.0)
    )
    # Identity gate reject is NOT coverage-bound.
    assert not is_coverage_bound(_evidence(reason="MODEL_MISMATCH", identity_relation="MISMATCH"))
    # Retrieval miss is NOT coverage-bound.
    assert not is_coverage_bound(
        _evidence(reason="NO_CANDIDATE", identity_relation="UNKNOWN", has_candidates=False)
    )
    # Contract coverage reject with UNKNOWN identity is NOT coverage-bound.
    assert not is_coverage_bound(_evidence(reason="PARTIAL_EVIDENCE_ONLY", identity_relation="UNKNOWN"))
