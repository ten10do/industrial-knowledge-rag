"""V3.76 Retrieval Score Lineage Fidelity tests.

These tests verify the score-propagation boundary fixes the V3.75 fidelity
defect without touching ranking, threshold, or Evidence semantics. They are
pure (no embedding model / Chroma index): documents and candidates are synthetic.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.evaluation.score_lineage import (
    assert_lineage_fidelity,
    build_retrieval_result,
)
from backend.retrieval.evidence import default_policy

MAX_VECTOR_DISTANCE = 13.234710693359375


def _doc(chunk_id):
    return SimpleNamespace(page_content="content", metadata={"chunk_id": chunk_id})


def _scored(chunk_ids_with_scores):
    return [(_doc(cid), score) for cid, score in chunk_ids_with_scores]


# --- Real score propagated ---------------------------------------------------


def test_build_retrieval_result_wires_real_score():
    scored = _scored([("c1", 10.5), ("c2", 12.0), ("c3", 15.5)])
    rr = build_retrieval_result(scored)
    assert len(rr.candidates) == 3
    assert rr.candidates[0].vector_score == 10.5
    assert rr.candidates[1].vector_score == 12.0
    assert rr.candidates[2].vector_score == 15.5


def test_build_retrieval_result_preserves_ranking_and_count():
    scored = _scored([("c1", 10.5), ("c2", 12.0), ("c3", 15.5)])
    rr = build_retrieval_result(scored)
    ids = [str(c.document.metadata["chunk_id"]) for c in rr.candidates]
    assert ids == ["c1", "c2", "c3"]  # rank order preserved exactly


def test_build_retrieval_result_does_not_rescore_or_reorder():
    """The adapter must be a pure conversion: no score recomputation, no reorder."""
    scored = _scored([("c2", 9.9), ("c1", 10.1)])  # deliberately non-monotonic
    rr = build_retrieval_result(scored)
    assert rr.candidates[0].document.metadata["chunk_id"] == "c2"
    assert rr.candidates[0].vector_score == 9.9  # value untouched
    assert rr.candidates[1].document.metadata["chunk_id"] == "c1"
    assert rr.candidates[1].vector_score == 10.1


def test_missing_score_stays_none_not_guessed():
    scored = _scored([("c1", None)])
    rr = build_retrieval_result(scored)
    assert rr.candidates[0].vector_score is None  # None -> None, never 0 / pass-through


# --- Score lineage fidelity roundtrip ----------------------------------------


def test_assert_lineage_fidelity_roundtrip_ok():
    from backend.retrieval.evidence import RetrievalEvidence

    scored = _scored([("c1", 12.285356521606445), ("c2", 13.7)])
    rr = build_retrieval_result(scored)
    evidence = RetrievalEvidence(
        has_candidates=True, exact_identifier_match=False, exact_model_match=False,
        lexical_score=None, lexical_margin=None, vector_distance=12.285356521606445,
        vector_margin=None, top1_top2_margin=None, metadata_consistency=True,
        retrieval_mode="v", effective_mode="v", decision="ANSWER",
        reason="STRONG_VECTOR_EVIDENCE", identity_relation="UNKNOWN",
    )
    record = assert_lineage_fidelity(scored, rr.candidates, evidence)
    assert record["fidelity_ok"] is True
    assert record["raw_top1"] == 12.285356521606445
    assert record["candidate_top1_vector_score"] == 12.285356521606445
    assert record["evidence_vector_distance"] == 12.285356521606445


def test_assert_lineage_fidelity_detects_mismatch():
    from backend.retrieval.evidence import RetrievalEvidence

    scored = _scored([("c1", 10.0)])
    rr = build_retrieval_result(scored)
    evidence = RetrievalEvidence(
        has_candidates=True, exact_identifier_match=False, exact_model_match=False,
        lexical_score=None, lexical_margin=None, vector_distance=99.0,  # wrong value
        vector_margin=None, top1_top2_margin=None, metadata_consistency=True,
        retrieval_mode="v", effective_mode="v", decision="ABSTAIN",
        reason="WEAK_RETRIEVAL_EVIDENCE", identity_relation="UNKNOWN",
    )
    record = assert_lineage_fidelity(scored, rr.candidates, evidence)
    assert record["fidelity_ok"] is False


# --- Threshold unchanged -----------------------------------------------------


def test_threshold_unchanged():
    assert default_policy().max_vector_distance == MAX_VECTOR_DISTANCE
