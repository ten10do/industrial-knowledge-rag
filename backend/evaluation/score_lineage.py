"""V3.76 authoritative retrieval-score propagation boundary.

Converts real Chroma retrieval output (``list[(Document, distance)]``) into a
``RetrievalResult`` whose ``RetrievalCandidate.vector_score`` carries the real
distance, so the Evidence runtime reads the same score the retriever produced.

Machine invariant:

    EVIDENCE_SCORE MUST_EQUAL RETRIEVAL_RUNTIME_SCORE

This module does NOT recompute any score, does NOT change ranking, does NOT
touch the Evidence policy/threshold, and does NOT modify Evidence semantics.
It only fixes the harness-level score propagation boundary identified in V3.75.
"""
from __future__ import annotations

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult


def build_retrieval_result(scored_docs, *, retrieval_mode: str = "vector_only_v369") -> RetrievalResult:
    """Build a RetrievalResult with the real Chroma distance on vector_score.

    ``scored_docs`` is the output of ``rag_core.retrieve_docs`` /
    ``similarity_search_with_score``: a list of ``(Document, distance)`` in
    rank order. Document order and count are preserved exactly.
    """
    candidates = []
    for rank, (document, score) in enumerate(scored_docs, start=1):
        distance = float(score) if score is not None else None
        candidate = RetrievalCandidate(
            document=document,
            retrieval_source="chroma",
            vector_rank=rank,
            vector_score=distance,
        )
        # Preserve the pre-existing fusion convention (negated distance); the
        # Evidence runtime reads vector_score, not fusion_score.
        candidate.fusion_score = float(-score) if score is not None else None
        candidates.append(candidate)
    return RetrievalResult(candidates=candidates, retrieval_mode=retrieval_mode)


def assert_lineage_fidelity(scored_docs, candidates, evidence) -> dict:
    """Verify raw distance -> candidate vector_score -> Evidence vector_distance.

    Returns a record with a ``fidelity_ok`` boolean. Uses only the scores that
    already flowed through the runtime; it never recomputes a score.
    """
    raw_top1 = (float(scored_docs[0][1]) if scored_docs and scored_docs[0][1] is not None else None)
    candidate_top1 = (
        None if not candidates or candidates[0].vector_score is None
        else float(candidates[0].vector_score)
    )
    evidence_vd = None if evidence.vector_distance is None else float(evidence.vector_distance)

    def _close(a, b):
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) <= 1e-6

    fidelity_ok = _close(raw_top1, candidate_top1) and _close(candidate_top1, evidence_vd)
    return {
        "raw_top1": raw_top1,
        "candidate_top1_vector_score": candidate_top1,
        "evidence_vector_distance": evidence_vd,
        "fidelity_ok": fidelity_ok,
    }
