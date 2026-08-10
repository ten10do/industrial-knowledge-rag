from __future__ import annotations

from .candidates import RetrievalCandidate


def _key(candidate: RetrievalCandidate) -> str:
    return candidate.chunk_id or str(id(candidate.document))


def rrf_fuse(
    lexical: list[RetrievalCandidate],
    vector: list[RetrievalCandidate],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[RetrievalCandidate]:
    merged: dict[str, RetrievalCandidate] = {}
    for source, candidates in (("lexical", lexical), ("vector", vector)):
        for rank, candidate in enumerate(candidates, start=1):
            key = _key(candidate)
            current = merged.get(key)
            if current is None:
                current = RetrievalCandidate(
                    document=candidate.document,
                    retrieval_source=source,
                    lexical_rank=candidate.lexical_rank,
                    vector_rank=candidate.vector_rank,
                    lexical_score=candidate.lexical_score,
                    vector_score=candidate.vector_score,
                    evidence_score=candidate.evidence_score,
                    exact_metadata_match=candidate.exact_metadata_match,
                )
                merged[key] = current
            elif source == "vector":
                current.vector_rank = candidate.vector_rank
                current.vector_score = candidate.vector_score
                current.evidence_score = candidate.evidence_score
                current.exact_metadata_match |= candidate.exact_metadata_match
            current.fusion_score = (current.fusion_score or 0.0) + 1.0 / (rrf_k + rank)
            if source not in current.retrieval_source.split("+"):
                current.retrieval_source = "+".join(sorted((*current.retrieval_source.split("+"), source)))
    ranked = sorted(
        merged.values(),
        key=lambda item: (-(item.fusion_score or 0.0), item.chunk_id),
    )[:top_k]
    for rank, candidate in enumerate(ranked, start=1):
        candidate.final_rank = rank
        candidate.retrieval_source = "hybrid" if "+" in candidate.retrieval_source else candidate.retrieval_source
    return ranked
