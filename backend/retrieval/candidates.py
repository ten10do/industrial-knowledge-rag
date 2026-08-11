from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalCandidate:
    document: object
    retrieval_source: str
    lexical_rank: int | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    vector_score: float | None = None
    fusion_score: float | None = None
    final_rank: int | None = None
    pre_rerank_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    evidence_score: float = 0.0
    exact_metadata_match: bool = False
    identity_relation: str = "UNKNOWN"
    scope_match: str = "none"
    scope_level: str = "GLOBAL_SCOPE"
    section_expanded: bool = False
    section_rank: int | None = None
    neighbor_distance: int | None = None
    pre_section_rank: int | None = None
    section_candidate_source: str = ""

    @property
    def metadata(self) -> dict:
        return getattr(self.document, "metadata", {}) or {}

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))


class RetrievalResult(list):
    """Legacy tuple-compatible retrieval output plus V2 candidate details."""

    def __init__(
        self,
        candidates: list[RetrievalCandidate],
        *,
        query_analysis=None,
        corpus_documents: list | None = None,
        retrieval_mode: str = "",
        scope_decision=None,
        section_report=None,
        trace=None,
    ):
        self.candidates = candidates
        self.query_analysis = query_analysis
        self.corpus_documents = corpus_documents or []
        self.retrieval_mode = retrieval_mode
        self.scope_decision = scope_decision
        self.section_report = section_report
        self.trace = trace
        super().__init__((candidate.document, candidate.evidence_score) for candidate in candidates)
