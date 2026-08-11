"""Disabled-by-default lifecycle tracing for retrieval candidates."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


MAX_EVENTS_PER_CANDIDATE = 32


@dataclass
class CandidateEvent:
    sequence: int
    event: str
    stage: str
    details: dict = field(default_factory=dict)


@dataclass
class CandidateTrace:
    query_id: str
    chunk_id: str
    document_id: str = ""
    source: str = ""
    page: int | None = None
    section: str = ""
    subsection: str = ""
    equipment_model: str = ""
    product_family: str = ""
    identity_relation: str = "UNKNOWN"
    scope_match: str = "none"
    scope_level: str = "GLOBAL_SCOPE"
    lexical_rank: int | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    vector_distance: float | None = None
    fusion_rank: int | None = None
    fusion_score: float | None = None
    candidate_source: str = "ORIGINAL_RETRIEVAL"
    preservation_class: str = "NORMAL"
    section_candidate: bool = False
    section_rank: int | None = None
    section_expanded: bool = False
    neighbor_distance: int | None = None
    expansion_origin_chunk_id: str = ""
    expansion_origin_rank: int | None = None
    expansion_type: str = ""
    section_score: float | None = None
    section_score_breakdown: dict = field(default_factory=dict)
    pre_budget_rank: int | None = None
    pre_budget_priority: int | None = None
    budget_lane: str = ""
    budget_selected: bool | None = None
    budget_reason: str = ""
    budget_reject_reason: str = ""
    identifier_protected: bool = False
    pre_rerank_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    final_selection_source: str = ""
    final_selection_reason: str = ""
    protected_candidate: bool = False
    rescue_candidate: bool = False
    candidate_replaced: bool = False
    replacement_reason: str = ""
    final_selected: bool = False
    final_rank: int | None = None
    drop_reason: str = ""
    is_relevant: bool | None = None
    is_expected_section: bool | None = None
    is_expected_model: bool | None = None
    events: list[CandidateEvent] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["events"] = [asdict(item) for item in self.events]
        return payload


@dataclass
class CandidateDisplacement:
    displaced_chunk: str
    replacement_chunk: str
    reason: str
    displaced_relevant: bool | None = None
    replacement_relevant: bool | None = None
    classification: str = "UNKNOWN_DISPLACEMENT"


class RetrievalTrace:
    """Collect trace state without participating in retrieval decisions."""

    def __init__(self, query: str, query_id: str = ""):
        self.query = query
        self.query_id = query_id
        self.query_intent = ""
        self.query_identity: dict = {}
        self.identifiers: list[str] = []
        self.section_hint: dict = {}
        self.scope: dict = {}
        self.identifier_protection: dict = {
            "identifier_extracted": [],
            "identifier_exists": False,
            "protected_candidates": [],
            "protection_applied": False,
        }
        self.candidates: dict[str, CandidateTrace] = {}
        self.stage_chunk_ids: dict[str, list[str]] = {}
        self.displacements: list[CandidateDisplacement] = []
        self._sequence = 0

    @staticmethod
    def _metadata(candidate) -> dict:
        return getattr(candidate, "metadata", None) or getattr(
            getattr(candidate, "document", None), "metadata", {}
        ) or {}

    def _ensure(self, candidate) -> CandidateTrace:
        metadata = self._metadata(candidate)
        chunk_id = str(metadata.get("chunk_id", "")) or str(id(getattr(candidate, "document", candidate)))
        item = self.candidates.get(chunk_id)
        if item is None:
            item = CandidateTrace(query_id=self.query_id, chunk_id=chunk_id)
            self.candidates[chunk_id] = item
        item.document_id = str(metadata.get("document_id", item.document_id))
        item.source = str(getattr(candidate, "retrieval_source", item.source) or item.source)
        try:
            page = metadata.get("page")
            item.page = int(page) if page is not None else item.page
        except (TypeError, ValueError):
            pass
        item.section = str(metadata.get("section", item.section))
        item.subsection = str(metadata.get("subsection", item.subsection))
        item.equipment_model = str(metadata.get("equipment_model", item.equipment_model))
        item.product_family = str(metadata.get("product_family", item.product_family))
        for name in (
            "identity_relation", "scope_match", "scope_level", "lexical_rank",
            "lexical_score", "vector_rank", "fusion_score", "section_rank",
            "section_expanded", "neighbor_distance", "pre_rerank_rank",
            "rerank_score", "rerank_rank",
            "final_selection_source", "final_selection_reason", "protected_candidate",
            "rescue_candidate", "candidate_replaced", "replacement_reason",
            "candidate_source", "preservation_class",
        ):
            value = getattr(candidate, name, None)
            if value is not None:
                setattr(item, name, value)
        vector_score = getattr(candidate, "vector_score", None)
        if vector_score is not None:
            item.vector_distance = float(vector_score)
        item.section_candidate = bool(item.section_candidate or item.section_expanded)
        return item

    def configure_query(self, analysis, scope) -> None:
        self.query_intent = str(getattr(analysis, "knowledge_type", "") or "")
        self.query_identity = {
            "manufacturer": str(getattr(analysis, "manufacturer", "") or ""),
            "product_family": str(getattr(analysis, "product_family", "") or ""),
            "product_series": str(getattr(analysis, "product_series", "") or ""),
            "equipment_model": str(getattr(analysis, "equipment_model", "") or ""),
        }
        self.identifiers = list(getattr(analysis, "identifiers", ()) or ())
        self.scope = scope.as_dict() if scope else {}
        self.identifier_protection.update({
            "identifier_extracted": list(self.identifiers),
            "identifier_exists": bool(getattr(scope, "identifier_found", False)),
        })

    def event(self, candidate, event: str, stage: str, **details) -> CandidateTrace:
        item = self._ensure(candidate)
        if len(item.events) < MAX_EVENTS_PER_CANDIDATE:
            self._sequence += 1
            item.events.append(CandidateEvent(self._sequence, event, stage, details))
        return item

    def mark_stage(self, stage: str, candidates: list, event: str = "") -> None:
        ids = []
        for candidate in candidates:
            item = self._ensure(candidate)
            ids.append(item.chunk_id)
            if event:
                self.event(candidate, event, stage)
        self.stage_chunk_ids[stage] = ids

    def drop(self, candidate, event: str, stage: str, reason: str, **details) -> None:
        item = self.event(candidate, event, stage, reason=reason, **details)
        item.drop_reason = reason or "UNKNOWN_DROP_REASON"

    def budget(
        self,
        candidate,
        *,
        selected: bool,
        priority: int,
        lane: str,
        reason: str = "",
        protected: bool = False,
    ) -> None:
        item = self._ensure(candidate)
        item.pre_budget_rank = getattr(candidate, "pre_section_rank", None)
        item.pre_budget_priority = priority
        item.budget_lane = lane
        item.budget_selected = selected
        item.budget_reason = reason or ("SELECTED" if selected else "UNKNOWN_DROP_REASON")
        item.budget_reject_reason = "" if selected else (reason or "UNKNOWN_DROP_REASON")
        item.identifier_protected = protected
        if protected and item.chunk_id not in self.identifier_protection["protected_candidates"]:
            self.identifier_protection["protected_candidates"].append(item.chunk_id)
            self.identifier_protection["protection_applied"] = True
        if selected:
            self.event(candidate, "BUDGET_SELECTED", "BUDGET", lane=lane, priority=priority, reason=item.budget_reason)
        else:
            self.drop(candidate, "BUDGET_REJECTED", "BUDGET", item.budget_reject_reason, lane=lane, priority=priority)

    def section_candidate(
        self,
        candidate,
        *,
        origin_chunk_id: str,
        origin_rank: int | None,
        expansion_type: str,
        score: float,
        score_breakdown: dict,
    ) -> None:
        item = self.event(candidate, "SECTION_ADDED", "SECTION_MERGE", expansion_type=expansion_type)
        item.section_candidate = True
        item.section_expanded = True
        item.expansion_origin_chunk_id = origin_chunk_id
        item.expansion_origin_rank = origin_rank
        item.expansion_type = expansion_type
        item.section_score = score
        item.section_score_breakdown = dict(score_breakdown)

    def displacement(self, displaced, replacement, reason: str) -> None:
        self.displacements.append(CandidateDisplacement(
            displaced_chunk=self._ensure(displaced).chunk_id,
            replacement_chunk=self._ensure(replacement).chunk_id,
            reason=reason,
        ))

    def finalize(self, candidates: list) -> None:
        selected = []
        for rank, candidate in enumerate(candidates, start=1):
            item = self.event(candidate, "FINAL_CONTEXT", "FINAL", rank=rank)
            item.final_selected = True
            item.final_rank = rank
            selected.append(candidate)
        self.mark_stage("FINAL", selected)

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_intent": self.query_intent,
            "query_identity": dict(self.query_identity),
            "identifiers": list(self.identifiers),
            "section_hint": dict(self.section_hint),
            "scope": dict(self.scope),
            "identifier_protection": dict(self.identifier_protection),
            "candidate_counts_by_stage": {
                stage: len(chunk_ids) for stage, chunk_ids in self.stage_chunk_ids.items()
            },
            "stage_chunk_ids": {key: list(value) for key, value in self.stage_chunk_ids.items()},
            "candidates": [item.as_dict() for item in self.candidates.values()],
            "displacements": [asdict(item) for item in self.displacements],
        }


def tracing_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return os.getenv("RETRIEVAL_TRACE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def create_trace(query: str, query_id: str = "", enabled: bool | None = None) -> RetrievalTrace | None:
    return RetrievalTrace(query, query_id) if tracing_enabled(enabled) else None
