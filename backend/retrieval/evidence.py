"""Explainable post-retrieval evidence decisions; RRF remains ranking-only."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from enum import Enum

from .filters import QueryAnalysis, analyze_query


DETAIL_MARKERS = (
    "n·m", "nm", "扭矩", "牌号", "年限", "校准", "证书", "备份", "耐压",
    "固件", "igbt", "pcb", "轴承", "润滑脂", "更换周期", "firmware", "backup",
    "certificate", "calibration", "torque", "bearing", "grease", "insulation",
)
EVIDENCE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:0x[0-9a-f]+|[faep]\d{2,5}|mw\d{1,5}|4\d{4})(?![a-z0-9])",
    re.IGNORECASE,
)
EVIDENCE_MODEL_PATTERN = re.compile(
    r"\b(?:[a-z]+\d*(?:-[a-z0-9]+)+|[a-z]{2,12}\s+(?:[a-z]{2,12}\s+)?\d{2,5}|[a-z]{2,8}\d{2,5})\b",
    re.IGNORECASE,
)


class Decision(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


class DecisionReason(str, Enum):
    EXACT_IDENTIFIER_EVIDENCE = "EXACT_IDENTIFIER_EVIDENCE"
    STRONG_LEXICAL_EVIDENCE = "STRONG_LEXICAL_EVIDENCE"
    STRONG_VECTOR_EVIDENCE = "STRONG_VECTOR_EVIDENCE"
    COMBINED_EVIDENCE = "COMBINED_EVIDENCE"
    NO_CANDIDATE = "NO_CANDIDATE"
    UNKNOWN_IDENTIFIER = "UNKNOWN_IDENTIFIER"
    WEAK_RETRIEVAL_EVIDENCE = "WEAK_RETRIEVAL_EVIDENCE"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class EvidencePolicy:
    """Calibrated on a separate fixture; distances are Chroma distances (lower is closer)."""

    max_vector_distance: float = 13.234710693359375
    min_vector_margin: float = 0.0


@dataclass(frozen=True)
class RetrievalEvidence:
    has_candidates: bool
    exact_identifier_match: bool
    exact_model_match: bool
    lexical_score: float | None
    lexical_margin: float | None
    vector_distance: float | None
    vector_margin: float | None
    top1_top2_margin: float | None
    metadata_consistency: bool
    retrieval_mode: str
    effective_mode: str
    decision: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def default_policy() -> EvidencePolicy:
    return EvidencePolicy(
        max_vector_distance=float(os.getenv("EVIDENCE_MAX_VECTOR_DISTANCE", "13.234710693359375")),
        min_vector_margin=float(os.getenv("EVIDENCE_MIN_VECTOR_MARGIN", "0.0")),
    )


def _metadata_values(documents: list, field: str) -> set[str]:
    return {
        str((getattr(document, "metadata", {}) or {}).get(field, "")).lower()
        for document in documents
        if str((getattr(document, "metadata", {}) or {}).get(field, "")).strip()
    }


def _evidence_analysis(query: str, documents: list, base: QueryAnalysis | None) -> QueryAnalysis:
    base = base or analyze_query(query, documents)
    identifier = EVIDENCE_IDENTIFIER_PATTERN.search(query or "")
    model = base.equipment_model
    if not model:
        model_match = EVIDENCE_MODEL_PATTERN.search(query or "")
        model = model_match.group(0) if model_match else ""
    return QueryAnalysis(
        error_code=identifier.group(0).upper() if identifier else base.error_code,
        equipment_model=model,
        manufacturer=base.manufacturer,
        equipment_type=base.equipment_type,
        document_type=base.document_type,
        knowledge_type=base.knowledge_type,
    )


def _margin(values: list[float], *, higher_is_better: bool) -> float | None:
    if len(values) < 2:
        return None
    return values[0] - values[1] if higher_is_better else values[1] - values[0]


def _candidate_values(candidates: list, field: str) -> list[float]:
    return [
        float(value)
        for candidate in candidates
        if (value := getattr(candidate, field, None)) is not None
    ]


def _detail_request_lacks_support(query: str, candidates: list) -> bool:
    """Detect a requested concrete detail that is absent from all retrieved text."""
    normalized_query = (query or "").lower().replace("·", "")
    content = "\n".join(
        str(getattr(candidate.document, "page_content", "")).lower().replace("·", "")
        for candidate in candidates
    )
    requested = [
        marker for marker in DETAIL_MARKERS
        if marker.replace("·", "") in normalized_query
    ]
    return bool(requested) and not any(
        marker.replace("·", "") in content for marker in requested
    )


def analyze_retrieval_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    policy: EvidencePolicy | None = None,
) -> RetrievalEvidence:
    """Return an ANSWER/ABSTAIN decision without invoking an LLM."""
    policy = policy or default_policy()
    candidates = list(getattr(result, "candidates", []) or [])
    analysis = _evidence_analysis(
        query,
        documents,
        getattr(result, "query_analysis", None),
    )
    identifiers = _metadata_values(documents, "error_code")
    models = _metadata_values(documents, "equipment_model")
    top = candidates[0] if candidates else None
    top_metadata = top.metadata if top else {}
    exact_identifier = bool(
        analysis.error_code
        and str(top_metadata.get("error_code", "")).lower() == analysis.error_code.lower()
    )
    exact_model = bool(
        analysis.equipment_model
        and str(top_metadata.get("equipment_model", "")).lower() == analysis.equipment_model.lower()
    )
    lexical_scores = _candidate_values(candidates, "lexical_score")
    vector_distances = _candidate_values(candidates, "vector_score")
    lexical_margin = _margin(lexical_scores, higher_is_better=True)
    vector_margin = _margin(vector_distances, higher_is_better=False)
    top1_top2_margin = vector_margin if vector_margin is not None else lexical_margin
    vector_distance = float(top.vector_score) if top and top.vector_score is not None else None
    lexical_score = float(top.lexical_score) if top and top.lexical_score is not None else None
    metadata_consistency = bool(
        (not analysis.error_code or exact_identifier)
        and (not analysis.equipment_model or exact_model)
    )

    unsupported_detail = _detail_request_lacks_support(query, candidates)
    if analysis.error_code and analysis.error_code.lower() not in identifiers:
        decision, reason = Decision.ABSTAIN, DecisionReason.UNKNOWN_IDENTIFIER
    elif analysis.equipment_model and analysis.equipment_model.lower() not in models:
        decision, reason = Decision.ABSTAIN, DecisionReason.MODEL_MISMATCH
    elif not candidates:
        decision, reason = Decision.ABSTAIN, DecisionReason.NO_CANDIDATE
    elif analysis.equipment_model and not exact_model:
        decision, reason = Decision.ABSTAIN, DecisionReason.MODEL_MISMATCH
    elif unsupported_detail:
        decision, reason = Decision.ABSTAIN, DecisionReason.INSUFFICIENT_EVIDENCE
    elif exact_identifier:
        decision, reason = Decision.ANSWER, DecisionReason.EXACT_IDENTIFIER_EVIDENCE
    elif vector_distance is not None and vector_distance <= policy.max_vector_distance:
        decision, reason = Decision.ANSWER, DecisionReason.STRONG_VECTOR_EVIDENCE
    elif lexical_score is not None and exact_model:
        decision, reason = Decision.ANSWER, DecisionReason.STRONG_LEXICAL_EVIDENCE
    elif lexical_score is not None and vector_distance is not None:
        decision, reason = Decision.ABSTAIN, DecisionReason.WEAK_RETRIEVAL_EVIDENCE
    else:
        decision, reason = Decision.ABSTAIN, DecisionReason.INSUFFICIENT_EVIDENCE

    return RetrievalEvidence(
        has_candidates=bool(candidates),
        exact_identifier_match=exact_identifier,
        exact_model_match=exact_model,
        lexical_score=lexical_score,
        lexical_margin=lexical_margin,
        vector_distance=vector_distance,
        vector_margin=vector_margin,
        top1_top2_margin=top1_top2_margin,
        metadata_consistency=metadata_consistency,
        retrieval_mode=retrieval_mode,
        effective_mode=retrieval_mode,
        decision=decision.value,
        reason=reason.value,
    )
