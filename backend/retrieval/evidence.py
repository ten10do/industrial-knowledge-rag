"""Explainable post-retrieval evidence decisions; RRF remains ranking-only."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .filters import QueryAnalysis, analyze_query
from .evidence_contract import evaluate_evidence_contract
from .product_identity import (
    IdentityRelation,
    ProductIdentity,
    identities_from_documents,
    identity_from_metadata,
    identity_is_compatible,
    identity_relation,
)
from .technical import (
    contains_parameter_identifier,
    extract_parameter_references,
    foreign_equipment_signal,
    normalize_technical_text,
)


DETAIL_MARKERS = (
    "n·m", "nm", "扭矩", "牌号", "年限", "校准", "证书", "备份", "耐压",
    "固件", "igbt", "pcb", "轴承", "润滑脂", "更换周期", "firmware", "backup",
    "certificate", "calibration", "torque", "bearing", "grease", "insulation",
    "bluetooth", "pairing", "station name", "mqtt", "broker",
    "bacnet", "controlnet", "5g", "sil 3", "certif",
    "配对", "默认端口", "预防性更换", "主板电容",
)
EVIDENCE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:0x[0-9a-f]+|[faceps]\d{2,5}|mw\d{1,5}|4\d{4})(?![a-z0-9])",
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
    EXACT_MODEL_EVIDENCE = "EXACT_MODEL_EVIDENCE"
    FAMILY_COMPATIBLE_EVIDENCE = "FAMILY_COMPATIBLE_EVIDENCE"
    NO_CANDIDATE = "NO_CANDIDATE"
    UNKNOWN_IDENTIFIER = "UNKNOWN_IDENTIFIER"
    WEAK_RETRIEVAL_EVIDENCE = "WEAK_RETRIEVAL_EVIDENCE"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
    CROSS_EQUIPMENT = "CROSS_EQUIPMENT"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    UNSUPPORTED_PROCEDURE = "UNSUPPORTED_PROCEDURE"
    IDENTIFIER_NOT_IN_EVIDENCE = "IDENTIFIER_NOT_IN_EVIDENCE"
    MISSING_ATTRIBUTE_EVIDENCE = "MISSING_ATTRIBUTE_EVIDENCE"
    MISSING_VALUE_EVIDENCE = "MISSING_VALUE_EVIDENCE"
    MISSING_REQUIREMENT_EVIDENCE = "MISSING_REQUIREMENT_EVIDENCE"
    MISSING_ACTION_EVIDENCE = "MISSING_ACTION_EVIDENCE"
    PARTIAL_EVIDENCE_ONLY = "PARTIAL_EVIDENCE_ONLY"
    CONTRACT_REQUIREMENTS_COVERED = "CONTRACT_REQUIREMENTS_COVERED"


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
    query_identity: dict = field(default_factory=dict)
    candidate_identity: dict = field(default_factory=dict)
    identity_relation: str = IdentityRelation.UNKNOWN.value
    contract: dict = field(default_factory=dict)

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


def _document_identifiers(documents: list) -> set[str]:
    values = _metadata_values(documents, "error_code")
    for document in documents:
        content = str(getattr(document, "page_content", "") or "")
        values.update(match.group(0).casefold() for match in EVIDENCE_IDENTIFIER_PATTERN.finditer(content))
    return values


def _evidence_analysis(query: str, documents: list, base: QueryAnalysis | None) -> QueryAnalysis:
    base = base or analyze_query(query, documents)
    identifier = EVIDENCE_IDENTIFIER_PATTERN.search(query or "")
    model = base.equipment_model
    if not model and not base.product_series and not base.product_family:
        model_match = EVIDENCE_MODEL_PATTERN.search(query or "")
        model = model_match.group(0) if model_match else ""
    parameter_identifiers = {item.identifier for item in extract_parameter_references(query)}
    generic_identifier = identifier.group(0).upper() if identifier else base.error_code
    if generic_identifier.upper() in parameter_identifiers:
        generic_identifier = ""
    return QueryAnalysis(
        error_code=generic_identifier,
        equipment_model=model,
        manufacturer=base.manufacturer,
        equipment_type=base.equipment_type,
        document_type=base.document_type,
        knowledge_type=base.knowledge_type,
        product_family=base.product_family,
        product_series=base.product_series,
        identity_confidence=base.identity_confidence,
        identifiers=base.identifiers,
        product_identities=base.product_identities,
    )


def _legacy_evidence_analysis(query: str, documents: list, base: QueryAnalysis | None) -> QueryAnalysis:
    """Preserve the V3.1 string-equality behavior for calibration comparisons only."""
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
        identifiers=base.identifiers,
        product_identities=base.product_identities,
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


def _candidate_text(candidates: list) -> str:
    return "\n".join(
        str(getattr(candidate.document, "page_content", "") or "")
        for candidate in candidates
    )


def _unknown_parameter(
    query: str,
    candidates: list,
    documents: list,
    analysis: QueryAnalysis,
) -> str | None:
    """Return a parameter/register literal absent from its product scope."""
    references = extract_parameter_references(query)
    if not references:
        return None
    identities = analysis.product_identities
    if not identities:
        identity = ProductIdentity(
            manufacturer=analysis.manufacturer,
            product_family=analysis.product_family,
            product_series=analysis.product_series,
            equipment_model=analysis.equipment_model,
            aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
        )
        identities = (identity,) if any(
            (identity.product_family, identity.product_series, identity.equipment_model)
        ) else ()
    scoped_documents = [
        document for document in documents
        if not identities or any(
            identity_is_compatible(
                identity,
                identity_from_metadata(getattr(document, "metadata", {}) or {}),
            )
            for identity in identities
        )
    ]
    evidence_documents = [candidate.document for candidate in candidates] + scoped_documents
    for reference in references:
        if not any(
            contains_parameter_identifier(
                getattr(document, "page_content", ""), reference.identifier,
            )
            for document in evidence_documents
        ):
            return reference.identifier
    return None


def _security_bypass_signal(query: str) -> bool:
    """Detect a request to bypass authentication (reset/recover a credential the
    requester no longer has). Manuals describe changing a password you already
    know; they do not document recovering an unknown or lost administrator
    credential without physical access."""
    text = normalize_technical_text(query or "")
    has_recovery = re.search(r"\b(reset\w*|recover\w*|restor\w*|bypass\w*|regain\w*|unlock\w*)", text) is not None
    has_credential = re.search(r"\b(password\w*|passcode|credential|login|administrator account)", text) is not None
    has_no_access = re.search(
        r"\b(lost|forgot|forgotten|unknown|cannot|can't|without|no longer|no physical access|unable to reach)\b",
        text,
    ) is not None
    return has_recovery and has_credential and has_no_access


def analyze_retrieval_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    policy: EvidencePolicy | None = None,
    identity_matching: bool = True,
) -> RetrievalEvidence:
    """Return an ANSWER/ABSTAIN decision without invoking an LLM."""
    policy = policy or default_policy()
    candidates = list(getattr(result, "candidates", []) or [])
    analysis_factory = _evidence_analysis if identity_matching else _legacy_evidence_analysis
    analysis = analysis_factory(query, documents, getattr(result, "query_analysis", None))
    identifiers = _document_identifiers(documents)
    top = candidates[0] if candidates else None
    top_metadata = top.metadata if top else {}
    query_identity = ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    )
    candidate_identity = identity_from_metadata(top_metadata)
    candidate_identities = [identity_from_metadata(candidate.metadata) for candidate in candidates]
    query_identities = analysis.product_identities or (query_identity,)
    has_query_identity = any(
        identity.product_family or identity.product_series or identity.equipment_model
        for identity in query_identities
    )
    if identity_matching:
        relations = {
            identity_relation(identity, candidate_item)
            for identity in query_identities
            for candidate_item in candidate_identities
        }
        relation = next(
            (
                value for value in (
                    IdentityRelation.EXACT_MODEL,
                    IdentityRelation.SAME_SERIES,
                    IdentityRelation.SAME_FAMILY,
                    IdentityRelation.MISMATCH,
                )
                if value in relations
            ),
            IdentityRelation.UNKNOWN,
        )
        compatible_identity = any(
            identity_is_compatible(identity, candidate_item)
            for identity in query_identities
            for candidate_item in candidate_identities
        )
        corpus_identities = identities_from_documents(documents)
        known_identity = all(
            any(identity_is_compatible(query_item, candidate_item) for candidate_item in corpus_identities)
            for query_item in query_identities
        ) if has_query_identity else True
    else:
        models = _metadata_values(documents, "equipment_model")
        candidate_model = str(top_metadata.get("equipment_model", ""))
        exact_legacy_model = bool(
            analysis.equipment_model
            and candidate_model.casefold() == analysis.equipment_model.casefold()
        )
        relation = (
            IdentityRelation.EXACT_MODEL
            if exact_legacy_model
            else IdentityRelation.MISMATCH if analysis.equipment_model else IdentityRelation.UNKNOWN
        )
        compatible_identity = exact_legacy_model if analysis.equipment_model else True
        known_identity = analysis.equipment_model.casefold() in models if analysis.equipment_model else True
    exact_identifier = bool(
        analysis.error_code
        and any(
            str(candidate.metadata.get("error_code", "")).casefold() == analysis.error_code.casefold()
            or analysis.error_code.casefold() in {
                match.group(0).casefold()
                for match in EVIDENCE_IDENTIFIER_PATTERN.finditer(
                    str(getattr(candidate.document, "page_content", "") or "")
                )
            }
            for candidate in candidates
        )
    )
    exact_model = relation == IdentityRelation.EXACT_MODEL
    lexical_scores = _candidate_values(candidates, "lexical_score")
    vector_distances = _candidate_values(candidates, "vector_score")
    lexical_margin = _margin(lexical_scores, higher_is_better=True)
    vector_margin = _margin(vector_distances, higher_is_better=False)
    top1_top2_margin = vector_margin if vector_margin is not None else lexical_margin
    vector_distance = float(top.vector_score) if top and top.vector_score is not None else None
    lexical_score = float(top.lexical_score) if top and top.lexical_score is not None else None
    metadata_consistency = bool(
        (not analysis.error_code or exact_identifier)
        and (not has_query_identity or compatible_identity)
    )

    unsupported_detail = _detail_request_lacks_support(query, candidates)
    foreign_equipment = foreign_equipment_signal(query, documents)
    unknown_parameter = _unknown_parameter(query, candidates, documents, analysis)
    contract = evaluate_evidence_contract(query, candidates, documents, analysis)
    contract_reason_map = {
        "IDENTIFIER": DecisionReason.IDENTIFIER_NOT_IN_EVIDENCE,
        "PROTOCOL": DecisionReason.PROTOCOL_MISMATCH,
        "ATTRIBUTE": DecisionReason.MISSING_ATTRIBUTE_EVIDENCE,
        "ACTION": DecisionReason.MISSING_ACTION_EVIDENCE,
        "VALUE": DecisionReason.MISSING_VALUE_EVIDENCE,
        "UNIT": DecisionReason.MISSING_VALUE_EVIDENCE,
        "VALUE_KIND": DecisionReason.MISSING_VALUE_EVIDENCE,
        "POLARITY": DecisionReason.MISSING_VALUE_EVIDENCE,
        "REQUIREMENT_TYPE": DecisionReason.MISSING_REQUIREMENT_EVIDENCE,
    }
    if analysis.error_code and analysis.error_code.lower() not in identifiers:
        decision, reason = Decision.ABSTAIN, DecisionReason.UNKNOWN_IDENTIFIER
    elif has_query_identity and not known_identity:
        decision, reason = Decision.ABSTAIN, DecisionReason.MODEL_MISMATCH
    elif not candidates:
        decision, reason = Decision.ABSTAIN, DecisionReason.NO_CANDIDATE
    elif has_query_identity and not compatible_identity:
        decision, reason = Decision.ABSTAIN, DecisionReason.MODEL_MISMATCH
    elif analysis.error_code and not exact_identifier:
        decision, reason = Decision.ABSTAIN, DecisionReason.UNKNOWN_IDENTIFIER
    elif foreign_equipment is not None:
        decision, reason = Decision.ABSTAIN, DecisionReason.CROSS_EQUIPMENT
    elif unknown_parameter is not None:
        decision, reason = Decision.ABSTAIN, DecisionReason.UNKNOWN_PARAMETER
    elif _security_bypass_signal(query):
        decision, reason = Decision.ABSTAIN, DecisionReason.UNSUPPORTED_PROCEDURE
    elif contract.has_critical_requirements and not contract.sufficient:
        missing_kind = contract.missing[0].split(":", 1)[0].upper() if contract.missing else ""
        decision = Decision.ABSTAIN
        reason = contract_reason_map.get(missing_kind, DecisionReason.PARTIAL_EVIDENCE_ONLY)
    elif unsupported_detail:
        decision, reason = Decision.ABSTAIN, DecisionReason.INSUFFICIENT_EVIDENCE
    elif contract.has_critical_requirements:
        decision = Decision.ANSWER
        if exact_identifier:
            reason = DecisionReason.EXACT_IDENTIFIER_EVIDENCE
        elif identity_matching and relation == IdentityRelation.EXACT_MODEL:
            reason = DecisionReason.EXACT_MODEL_EVIDENCE
        elif identity_matching and relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            reason = DecisionReason.FAMILY_COMPATIBLE_EVIDENCE
        else:
            reason = DecisionReason.CONTRACT_REQUIREMENTS_COVERED
    elif exact_identifier:
        decision, reason = Decision.ANSWER, DecisionReason.EXACT_IDENTIFIER_EVIDENCE
    elif vector_distance is not None and vector_distance <= policy.max_vector_distance:
        if identity_matching and relation == IdentityRelation.EXACT_MODEL:
            decision, reason = Decision.ANSWER, DecisionReason.EXACT_MODEL_EVIDENCE
        elif identity_matching and relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            decision, reason = Decision.ANSWER, DecisionReason.FAMILY_COMPATIBLE_EVIDENCE
        else:
            decision, reason = Decision.ANSWER, DecisionReason.STRONG_VECTOR_EVIDENCE
    elif lexical_score is not None and compatible_identity:
        if not identity_matching:
            reason = DecisionReason.STRONG_LEXICAL_EVIDENCE
            decision = Decision.ANSWER
        elif relation == IdentityRelation.EXACT_MODEL:
            reason = DecisionReason.EXACT_MODEL_EVIDENCE
            decision = Decision.ANSWER
        elif relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            reason = DecisionReason.FAMILY_COMPATIBLE_EVIDENCE
            decision = Decision.ANSWER
        else:
            decision, reason = Decision.ABSTAIN, DecisionReason.INSUFFICIENT_EVIDENCE
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
        query_identity=(
            {"identities": [identity.as_dict() for identity in query_identities]}
            if len(query_identities) > 1 else query_identity.as_dict()
        ),
        candidate_identity=candidate_identity.as_dict(),
        identity_relation=relation.value,
        contract=contract.as_dict(),
    )
