"""Deterministic claim-level identity boundary for the V3.36 utility candidate.

This module is additive.  It does not change retrieval or the existing Evidence
contract.  Reliable identity mismatches abstain before the frozen V3.32 mixed
candidate; compatible and unknown cases delegate unchanged.
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

from .evidence_contract import build_typed_requirement
from .candidates import RetrievalResult
from .evidence_mixed import analyze_mixed_evidence
from .evidence_querytype import EvidenceQueryType, route_query_type
from .filters import analyze_query
from .product_identity import normalize_identity_text
from .technical import PROTOCOL_ALIASES, matched_terms


IDENTITY_AWARE_CANDIDATE_VERSION = "identity-v336-candidate"
IDENTITY_AWARE_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"


class ScopeLevel(str, Enum):
    GLOBAL = "GLOBAL"
    FAMILY = "FAMILY"
    SERIES = "SERIES"
    MODEL = "MODEL"
    MODULE = "MODULE"
    OPTION = "OPTION"


class IdentityCompatibility(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProductIdentity:
    manufacturer: str = ""
    family: str = ""
    series: str = ""
    model: str = ""
    module: str = ""
    option: str = ""
    firmware: str = ""
    firmware_floor: bool = False
    protocol: tuple[str, ...] = ()
    parameter: str = ""
    covered_models: tuple[str, ...] = ()
    scope_universal: bool = False
    scope_level: str = ScopeLevel.GLOBAL.value

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["protocol"] = list(self.protocol)
        payload["covered_models"] = list(self.covered_models)
        return payload

    @property
    def constrained(self) -> bool:
        return any((self.family, self.series, self.model, self.module, self.option,
                    self.firmware, self.parameter))


@dataclass(frozen=True)
class CandidateIdentityCheck:
    chunk_id: str
    compatibility: str
    reason: str
    identity: ProductIdentity

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "compatibility": self.compatibility,
            "reason": self.reason,
            "identity": self.identity.as_dict(),
        }


@dataclass(frozen=True)
class IdentityBoundaryResult:
    status: str
    reason: str
    query_identity: ProductIdentity
    candidates: tuple[CandidateIdentityCheck, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "query_identity": self.query_identity.as_dict(),
            "candidates": [item.as_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class IdentityAwareEvidenceDecision:
    query: str
    query_path: str
    decision: str
    reason: str
    final_decision_source: str
    identity_boundary: dict[str, Any]
    delegated_to_existing_evidence: bool
    identity_alignment_applied: bool
    existing_evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_IDENTITY_FIELDS = (
    "manufacturer", "product_family", "product_series", "equipment_model",
    "module", "module_model", "communication_module", "option", "option_code",
    "accessory", "extension", "adapter", "firmware", "firmware_version",
)
_MODULE_WORDS = r"module|adapter|coupler|gateway"
_OPTION_WORDS = r"option|accessory|extension|add-on|expansion"
_IDENTIFIER = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\d[a-z0-9-]*"


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized(value: object) -> str:
    return normalize_identity_text(value)


def _mentioned(text: str, value: str) -> bool:
    needle = _normalized(value)
    return bool(needle and re.search(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", _normalized(text),
    ))


def _metadata_values(metadata: dict, *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key, "")
        parts = raw if isinstance(raw, (list, tuple, set)) else re.split(r"[|;,]", str(raw or ""))
        values.extend(value for part in parts if (value := _text(part)))
    return tuple(dict.fromkeys(values))


def _corpus_metadata(documents: list) -> tuple[dict, ...]:
    return tuple((getattr(document, "metadata", {}) or {}) for document in documents)


def _context_identifier(text: str, words: str) -> str:
    normalized = _normalized(text)
    patterns = (
        rf"(?P<identifier>{_IDENTIFIER})\s+(?:{words})\b",
        rf"\b(?:{words})\s+(?P<identifier>{_IDENTIFIER})\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, normalized, re.IGNORECASE):
            return _text(match.group("identifier"))
    return ""


def _firmware(text: str) -> tuple[str, bool]:
    normalized = _normalized(text)
    match = re.search(
        r"\b(?:firmware|fw|revision|version)\s*(?:version|rev\.?|v)?\s*[:=]?\s*(\d+(?:\.\d+)+)\b",
        normalized, re.IGNORECASE,
    )
    if not match:
        return "", False
    suffix = normalized[match.end():match.end() + 24]
    floor = bool(re.match(r"\s*(?:or|and)\s+(?:later|newer|higher)\b", suffix, re.IGNORECASE))
    return match.group(1), floor


def _parameter(text: str) -> str:
    normalized = _normalized(text)
    patterns = (
        r"\b(?:parameter|register|object)\s*(?:number|no\.?|id)?\s*[:=]?\s*([a-z]{0,3}\d+(?:\.\d+|-[a-z0-9]+)?)\b",
        r"\b([a-z]{1,3}\d+(?:\.\d+|-[a-z0-9]+)?)\s+(?:parameter|register|object)\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, normalized, re.IGNORECASE):
            return match.group(1).upper()
    return ""


def _protocols(text: str) -> tuple[str, ...]:
    return matched_terms(text, PROTOCOL_ALIASES)


def _matching_metadata(text: str, metadata_items: tuple[dict, ...]) -> tuple[dict, str, str]:
    """Return metadata, matched value, and semantic field for the best mention."""
    matches: list[tuple[int, int, dict, str, str]] = []
    priority = {
        "module": 0, "module_model": 0, "communication_module": 0,
        "option": 1, "option_code": 1, "accessory": 1, "extension": 1, "adapter": 1,
        "model_aliases": 2, "aliases": 2, "equipment_model": 3,
        "product_series": 4, "product_family": 5, "manufacturer": 6,
    }
    for metadata in metadata_items:
        for field in priority:
            for value in _metadata_values(metadata, field):
                if _mentioned(text, value):
                    matches.append((priority[field], -len(_normalized(value)), metadata, value, field))
    if not matches:
        return {}, "", ""
    _, _, metadata, value, field = min(matches, key=lambda item: (item[0], item[1]))
    return metadata, value, field


def _scope_for_match(field: str, value: str, metadata: dict, text: str) -> ScopeLevel:
    if field in {"module", "module_model", "communication_module"}:
        return ScopeLevel.MODULE
    if field in {"option", "option_code", "accessory", "extension", "adapter"}:
        return ScopeLevel.OPTION
    if field == "product_family" or re.search(r"\bfamily\b", _normalized(text)):
        return ScopeLevel.FAMILY
    if field == "product_series" or re.search(r"\bseries\b", _normalized(value)):
        return ScopeLevel.SERIES
    if field in {"model_aliases", "aliases"}:
        broad = {_normalized(item) for item in _metadata_values(metadata, "product_family", "product_series", "equipment_model")}
        return ScopeLevel.SERIES if _normalized(value) in broad and re.search(r"\bseries\b", _normalized(value)) else ScopeLevel.MODEL
    if field == "equipment_model":
        return ScopeLevel.SERIES if re.search(r"\bseries\b|[/()]", _normalized(value)) else ScopeLevel.MODEL
    return ScopeLevel.GLOBAL


def extract_query_identity(query: str, documents: list) -> ProductIdentity:
    metadata_items = _corpus_metadata(documents)
    metadata, matched, field = _matching_metadata(query, metadata_items)
    module = _context_identifier(query, _MODULE_WORDS)
    option = _context_identifier(query, _OPTION_WORDS)
    firmware, firmware_floor = _firmware(query)
    parameter = _parameter(query)
    scope = _scope_for_match(field, matched, metadata, query) if matched else ScopeLevel.GLOBAL

    if module:
        scope = ScopeLevel.MODULE
    elif option:
        scope = ScopeLevel.OPTION
    elif scope == ScopeLevel.MODULE:
        module = matched
    elif scope == ScopeLevel.OPTION:
        option = matched

    model = matched if scope == ScopeLevel.MODEL else ""
    series = matched if scope == ScopeLevel.SERIES else _text(metadata.get("product_series", ""))
    family = matched if scope == ScopeLevel.FAMILY else _text(metadata.get("product_family", ""))
    if not model and not module and not option:
        # Resolve a model-like token that is not in aliases, but only when it
        # shares a known family prefix. This is deterministic and corpus-bound.
        for known_family in _metadata_values(metadata, "product_family"):
            match = re.search(
                rf"(?<![a-z0-9])({re.escape(_normalized(known_family))}\s*[- ]?\s*\d[a-z0-9-]*)(?![a-z0-9])",
                _normalized(query), re.IGNORECASE,
            )
            if match:
                model, scope = _text(match.group(1)), ScopeLevel.MODEL
                family = known_family
                break

    return ProductIdentity(
        manufacturer=_text(metadata.get("manufacturer", "")),
        family=family,
        series=series,
        model=model,
        module=module,
        option=option,
        firmware=firmware,
        firmware_floor=firmware_floor,
        protocol=_protocols(query),
        parameter=parameter,
        scope_level=scope.value,
    )


def _single_model_fallback(metadata: dict) -> str:
    model = _text(metadata.get("equipment_model", ""))
    aliases = _metadata_values(metadata, "model_aliases", "aliases")
    if not model or re.search(r"\bseries\b|[/()]", _normalized(model)):
        return ""
    distinct = {_normalized(value) for value in aliases if _normalized(value) != _normalized(model)}
    return "" if distinct else model


def extract_claim_identity(text: str, metadata: dict) -> ProductIdentity:
    metadata_items = (metadata,)
    matched_metadata, matched, field = _matching_metadata(text, metadata_items)
    module = _context_identifier(text, _MODULE_WORDS)
    option = _context_identifier(text, _OPTION_WORDS)
    firmware, firmware_floor = _firmware(text)
    parameter = _parameter(text)
    scope = _scope_for_match(field, matched, matched_metadata, text) if matched else ScopeLevel.GLOBAL

    if module:
        scope = ScopeLevel.MODULE
    elif option:
        scope = ScopeLevel.OPTION
    elif scope == ScopeLevel.MODULE:
        module = matched
    elif scope == ScopeLevel.OPTION:
        option = matched

    model = matched if scope == ScopeLevel.MODEL else ""
    series = matched if scope == ScopeLevel.SERIES else ""
    family = matched if scope == ScopeLevel.FAMILY else _text(metadata.get("product_family", ""))
    if scope == ScopeLevel.GLOBAL and (fallback := _single_model_fallback(metadata)):
        model, scope = fallback, ScopeLevel.MODEL
    if scope in {ScopeLevel.MODEL, ScopeLevel.MODULE, ScopeLevel.OPTION}:
        series = _text(metadata.get("product_series", ""))
        family = _text(metadata.get("product_family", ""))

    covered_models = _metadata_values(metadata, "document_scope_models")
    scope_universal = bool(
        str(metadata.get("document_scope_policy", "")).upper() == "ALL_LISTED_MODELS"
        and re.search(r"\b(?:all|every|each)\b", _normalized(text), re.IGNORECASE)
    )

    return ProductIdentity(
        manufacturer=_text(metadata.get("manufacturer", "")),
        family=family,
        series=series,
        model=model,
        module=module,
        option=option,
        firmware=firmware,
        firmware_floor=firmware_floor,
        protocol=_protocols(text),
        parameter=parameter,
        covered_models=covered_models,
        scope_universal=scope_universal,
        scope_level=scope.value,
    )


def _same(left: str, right: str) -> bool:
    return bool(left and right and _normalized(left) == _normalized(right))


def _descendant_variant(requested: str, observed: str) -> bool:
    """A base identity may be covered by an explicitly named child variant."""
    base, child = _normalized(requested), _normalized(observed)
    return bool(base and child.startswith(base + " "))


def _version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def _covered_model(query_model: str, evidence: ProductIdentity) -> bool:
    return any(
        _same(query_model, covered) or _descendant_variant(query_model, covered)
        for covered in evidence.covered_models
    )


def identity_compatibility(query: ProductIdentity, evidence: ProductIdentity) -> tuple[IdentityCompatibility, str]:
    if not query.constrained:
        return IdentityCompatibility.UNKNOWN, "QUERY_IDENTITY_UNKNOWN"
    if query.manufacturer and evidence.manufacturer and not _same(query.manufacturer, evidence.manufacturer):
        return IdentityCompatibility.INCOMPATIBLE, "MANUFACTURER_MISMATCH"

    firmware_range = False
    if query.firmware:
        if evidence.firmware and not _same(query.firmware, evidence.firmware):
            requested, observed = _version(query.firmware), _version(evidence.firmware)
            if evidence.firmware_floor and not query.firmware_floor and requested and observed and requested >= observed:
                firmware_range = True
            else:
                return IdentityCompatibility.INCOMPATIBLE, "FIRMWARE_MISMATCH"
        elif not evidence.firmware and evidence.constrained:
            return IdentityCompatibility.INCOMPATIBLE, "FIRMWARE_MISMATCH"
    if query.parameter:
        if evidence.parameter and not _same(query.parameter, evidence.parameter):
            return IdentityCompatibility.INCOMPATIBLE, "PARAMETER_SCOPE_MISMATCH"
        if not evidence.parameter and evidence.constrained:
            return IdentityCompatibility.INCOMPATIBLE, "PARAMETER_SCOPE_MISMATCH"

    def compatible(reason: str) -> tuple[IdentityCompatibility, str]:
        return IdentityCompatibility.COMPATIBLE, (
            "FIRMWARE_RANGE_COVERS_QUERY" if firmware_range else reason
        )

    if query.module:
        if evidence.module:
            if _same(query.module, evidence.module):
                return compatible("EXACT_MODULE")
            if _descendant_variant(query.module, evidence.module):
                return compatible("MODULE_VARIANT_DESCENDANT")
            return IdentityCompatibility.INCOMPATIBLE, "MODULE_MISMATCH"
        return (IdentityCompatibility.INCOMPATIBLE, "MODULE_TO_CONTROLLER_LEAKAGE") if evidence.constrained else (IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN")

    if query.option:
        if evidence.option:
            if _same(query.option, evidence.option):
                return compatible("EXACT_OPTION")
            if _descendant_variant(query.option, evidence.option):
                return compatible("OPTION_VARIANT_DESCENDANT")
            return IdentityCompatibility.INCOMPATIBLE, "OPTION_MISMATCH"
        return (IdentityCompatibility.INCOMPATIBLE, "OPTION_SCOPE_MISMATCH") if evidence.constrained else (IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN")

    if query.model:
        if evidence.model:
            if _same(query.model, evidence.model):
                return compatible("EXACT_MODEL")
            if _descendant_variant(query.model, evidence.model):
                return compatible("MODEL_VARIANT_DESCENDANT")
            return IdentityCompatibility.INCOMPATIBLE, "MODEL_MISMATCH"
        if evidence.scope_level in {ScopeLevel.FAMILY.value, ScopeLevel.SERIES.value}:
            if evidence.scope_universal and _covered_model(query.model, evidence):
                return compatible("DOCUMENT_SCOPE_COVERS_MODEL")
            return IdentityCompatibility.INCOMPATIBLE, "BROADER_EVIDENCE_SCOPE"
        return IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN"

    if query.series:
        if evidence.model and query.family and _same(query.family, evidence.family):
            return compatible("MODEL_DESCENDS_FROM_SERIES")
        if evidence.series:
            return compatible("SAME_SERIES") if _same(query.series, evidence.series) else (IdentityCompatibility.INCOMPATIBLE, "SERIES_MISMATCH")
        if evidence.family:
            return IdentityCompatibility.INCOMPATIBLE, "BROADER_EVIDENCE_SCOPE"
        return IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN"

    if query.family:
        if evidence.family:
            return compatible("SAME_FAMILY") if _same(query.family, evidence.family) else (IdentityCompatibility.INCOMPATIBLE, "FAMILY_MISMATCH")
        return IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN"
    return IdentityCompatibility.UNKNOWN, "QUERY_IDENTITY_UNKNOWN"


def analyze_identity_boundary(query: str, result, documents: list) -> IdentityBoundaryResult:
    query_identity = extract_query_identity(query, documents)
    checks: list[CandidateIdentityCheck] = []
    for candidate in list(getattr(result, "candidates", []) or []):
        metadata = getattr(candidate, "metadata", {}) or {}
        claim_text = str(getattr(getattr(candidate, "document", None), "page_content", "") or "")
        identity = extract_claim_identity(claim_text, metadata)
        compatibility, reason = identity_compatibility(query_identity, identity)
        checks.append(CandidateIdentityCheck(
            chunk_id=str(metadata.get("chunk_id", "")),
            compatibility=compatibility.value,
            reason=reason,
            identity=identity,
        ))

    statuses = {item.compatibility for item in checks}
    if IdentityCompatibility.COMPATIBLE.value in statuses:
        status, reason = IdentityCompatibility.COMPATIBLE, "COMPATIBLE_CLAIM_PRESENT"
    elif IdentityCompatibility.INCOMPATIBLE.value in statuses:
        status, reason = IdentityCompatibility.INCOMPATIBLE, "NO_COMPATIBLE_IDENTITY_CLAIM"
    else:
        status, reason = IdentityCompatibility.UNKNOWN, "IDENTITY_EVIDENCE_UNKNOWN"
    return IdentityBoundaryResult(status.value, reason, query_identity, tuple(checks))


def _query_path(query: str, documents: list, analysis: Any) -> str:
    route = route_query_type(query, build_typed_requirement(query, documents, analysis))
    if route.query_type == EvidenceQueryType.EXTRACTION.value:
        return "OPEN"
    if route.query_type == EvidenceQueryType.VERIFICATION.value:
        return "VERIFICATION"
    return "FALLBACK"


def _aligned_evidence_view(query: str, identity: ProductIdentity, boundary: IdentityBoundaryResult, result, documents: list):
    """Align proven-compatible claim metadata for the unchanged Evidence contract.

    This view is local to the candidate call. It never mutates retrieval output,
    stored chunks, or the caller's documents, and is never applied to UNKNOWN or
    INCOMPATIBLE claims.
    """
    target = identity.module or identity.option or identity.model or identity.series or identity.family
    if not target:
        return result, documents, False
    compatible_ids = {
        check.chunk_id for check in boundary.candidates
        if check.compatibility == IdentityCompatibility.COMPATIBLE.value
    }
    clones: dict[int, object] = {}
    candidates = []
    for candidate in list(getattr(result, "candidates", []) or []):
        metadata = getattr(candidate, "metadata", {}) or {}
        if str(metadata.get("chunk_id", "")) not in compatible_ids:
            candidates.append(candidate)
            continue
        document = copy(candidate.document)
        aligned = dict(metadata)
        aliases = list(_metadata_values(aligned, "model_aliases", "aliases"))
        if target not in aliases:
            aliases.append(target)
        aligned["equipment_model"] = target
        aligned["model_aliases"] = "|".join(aliases)
        document.metadata = aligned
        clones[id(candidate.document)] = document
        candidates.append(replace(candidate, document=document, exact_metadata_match=True))
    if not clones:
        return result, documents, False
    aligned_documents = [clones.get(id(document), document) for document in documents]
    aligned_result = RetrievalResult(
        candidates,
        query_analysis=analyze_query(query, aligned_documents),
        corpus_documents=aligned_documents,
        retrieval_mode=getattr(result, "retrieval_mode", ""),
        scope_decision=getattr(result, "scope_decision", None),
        section_report=getattr(result, "section_report", None),
        trace=getattr(result, "trace", None),
    )
    return aligned_result, aligned_documents, True


def analyze_identity_aware_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    policy: Any = None,
    identity_matching: bool = True,
    requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> IdentityAwareEvidenceDecision:
    """Apply the V3.36 utility boundary, then delegate to frozen Evidence."""
    boundary = analyze_identity_boundary(query, result, documents)
    if boundary.status == IdentityCompatibility.INCOMPATIBLE.value:
        analysis = getattr(result, "query_analysis", None) or analyze_query(query, documents)
        return IdentityAwareEvidenceDecision(
            query=query,
            query_path=_query_path(query, documents, analysis),
            decision="ABSTAIN",
            reason="IDENTITY_SCOPE_MISMATCH",
            final_decision_source="IDENTITY_BOUNDARY",
            identity_boundary=boundary.as_dict(),
            delegated_to_existing_evidence=False,
            identity_alignment_applied=False,
        )

    evidence_result, evidence_documents, aligned = result, documents, False
    if boundary.status == IdentityCompatibility.COMPATIBLE.value:
        evidence_result, evidence_documents, aligned = _aligned_evidence_view(
            query, boundary.query_identity, boundary, result, documents,
        )
    existing = analyze_mixed_evidence(
        query, evidence_result, evidence_documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    return IdentityAwareEvidenceDecision(
        query=query,
        query_path=existing.query_path,
        decision=existing.decision,
        reason=existing.reason,
        final_decision_source=existing.final_decision_source,
        identity_boundary=boundary.as_dict(),
        delegated_to_existing_evidence=True,
        identity_alignment_applied=aligned,
        existing_evidence=existing.as_dict(),
    )
