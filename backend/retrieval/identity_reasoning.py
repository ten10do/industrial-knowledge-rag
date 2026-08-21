"""Deterministic claim-level identity boundary for the V3.34 candidate.

This module is additive.  It does not change retrieval or the existing Evidence
contract.  Reliable identity mismatches abstain before the frozen V3.32 mixed
candidate; compatible and unknown cases delegate unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .evidence_contract import build_typed_requirement
from .evidence_mixed import analyze_mixed_evidence
from .evidence_querytype import EvidenceQueryType, route_query_type
from .filters import analyze_query
from .product_identity import normalize_identity_text
from .technical import PROTOCOL_ALIASES, matched_terms


IDENTITY_AWARE_CANDIDATE_VERSION = "identity-aware-evidence-v334-candidate"
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
    protocol: tuple[str, ...] = ()
    parameter: str = ""
    scope_level: str = ScopeLevel.GLOBAL.value

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["protocol"] = list(self.protocol)
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


def _firmware(text: str) -> str:
    match = re.search(
        r"\b(?:firmware|fw|revision|version)\s*(?:version|rev\.?|v)?\s*[:=]?\s*(\d+(?:\.\d+)+)\b",
        _normalized(text), re.IGNORECASE,
    )
    return match.group(1) if match else ""


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
    firmware = _firmware(query)
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
    firmware = _firmware(text)
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

    return ProductIdentity(
        manufacturer=_text(metadata.get("manufacturer", "")),
        family=family,
        series=series,
        model=model,
        module=module,
        option=option,
        firmware=firmware,
        protocol=_protocols(text),
        parameter=parameter,
        scope_level=scope.value,
    )


def _same(left: str, right: str) -> bool:
    return bool(left and right and _normalized(left) == _normalized(right))


def identity_compatibility(query: ProductIdentity, evidence: ProductIdentity) -> tuple[IdentityCompatibility, str]:
    if not query.constrained:
        return IdentityCompatibility.UNKNOWN, "QUERY_IDENTITY_UNKNOWN"
    if query.manufacturer and evidence.manufacturer and not _same(query.manufacturer, evidence.manufacturer):
        return IdentityCompatibility.INCOMPATIBLE, "MANUFACTURER_MISMATCH"

    for field, reason in (("firmware", "FIRMWARE_MISMATCH"), ("parameter", "PARAMETER_SCOPE_MISMATCH")):
        requested = getattr(query, field)
        observed = getattr(evidence, field)
        if requested and observed and not _same(requested, observed):
            return IdentityCompatibility.INCOMPATIBLE, reason
        if requested and not observed and evidence.constrained:
            return IdentityCompatibility.INCOMPATIBLE, reason

    if query.module:
        if evidence.module:
            return (IdentityCompatibility.COMPATIBLE, "EXACT_MODULE") if _same(query.module, evidence.module) else (IdentityCompatibility.INCOMPATIBLE, "MODULE_MISMATCH")
        return (IdentityCompatibility.INCOMPATIBLE, "MODULE_TO_CONTROLLER_LEAKAGE") if evidence.constrained else (IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN")

    if query.option:
        if evidence.option:
            return (IdentityCompatibility.COMPATIBLE, "EXACT_OPTION") if _same(query.option, evidence.option) else (IdentityCompatibility.INCOMPATIBLE, "OPTION_MISMATCH")
        return (IdentityCompatibility.INCOMPATIBLE, "OPTION_SCOPE_MISMATCH") if evidence.constrained else (IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN")

    if query.model:
        if evidence.model:
            return (IdentityCompatibility.COMPATIBLE, "EXACT_MODEL") if _same(query.model, evidence.model) else (IdentityCompatibility.INCOMPATIBLE, "MODEL_MISMATCH")
        if evidence.scope_level in {ScopeLevel.FAMILY.value, ScopeLevel.SERIES.value}:
            return IdentityCompatibility.INCOMPATIBLE, "BROADER_EVIDENCE_SCOPE"
        return IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN"

    if query.series:
        if evidence.model and query.family and _same(query.family, evidence.family):
            return IdentityCompatibility.COMPATIBLE, "MODEL_DESCENDS_FROM_SERIES"
        if evidence.series:
            return (IdentityCompatibility.COMPATIBLE, "SAME_SERIES") if _same(query.series, evidence.series) else (IdentityCompatibility.INCOMPATIBLE, "SERIES_MISMATCH")
        if evidence.family:
            return IdentityCompatibility.INCOMPATIBLE, "BROADER_EVIDENCE_SCOPE"
        return IdentityCompatibility.UNKNOWN, "CLAIM_IDENTITY_UNKNOWN"

    if query.family:
        if evidence.family:
            return (IdentityCompatibility.COMPATIBLE, "SAME_FAMILY") if _same(query.family, evidence.family) else (IdentityCompatibility.INCOMPATIBLE, "FAMILY_MISMATCH")
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
    """Apply the V3.34 boundary, then delegate unchanged when it permits it."""
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
        )

    existing = analyze_mixed_evidence(
        query, result, documents, retrieval_mode,
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
        existing_evidence=existing.as_dict(),
    )
