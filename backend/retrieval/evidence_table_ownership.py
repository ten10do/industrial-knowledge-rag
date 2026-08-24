"""V3.49 bounded table-ownership candidate.

This candidate composes the frozen V3.47 decision. It never upgrades an
abstention. Existing answers are preserved only when an explicit structured
claim can be bound to one target-owned table region without conflicting scope.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import evidence_claim_binding_v347 as v347


EVIDENCE_TABLE_OWNERSHIP_VERSION = "evidence-v349-table-ownership-candidate"
EVIDENCE_TABLE_OWNERSHIP_STATUS = "EXPERIMENTAL_CANDIDATE"


class TableOwnershipRelation(str, Enum):
    DIRECT_ROW = "DIRECT_ROW"
    COLUMN_BOUND = "COLUMN_BOUND"
    HEADER_INHERITED = "HEADER_INHERITED"
    SECTION_INHERITED = "SECTION_INHERITED"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class TableOwnershipClaim:
    target: str
    relation: str
    attribute: str
    value_or_action: str
    section: str = ""


@dataclass(frozen=True)
class TableOwnershipDecision:
    ownership_relation: str
    confidence: float
    reason_code: str
    target: str
    attribute: str
    value_or_action: str
    chunk_id: str = ""
    document_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceTableOwnershipOutcome:
    query: str
    decision: str
    action: str
    ownership_relation: str
    confidence: float
    reason_code: str
    baseline_decision: str
    ownership: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_TABLE_MARKER = re.compile(
    r"\b(?:table|technical data|technical specifications?|specifications?|parameters?|"
    r"settings?|configuration|limits?|ratings?|data type|default)\b",
    re.IGNORECASE,
)
_REFERENCE_MARKER = re.compile(r"\b(?:page|refer|reference|see)\b|\[\s*}\s*\d+\s*\]", re.IGNORECASE)
_MODELISH = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b", re.IGNORECASE)
_QUALIFIER = re.compile(r"\b(?:defaults?|factory|preset|preconfigured|delivery state)\b", re.IGNORECASE)
_SECTION_QUERY = re.compile(r"\b(?:under|within|in)\s+(.{3,80}?)(?:\s+section)?\s*,", re.IGNORECASE)
_ATTRIBUTE_ALIASES = {
    "voltage": ("voltage", "rated voltage", "nominal voltage", "supply voltage"),
    "current": ("current", "input current", "output current", "rated current", "current consumption"),
    "weight": ("weight", "mass"),
    "dimensions": ("dimensions", "dimension", "size", "width", "height", "depth"),
    "filter": ("filter time", "input filter", "input delay", "filter"),
    "temperature": ("temperature", "ambient temperature", "operating temperature", "storage temperature"),
    "power": ("power", "power loss", "power consumption"),
    "inputs": ("digital inputs", "number of inputs", "input count", "inputs"),
    "outputs": ("digital outputs", "number of outputs", "output count", "outputs"),
    "protection": ("protection rating", "degree of protection", "ip rating"),
    "cable": ("cable length", "maximum cable length", "cable"),
    "torque": ("torque", "tightening torque"),
    "speed": ("speed", "rated speed", "maximum speed"),
    "address": ("address space", "process image", "input bytes", "output bytes"),
    "diagnostic": ("diagnostic", "diagnostics", "fault code"),
}


def _norm(value: Any) -> str:
    text = str(value or "").casefold().replace("μ", "u").replace("µ", "u")
    return " ".join(re.sub(r"[^a-z0-9.%+/-]+", " ", text).split())


def _aliases(metadata: dict) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("equipment_model", "product_series"):
        if metadata.get(key):
            values.append(str(metadata[key]))
    raw = metadata.get("model_aliases", [])
    values.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])
    return tuple(dict.fromkeys(_norm(value) for value in values if _norm(value)))


def _attribute_terms(attribute: str) -> tuple[str, ...]:
    normalized = _norm(attribute)
    terms = [normalized]
    for key, aliases in _ATTRIBUTE_ALIASES.items():
        if key in normalized or any(alias in normalized for alias in aliases):
            terms.extend(_norm(alias) for alias in aliases)
    return tuple(dict.fromkeys(term for term in terms if term))


def _contains(text: str, value: str) -> bool:
    normalized_text = f" {_norm(text)} "
    normalized_value = _norm(value)
    return bool(normalized_value) and f" {normalized_value} " in normalized_text


def _line_matches_attribute(line: str, terms: tuple[str, ...]) -> bool:
    normalized = _norm(line)
    return any(term in normalized for term in terms)


def _target_scope(target: str, text: str, metadata: dict) -> tuple[bool, bool]:
    normalized_target = _norm(target)
    aliases = _aliases(metadata)
    metadata_owned = normalized_target in aliases
    text_owned = normalized_target in _norm(text)
    return metadata_owned or text_owned, text_owned


def _model_conflict(lines: list[str], target: str, start: int, end: int) -> bool:
    target_norm = _norm(target)
    for line in lines[max(0, start):min(len(lines), end + 1)]:
        for model in _MODELISH.findall(line):
            if _norm(model) != target_norm:
                return True
    return False


def _unsupported(claim: TableOwnershipClaim, reason: str) -> TableOwnershipDecision:
    return TableOwnershipDecision(
        TableOwnershipRelation.UNSUPPORTED.value, 1.0, reason,
        claim.target, claim.attribute, claim.value_or_action,
    )


def analyze_table_ownership(
    claim: TableOwnershipClaim, candidates: list[Any],
) -> TableOwnershipDecision:
    if not all(_norm(value) for value in (
        claim.target, claim.relation, claim.attribute, claim.value_or_action,
    )):
        return _unsupported(claim, "INCOMPLETE_CLAIM")
    terms = _attribute_terms(claim.attribute)
    section = _norm(claim.section)
    qualifier_required = bool(_QUALIFIER.search(
        f"{claim.relation} {claim.attribute} {claim.value_or_action}"
    ))
    reference_required = bool(_REFERENCE_MARKER.search(
        f"{claim.relation} {claim.attribute} {claim.value_or_action}"
    ))
    successes: list[TableOwnershipDecision] = []
    saw_target = False
    saw_table = False
    saw_attribute = False
    saw_value = False

    for candidate in candidates:
        document = getattr(candidate, "document", candidate)
        metadata = getattr(document, "metadata", {}) or {}
        text = str(getattr(document, "page_content", "") or "")
        target_owned, target_in_text = _target_scope(claim.target, text, metadata)
        if not target_owned:
            continue
        saw_target = True
        table_region = bool(_TABLE_MARKER.search(text) or _TABLE_MARKER.search(
            str(metadata.get("section", ""))
        ))
        if not table_region:
            continue
        saw_table = True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        attribute_rows = [i for i, line in enumerate(lines) if _line_matches_attribute(line, terms)]
        value_rows = [i for i, line in enumerate(lines) if _contains(line, claim.value_or_action)]
        saw_attribute = saw_attribute or bool(attribute_rows)
        saw_value = saw_value or bool(value_rows)
        if not attribute_rows or not value_rows:
            continue
        if section:
            section_rows = [i for i, line in enumerate(lines) if section in _norm(line)]
            section_metadata = section in _norm(metadata.get("section", ""))
            if not section_rows and not section_metadata:
                continue
        else:
            section_rows = []
            section_metadata = False

        for attribute_row in attribute_rows:
            for value_row in value_rows:
                low, high = sorted((attribute_row, value_row))
                if high - low > 3:
                    continue
                local = " ".join(lines[max(0, low - 2):min(len(lines), high + 3)])
                if _model_conflict(lines, claim.target, low - 2, high + 2):
                    continue
                if qualifier_required and not _QUALIFIER.search(local):
                    continue
                if reference_required and not _REFERENCE_MARKER.search(local):
                    continue
                if section_rows and not any(
                    section_row <= low and low - section_row <= 12 for section_row in section_rows
                ):
                    continue

                if reference_required:
                    relation = TableOwnershipRelation.CROSS_REFERENCE
                    reason = "EXPLICIT_REFERENCE_OWNERSHIP"
                    confidence = 0.99
                elif attribute_row == value_row and _contains(lines[attribute_row], claim.target):
                    relation = TableOwnershipRelation.COLUMN_BOUND
                    reason = "TARGET_ATTRIBUTE_VALUE_SAME_ROW"
                    confidence = 0.995
                elif attribute_row == value_row:
                    relation = TableOwnershipRelation.DIRECT_ROW
                    reason = "UNIQUE_MODEL_ATTRIBUTE_VALUE_ROW"
                    confidence = 0.99
                elif section_rows or section_metadata:
                    relation = TableOwnershipRelation.SECTION_INHERITED
                    reason = "EXPLICIT_SECTION_TABLE_SCOPE"
                    confidence = 0.98
                else:
                    relation = TableOwnershipRelation.HEADER_INHERITED
                    reason = "ADJACENT_HEADER_VALUE_BINDING"
                    confidence = 0.97
                successes.append(TableOwnershipDecision(
                    relation.value, confidence, reason,
                    claim.target, claim.attribute, claim.value_or_action,
                    str(metadata.get("chunk_id", "")), str(metadata.get("document_id", "")),
                ))

    if successes:
        return max(successes, key=lambda item: item.confidence)
    if not saw_target:
        return _unsupported(claim, "MODEL_SCOPE_NOT_OWNED")
    if not saw_table:
        return _unsupported(claim, "SAME_TABLE_REGION_NOT_PROVEN")
    if not saw_attribute:
        return _unsupported(claim, "PARAMETER_SCOPE_NOT_OWNED")
    if not saw_value:
        return _unsupported(claim, "REQUESTED_VALUE_NOT_OWNED")
    return _unsupported(claim, "CONFLICTING_OR_AMBIGUOUS_OWNERSHIP")


def analyze_evidence_table_ownership(
    query: str, result: Any, documents: list, retrieval_mode: str, *,
    claim: TableOwnershipClaim, judge: Any = None, policy: Any = None,
    identity_matching: bool = True, requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> EvidenceTableOwnershipOutcome:
    baseline = v347.analyze_evidence_claim_binding(
        query, result, documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    baseline_dict = baseline.as_dict()
    if baseline.decision == "ABSTAIN":
        ownership = _unsupported(claim, "V347_ABSTAIN_PRESERVED")
        return EvidenceTableOwnershipOutcome(
            query, "ABSTAIN", "PRESERVE", ownership.ownership_relation,
            ownership.confidence, ownership.reason_code, baseline.decision,
            ownership.as_dict(), baseline_dict,
        )
    ownership = analyze_table_ownership(
        claim, list(getattr(result, "candidates", []) or []),
    )
    supported = ownership.ownership_relation != TableOwnershipRelation.UNSUPPORTED.value
    return EvidenceTableOwnershipOutcome(
        query, "ANSWER" if supported else "ABSTAIN",
        "PRESERVE" if supported else "VETO", ownership.ownership_relation,
        ownership.confidence, ownership.reason_code, baseline.decision,
        ownership.as_dict(), baseline_dict,
    )
