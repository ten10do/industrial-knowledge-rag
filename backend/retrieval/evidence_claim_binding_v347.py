"""Bounded Evidence claim-binding veto candidate for V3.47.

The candidate composes the frozen V3.45 decision.  It never upgrades an
abstention and never treats missing evidence as a conflict.  Existing answers
may be vetoed only when one target-owned line exposes an explicit competing
attribute value, qualifier, section, or reference.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import evidence_decision_scope_v345 as v345


EVIDENCE_CLAIM_BINDING_CANDIDATE_VERSION = "evidence-v347-bounded-claim-binding-candidate"
EVIDENCE_CLAIM_BINDING_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
VETO_CONFIDENCE_FLOOR = 0.97


class EvidenceClaimBindingRelation(str, Enum):
    OWNED_SUPPORTED = "OWNED_SUPPORTED"
    OWNED_CONFLICT = "OWNED_CONFLICT"
    UNBOUND = "UNBOUND"


class EvidenceClaimBindingAction(str, Enum):
    VETO = "VETO"
    PRESERVE = "PRESERVE"


_ATTRIBUTE_PATTERNS = {
    "weight": re.compile(r"\b(?:weight|weighs?|mass)\b", re.IGNORECASE),
    "pwr_led_color": re.compile(
        r"\b(?:pwr[ -]?led|power(?: supply)? led|power indicator)\b", re.IGNORECASE,
    ),
    "readback_time": re.compile(r"\breadback(?:[ -]time)?\b", re.IGNORECASE),
    "input_filter": re.compile(r"\b(?:input filter|filter time|input delay)\b", re.IGNORECASE),
    "dimensions": re.compile(r"\b(?:dimensions?|dimension drawing|size)\b", re.IGNORECASE),
    "cable_length": re.compile(r"\b(?:cable length|maximum cable run|max\. cable length)\b", re.IGNORECASE),
    "input_current": re.compile(r"\b(?:input current|input amperage|current consumption)\b", re.IGNORECASE),
    "power_loss": re.compile(r"\bpower loss\b", re.IGNORECASE),
    "number_outputs": re.compile(r"\b(?:number of digital outputs|digital outputs|output count)\b", re.IGNORECASE),
    "diagnostics": re.compile(r"\b(?:diagnostic|diagnostics|overtemperature)\b", re.IGNORECASE),
    "address_space": re.compile(r"\b(?:address space|process image|pii assignment)\b", re.IGNORECASE),
}

_NUMERIC_VALUE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ns|us|μs|ms|ma|a|vdc|v|mw|w|kg|g|mm|m|bytes?|bits?|outputs?|inputs?|%)\b",
    re.IGNORECASE,
)
_PAGE_VALUE = re.compile(r"\bpage\s*(\d{1,3})\b", re.IGNORECASE)
_COLOR_VALUE = re.compile(r"\b(?:green|red|yellow|amber|blue|white)\b", re.IGNORECASE)
_SECTION_QUERY = re.compile(r"\b(?:under|within|in)\s+(.{3,80}?)(?:\s+section)?\s*,", re.IGNORECASE)
_QUALIFIER_QUERY = re.compile(r"\b(?:factory|default|preset|preconfigured)(?:\s+setting|\s+value)?\b", re.IGNORECASE)
_REFERENCE_QUERY = re.compile(r"\b(?:refer|reference|point|page)\w*\b", re.IGNORECASE)
_MODELISH = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+){1,}\b", re.IGNORECASE)
_HEADINGS = re.compile(
    r"^(?:technical specifications?|dimensions?|weights?|digital outputs?|digital inputs?|"
    r"power loss|input current|cable length|diagnostics?|address space|parameters?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimBindingSignal:
    relation: str
    reason_code: str
    confidence: float
    attribute: str
    requested_values: tuple[str, ...] = ()
    observed_values: tuple[str, ...] = ()
    chunk_id: str = ""
    document_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceClaimBindingDecision:
    query: str
    decision: str
    action: str
    reason_code: str
    confidence: float
    final_decision_source: str
    baseline_decision: str
    baseline_reason: str
    relation: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(value: Any) -> str:
    text = str(value or "").casefold().replace("μ", "u").replace("µ", "u")
    return " ".join(re.sub(r"[^a-z0-9.%+-]+", " ", text).split())


def _aliases(metadata: dict) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("equipment_model", "product_series", "product_family"):
        if metadata.get(key):
            values.append(str(metadata[key]))
    raw = metadata.get("model_aliases", [])
    values.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])
    return tuple(dict.fromkeys(_norm(value) for value in values if _norm(value)))


def _query_target_aliases(query: str, candidates: list[Any]) -> tuple[str, ...]:
    query_norm = _norm(query)
    known = {
        alias
        for candidate in candidates
        for alias in _aliases(getattr(candidate.document, "metadata", {}) or {})
        if alias in query_norm
    }
    if known:
        return tuple(sorted(known, key=len, reverse=True))
    return tuple(dict.fromkeys(_norm(value) for value in _MODELISH.findall(query)))


def _attribute(query: str) -> tuple[str, re.Pattern[str]] | None:
    return next(
        ((name, pattern) for name, pattern in _ATTRIBUTE_PATTERNS.items() if pattern.search(query)),
        None,
    )


def _canonical_value(value: str) -> str:
    normalized = _norm(value).replace(" ", "")
    if normalized.startswith("page"):
        return normalized
    return normalized


def _values(text: str, attribute: str) -> tuple[str, ...]:
    values: list[str] = []
    if attribute in {"readback_time", "diagnostics", "address_space", "dimensions"}:
        values.extend(f"page{value}" for value in _PAGE_VALUE.findall(text))
    if attribute == "pwr_led_color":
        values.extend(match.group(0) for match in _COLOR_VALUE.finditer(text))
    values.extend(match.group(0) for match in _NUMERIC_VALUE.finditer(text))
    return tuple(dict.fromkeys(_canonical_value(value) for value in values))


def _dimension(value: str) -> str:
    if value.startswith("page"):
        return "page"
    if value in {"green", "red", "yellow", "amber", "blue", "white"}:
        return "color"
    match = re.search(r"(?:ns|us|ms|ma|a|vdc|v|mw|w|kg|g|mm|m|bytes?|bits?|outputs?|inputs?|%)$", value)
    unit = match.group(0) if match else ""
    if unit in {"ns", "us", "ms"}:
        return "time"
    if unit in {"ma", "a"}:
        return "current"
    if unit in {"mw", "w"}:
        return "power"
    if unit in {"kg", "g"}:
        return "weight"
    return unit


def _competes(requested: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    if not requested or not observed:
        return False
    return all(any(_dimension(value) == _dimension(other) and value != other for other in observed)
               for value in requested)


def _target_owned(line: str, text: str, targets: tuple[str, ...], metadata: dict) -> bool:
    if not targets:
        return False
    line_norm = _norm(line)
    text_norm = _norm(text)
    metadata_aliases = set(_aliases(metadata))
    for target in targets:
        if target in line_norm:
            return True
        if target in metadata_aliases and target in text_norm:
            return True
    return False


def _attribute_lines(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    matches = [(index, line) for index, line in enumerate(lines) if pattern.search(line)]
    if matches:
        return matches
    return [(-1, text[match.start() - 160:match.end() + 240]) for match in pattern.finditer(text)]


def _section_signal(
    query: str, text: str, pattern: re.Pattern[str], requested: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    match = _SECTION_QUERY.search(query)
    if not match:
        return None
    section = _norm(match.group(1))
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    section_indexes = [index for index, line in enumerate(lines) if section in _norm(line)]
    if not section_indexes:
        return None
    owned_values: list[str] = []
    outside_values: list[str] = []
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        values = _values(line, _attribute(query)[0])
        if any(abs(index - section_index) <= 3 for section_index in section_indexes):
            owned_values.extend(values)
        else:
            outside_values.extend(values)
    owned = tuple(dict.fromkeys(owned_values))
    outside = tuple(dict.fromkeys(outside_values))
    if all(value in owned for value in requested):
        return "SUPPORTED", owned
    if all(value in outside for value in requested) and not all(value in owned for value in requested):
        return "CONFLICT", outside
    return None


def _qualifier_values(line: str, attribute: str) -> tuple[str, ...]:
    if not re.search(r"\b(?:factory|default|preset|preconfigured)\b", line, re.IGNORECASE):
        return ()
    values = _values(line, attribute)
    anchors = list(re.finditer(r"\b(?:factory|default|preset|preconfigured)\b", line, re.IGNORECASE))
    raw_values = list(_NUMERIC_VALUE.finditer(line))
    if not anchors or not raw_values:
        return values
    nearest = min(raw_values, key=lambda value: min(abs(value.start() - anchor.start()) for anchor in anchors))
    return (_canonical_value(nearest.group(0)),)


def classify_bounded_claim_binding(query: str, candidates: list[Any]) -> ClaimBindingSignal:
    attribute_match = _attribute(query)
    if attribute_match is None:
        return ClaimBindingSignal(
            EvidenceClaimBindingRelation.UNBOUND.value, "ATTRIBUTE_UNBOUND_PRESERVED",
            1.0, "",
        )
    attribute, pattern = attribute_match
    requested = _values(query, attribute)
    if not requested:
        return ClaimBindingSignal(
            EvidenceClaimBindingRelation.UNBOUND.value, "VALUE_UNBOUND_PRESERVED",
            1.0, attribute,
        )
    targets = _query_target_aliases(query, candidates)
    if not targets:
        return ClaimBindingSignal(
            EvidenceClaimBindingRelation.UNBOUND.value, "TARGET_UNBOUND_PRESERVED",
            1.0, attribute, requested,
        )

    supports: list[ClaimBindingSignal] = []
    conflicts: list[ClaimBindingSignal] = []
    for candidate in candidates:
        document = candidate.document
        metadata = getattr(document, "metadata", {}) or {}
        text = str(getattr(document, "page_content", "") or "")
        if not _target_owned(text, text, targets, metadata):
            continue
        chunk_id = str(metadata.get("chunk_id", ""))
        document_id = str(metadata.get("document_id", ""))

        section = _section_signal(query, text, pattern, requested)
        if section is not None:
            state, observed = section
            signal = ClaimBindingSignal(
                EvidenceClaimBindingRelation.OWNED_SUPPORTED.value if state == "SUPPORTED"
                else EvidenceClaimBindingRelation.OWNED_CONFLICT.value,
                "EXPLICIT_SECTION_SUPPORTED" if state == "SUPPORTED" else "EXPLICIT_SECTION_OWNERSHIP_CONFLICT",
                0.99, attribute, requested, observed, chunk_id, document_id,
            )
            (supports if state == "SUPPORTED" else conflicts).append(signal)
            continue

        for _, line in _attribute_lines(text, pattern):
            if not _target_owned(line, text, targets, metadata):
                continue
            observed = _values(line, attribute)
            if _QUALIFIER_QUERY.search(query):
                qualified = _qualifier_values(line, attribute)
                if not qualified:
                    continue
                if all(value in qualified for value in requested):
                    supports.append(ClaimBindingSignal(
                        EvidenceClaimBindingRelation.OWNED_SUPPORTED.value,
                        "EXPLICIT_QUALIFIER_SUPPORTED", 0.99, attribute,
                        requested, qualified, chunk_id, document_id,
                    ))
                elif _competes(requested, qualified):
                    conflicts.append(ClaimBindingSignal(
                        EvidenceClaimBindingRelation.OWNED_CONFLICT.value,
                        "EXPLICIT_QUALIFIER_VALUE_CONFLICT", 0.99, attribute,
                        requested, qualified, chunk_id, document_id,
                    ))
                continue

            if _REFERENCE_QUERY.search(query) and any(value.startswith("page") for value in requested):
                pages = tuple(value for value in observed if value.startswith("page"))
                if all(value in pages for value in requested):
                    supports.append(ClaimBindingSignal(
                        EvidenceClaimBindingRelation.OWNED_SUPPORTED.value,
                        "EXPLICIT_REFERENCE_SUPPORTED", 0.99, attribute,
                        requested, pages, chunk_id, document_id,
                    ))
                elif _competes(requested, pages):
                    conflicts.append(ClaimBindingSignal(
                        EvidenceClaimBindingRelation.OWNED_CONFLICT.value,
                        "EXPLICIT_REFERENCE_OWNERSHIP_CONFLICT", 0.99, attribute,
                        requested, pages, chunk_id, document_id,
                    ))
                continue

            if all(value in observed for value in requested):
                supports.append(ClaimBindingSignal(
                    EvidenceClaimBindingRelation.OWNED_SUPPORTED.value,
                    "EXPLICIT_ATTRIBUTE_VALUE_SUPPORTED", 0.98, attribute,
                    requested, observed, chunk_id, document_id,
                ))
            elif _competes(requested, observed):
                conflicts.append(ClaimBindingSignal(
                    EvidenceClaimBindingRelation.OWNED_CONFLICT.value,
                    "EXPLICIT_ATTRIBUTE_VALUE_CONFLICT", 0.98, attribute,
                    requested, observed, chunk_id, document_id,
                ))

    if supports:
        return max(supports, key=lambda signal: signal.confidence)
    if conflicts:
        return max(conflicts, key=lambda signal: (
            signal.confidence,
            signal.reason_code == "EXPLICIT_QUALIFIER_VALUE_CONFLICT",
            signal.reason_code == "EXPLICIT_REFERENCE_OWNERSHIP_CONFLICT",
        ))
    return ClaimBindingSignal(
        EvidenceClaimBindingRelation.UNBOUND.value, "NO_EXPLICIT_OWNED_SIGNAL_PRESERVED",
        1.0, attribute, requested,
    )


def analyze_evidence_claim_binding(
    query: str,
    result: Any,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    policy: Any = None,
    identity_matching: bool = True,
    requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> EvidenceClaimBindingDecision:
    baseline = v345.analyze_evidence_decision_scope(
        query, result, documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    baseline_dict = baseline.as_dict()
    if baseline.decision == "ABSTAIN":
        relation = ClaimBindingSignal(
            EvidenceClaimBindingRelation.UNBOUND.value, "V345_ABSTAIN_PRESERVED",
            1.0, "",
        )
        return EvidenceClaimBindingDecision(
            query, "ABSTAIN", EvidenceClaimBindingAction.PRESERVE.value,
            relation.reason_code, relation.confidence, baseline.final_decision_source,
            baseline.decision, baseline.reason_code, relation.as_dict(), baseline_dict,
        )

    relation = classify_bounded_claim_binding(
        query, list(getattr(result, "candidates", []) or []),
    )
    veto = (
        relation.relation == EvidenceClaimBindingRelation.OWNED_CONFLICT.value
        and relation.confidence >= VETO_CONFIDENCE_FLOOR
    )
    return EvidenceClaimBindingDecision(
        query, "ABSTAIN" if veto else "ANSWER",
        EvidenceClaimBindingAction.VETO.value if veto else EvidenceClaimBindingAction.PRESERVE.value,
        relation.reason_code, relation.confidence,
        "V347_EVIDENCE_CLAIM_BINDING" if veto else baseline.final_decision_source,
        baseline.decision, baseline.reason_code, relation.as_dict(), baseline_dict,
    )
