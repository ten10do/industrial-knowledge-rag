"""Layered, typed Evidence coverage contract used by evidence-v320.1.

This module decides whether retrieved candidates justify entering the answer
path.  It does not replace the independent final Support validator.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations

from .evidence_support import (
    ACTION_ALIASES, EvidenceIntent, _attribute_supported, _concept_supported,
    _contains_alias, _qualifier_supported, _requirement_type_supported,
    _unit_supported, _value_kind_supported, build_evidence_requirement,
)
from .product_identity import ProductIdentity, identity_from_metadata, identity_is_compatible
from .technical import PROTOCOL_ALIASES, contains_parameter_identifier, normalize_technical_text


class RequirementCriticality(str, Enum):
    CRITICAL = "CRITICAL"
    OPTIONAL = "OPTIONAL"


class RequirementMatchMode(str, Enum):
    EXACT = "EXACT"
    NORMALIZED = "NORMALIZED"
    SEMANTIC_EQUIVALENT = "SEMANTIC_EQUIVALENT"
    LOCAL_VALUE_ASSOCIATION = "LOCAL_VALUE_ASSOCIATION"
    SCOPE_AGGREGATED = "SCOPE_AGGREGATED"
    OPTIONAL = "OPTIONAL"


class AggregationLevel(str, Enum):
    SAME_CANDIDATE = "SAME_CANDIDATE"
    SAME_PARAMETER_BLOCK = "SAME_PARAMETER_BLOCK"
    SAME_SECTION = "SAME_SECTION"
    ADJACENT_SECTION = "ADJACENT_SECTION"
    DOCUMENT_GLOBAL = "DOCUMENT_GLOBAL"
    NONE = "NONE"


@dataclass(frozen=True)
class TypedRequirementItem:
    kind: str
    value: str
    criticality: str
    match_mode: str


@dataclass(frozen=True)
class TypedEvidenceRequirement:
    identity: dict
    intent: str
    specificity: str
    location: bool
    items: tuple[TypedRequirementItem, ...]

    def as_dict(self) -> dict:
        return {"identity": self.identity, "intent": self.intent, "specificity": self.specificity,
                "location": self.location,
                "items": [asdict(item) for item in self.items]}


@dataclass(frozen=True)
class CandidateClaim:
    chunk_id: str
    document_id: str
    identity: dict
    section: str
    subsection: str
    page: int | None
    text: str
    segments: tuple[str, ...]
    matched: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        payload = asdict(self); payload["segments"] = list(self.segments); payload["matched"] = list(self.matched); return payload


@dataclass(frozen=True)
class EvidenceContractResult:
    sufficient: bool
    has_critical_requirements: bool
    reason: str
    aggregation_level: str
    requirement: TypedEvidenceRequirement
    claims: tuple[CandidateClaim, ...]
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"sufficient": self.sufficient, "has_critical_requirements": self.has_critical_requirements,
                "reason": self.reason, "aggregation_level": self.aggregation_level,
                "requirement": self.requirement.as_dict(), "claims": [claim.as_dict() for claim in self.claims],
                "covered": list(self.covered), "missing": list(self.missing)}


EVIDENCE_PROTOCOL_ALIASES = {
    **PROTOCOL_ALIASES,
    "bacnet": ("bacnet", "bacnet ms/tp", "bacnet mstp"),
    "controlnet": ("controlnet",),
    "mqtt": ("mqtt",),
    "5g": ("5g", "5g cellular"),
}
SEMANTIC_ATTRIBUTE_ALIASES = {
    "waiting_time": ("delay", "time between", "wait", "waiting time", "restart delay"),
    "cause": ("because", "therefore", "to ensure", "to prevent", "allows", "enables", "may cause", "may result"),
    "quantity": ("maximum number", "up to", "number of", "or fewer"),
}
SEMANTIC_VALUE_KIND_ALIASES = {
    "duration": ("time between", "delay", "waiting time", "elapsed time"),
}
EXPLICIT_DETAIL_ALIASES = {
    "broker_port": ("broker port",), "publish_topic": ("publish topic", "mqtt topic"),
    "communication_speed": ("communication speed", "link speed"),
    "power_line_separation": ("separation", "away from power lines", "between", "power lines"),
    "factory_reset": ("factory reset", "factory-reset", "restore default", "factory defaults", "恢复出厂"),
    "configuration_length": ("configuration length", "configuration data length", "process-data configuration length", "数据的总长度"),
    "restart_attempts": ("restart attempts", "restart tries", "attempts to reset a fault and restart", "auto rstrt tries"),
    "restart_delay": ("delay between attempts", "time between restart attempts", "restart delay", "auto rstrt delay"),
}
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30", "sixty": "60"}


def _critical(kind: str, value: str, mode: RequirementMatchMode) -> TypedRequirementItem:
    return TypedRequirementItem(kind, value, RequirementCriticality.CRITICAL.value, mode.value)


def _optional(kind: str, value: str) -> TypedRequirementItem:
    return TypedRequirementItem(kind, value, RequirementCriticality.OPTIONAL.value, RequirementMatchMode.OPTIONAL.value)


def _explicit_values(query: str, excluded: tuple[str, ...]) -> tuple[str, ...]:
    text = normalize_technical_text(query)
    values = [match.group(0).strip() for match in re.finditer(
        r"(?<![a-z0-9])\d+(?:\.\d+)?(?:\s*(?:%|percent|µj|uj|j|mm|cm|m|ms|s|seconds?|minutes?|v|vac|vdc|hz|mbps|kbyte|byte))?(?![a-z0-9])",
        text, re.IGNORECASE,
    )]
    values.extend(f"{number} {unit}" for word, number in NUMBER_WORDS.items() for unit in ("minute", "second") if re.search(rf"\b{word}[- ]{unit}s?\b", text))
    excluded_numbers = {
        _normalized_value(number)[0]
        for excluded_value in excluded
        for number in re.findall(r"\d+(?:\.\d+)?", normalize_technical_text(excluded_value))
    }
    # Product/model and parameter digits are identity, not requested values.
    return tuple(dict.fromkeys(
        value for value in values if _normalized_value(value)[0] not in excluded_numbers
    ))


def _polarity_requirements(query: str) -> tuple[str, ...]:
    text = normalize_technical_text(query)
    values = []
    if re.search(r"\b(?:power|controller power).{0,20}\bon\b|\bwhile.{0,20}power.{0,10}on\b", text): values.append("power_on")
    if re.search(r"\b(?:power|controller power).{0,20}\boff\b|turn off.{0,20}power", text): values.append("power_off")
    if re.search(r"\b(?:needs?|requires?).{0,20}\bno configuration|without configuration", text): values.append("no_configuration")
    if re.search(r"\brequires?.{0,30}configuration data", text): values.append("requires_configuration")
    if "power cycle" in text: values.append("power_cycle")
    return tuple(values)


def build_typed_requirement(query: str, documents: list, analysis) -> TypedEvidenceRequirement:
    raw = build_evidence_requirement(query, documents, analysis)
    items: list[TypedRequirementItem] = []
    explicit_identifiers = tuple(dict.fromkeys(
        match.group("identifier").upper()
        for match in re.finditer(
            r"\b(?:parameter|item|register|object|index)\s+(?P<identifier>[a-z]?\d+(?:\.\d+)?)\b",
            query,
            re.IGNORECASE,
        )
    ))
    identifiers = tuple(dict.fromkeys((*raw.identifiers, *explicit_identifiers)))
    multiple_identifiers_required = bool(re.search(r"\b(?:and|both)\b", query, re.IGNORECASE))
    for index, identifier in enumerate(identifiers):
        # The legacy extractor may join an article and a unit-bearing number
        # (for example, "a 60 second") into a synthetic A60 identifier.
        if "." in identifier or re.search(rf"(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])", query, re.IGNORECASE):
            item = _critical("identifier", identifier, RequirementMatchMode.EXACT)
            items.append(item if index == 0 or multiple_identifiers_required else _optional("identifier", identifier))
    protocols = [name for name, aliases in EVIDENCE_PROTOCOL_ALIASES.items() if _contains_alias(normalize_technical_text(query), aliases)]
    for protocol in dict.fromkeys((*raw.requested_protocol, *protocols)):
        items.append(_critical("protocol", protocol, RequirementMatchMode.NORMALIZED))
    for attribute in raw.requested_attributes:
        if attribute == "requirements" and raw.requested_requirement_type == "general": items.append(_optional("attribute", attribute))
        else: items.append(_critical("attribute", attribute, RequirementMatchMode.SEMANTIC_EQUIVALENT))
    action_is_critical = EvidenceIntent(raw.intent) in {EvidenceIntent.PROCEDURE, EvidenceIntent.FAULT_ACTION, EvidenceIntent.SAFETY_REQUIREMENT, EvidenceIntent.MAINTENANCE} or bool(re.search(r"\bhow\b", query, re.IGNORECASE))
    for action in raw.requested_action:
        items.append(_critical("action", action, RequirementMatchMode.SEMANTIC_EQUIVALENT) if action_is_critical else _optional("action", action))
    for concept in raw.requested_concepts:
        critical = concept == "configuration_data_required"
        items.append(_critical("concept", concept, RequirementMatchMode.SEMANTIC_EQUIVALENT) if critical else _optional("concept", concept))
    identity_values = tuple(str(value) for value in raw.target_identity.values() if value)
    excluded_values = identifiers + tuple(raw.requested_qualifiers) + identity_values
    value_assertion = bool(
        re.search(r"^\s*(?:is|are|can|does|do|should|may|must|will)\b", query, re.IGNORECASE)
        or re.search(r"\d+(?:\.\d+)?\s*(?:%|percent|mm|cm|m|ms|s|seconds?|minutes?|v|vac|vdc|hz|mbps|kbyte|byte)\b", query, re.IGNORECASE)
    )
    for value in _explicit_values(query, excluded_values) if value_assertion else ():
        items.append(_critical("value", value, RequirementMatchMode.LOCAL_VALUE_ASSOCIATION))
    if raw.requested_unit:
        items.append(_critical("unit", raw.requested_unit, RequirementMatchMode.LOCAL_VALUE_ASSOCIATION))
    for kind in raw.requested_value_kind:
        items.append(_critical("value_kind", kind, RequirementMatchMode.LOCAL_VALUE_ASSOCIATION))
    if raw.requested_requirement_type in {"compatibility", "prerequisite", "version"}:
        items.append(_critical("requirement_type", raw.requested_requirement_type, RequirementMatchMode.SEMANTIC_EQUIVALENT))
    else:
        items.append(_optional("requirement_type", raw.requested_requirement_type))
    for qualifier in raw.requested_qualifiers:
        critical = bool(protocols) or raw.requested_requirement_type in {"compatibility", "version"} or bool(
            re.search(r"\b(?:supported|compatible|corresponds?|provides?)\b", query, re.IGNORECASE)
        )
        items.append(_critical("qualifier", qualifier, RequirementMatchMode.NORMALIZED) if critical else _optional("qualifier", qualifier))
    for polarity in _polarity_requirements(query):
        items.append(_critical("polarity", polarity, RequirementMatchMode.EXACT))
    normalized = normalize_technical_text(query)
    for detail, aliases in EXPLICIT_DETAIL_ALIASES.items():
        if detail == "power_line_separation":
            matched = "power line" in normalized and any(word in normalized for word in ("separation", "away", "between"))
        else:
            matched = _contains_alias(normalized, aliases)
        if matched:
            mode = (
                RequirementMatchMode.SCOPE_AGGREGATED
                if detail in {"restart_attempts", "restart_delay"}
                else RequirementMatchMode.LOCAL_VALUE_ASSOCIATION
                if re.search(r"^\s*(?:does|is|can)\b", query, re.IGNORECASE)
                else RequirementMatchMode.SEMANTIC_EQUIVALENT
            )
            items.append(_critical("detail", detail, mode))
    unique = {(item.kind, item.value, item.criticality): item for item in items}
    return TypedEvidenceRequirement(raw.target_identity, raw.intent, raw.specificity, raw.requested_location, tuple(unique.values()))


def extract_candidate_claim(candidate, requirement: TypedEvidenceRequirement) -> CandidateClaim:
    metadata = candidate.metadata or {}
    raw = str(candidate.document.page_content or "")
    text = normalize_technical_text(raw)
    segments = tuple(normalize_technical_text(segment) for segment in raw.splitlines() if segment.strip()) or (text,)
    document_id = str(metadata.get("document_id") or metadata.get("source") or metadata.get("file_name") or "")
    claim = CandidateClaim(candidate.chunk_id, document_id, identity_from_metadata(metadata).as_dict(),
                           str(metadata.get("section", "")), str(metadata.get("subsection", "")), metadata.get("page"), text, segments)
    matched = tuple(_item_key(item) for item in requirement.items if _item_supported(item, claim))
    return CandidateClaim(**{**asdict(claim), "segments": claim.segments, "matched": matched})


def _item_key(item: TypedRequirementItem) -> str:
    return f"{item.kind}:{item.value}"


def _normalized_value(value: str) -> tuple[str, str]:
    match = re.match(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>.*)", value)
    if not match: return value, ""
    number = match.group("number").rstrip("0").rstrip(".") if "." in match.group("number") else match.group("number")
    unit = match.group("unit").strip().replace("percent", "%").replace("minutes", "minute").replace("seconds", "second")
    return number, unit


def _value_supported(value: str, text: str) -> bool:
    number, unit = _normalized_value(value)
    if not re.search(rf"(?<![\d.]){re.escape(number)}(?:\.0+)?(?![\d.])", text): return False
    if not unit: return True
    aliases = {"%": ("%", "percent"), "minute": ("minute", "minutes", "min"), "second": ("second", "seconds", "sec", "s"), "µj": ("µj", "uj"), "mbps": ("mbps",)}
    return any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) for alias in aliases.get(unit, (unit,)))


def _item_supported(item: TypedRequirementItem, claim: CandidateClaim) -> bool:
    text = claim.text
    if item.kind == "identifier":
        pattern = re.compile(rf"(?<![a-z0-9.]){re.escape(item.value)}(?![a-z0-9.])", re.IGNORECASE)
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 80):match.start()]
            if not re.search(r"related\s+parameters?\s*:[^\n]{0,60}$", prefix, re.IGNORECASE):
                return True
        return False
    if item.kind == "protocol": return _contains_alias(text, EVIDENCE_PROTOCOL_ALIASES[item.value])
    if item.kind == "concept": return _concept_supported(item.value, text)
    if item.kind == "attribute": return _attribute_supported(item.value, text) or _contains_alias(text, SEMANTIC_ATTRIBUTE_ALIASES.get(item.value, ()))
    if item.kind == "action":
        if not _contains_alias(text, ACTION_ALIASES[item.value]):
            return False
        return re.search(
            r"\b(?:steps?|select|click|enter|set|assign|turn|disconnect|connect|clear|remove|install|restart|restore|wait|verify|confirm|authorize|cycle|test|must|before|after|by)\b|"
            r"(?:输入|选择|双击|设置|必须|通过|断开|关闭|恢复)",
            text,
            re.IGNORECASE,
        ) is not None
    if item.kind == "value": return _value_supported(item.value, text)
    if item.kind == "unit": return _unit_supported(item.value, text)
    if item.kind == "value_kind": return _value_kind_supported(item.value, text) or _contains_alias(text, SEMANTIC_VALUE_KIND_ALIASES.get(item.value, ()))
    if item.kind == "requirement_type":
        raw = type("Requirement", (), {"requested_requirement_type": item.value, "requested_qualifiers": ()})()
        return _requirement_type_supported(raw, text)
    if item.kind == "qualifier": return _qualifier_supported(item.value, text)
    if item.kind == "polarity":
        patterns = {"power_on": r"power.{0,25}\bon\b|power on", "power_off": r"turn off.{0,25}power|power.{0,25}\boff\b",
                    "no_configuration": r"without configuration|no configuration|requires? no configuration|无需配置(?:数据)?|无须配置",
                    "requires_configuration": r"requires?.{0,30}configuration data", "power_cycle": r"power cycle|cycle power"}
        return re.search(patterns[item.value], text, re.IGNORECASE) is not None
    if item.kind == "detail":
        if item.value == "restart_attempts":
            return re.search(r"(?:number|maximum).{0,25}restart attempts|restart tries|attempts to reset.{0,20}restart", text, re.IGNORECASE) is not None
        return _contains_alias(text, EXPLICIT_DETAIL_ALIASES[item.value])
    return False


def _identity_from_dict(payload: dict) -> ProductIdentity:
    return ProductIdentity(
        manufacturer=str(payload.get("manufacturer", "")),
        product_family=str(payload.get("product_family", "")),
        product_series=str(payload.get("product_series", "")),
        equipment_model=str(payload.get("equipment_model", "")),
        aliases=tuple(payload.get("aliases", ())),
    )


def _claim_matches_target(requirement: TypedEvidenceRequirement, claim: CandidateClaim) -> bool:
    payloads = requirement.identity.get("identities", [requirement.identity])
    targets = [_identity_from_dict(payload) for payload in payloads]
    targets = [target for target in targets if any((target.product_family, target.product_series, target.equipment_model))]
    return not targets or any(identity_is_compatible(target, _identity_from_dict(claim.identity)) for target in targets)


def _scope_compatible(claims: tuple[CandidateClaim, ...], requirement: TypedEvidenceRequirement) -> tuple[bool, AggregationLevel]:
    if len(claims) == 1: return True, AggregationLevel.SAME_CANDIDATE
    if len({claim.document_id for claim in claims}) != 1 or not claims[0].document_id: return False, AggregationLevel.NONE
    identities = [_identity_from_dict(claim.identity) for claim in claims]
    if any(not identity_is_compatible(identities[0], identity) for identity in identities[1:]): return False, AggregationLevel.NONE
    critical = [item for item in requirement.items if item.criticality == RequirementCriticality.CRITICAL.value]
    identifiers = [item for item in critical if item.kind == "identifier"]
    if len(identifiers) == 1 and not all(_item_supported(identifiers[0], claim) for claim in claims):
        return False, AggregationLevel.NONE
    subsections = {normalize_technical_text(claim.subsection) for claim in claims if claim.subsection}
    if len(subsections) == 1 and subsections:
        return True, AggregationLevel.SAME_PARAMETER_BLOCK
    sections = {normalize_technical_text(claim.section) for claim in claims if claim.section}
    pages = [int(claim.page) for claim in claims if isinstance(claim.page, int)]
    if len(sections) <= 1: return True, AggregationLevel.SAME_SECTION
    if pages and max(pages) - min(pages) <= 2: return True, AggregationLevel.ADJACENT_SECTION
    if any(item.match_mode == RequirementMatchMode.LOCAL_VALUE_ASSOCIATION.value for item in critical):
        return False, AggregationLevel.NONE
    return True, AggregationLevel.DOCUMENT_GLOBAL


def _association_safe(requirement: TypedEvidenceRequirement, claims: tuple[CandidateClaim, ...]) -> bool:
    critical = [item for item in requirement.items if item.criticality == RequirementCriticality.CRITICAL.value]
    identifiers = [item for item in critical if item.kind == "identifier"]
    protocols = [item for item in critical if item.kind == "protocol"]
    qualifiers = [item for item in critical if item.kind == "qualifier"]
    local = [item for item in critical if item.kind in {"value", "unit", "value_kind", "attribute", "detail", "polarity"}]
    if len(identifiers) > 1 and not any(all(_item_supported(item, claim) for item in identifiers) for claim in claims): return False
    if identifiers and local:
        same_claim = any(
            any(_identifier_anchors_claim(identifier, claim) for identifier in identifiers)
            and all(_item_supported(item, claim) for item in local)
            for claim in claims
        )
        same_parameter_block = len(claims) > 1 and all(
            any(_identifier_anchors_claim(identifier, claim) for identifier in identifiers)
            for claim in claims
        )
        if not same_claim and not same_parameter_block:
            return False
    explicit_values = [item for item in local if item.kind == "value"]
    value_context = [item for item in local if item.kind in {"attribute", "detail", "polarity"}]
    if explicit_values and value_context:
        for value in explicit_values:
            if not any(
                _item_supported(value, CandidateClaim(**{**asdict(claim), "text": segment, "segments": (segment,)}))
                and any(_item_supported(context, CandidateClaim(**{**asdict(claim), "text": segment, "segments": (segment,)})) for context in value_context)
                for claim in claims for segment in claim.segments
            ):
                return False
    if protocols and qualifiers:
        for protocol in protocols:
            for qualifier in qualifiers:
                if not any(
                    any(
                        _item_supported(protocol, CandidateClaim(**{**asdict(claim), "text": " ".join(claim.segments[index:index + 2]), "segments": claim.segments[index:index + 2]}))
                        and _item_supported(qualifier, CandidateClaim(**{**asdict(claim), "text": " ".join(claim.segments[index:index + 2]), "segments": claim.segments[index:index + 2]}))
                        for index in range(len(claim.segments) - 1)
                    ) for claim in claims
                ): return False
    local_details = [
        item for item in critical
        if item.kind == "detail" and item.match_mode == RequirementMatchMode.LOCAL_VALUE_ASSOCIATION.value
    ]
    if len(local_details) > 1 and not any(all(_item_supported(item, claim) for item in local_details) for claim in claims): return False
    return True


def _identifier_anchors_claim(identifier: TypedRequirementItem, claim: CandidateClaim) -> bool:
    """Return whether an identifier names this claim's parameter block.

    Mentions in a related-parameters list or a later procedure step establish
    existence, but do not re-scope a value table to that identifier.
    """
    scope_text = normalize_technical_text(" ".join((claim.subsection, claim.segments[0] if claim.segments else "")))
    scope_text = re.split(r"\brelated\s+parameters?\s*:", scope_text, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.search(
        rf"(?<![a-z0-9.]){re.escape(identifier.value)}(?![a-z0-9.])",
        scope_text,
        re.IGNORECASE,
    ) is not None


def evaluate_evidence_contract(query: str, candidates: list, documents: list, analysis) -> EvidenceContractResult:
    requirement = build_typed_requirement(query, documents, analysis)
    claims = tuple(extract_candidate_claim(candidate, requirement) for candidate in candidates)
    critical = tuple(item for item in requirement.items if item.criticality == RequirementCriticality.CRITICAL.value)
    if not critical:
        return EvidenceContractResult(False, False, "NO_CRITICAL_REQUIREMENT", AggregationLevel.NONE.value, requirement, claims)
    best_covered: set[str] = set()
    best_level = AggregationLevel.NONE
    critical_keys = {_item_key(item) for item in critical}
    relevant = tuple(
        claim for claim in claims
        if _claim_matches_target(requirement, claim) and set(claim.matched) & critical_keys
    )
    groups = [(claim,) for claim in relevant]
    for size in range(2, min(len(relevant), len(critical)) + 1):
        groups.extend(combinations(relevant, size))
    for group in groups:
        compatible, level = _scope_compatible(group, requirement)
        if not compatible: continue
        covered = {_item_key(item) for item in critical if any(_item_supported(item, claim) for claim in group)}
        if len(covered) > len(best_covered): best_covered, best_level = covered, level
        if len(covered) == len(critical) and _association_safe(requirement, group):
            return EvidenceContractResult(True, True, "CRITICAL_REQUIREMENTS_COVERED", level.value, requirement, claims, tuple(sorted(covered)), ())
    missing = tuple(_item_key(item) for item in critical if _item_key(item) not in best_covered)
    reason = "UNSAFE_REQUIREMENT_ASSOCIATION" if not missing else f"MISSING_{missing[0].split(':',1)[0].upper()}_CLAIM"
    return EvidenceContractResult(False, True, reason, best_level.value, requirement, claims, tuple(sorted(best_covered)), missing)
