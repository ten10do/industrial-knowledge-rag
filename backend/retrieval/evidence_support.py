"""Explainable evidence-sufficiency validation over a retrieved Top-K set."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .filters import IDENTIFIER_PATTERN, QueryAnalysis, analyze_query
from .product_identity import (
    IdentityRelation,
    ProductIdentity,
    identity_from_metadata,
    identity_is_compatible,
)
from .technical import (
    PROTOCOL_ALIASES,
    contains_parameter_identifier,
    contains_term,
    extract_parameter_references,
    normalize_technical_text,
)


class EvidenceIntent(str, Enum):
    FACT = "FACT"
    PARAMETER_VALUE = "PARAMETER_VALUE"
    PROCEDURE = "PROCEDURE"
    FAULT_CAUSE = "FAULT_CAUSE"
    FAULT_ACTION = "FAULT_ACTION"
    SAFETY_REQUIREMENT = "SAFETY_REQUIREMENT"
    MAINTENANCE = "MAINTENANCE"
    COMPARISON = "COMPARISON"
    IDENTIFIER_LOOKUP = "IDENTIFIER_LOOKUP"
    GENERAL = "GENERAL"


class EvidenceSpecificity(str, Enum):
    GENERAL = "GENERAL"
    SPECIFIC = "SPECIFIC"
    HIGHLY_SPECIFIC = "HIGHLY_SPECIFIC"


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"


class SupportReason(str, Enum):
    DETAIL_SUPPORTED = "DETAIL_SUPPORTED"
    MISSING_REQUIRED_CONCEPT = "MISSING_REQUIRED_CONCEPT"
    MISSING_REQUESTED_ACTION = "MISSING_REQUESTED_ACTION"
    MISSING_PARAMETER_VALUE = "MISSING_PARAMETER_VALUE"
    MISSING_IDENTIFIER_SUPPORT = "MISSING_IDENTIFIER_SUPPORT"
    PARTIAL_EVIDENCE_ONLY = "PARTIAL_EVIDENCE_ONLY"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    NO_EVIDENCE = "NO_EVIDENCE"
    NO_EXTRACTABLE_REQUIREMENT = "NO_EXTRACTABLE_REQUIREMENT"
    BASE_EVIDENCE_ABSTAIN = "BASE_EVIDENCE_ABSTAIN"


# PROTOCOL_ALIASES is imported from .technical so the protocol registry stays in
# a single, shared location for both the base Evidence gate and the Support gate.

# Small, general industrial synonym groups. These bridge common Chinese query
# phrasing to English manuals without claiming semantic support from similarity.
CONCEPT_ALIASES = {
    "parameter": ("parameter", "param", "参数"),
    "register": ("register", "寄存器"),
    "device_name": ("设备名称", "设备名", "device name"),
    "ip_address": ("ip 地址", "ip地址", "ip address"),
    "node_address": ("节点地址", "node address"),
    "baud_rate": ("波特率", "baud rate"),
    "residual_energy": ("残余能量", "stored energy", "residual energy"),
    "time_sync": ("时间同步", "time synchronization", "time sync"),
    "safety_signature": ("安全签名", "safety signature"),
    "dynamic_brake": ("动态制动", "dynamic brake", "dynamic braking"),
    "braking_resistor": ("制动电阻", "braking resistor", "db resistor"),
    "analog_output": ("模拟输出", "analog output"),
    "accuracy": ("精度", "accuracy", "precision"),
    "dc_bus": ("直流母线", "dc bus", "dc-link", "dc link"),
    "network_topology": ("网络拓扑", "network topology"),
    "thermal_model": ("热模型", "thermal model"),
    "input_power": (
        "主电源", "输入电源", "input power", "main power", "supply power",
        "field-side power", "disconnect power", "remove power",
        "power is removed", "power removed", "power supply",
    ),
}

ACTION_ALIASES = {
    "configure": ("配置", "设置", "configure", "configuration", "set", "assign", "select"),
    "commission": ("投入运行", "投运", "commission", "startup", "authorize the system for use"),
    "wait": ("等待", "多久", "wait", "after", "minute", "second"),
    "transport": ("搬运", "运输", "transport", "move the controller"),
    "measure": ("测量", "确认无电", "measure", "test point", "verify.*voltage"),
    "recover": ("恢复", "recover", "recovery", "restore", "reset", "power cycle"),
    "restart": ("重启", "restart", "auto-reset/run"),
    "enable": ("启用", "允许", "enable", "allow", "set.*(?:other than|greater than).*0"),
    "replace": ("替换", "更换", "replace", "replacement"),
    "remove_power": (
        "断电", "切断电源", "断开主电源", "断开输入电源", "remove.*power",
        "disconnect.*power", "turn off power", "cut.*power", "cut off power",
        "shut off power", "isolate.*power",
    ),
}

ATTRIBUTE_ALIASES = {
    "waiting_time": ("等待多久", "等待时间", "多久", "wait", "minute", "second"),
    "threshold": ("阈值", "threshold", "limit", "percent", "%"),
    "accuracy": CONCEPT_ALIASES["accuracy"],
    "voltage": ("电压", "voltage", "volt"),
    "quantity": ("多少个", "最大数量", "最多", "maximum", "max."),
    "status": ("什么状态", "状态", "status", "state", "fault"),
    "cause": (
        "为什么", "原因", "cause", "reason", "because", "verif", "mandatory",
        "required", "grandmaster", "ensure", "so that", "allows", "enables",
    ),
    "resistance": ("阻值", "欧姆", "resistance", "ohm", "ω", "Ω"),
}

UNIT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:n\s*[·路.]?\s*m|v|a|hz|khz|mhz|s|ms|μs|us|%|°c)(?![a-z0-9])",
    re.IGNORECASE,
)
VALUE_PATTERN = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:%|[a-zμ°]+)?", re.IGNORECASE)
REQUESTED_VALUE_PATTERN = re.compile(
    r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:%|n\s*[·路.]?\s*m|v|a|hz|khz|mhz|ms|μs|us|°c)(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceRequirement:
    intent: str
    target_identity: dict = field(default_factory=dict)
    identifiers: tuple[str, ...] = ()
    requested_concepts: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()
    requested_action: tuple[str, ...] = ()
    requested_value: str = ""
    requested_unit: str = ""
    requested_protocol: tuple[str, ...] = ()
    specificity: str = EvidenceSpecificity.GENERAL.value

    def as_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "identifiers", "requested_concepts", "requested_attributes",
            "requested_action", "requested_protocol",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class EvidenceSupport:
    status: str
    reason: str
    coverage: dict
    missing_requirements: tuple[str, ...] = ()
    supporting_chunks: tuple[str, ...] = ()
    requirement: EvidenceRequirement | None = None
    support_validator_requested: bool = True
    support_validator_effective: bool = True
    support_validator_backend: str = "rule_v1"
    fallback_reason: str = ""

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["missing_requirements"] = list(self.missing_requirements)
        payload["supporting_chunks"] = list(self.supporting_chunks)
        if self.requirement:
            payload["requirement"] = self.requirement.as_dict()
        return payload


def support_gate_enabled() -> bool:
    return os.getenv("SUPPORT_GATE_ENABLED", "false").strip().casefold() in {
        "1", "true", "yes", "on",
    }


def _normalize(value: object) -> str:
    return normalize_technical_text(value)


def _contains_alias(text: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if any(token in alias for token in (".*", "(", "\\", "|", "[", "]")):
            if re.search(alias, text, re.IGNORECASE):
                return True
        elif contains_term(text, (alias,)):
            return True
    return False


def _matched_groups(text: str, groups: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(name for name, aliases in groups.items() if _contains_alias(text, aliases))


def _intent(query: str, identifiers: tuple[str, ...]) -> EvidenceIntent:
    if re.search(r"\b(?:vs\.?|versus|compare)\b|对比|比较|区别|不同", query, re.IGNORECASE):
        return EvidenceIntent.COMPARISON
    if identifiers:
        if re.search(r"原因|为什么|cause|reason", query, re.IGNORECASE):
            return EvidenceIntent.FAULT_CAUSE
        if re.search(r"处理|复位|恢复|解决|action|reset|recover", query, re.IGNORECASE):
            return EvidenceIntent.FAULT_ACTION
        return EvidenceIntent.IDENTIFIER_LOOKUP
    if re.search(r"安全|危险|必须|能否|是否|safety|hazard", query, re.IGNORECASE):
        return EvidenceIntent.SAFETY_REQUIREMENT
    if re.search(r"维护|维修|保养|更换|maintenance|service|replace", query, re.IGNORECASE):
        return EvidenceIntent.MAINTENANCE
    if re.search(r"如何|怎样|步骤|怎么|procedure|configure|set up|commission", query, re.IGNORECASE):
        return EvidenceIntent.PROCEDURE
    if re.search(r"多少|数值|值|阈值|精度|电压|电流|扭矩|频率|多久|value|threshold|accuracy", query, re.IGNORECASE):
        return EvidenceIntent.PARAMETER_VALUE
    if re.search(r"为什么|原因|cause|reason", query, re.IGNORECASE):
        return EvidenceIntent.FACT
    return EvidenceIntent.GENERAL


def _identity_dict(analysis: QueryAnalysis) -> dict:
    if analysis.product_identities:
        return {"identities": [item.as_dict() for item in analysis.product_identities]}
    return ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    ).as_dict()


def build_evidence_requirement(
    query: str,
    documents: list,
    analysis: QueryAnalysis | None = None,
) -> EvidenceRequirement:
    analysis = analysis or analyze_query(query, documents)
    normalized = _normalize(query)
    parameter_identifiers = tuple(
        reference.identifier for reference in extract_parameter_references(query)
    )
    identifiers = tuple(dict.fromkeys(
        parameter_identifiers
        + tuple(match.upper() for match in IDENTIFIER_PATTERN.findall(query or ""))
    ))
    protocols = _matched_groups(normalized, PROTOCOL_ALIASES)
    concepts = _matched_groups(normalized, CONCEPT_ALIASES)
    attributes = _matched_groups(normalized, ATTRIBUTE_ALIASES)
    actions = _matched_groups(normalized, ACTION_ALIASES)
    unit_match = UNIT_PATTERN.search(normalized)
    requested_unit = _normalize(unit_match.group(0)) if unit_match else ""
    value_match = REQUESTED_VALUE_PATTERN.search(normalized)
    requested_value = value_match.group(0).strip() if value_match else ""
    intent = _intent(normalized, identifiers)
    has_identity = bool(
        analysis.equipment_model or analysis.product_series
        or analysis.product_family or analysis.product_identities
    )
    specificity = EvidenceSpecificity.GENERAL
    if protocols or identifiers or requested_unit or len(concepts) >= 2:
        specificity = EvidenceSpecificity.HIGHLY_SPECIFIC
    elif has_identity or concepts or attributes or actions:
        specificity = EvidenceSpecificity.SPECIFIC
    return EvidenceRequirement(
        intent=intent.value,
        target_identity=_identity_dict(analysis),
        identifiers=identifiers,
        requested_concepts=concepts,
        requested_attributes=attributes,
        requested_action=actions,
        requested_value=requested_value,
        requested_unit=requested_unit,
        requested_protocol=protocols,
        specificity=specificity.value,
    )


def _query_identities(analysis: QueryAnalysis) -> tuple[ProductIdentity, ...]:
    if analysis.product_identities:
        return analysis.product_identities
    identity = ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    )
    return (identity,) if any((identity.product_family, identity.product_series, identity.equipment_model)) else ()


def validate_evidence_support(query: str, result, documents: list | None = None) -> EvidenceSupport:
    candidates = list(getattr(result, "candidates", []) or [])
    corpus = list(documents if documents is not None else getattr(result, "corpus_documents", []) or [])
    analysis = getattr(result, "query_analysis", None) or analyze_query(query, corpus)
    requirement = build_evidence_requirement(query, corpus, analysis)
    evidence_text = _normalize("\n".join(str(candidate.document.page_content) for candidate in candidates))

    identities = _query_identities(analysis)
    identity_supported = not identities or all(
        any(identity_is_compatible(identity, identity_from_metadata(candidate.metadata)) for candidate in candidates)
        for identity in identities
    )
    parameter_identifiers = {
        reference.identifier for reference in extract_parameter_references(query)
    }
    identifier_hits = {
        identifier: (
            contains_parameter_identifier(evidence_text, identifier)
            if identifier in parameter_identifiers
            else re.search(
                rf"(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])",
                evidence_text,
                re.IGNORECASE,
            ) is not None
        )
        for identifier in requirement.identifiers
    }
    protocol_hits = {
        protocol: _contains_alias(evidence_text, PROTOCOL_ALIASES[protocol])
        for protocol in requirement.requested_protocol
    }
    identifier_concepts = {
        reference.concept for reference in extract_parameter_references(query)
    }
    concept_hits = {
        concept: (
            any(identifier_hits.get(identifier, False) for identifier in parameter_identifiers)
            if concept in identifier_concepts
            else _contains_alias(evidence_text, CONCEPT_ALIASES[concept])
        )
        for concept in requirement.requested_concepts
    }
    action_hits = {
        action: _contains_alias(evidence_text, ACTION_ALIASES[action])
        for action in requirement.requested_action
    }
    attribute_hits = {
        attribute: _contains_alias(evidence_text, ATTRIBUTE_ALIASES[attribute])
        for attribute in requirement.requested_attributes
    }
    unit_supported = not requirement.requested_unit or bool(
        re.search(re.escape(requirement.requested_unit), evidence_text, re.IGNORECASE)
    )
    value_supported = not requirement.requested_value or requirement.requested_value.casefold() in evidence_text
    parameter_value_supported = bool(VALUE_PATTERN.search(evidence_text))

    missing = []
    if not identity_supported:
        missing.append("target_identity")
    missing.extend(f"identifier:{name}" for name, supported in identifier_hits.items() if not supported)
    missing.extend(f"protocol:{name}" for name, supported in protocol_hits.items() if not supported)
    missing.extend(f"concept:{name}" for name, supported in concept_hits.items() if not supported)
    action_required = EvidenceIntent(requirement.intent) in {
        EvidenceIntent.PROCEDURE,
        EvidenceIntent.FAULT_ACTION,
        EvidenceIntent.SAFETY_REQUIREMENT,
        EvidenceIntent.MAINTENANCE,
    }
    if action_required:
        missing.extend(f"action:{name}" for name, supported in action_hits.items() if not supported)
    missing.extend(f"attribute:{name}" for name, supported in attribute_hits.items() if not supported)
    if not unit_supported:
        missing.append(f"unit:{requirement.requested_unit}")
    if not value_supported:
        missing.append(f"value:{requirement.requested_value}")

    intent = EvidenceIntent(requirement.intent)
    if intent == EvidenceIntent.PARAMETER_VALUE and not requirement.requested_attributes and not parameter_value_supported:
        missing.append("parameter_value")

    coverage = {
        "identity": identity_supported,
        "identifiers": identifier_hits,
        "protocols": protocol_hits,
        "technical_concepts": concept_hits,
        "requested_actions": action_hits,
        "requested_attributes": attribute_hits,
        "requested_unit": unit_supported,
        "requested_value": value_supported,
        "parameter_value": parameter_value_supported,
        "candidate_count": len(candidates),
    }
    supporting_chunks = tuple(
        candidate.chunk_id for candidate in candidates
        if candidate.chunk_id and any((
            not identities or any(
                identity_is_compatible(identity, identity_from_metadata(candidate.metadata))
                for identity in identities
            ),
            any(_normalize(identifier) in _normalize(candidate.document.page_content) for identifier in requirement.identifiers),
            any(_contains_alias(_normalize(candidate.document.page_content), PROTOCOL_ALIASES[name]) for name in requirement.requested_protocol),
            any(_contains_alias(_normalize(candidate.document.page_content), CONCEPT_ALIASES[name]) for name in requirement.requested_concepts),
        ))
    )

    if not candidates:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.NO_EVIDENCE
    elif "target_identity" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MODEL_MISMATCH
    elif any(item.startswith("identifier:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_IDENTIFIER_SUPPORT
    elif any(item.startswith(("protocol:", "concept:")) for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUIRED_CONCEPT
    elif any(item.startswith("action:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUESTED_ACTION
    elif any(item.startswith(("attribute:", "unit:", "value:")) for item in missing) or "parameter_value" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_PARAMETER_VALUE
    elif not any((identities, requirement.identifiers, requirement.requested_protocol,
                  requirement.requested_concepts, requirement.requested_attributes,
                  requirement.requested_action, requirement.requested_unit,
                  requirement.requested_value)):
        status, reason = SupportStatus.UNKNOWN, SupportReason.NO_EXTRACTABLE_REQUIREMENT
    elif missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.PARTIAL_EVIDENCE_ONLY
    else:
        status, reason = SupportStatus.SUPPORTED, SupportReason.DETAIL_SUPPORTED
    return EvidenceSupport(
        status=status.value,
        reason=reason.value,
        coverage=coverage,
        missing_requirements=tuple(missing),
        supporting_chunks=tuple(dict.fromkeys(supporting_chunks)),
        requirement=requirement,
    )


def skipped_support(reason: str = SupportReason.BASE_EVIDENCE_ABSTAIN.value) -> EvidenceSupport:
    return EvidenceSupport(
        status=SupportStatus.UNKNOWN.value,
        reason=reason,
        coverage={},
        support_validator_effective=False,
        fallback_reason="Base evidence gate did not produce an answerable candidate.",
    )
