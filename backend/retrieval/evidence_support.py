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


SUPPORT_RULE_VERSION = "support-v316.1"


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
    MISSING_REQUESTED_LOCATION = "MISSING_REQUESTED_LOCATION"
    MISSING_PARAMETER_VALUE = "MISSING_PARAMETER_VALUE"
    MISSING_ATTRIBUTE_SUPPORT = "MISSING_ATTRIBUTE_SUPPORT"
    MISSING_VALUE_SUPPORT = "MISSING_VALUE_SUPPORT"
    MISSING_UNIT_SUPPORT = "MISSING_UNIT_SUPPORT"
    MISSING_REQUIREMENT_SUPPORT = "MISSING_REQUIREMENT_SUPPORT"
    MISSING_COMPATIBILITY_SUPPORT = "MISSING_COMPATIBILITY_SUPPORT"
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
    "firewall": ("firewall", "network firewall"),
    "tls_cipher": ("tls cipher", "cipher suite", "cipher suites", "cipher restriction"),
    "subnet_mask": ("subnet mask", "network mask", "子网掩码", "网络掩码"),
    "temperature_sensor": ("temperature sensor", "thermal sensor"),
    "megger_test": ("megger test", "insulation resistance measurement"),
    "dip_switch": ("dip switch", "dip-switch"),
    "configuration_data_required": (
        r"configuration.*data values", r"cannot.*without configuration",
        r"requires?.*configuration data",
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
    "install": ("install", "installation", "mount", "mounting"),
    "reinstall": ("reinstall", "re-install"),
    "disconnect": ("disconnect", "remove connection", "remove all connections"),
}

ATTRIBUTE_ALIASES = {
    "waiting_time": ("等待多久", "等待时间", "多久", "wait", "minute", "second"),
    "threshold": ("阈值", "threshold", "limit", "percent", "%"),
    "accuracy": CONCEPT_ALIASES["accuracy"],
    "voltage": ("电压", "voltage", "volt"),
    "quantity": ("多少个", "最大数量", "最多", "maximum", "max."),
    "status": ("什么状态", "状态", "status", "state", "fault status"),
    "cause": (
        "为什么", "原因", "why", "cause", "reason", "because",
    ),
    "resistance": ("阻值", "欧姆", "resistance", "ohm", "ω", "Ω"),
    "compatibility": ("compatibility", "compatible", "compatible with"),
    "requirements": (
        "requirement", "requirements", "prerequisite", "prerequisites",
        "restriction", "restrictions", "condition", "conditions",
    ),
    "version": ("version", "versions", "revision", "revisions", "device model", "devicemodel"),
    "default_value": ("default value", "by default", "default"),
    "range": ("supported range", "permitted range", "acceptable range", "range"),
    "rated_value": ("rated value", "rated voltage", "rated current", "rated"),
    "motor_voltage": ("motor voltage", "servomotor voltage", "motor winding voltage", "servomotor winding voltage"),
    "temperature": ("ambient temperature", "temperature"),
    "cable_length": ("cable length", "segment length", "transmission distance"),
    "data_size": ("data size", "data sizes", "input and output data", "process data size"),
    "subnet_mask": CONCEPT_ALIASES["subnet_mask"],
    "username": ("username", "user name", "login name"),
    "password": ("password", "login password"),
    "switch_position": ("switch position", "dip-switch position", "dip switch position"),
    "torque": ("torque", "tightening torque"),
    "manufacturer": ("manufacturer", "vendor", "maker"),
    "timer_values": ("timer values", "timer number", "completion flag", "present value"),
    "task_period": ("task period", "task cycle", "cycle time", "task timing"),
}

ATTRIBUTE_EVIDENCE_ALIASES = {
    **ATTRIBUTE_ALIASES,
    "threshold": (*ATTRIBUTE_ALIASES["threshold"], "or fewer", "or less", "at least", "at most"),
    "quantity": (*ATTRIBUTE_ALIASES["quantity"], "maximum", "minimum", "max", "min", "数量", "最大", "最小"),
    "version": (*ATTRIBUTE_ALIASES["version"], "firmware", "software", "build", "or later", "and above", "版本"),
    "requirements": (*ATTRIBUTE_ALIASES["requirements"], "required", "mandatory", "must", "before", "only if"),
    "default_value": (*ATTRIBUTE_ALIASES["default_value"], "initial", "factory default"),
    "range": (*ATTRIBUTE_ALIASES["range"], "variation", "from", " to ", "...", "…"),
    "rated_value": ATTRIBUTE_ALIASES["rated_value"],
    "motor_voltage": (*ATTRIBUTE_ALIASES["motor_voltage"], "motor rated voltage", "motor nominal voltage"),
    "cable_length": (*ATTRIBUTE_ALIASES["cable_length"], "电缆长度", "最长"),
    "data_size": (*ATTRIBUTE_ALIASES["data_size"], "input data", "output data", "kbyte", "byte", "数据量", "输入数据", "输出数据"),
    "switch_position": (*ATTRIBUTE_ALIASES["switch_position"], "position"),
    "task_period": (*ATTRIBUTE_ALIASES["task_period"], "任务周期", "任务时间"),
}

VALUE_KIND_PATTERNS = {
    "default": re.compile(r"\bdefault\b|factory default", re.IGNORECASE),
    "range": re.compile(r"\brange\b|permitted|acceptable variation", re.IGNORECASE),
    "maximum": re.compile(r"\bmaximum\b|\bmax\.?\b", re.IGNORECASE),
    "minimum": re.compile(r"\bminimum\b|\bmin\.?\b", re.IGNORECASE),
    "rated": re.compile(r"\brated\b", re.IGNORECASE),
    "nominal": re.compile(r"\bnominal\b", re.IGNORECASE),
    "recommended": re.compile(r"\brecommended\b|\badvised\b", re.IGNORECASE),
    "duration": re.compile(r"\bhow long\b|waiting time|wait time", re.IGNORECASE),
    "exact": re.compile(r"\b(?:what|which)\b.{0,80}\b(?:value|limit|length|size|voltage|temperature|torque|address)\b", re.IGNORECASE),
}

REQUIREMENT_TYPE_PATTERNS = (
    ("compatibility", re.compile(r"compatib|supported combination", re.IGNORECASE)),
    ("prerequisite", re.compile(r"prerequisite|\bbefore\b|prior to", re.IGNORECASE)),
    ("version", re.compile(r"version|revision|firmware|software build", re.IGNORECASE)),
    ("installation", re.compile(r"install|mount|wiring", re.IGNORECASE)),
    ("configuration", re.compile(r"configur|\bset(?:ting)?\b|assign", re.IGNORECASE)),
    ("safety", re.compile(r"safety|hazard|danger|warning", re.IGNORECASE)),
    ("maintenance", re.compile(r"maintenance|service|replace|restore", re.IGNORECASE)),
)

UNIT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:n\s*[·路.]?\s*m|v|volt|volts|amp|amps|ampere|amperes|hz|khz|mhz|s|sec|second|seconds|ms|μs|us|%|°c)(?![a-z0-9])",
    re.IGNORECASE,
)
VALUE_PATTERN = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:%|[a-zμ°]+)?", re.IGNORECASE)
REQUESTED_VALUE_PATTERN = re.compile(
    r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:%|n\s*[·路.]?\s*m|v|a|hz|khz|mhz|ms|μs|us|°c)(?![a-z0-9])",
    re.IGNORECASE,
)

V315_UNIT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:n\s*[·⋅-]?\s*m|lb\s*[·⋅-]?\s*in|v(?:ac|dc)?|volt|volts|"
    r"amp|amps|ampere|amperes|hz|khz|mhz|rpm|s|sec|second|seconds|ms|μs|us|"
    r"%|°c|°f|mm|cm|m|ft|byte|kbyte|kb)(?![a-z0-9])",
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
    requested_location: bool = False
    specificity: str = EvidenceSpecificity.GENERAL.value
    requested_value_kind: tuple[str, ...] = ()
    requested_requirement_type: str = "general"
    requested_qualifiers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "identifiers", "requested_concepts", "requested_attributes",
            "requested_action", "requested_protocol", "requested_value_kind",
            "requested_qualifiers",
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


def _requested_value_kinds(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    kinds = [name for name, pattern in VALUE_KIND_PATTERNS.items() if pattern.search(normalized)]
    location_cue = re.search(
        r"\b(?:where|at\s+which|which\s+(?:parts?|points?|terminals?|locations?))\b|"
        r"\bwhat\s+(?:physical\s+)?(?:points?|terminals?|locations?)\b|"
        r"在哪|哪些端子|哪些位置|何处",
        normalized, re.IGNORECASE,
    )
    if location_cue and not re.search(r"\b(?:value|limit|range|maximum|minimum|rated|nominal|default)\b", normalized, re.IGNORECASE):
        kinds = [name for name in kinds if name != "exact"]
    return tuple(kinds)


def _requested_requirement_type(query: str) -> str:
    normalized = _normalize(query)
    for name, pattern in REQUIREMENT_TYPE_PATTERNS:
        if pattern.search(normalized):
            return name
    return "general"


def _requested_qualifiers(
    query: str, analysis: QueryAnalysis, identifiers: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract model-like qualifiers without adding vendor-specific knowledge."""
    model = _normalize(analysis.equipment_model)
    identifier_set = {_normalize(item) for item in identifiers}
    tokens = re.findall(
        r"(?<![a-z0-9-])(?:[a-z]{2,}[a-z0-9-]*\d[a-z0-9-]*|[a-z]+\d[a-z0-9-]+)(?![a-z0-9-])",
        _normalize(query), re.IGNORECASE,
    )
    tokens.extend(re.findall(
        r"(?<![a-z0-9-])([a-z][a-z0-9-]{2,})\s+(?:firmware\s+|software\s+|unit\s+)?(?:versions?|revision)",
        _normalize(query), re.IGNORECASE,
    ))
    return tuple(dict.fromkeys(
        token.upper() for token in tokens
        if _normalize(token) not in identifier_set
        and _normalize(token) != model
        and _normalize(token) not in model
        and _normalize(token) not in {"which", "what", "minimum", "required"}
    ))


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
    if re.search(r"如何|怎样|步骤|怎么|\bhow\b|procedure|configure|set up|commission|reinstall", query, re.IGNORECASE):
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
    requested_location = bool(re.search(
        r"\bwhere\b|\bwhich\s+(?:section|chapter|appendix)\b|"
        r"\bon\s+what\s+page\b|\bwhere\s+in\s+the\s+manual\b",
        normalized,
        re.IGNORECASE,
    ))
    unit_match = V315_UNIT_PATTERN.search(normalized)
    requested_unit = _normalize(unit_match.group(0)) if unit_match else ""
    value_match = REQUESTED_VALUE_PATTERN.search(normalized)
    requested_value = value_match.group(0).strip() if value_match else ""
    requested_value_kind = _requested_value_kinds(normalized)
    requested_requirement_type = _requested_requirement_type(normalized)
    requested_qualifiers = _requested_qualifiers(query, analysis, identifiers)
    intent = _intent(normalized, identifiers)
    has_identity = bool(
        analysis.equipment_model or analysis.product_series
        or analysis.product_family or analysis.product_identities
    )
    specificity = EvidenceSpecificity.GENERAL
    if protocols or identifiers or requested_unit or requested_location or requested_qualifiers or len(concepts) >= 2:
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
        requested_location=requested_location,
        specificity=specificity.value,
        requested_value_kind=requested_value_kind,
        requested_requirement_type=requested_requirement_type,
        requested_qualifiers=requested_qualifiers,
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


def _unit_supported(requested_unit: str, evidence_text: str) -> bool:
    if not requested_unit:
        return True
    aliases = {
        "v": ("v", "volt", "volts"),
        "volt": ("v", "volt", "volts"),
        "volts": ("v", "volt", "volts"),
        "amp": ("a", "amp", "amps", "ampere", "amperes"),
        "amps": ("a", "amp", "amps", "ampere", "amperes"),
        "ampere": ("a", "amp", "amps", "ampere", "amperes"),
        "amperes": ("a", "amp", "amps", "ampere", "amperes"),
        "s": ("s", "sec", "second", "seconds"),
        "sec": ("s", "sec", "second", "seconds"),
        "second": ("s", "sec", "second", "seconds"),
        "seconds": ("s", "sec", "second", "seconds"),
        "n m": (r"n\s*[·⋅-]?\s*m",),
        "n·m": (r"n\s*[·⋅-]?\s*m",),
        "nm": (r"n\s*[·⋅-]?\s*m",),
        "v ac": ("v ac", "vac", "v"),
        "vac": ("vac", "v ac", "v"),
        "v dc": ("v dc", "vdc", "v"),
        "vdc": ("vdc", "v dc", "v"),
    }
    normalized_unit = _normalize(requested_unit)
    for alias in aliases.get(normalized_unit, (normalized_unit,)):
        pattern = alias if "\\s" in alias else re.escape(alias)
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", evidence_text, re.IGNORECASE):
            return True
    return False


def _location_supported(candidates: list) -> bool:
    for candidate in candidates:
        metadata = candidate.metadata or {}
        if any(str(metadata.get(key, "")).strip() for key in ("section", "subsection")):
            return True
        if metadata.get("page") is not None or metadata.get("page_number") is not None:
            return True
    return False


def _value_kind_supported(kind: str, text: str) -> bool:
    has_value = bool(
        VALUE_PATTERN.search(text)
        or re.search(r"\b(?:enabled|disabled|enable|disable|admin)\b", text, re.IGNORECASE)
        or re.search(r"\b(?:0x[0-9a-f]+|\d{1,3}(?:\.\d{1,3}){3})\b", text, re.IGNORECASE)
    )
    patterns = {
        "default": r"\bdefault\b|factory default|initial|\bdef\b",
        "range": r"\brange\b|variation|\d(?:\.\d+)?\s*(?:…|\.\.\.|to)\s*\d|范围|变化范围",
        "maximum": r"\bmaximum\b|\bmax\.?\b|at most|no more than|or fewer|最大|最长",
        "minimum": r"\bminimum\b|\bmin\.?\b|at least|or later|以上|最小",
        "rated": r"\brated\b|额定",
        "nominal": r"\bnominal\b|标称",
        "recommended": r"\brecommended\b|\badvised\b|建议|推荐",
        "duration": r"\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|mins?|hours?|hrs?)\b",
        "exact": r"",
    }
    if kind == "default":
        default_marker = bool(
            re.search(patterns[kind], text, re.IGNORECASE)
            or contains_term(text, ("default", "factory default", "initial"))
        )
        if default_marker:
            return has_value
        # Common parameter tables put the default on one line immediately before the range.
        return bool(re.search(r"\d+(?:\.\d+)?\s*%?\s+\d+(?:\.\d+)?\s*(?:…|\.\.\.|to)\s*\d", text, re.IGNORECASE))
    return has_value and (not patterns[kind] or bool(re.search(patterns[kind], text, re.IGNORECASE)))


def _attribute_supported(attribute: str, text: str) -> bool:
    if attribute == "cause":
        return bool(re.search(
            r"\b(?:because|therefore|thus|hence|so that|in order to|to prevent|to ensure|"
            r"enables?|allows?|avoids?)\b|"
            r"\bwithout\b.{0,100}\b(?:cannot|can't|won't|unable)\b|"
            r"\bif\b.{0,120}\b(?:may|can|will)\b.{0,50}\b(?:result|cause|lead)\b|"
            r"\bverif(?:y|ies|ied)\b.{0,60}\b(?:integrity|validity|correctness|safety)\b|"
            r"因为|因此|从而|以便|为了|否则|导致|确保|防止|避免",
            text, re.IGNORECASE,
        ))
    if attribute == "subnet_mask" and re.search(
        r"\b(?:no|without)\s+(?:default\s+)?subnet mask\b|未(?:指定|提供).{0,12}子网掩码",
        text, re.IGNORECASE,
    ):
        return False
    value_kind = {
        "default_value": "default", "range": "range", "rated_value": "rated",
    }.get(attribute)
    if value_kind and _value_kind_supported(value_kind, text):
        return True
    return _contains_alias(text, ATTRIBUTE_EVIDENCE_ALIASES[attribute])


def _concept_supported(concept: str, text: str) -> bool:
    if concept == "configuration_data_required":
        if re.search(
            r"(?:without|no|zero)\s+configuration|requires?\s+no\s+configuration|"
            r"only\s+supports?.{0,40}(?:without|no)\s+configuration|"
            r"无需配置|无须配置|不需要配置",
            text, re.IGNORECASE,
        ):
            return False
        return bool(re.search(
            r"requires?.{0,40}configuration data|configuration.?data.{0,40}(?:value|instance|size)|"
            r"配置数据.{0,30}(?:值|实例|大小)",
            text, re.IGNORECASE,
        ))
    return _contains_alias(text, CONCEPT_ALIASES[concept])


def _qualifier_supported(qualifier: str, text: str) -> bool:
    normalized = _normalize(qualifier)
    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized)}(?:[a-z0-9]{{1,4}}(?![a-z0-9])|(?=[^a-z0-9])|$)",
        text, re.IGNORECASE,
    ) is not None


def _version_requirement_supported(requirement: EvidenceRequirement, text: str) -> bool:
    value_markers = list(re.finditer(
        r"(?:version|revision|firmware|build|版本)\s*[:：]?\s*\d+(?:\.\d+)*|"
        r"\d+(?:\.\d+)+\s*(?:版本|version|revision|build)",
        text, re.IGNORECASE,
    ))
    if not value_markers:
        return False
    if not requirement.requested_qualifiers:
        return True
    for qualifier in requirement.requested_qualifiers:
        qualifier_matches = list(re.finditer(
            rf"(?<![a-z0-9]){re.escape(_normalize(qualifier))}[a-z0-9]{{0,4}}(?![a-z0-9])",
            text, re.IGNORECASE,
        ))
        if not qualifier_matches or not any(
            abs(marker.start() - match.end()) <= 60
            for marker in value_markers for match in qualifier_matches
        ):
            return False
    return True


def _requirement_type_supported(requirement: EvidenceRequirement, text: str) -> bool:
    requirement_type = requirement.requested_requirement_type
    if requirement_type == "compatibility":
        return bool(re.search(r"compatib|\bsupports?\b|required.{0,50}(?:adapter|module)|兼容|支持", text, re.IGNORECASE))
    if requirement_type == "prerequisite":
        return bool(re.search(r"prerequisite|\bbefore\b|prior to|\brequired\b|\bmust\b|前|必须", text, re.IGNORECASE))
    if requirement_type == "version":
        return _version_requirement_supported(requirement, text)
    return True


def _implicit_unit_supported(requirement: EvidenceRequirement, text: str) -> bool:
    attributes = set(requirement.requested_attributes)
    concepts = set(requirement.requested_concepts)
    checks = []
    if "voltage" in attributes:
        checks.append(r"\d+(?:\.\d+)?\s*v(?:ac|dc)?\b")
    if "temperature" in attributes or "temperature_sensor" in concepts:
        checks.append(r"\d+(?:\.\d+)?\s*°?\s*[cf]\b")
    if "torque" in attributes:
        checks.append(r"\d+(?:\.\d+)?\s*(?:n\s*[·⋅-]?\s*m|lb\s*[·⋅-]?\s*in|%)")
    if "cable_length" in attributes:
        checks.append(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|ft)\b")
    if "data_size" in attributes:
        checks.append(r"\d+(?:\.\d+)?\s*(?:byte|kbyte|kb)\b")
    if "waiting_time" in attributes:
        checks.append(r"\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|mins?|hours?|hrs?)\b")
    if "residual_energy" in concepts:
        checks.append(r"\d+(?:\.\d+)?\s*(?:μj|uj|microjoules?|joules?|j)\b")
    return all(re.search(pattern, text, re.IGNORECASE) for pattern in checks)


def _identifier_attribute_associated(
    requirement: EvidenceRequirement, text: str,
) -> bool:
    if not requirement.identifiers:
        return True
    target_positions = [
        match.start() for identifier in requirement.identifiers
        for match in re.finditer(re.escape(_normalize(identifier)), text, re.IGNORECASE)
    ]
    if not target_positions:
        return False
    strong_attributes = [
        name for name in requirement.requested_attributes
        if name not in {"requirements", "cause", "status", "quantity"}
    ]
    if not strong_attributes:
        return True
    other_identifiers = [match.start() for match in re.finditer(
        r"(?<![a-z0-9])(?:\d{1,2}\.\d{1,2}|[a-z]\d{2,5})(?![a-z0-9])", text, re.IGNORECASE,
    ) if all(abs(match.start() - target) > 1 for target in target_positions)]
    attribute_positions = [
        match.start() for attribute in strong_attributes
        for alias in ATTRIBUTE_EVIDENCE_ALIASES[attribute]
        for match in re.finditer(re.escape(_normalize(alias)), text, re.IGNORECASE)
        if alias and not any(token in alias for token in (".*", "(", "\\", "|", "[", "]"))
    ]
    if not attribute_positions:
        return True
    for target in target_positions:
        for attribute in attribute_positions:
            low, high = sorted((target, attribute))
            if not any(low < other < high for other in other_identifiers):
                return True
    return False


def _text_supports_local_value(requirement: EvidenceRequirement, text: str) -> bool:
    if requirement.identifiers and not all(
        contains_parameter_identifier(text, identifier) for identifier in requirement.identifiers
    ):
        return False
    if requirement.requested_qualifiers and not all(
        _qualifier_supported(item, text) for item in requirement.requested_qualifiers
    ):
        return False
    if requirement.requested_protocol and not all(
        _contains_alias(text, PROTOCOL_ALIASES[item]) for item in requirement.requested_protocol
    ):
        return False
    if requirement.requested_concepts and not all(
        _concept_supported(item, text) for item in requirement.requested_concepts
    ):
        return False
    if not all(_attribute_supported(item, text) for item in requirement.requested_attributes):
        return False
    if not _identifier_attribute_associated(requirement, text):
        return False
    if requirement.requested_value and requirement.requested_value.casefold() not in text:
        return False
    if requirement.requested_unit and not _unit_supported(requirement.requested_unit, text):
        return False
    if not all(_value_kind_supported(kind, text) for kind in requirement.requested_value_kind):
        return False
    if not _implicit_unit_supported(requirement, text):
        return False
    return True


def _scope_group_key(candidate) -> tuple[str, ...] | None:
    metadata = candidate.metadata or {}
    document_id = _normalize(metadata.get("document_id", ""))
    scope = _normalize(metadata.get("subsection") or metadata.get("section") or "")
    if not document_id or not scope:
        return None
    return (
        document_id,
        _normalize(metadata.get("manufacturer", "")),
        _normalize(metadata.get("equipment_model", "")),
        scope,
    )


def _local_value_supported(requirement: EvidenceRequirement, candidates: list) -> bool:
    needs_value = bool(
        requirement.requested_value or requirement.requested_value_kind or requirement.requested_unit
    )
    if not needs_value and not requirement.requested_unit:
        return True
    for candidate in candidates:
        text = _normalize(candidate.document.page_content)
        if _text_supports_local_value(requirement, text):
            return True

    groups: dict[tuple[str, ...], list] = {}
    for candidate in candidates:
        key = _scope_group_key(candidate)
        if key is not None:
            groups.setdefault(key, []).append(candidate)
    for group in groups.values():
        if len(group) < 2:
            continue
        pages = [
            int(candidate.metadata.get("page")) for candidate in group
            if str(candidate.metadata.get("page", "")).isdigit()
        ]
        if pages and max(pages) - min(pages) > 2:
            continue
        texts = [_normalize(candidate.document.page_content) for candidate in group]
        if requirement.identifiers and not all(
            all(contains_parameter_identifier(text, identifier) for identifier in requirement.identifiers)
            for text in texts
        ):
            continue
        if _text_supports_local_value(requirement, "\n".join(texts)):
            return True
    return False


def validate_evidence_support(query: str, result, documents: list | None = None) -> EvidenceSupport:
    candidates = list(getattr(result, "candidates", []) or [])
    corpus = list(documents if documents is not None else getattr(result, "corpus_documents", []) or [])
    analysis = getattr(result, "query_analysis", None) or analyze_query(query, corpus)
    requirement = build_evidence_requirement(query, corpus, analysis)
    candidate_texts = [_normalize(candidate.document.page_content) for candidate in candidates]
    evidence_text = _normalize("\n".join(candidate_texts))

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
            else _concept_supported(concept, evidence_text)
        )
        for concept in requirement.requested_concepts
    }
    action_hits = {
        action: _contains_alias(evidence_text, ACTION_ALIASES[action])
        for action in requirement.requested_action
    }
    attribute_hits = {
        attribute: _attribute_supported(attribute, evidence_text)
        for attribute in requirement.requested_attributes
    }
    qualifier_hits = {
        qualifier: _qualifier_supported(qualifier, evidence_text)
        for qualifier in requirement.requested_qualifiers
    }
    requirement_type_supported = _requirement_type_supported(requirement, evidence_text)
    unit_supported = not requirement.requested_unit or any(
        _unit_supported(requirement.requested_unit, text) for text in candidate_texts
    )
    location_supported = not requirement.requested_location or _location_supported(candidates)
    value_supported = not requirement.requested_value or requirement.requested_value.casefold() in evidence_text
    parameter_value_supported = bool(VALUE_PATTERN.search(evidence_text))
    local_value_supported = _local_value_supported(requirement, candidates)

    missing = []
    if not identity_supported:
        missing.append("target_identity")
    missing.extend(f"identifier:{name}" for name, supported in identifier_hits.items() if not supported)
    missing.extend(f"protocol:{name}" for name, supported in protocol_hits.items() if not supported)
    missing.extend(f"concept:{name}" for name, supported in concept_hits.items() if not supported)
    missing.extend(f"qualifier:{name}" for name, supported in qualifier_hits.items() if not supported)
    if requirement.requested_requirement_type == "compatibility" and not requirement_type_supported:
        missing.append("requirement_type:compatibility")
    elif requirement.requested_requirement_type == "prerequisite" and not requirement_type_supported:
        missing.append("requirement_type:prerequisite")
    elif requirement.requested_requirement_type == "version" and not requirement_type_supported:
        missing.append("requirement_type:version")
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
    if not location_supported:
        missing.append("location")
    if not value_supported:
        missing.append(f"value:{requirement.requested_value}")
    if not local_value_supported:
        missing.append("value:local_association")

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
        "requested_qualifiers": qualifier_hits,
        "requested_requirement_type": requirement_type_supported,
        "requested_value_kinds": {
            kind: any(_value_kind_supported(kind, text) for text in candidate_texts)
            for kind in requirement.requested_value_kind
        },
        "requested_unit": unit_supported,
        "requested_location": location_supported,
        "requested_value": value_supported,
        "parameter_value": parameter_value_supported,
        "local_value_association": local_value_supported,
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
            any(_concept_supported(name, _normalize(candidate.document.page_content)) for name in requirement.requested_concepts),
        ))
    )

    if not candidates:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.NO_EVIDENCE
    elif "target_identity" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MODEL_MISMATCH
    elif any(item.startswith("identifier:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_IDENTIFIER_SUPPORT
    elif any(item.startswith(("protocol:", "concept:", "qualifier:")) for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUIRED_CONCEPT
    elif "requirement_type:compatibility" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_COMPATIBILITY_SUPPORT
    elif any(item.startswith("requirement_type:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUIREMENT_SUPPORT
    elif any(item.startswith("action:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUESTED_ACTION
    elif "location" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUESTED_LOCATION
    elif any(item.startswith("attribute:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_ATTRIBUTE_SUPPORT
    elif any(item.startswith("unit:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_UNIT_SUPPORT
    elif any(item.startswith("value:") for item in missing) or "parameter_value" in missing:
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_VALUE_SUPPORT
    elif any(item.startswith("requirement_type:") for item in missing):
        status, reason = SupportStatus.INSUFFICIENT, SupportReason.MISSING_REQUIREMENT_SUPPORT
    elif not any((identities, requirement.identifiers, requirement.requested_protocol,
                  requirement.requested_concepts, requirement.requested_attributes,
                  requirement.requested_action, requirement.requested_unit,
                  requirement.requested_value, requirement.requested_location,
                  requirement.requested_value_kind, requirement.requested_qualifiers,
                  requirement.requested_requirement_type != "general")):
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
