"""Identity-aware Evidence sufficiency candidate for V3.42.

The candidate composes on top of the unchanged V3.41 identity-claim decision.
It can reconsider only an identity-COMPATIBLE verification refusal and requires
one retrieved candidate to bind all four proposition components locally:
target, relation, attribute, and value/action.  It never treats compatible
identity as sufficient evidence and never joins components across chunks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .identity_claim_v341 import analyze_identity_claim_evidence
from .product_identity import normalize_identity_text


EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION = "evidence-v342-sufficiency-candidate"
EVIDENCE_SUFFICIENCY_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
SUPPORT_CONFIDENCE_FLOOR = 0.82


class EvidenceSufficiencyRelation(str, Enum):
    DIRECT_SUPPORTED = "DIRECT_SUPPORTED"
    SEMANTIC_SUPPORTED = "SEMANTIC_SUPPORTED"
    REFERENCE_SUPPORTED = "REFERENCE_SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNSAFE = "UNSAFE"


_SUPPORTED = frozenset({
    EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value,
    EvidenceSufficiencyRelation.SEMANTIC_SUPPORTED.value,
    EvidenceSufficiencyRelation.REFERENCE_SUPPORTED.value,
})
_MODEL = re.compile(
    r"\bEL\d{4}(?:-\d{4})?\b|\bBNI(?:\s+IOL)?-\d{3}-\d{3}-[A-Z]\d{3}\b|\bEL18XX\b",
    re.IGNORECASE,
)
_REFERENCE_QUERY = re.compile(r"\b(?:refer|reference|point|page|section)\w*\b", re.IGNORECASE)
_REFERENCE_TARGET = re.compile(r"\b(?:page|section)\s*(\d{1,3})\b", re.IGNORECASE)
_REFERENCE_MARKER = re.compile(r"\[\s*}\s*\d+\s*\]|\bsee\s+also\b|\b(?:refer|reference|link|point)\w*\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|without|never|bypass|regardless)\b", re.IGNORECASE)
_KNOWN_MANUFACTURERS = (
    "balluff", "beckhoff", "siemens", "mitsubishi", "omron", "rockwell",
    "allen bradley", "schneider", "abb", "festo", "phoenix contact", "wago",
    "turck", "sick", "keyence", "panasonic", "hitachi", "weg", "invertek",
)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
    "from", "has", "have", "in", "is", "it", "its", "manual", "of", "on",
    "or", "the", "this", "to", "under", "with", "within",
})

# The vocabulary is deliberately closed and industrial: every entry denotes
# one attribute, never a product or family expansion.  Both query and evidence
# must resolve to the same canonical attribute before support is possible.
_ATTRIBUTE_GROUPS: dict[str, tuple[str, ...]] = {
    "input_filter": ("input filter", "filter time", "input delay"),
    "number_of_inputs": ("number of inputs", "input count", "digital inputs", "inputs outputs"),
    "input_current": ("input current", "input amperage"),
    "nominal_input_voltage": ("nominal voltage of inputs", "nominal input voltage"),
    "signal_voltage_zero": ("signal voltage 0", "logic zero voltage", "low signal voltage"),
    "signal_voltage_one": ("signal voltage 1", "logic one voltage", "high signal voltage"),
    "ebus_current": ("current consumption via e bus", "e bus current draw", "e bus consumption"),
    "power_contact_current": ("current consumption power contacts", "power contact current draw"),
    "electrical_isolation": ("electrical isolation", "electrical separation", "galvanic isolation"),
    "process_image_width": ("bit width in the process image", "process image width"),
    "configuration": ("configuration", "configured", "configuration method"),
    "rated_cross_section": ("rated cross section", "wire size", "conductor cross section"),
    "weight": ("weight", "mass"),
    "operating_temperature": ("ambient temperature range during operation", "operating temperature range"),
    "storage_temperature": ("storage temperature",),
    "relative_humidity": ("relative humidity", "humidity limit"),
    "dimensions": ("dimensions", "size"),
    "mounting": ("mounting", "mounting rail", "rail installation"),
    "protection_class": ("protection class", "ingress protection", "ip rating"),
    "installation_position": ("installation position", "mounting orientation", "operating orientation"),
    "product_id": ("product id", "product identifier"),
    "device_id": ("device id", "device identifier"),
    "vendor_id": ("vendor id", "vendor identifier"),
    "application_tag": ("application specific tag", "application tag"),
    "input_inversion": ("inversion of the inputs", "input inversion"),
    "io_configuration": ("config inputs outputs", "input output configuration", "i o configuration"),
    "safe_state_pin4": ("safe state on pin 4", "pin 4 safe state"),
    "safe_state_pin2": ("safe state on pin 2", "pin 2 safe state"),
    "safe_state_output": ("safe state output", "safe state of outputs"),
    "voltage_monitoring": ("voltage monitoring",),
    "output_monitoring": ("output monitoring", "monitoring the outputs"),
    "actuator_warning": ("actuator warning",),
    "serial_number": ("serial number",),
    "extension_port": ("extension port",),
    "function_ground": ("function ground", "functional ground", "functional earth"),
    "cable_length": ("cable length", "maximum length", "maximum cable run"),
    "mechanical_attachment": ("attached", "attachment", "mounting hardware"),
    "supply_source": ("supply voltage connection", "supply source", "powered via"),
    "abbreviation_meaning": ("stand for", "means", "abbreviation", "expanded as"),
    "atex_approval": ("atex approval", "atex"),
    "iecex_approval": ("iecex approval", "iecex"),
    "culus_approval": ("culus approval", "culus"),
    "approval": ("approval", "marking"),
    "vibration_resistance": ("vibration shock resistance", "mechanical load capacity"),
}

_ABBREVIATIONS = {
    "bni": "balluff network interface",
    "dpp": "direct parameter page",
    "i o port": "digital input output port",
    "emc": "electromagnetic compatibility",
    "fe": "function ground",
    "iol": "io link",
    "lsb": "least significant bit",
    "msb": "most significant bit",
    "spdu": "service protocol data unit",
    "us": "sensor supply undervoltage",
    "ua": "actuator supply undervoltage",
}

_SECTION_ANCHORS = (
    "extension off",
    "extended with bni iol 302 002 e012",
    "valve terminal connector",
    "el1804 el1814 product description",
    "el1809 el1819 product description",
)


@dataclass(frozen=True)
class EvidenceRelationDecision:
    relation: str
    reason_code: str
    confidence: float
    chunk_id: str = ""
    document_id: str = ""
    target: str = ""
    relation_anchor: str = ""
    attribute_anchor: str = ""
    value_action_anchor: str = ""
    semantic_match: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    query: str
    decision: str
    reason: str
    confidence: float
    final_decision_source: str
    query_path: str
    baseline_decision: str
    baseline_reason: str
    identity_result: str
    relaxed: bool
    relation: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(value: object) -> str:
    text = str(value or "").replace("µ", "u").replace("²", "2").replace("/", " ")
    return " ".join(normalize_identity_text(text).split())


def _phrases_present(text: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)]


def _attribute(text: str) -> tuple[str, str] | None:
    matches: list[tuple[int, str, str]] = []
    for canonical, phrases in _ATTRIBUTE_GROUPS.items():
        for phrase in _phrases_present(text, phrases):
            matches.append((len(phrase), canonical, phrase))
    if not matches:
        return None
    _, canonical, phrase = max(matches)
    return canonical, phrase


def _aliases(metadata: dict) -> tuple[str, ...]:
    raw = metadata.get("model_aliases", "")
    values = raw if isinstance(raw, list) else str(raw or "").split("|")
    values.extend([
        str(metadata.get("equipment_model", "")),
        str(metadata.get("product_family", "")),
        str(metadata.get("product_series", "")),
    ])
    return tuple(dict.fromkeys(_norm(value) for value in values if _norm(value)))


def _owned_models(metadata: dict) -> tuple[str, ...]:
    raw = metadata.get("model_aliases", "")
    values = list(raw) if isinstance(raw, list) else str(raw or "").split("|")
    values.append(str(metadata.get("equipment_model", "")))
    return tuple(dict.fromkeys(_norm(value) for value in values if _norm(value)))


def _query_models(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_norm(match.group(0)) for match in _MODEL.finditer(query)))


def _document(candidate: Any) -> Any:
    return getattr(candidate, "document", candidate)


def _query_values(query: str, models: tuple[str, ...]) -> tuple[str, ...]:
    text = _norm(query)
    for model in models:
        text = re.sub(rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", " ", text)
    for target in _REFERENCE_TARGET.findall(query):
        text = re.sub(rf"\b(?:page|section)\s*{re.escape(target)}\b", " ", text)
    patterns = (
        r"\b0x(?:[0-9a-f]{2}(?:\s+|$)){1,8}",
        r"\b[0-9a-f]{2,4}hex\b",
        r"\bip\s*\d{2}\b",
        r"\b\d+(?:\.\d+)?\s*(?:us|ms|ma|vdc|v|mm2|mm|m|g|bytes?|bits?|wire|inputs?|outputs?|%)\b",
        r"\b\d+\s+m\d+\s+screws?\b",
        r"\b(?:read write|read only|variable|bni00ar|balluff|stainless steel|a coded male|low impedance|twincat system manager|io link interface|via power contacts)\b",
        r"\bindex\s+\d{1,3}\b",
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0) for match in re.finditer(pattern, text, re.IGNORECASE))
    return tuple(dict.fromkeys(_norm(value) for value in values))


def _value_present(text: str, value: str) -> bool:
    if value.startswith("index "):
        number = value.split()[-1]
        return re.search(rf"(?<!\d){re.escape(number)}(?!\d)", text) is not None
    return re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", text) is not None


def _local_windows(text: str, attribute_phrases: tuple[str, ...], radius: int = 28) -> list[str]:
    lines = [_norm(line) for line in str(text or "").splitlines() if _norm(line)]
    indexes = [
        index for index, line in enumerate(lines)
        if any(phrase in line for phrase in attribute_phrases)
    ]
    if not indexes:
        return []
    return ["\n".join(lines[max(0, index - radius):index + radius + 1]) for index in indexes]


def _parameter_windows(text: str, attribute_phrases: tuple[str, ...]) -> list[str]:
    """Return one extracted parameter-table row at a time.

    Balluff's PDF exposes a row as several physical lines.  The hexadecimal DPP
    cell begins the row, so it is the safe boundary for preventing a value or
    SPDU index from a neighboring parameter from being borrowed.
    """
    lines = [_norm(line) for line in str(text or "").splitlines() if _norm(line)]
    starts = [index for index, line in enumerate(lines) if re.match(r"^[0-9a-f]{2}hex\b", line)]
    windows: list[str] = []
    for index, line in enumerate(lines):
        if not any(phrase in line for phrase in attribute_phrases):
            continue
        previous = [start for start in starts if start <= index]
        following = [start for start in starts if start > index]
        start = previous[-1] if previous else max(0, index - 4)
        end = following[0] if following else min(len(lines), index + 8)
        windows.append("\n".join(lines[start:end]))
    return windows


def _reference_owned(text: str, attribute_phrases: tuple[str, ...], target: str) -> bool:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not any(phrase in _norm(line) for phrase in attribute_phrases):
            continue
        local = "\n".join(lines[max(0, index - 2):index + 5])
        for phrase in attribute_phrases:
            anchor = r"[\s/\-]+".join(re.escape(part) for part in phrase.split())
            span = 240 if phrase.startswith(("vibration", "mechanical load")) else 60
            if re.search(
                rf"{anchor}.{{0,{span}}}?\[\s*}}\s*{re.escape(target)}\s*\]",
                local, re.IGNORECASE | re.DOTALL,
            ):
                return True
    return False


def _section_required(query_norm: str) -> str:
    return next((anchor for anchor in _SECTION_ANCHORS if anchor in query_norm), "")


def _manufacturer_conflict(query_norm: str, metadata: dict) -> bool:
    named = [name for name in _KNOWN_MANUFACTURERS if name in query_norm]
    if not named:
        return False
    manufacturer = _norm(metadata.get("manufacturer", ""))
    return not any(name in manufacturer for name in named)


def _column_scope_conflict(
    query_model: str,
    attribute: str,
    values: tuple[str, ...],
    window: str,
) -> bool:
    """Reject a value that belongs to a sibling column in a two-model table."""
    if not query_model or not values:
        return False
    header = re.search(r"technical data\s+(el\d{4})\s+(el\d{4})", window)
    if not header:
        return False
    models = (_norm(header.group(1)), _norm(header.group(2)))
    if query_model not in models:
        return True
    phrases = _ATTRIBUTE_GROUPS[attribute]
    row = next((line for line in window.splitlines() if any(phrase in line for phrase in phrases)), "")
    if not row:
        return False
    cells = re.findall(r"\d+(?:\.\d+)?\s*(?:us|ms|ma|vdc|v|mm2|mm|m|g|bytes?|bits?|%)", row)
    if len(cells) != len(models):
        return False
    owned = _norm(cells[models.index(query_model)])
    return any(not _value_present(owned, value) for value in values)


def _sibling_row_conflict(query_model: str, values: tuple[str, ...], window: str) -> bool:
    if not query_model or not values:
        return False
    inline = re.findall(r"(el\d{4}(?:-\d{4})?)\s*:\s*([^;\n)]+)", window)
    if inline:
        owned = next((_norm(value) for model, value in inline if _norm(model) == query_model), "")
        if owned and any(not _value_present(owned, value) for value in values):
            return any(all(_value_present(_norm(value), item) for item in values) for _, value in inline)
    target_rows = []
    sibling_rows = []
    for line in window.splitlines():
        models = _query_models(line)
        if len(models) != 1:
            continue
        if models[0] == query_model:
            target_rows.append(line)
        else:
            sibling_rows.append(line)
    if not target_rows:
        return False
    owned = any(all(_value_present(line, value) for value in values) for line in target_rows)
    borrowed = any(all(_value_present(line, value) for value in values) for line in sibling_rows)
    return borrowed and not owned


def _abbreviation_relation(
    query_norm: str,
    candidate: Any,
    target: str,
) -> EvidenceRelationDecision | None:
    if not any(term in query_norm for term in ("stand for", "means", "abbreviation", "expanded as")):
        return None
    document = _document(candidate)
    metadata = getattr(document, "metadata", {}) or {}
    text = _norm(getattr(document, "page_content", ""))
    requested = re.search(r"\b(?:does|is)\s+([a-z0-9 ]+?)\s+(?:stand for|an abbreviation|expanded as)\b", query_norm)
    requested_key = requested.group(1).strip() if requested else ""
    acronym = (
        requested_key if requested_key in _ABBREVIATIONS
        else "" if requested_key
        else next((key for key in _ABBREVIATIONS if re.search(rf"\b{re.escape(key)}\b", query_norm)), "")
    )
    if not acronym:
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.INSUFFICIENT.value,
            "ABBREVIATION_TARGET_MISSING", 0.9, target=target,
            attribute_anchor="abbreviation_meaning",
        )
    expansion = _ABBREVIATIONS[acronym]
    claimed = re.search(r"\bstand for\s+(.+?)(?:\?|$)", query_norm)
    if claimed and expansion not in claimed.group(1):
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.UNSAFE.value,
            "ABBREVIATION_DEFINITION_CONFLICT", 0.99,
            chunk_id=str(metadata.get("chunk_id", "")),
            document_id=str(metadata.get("document_id", "")), target=target,
            relation_anchor="DEFINES", attribute_anchor="abbreviation_meaning",
            value_action_anchor=claimed.group(1).strip(),
        )
    if re.search(rf"\b{re.escape(acronym)}\b", text) and expansion in text:
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.SEMANTIC_SUPPORTED.value,
            "ABBREVIATION_DEFINITION_SUPPORTED", 0.94,
            chunk_id=str(metadata.get("chunk_id", "")),
            document_id=str(metadata.get("document_id", "")), target=target,
            relation_anchor="DEFINES", attribute_anchor="abbreviation_meaning",
            value_action_anchor=expansion, semantic_match=True,
        )
    return EvidenceRelationDecision(
        EvidenceSufficiencyRelation.UNSAFE.value,
        "ABBREVIATION_DEFINITION_CONFLICT", 0.96,
        chunk_id=str(metadata.get("chunk_id", "")),
        document_id=str(metadata.get("document_id", "")), target=target,
        relation_anchor="DEFINES", attribute_anchor="abbreviation_meaning",
    )


def classify_evidence_sufficiency_relation(
    query: str,
    candidates: list[Any],
    *,
    allowed_document_id: str = "",
) -> EvidenceRelationDecision:
    """Classify one bounded target-relation-attribute-value/action relation."""
    query_norm = _norm(query)
    query_models = _query_models(query)
    metadata_rows = [getattr(_document(candidate), "metadata", {}) or {} for candidate in candidates]
    known_aliases = {alias for metadata in metadata_rows for alias in _owned_models(metadata)}
    if query_models and any(
        not any(model == alias or model in alias or alias in model for alias in known_aliases)
        for model in query_models
    ):
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.UNSAFE.value,
            "CROSS_MODEL_LEAKAGE_BLOCKED", 0.99, target="|".join(query_models),
        )
    if _NEGATION.search(query):
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.UNSAFE.value,
            "NEGATED_RELAXATION_FORBIDDEN", 0.99, target="|".join(query_models),
        )

    query_attribute = _attribute(query_norm)
    if query_attribute is None:
        return EvidenceRelationDecision(
            EvidenceSufficiencyRelation.INSUFFICIENT.value,
            "ATTRIBUTE_NOT_BOUND", 0.9, target="|".join(query_models),
        )
    attribute, query_attribute_phrase = query_attribute
    values = _query_values(query, query_models)
    section = _section_required(query_norm)
    reference_targets = tuple(_REFERENCE_TARGET.findall(query))
    best_failure: EvidenceRelationDecision | None = None

    for candidate in candidates:
        document = _document(candidate)
        metadata = getattr(document, "metadata", {}) or {}
        document_id = str(metadata.get("document_id", ""))
        if allowed_document_id and document_id != allowed_document_id:
            continue
        if _manufacturer_conflict(query_norm, metadata):
            best_failure = EvidenceRelationDecision(
                EvidenceSufficiencyRelation.UNSAFE.value,
                "MANUFACTURER_EXPANSION_BLOCKED", 0.99,
                chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
            )
            continue
        aliases = _aliases(metadata)
        owned_models = _owned_models(metadata)
        explicit_target = next((model for model in query_models if any(
            model == alias or model in alias or alias in model for alias in owned_models
        )), "")
        target = explicit_target or next((alias for alias in aliases if alias in query_norm), "")
        if query_models and not explicit_target:
            continue
        if not target:
            if not allowed_document_id:
                continue
            target = allowed_document_id

        abbreviation = _abbreviation_relation(query_norm, candidate, target)
        if abbreviation is not None:
            if abbreviation.relation in _SUPPORTED:
                return abbreviation
            best_failure = abbreviation
            continue

        raw_text = str(getattr(document, "page_content", "") or "")
        text = _norm(raw_text)
        phrases = _ATTRIBUTE_GROUPS[attribute]
        windows = _local_windows(raw_text, phrases)
        if attribute in {
            "product_id", "device_id", "vendor_id", "application_tag", "input_inversion",
            "io_configuration", "safe_state_pin4", "safe_state_pin2", "safe_state_output",
            "voltage_monitoring", "output_monitoring", "actuator_warning", "serial_number",
            "extension_port",
        } or re.search(r"\b(?:dpp|spdu|parameter)\b", query_norm):
            parameter_windows = _parameter_windows(raw_text, phrases)
            if parameter_windows:
                windows = parameter_windows
        if not windows:
            best_failure = EvidenceRelationDecision(
                EvidenceSufficiencyRelation.INSUFFICIENT.value,
                "ATTRIBUTE_RELATION_MISSING", 0.88,
                chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                target=target, attribute_anchor=attribute,
            )
            continue
        for window in windows:
            if section and section not in text:
                best_failure = EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.UNSAFE.value,
                    "SECTION_SCOPE_MISMATCH", 0.96,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, attribute_anchor=attribute,
                )
                continue
            metadata_model = _norm(metadata.get("equipment_model", ""))
            if explicit_target and explicit_target != metadata_model and explicit_target not in text:
                continue
            window_models = _query_models(window)
            if explicit_target and explicit_target not in window_models and window_models:
                continue
            if _column_scope_conflict(explicit_target, attribute, values, window):
                best_failure = EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.UNSAFE.value,
                    "SIBLING_MODEL_VALUE_BLOCKED", 0.99,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, attribute_anchor=attribute,
                    value_action_anchor="|".join(values),
                )
                continue
            if _sibling_row_conflict(explicit_target, values, window):
                best_failure = EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.UNSAFE.value,
                    "SIBLING_MODEL_VALUE_BLOCKED", 0.99,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, attribute_anchor=attribute,
                    value_action_anchor="|".join(values),
                )
                continue

            if _REFERENCE_QUERY.search(query):
                if not reference_targets or not _REFERENCE_MARKER.search(raw_text):
                    best_failure = EvidenceRelationDecision(
                        EvidenceSufficiencyRelation.INSUFFICIENT.value,
                        "REFERENCE_TARGET_UNVERIFIED", 0.95,
                        chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                        target=target, relation_anchor="REFERENCES", attribute_anchor=attribute,
                    )
                    continue
                if not all(_reference_owned(raw_text, phrases, item) for item in reference_targets):
                    best_failure = EvidenceRelationDecision(
                        EvidenceSufficiencyRelation.UNSAFE.value,
                        "REFERENCE_TARGET_MISMATCH", 0.99,
                        chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                        target=target, relation_anchor="REFERENCES", attribute_anchor=attribute,
                        value_action_anchor="|".join(reference_targets),
                    )
                    continue
                return EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.REFERENCE_SUPPORTED.value,
                    "EXPLICIT_REFERENCE_SUPPORTED", 0.94,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, relation_anchor="REFERENCES", attribute_anchor=attribute,
                    value_action_anchor="|".join(reference_targets),
                )

            if not values:
                best_failure = EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.INSUFFICIENT.value,
                    "VALUE_OR_ACTION_NOT_BOUND", 0.9,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, relation_anchor="ASSERTS", attribute_anchor=attribute,
                )
                continue
            if not all(_value_present(window, value) for value in values):
                best_failure = EvidenceRelationDecision(
                    EvidenceSufficiencyRelation.UNSAFE.value,
                    "ATTRIBUTE_VALUE_MISMATCH", 0.97,
                    chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                    target=target, relation_anchor="ASSERTS_VALUE", attribute_anchor=attribute,
                    value_action_anchor="|".join(values),
                )
                continue

            semantic = query_attribute_phrase != _ATTRIBUTE_GROUPS[attribute][0] or bool(section)
            relation = (
                EvidenceSufficiencyRelation.SEMANTIC_SUPPORTED.value
                if semantic else EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value
            )
            return EvidenceRelationDecision(
                relation,
                "SEMANTIC_ATTRIBUTE_SUPPORTED" if semantic else "DIRECT_PROPOSITION_SUPPORTED",
                0.9 if semantic else 0.92,
                chunk_id=str(metadata.get("chunk_id", "")), document_id=document_id,
                target=target, relation_anchor="ASSERTS_VALUE", attribute_anchor=attribute,
                value_action_anchor="|".join(values), semantic_match=semantic,
            )

    return best_failure or EvidenceRelationDecision(
        EvidenceSufficiencyRelation.INSUFFICIENT.value,
        "NO_SINGLE_CHUNK_PROPOSITION", 0.9,
        target="|".join(query_models), attribute_anchor=attribute,
    )


def _dominant_document_id(result: Any) -> str:
    ids = {
        str((getattr(_document(candidate), "metadata", {}) or {}).get("document_id", ""))
        for candidate in list(getattr(result, "candidates", []) or [])
    }
    ids.discard("")
    return next(iter(ids)) if len(ids) == 1 else ""


def analyze_evidence_sufficiency(
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
) -> EvidenceSufficiencyDecision:
    """Run V3.41 unchanged, then apply the V3.42 relation gate."""
    baseline = analyze_identity_claim_evidence(
        query, result, documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    baseline_dict = baseline.as_dict()
    if baseline.decision == "ANSWER":
        relation = EvidenceRelationDecision(
            EvidenceSufficiencyRelation.DIRECT_SUPPORTED.value,
            "V341_ALREADY_SUPPORTED", 1.0,
        )
        return EvidenceSufficiencyDecision(
            query, "ANSWER", baseline.reason, 1.0, baseline.final_decision_source,
            baseline.query_path, baseline.decision, baseline.reason,
            baseline.identity_result, False, relation.as_dict(), baseline_dict,
        )
    if baseline.identity_result != "COMPATIBLE":
        relation = EvidenceRelationDecision(
            EvidenceSufficiencyRelation.UNSAFE.value,
            "IDENTITY_BOUNDARY_PRESERVED", 1.0,
        )
        return EvidenceSufficiencyDecision(
            query, "ABSTAIN", relation.reason_code, relation.confidence,
            baseline.final_decision_source, baseline.query_path, baseline.decision,
            baseline.reason, baseline.identity_result, False,
            relation.as_dict(), baseline_dict,
        )
    if baseline.query_path != "VERIFICATION":
        relation = EvidenceRelationDecision(
            EvidenceSufficiencyRelation.INSUFFICIENT.value,
            "NON_VERIFICATION_PATH_PRESERVED", 1.0,
        )
        return EvidenceSufficiencyDecision(
            query, "ABSTAIN", relation.reason_code, relation.confidence,
            baseline.final_decision_source, baseline.query_path, baseline.decision,
            baseline.reason, baseline.identity_result, False,
            relation.as_dict(), baseline_dict,
        )

    allowed_document_id = _dominant_document_id(result) if baseline.expanded else ""
    relation = classify_evidence_sufficiency_relation(
        query, list(getattr(result, "candidates", []) or []),
        allowed_document_id=allowed_document_id,
    )
    supported = relation.relation in _SUPPORTED and relation.confidence >= SUPPORT_CONFIDENCE_FLOOR
    return EvidenceSufficiencyDecision(
        query, "ANSWER" if supported else "ABSTAIN", relation.reason_code,
        relation.confidence,
        "V342_EVIDENCE_SUFFICIENCY" if supported else baseline.final_decision_source,
        baseline.query_path, baseline.decision, baseline.reason,
        baseline.identity_result, supported, relation.as_dict(), baseline_dict,
    )
