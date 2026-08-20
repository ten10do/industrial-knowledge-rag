"""Query-type-aware evidence architecture (V3.27).

V3.26's sealed gate showed that the selective local-NLI verifier is correct when
it fires (7/7 routed correct) but that the evidence architecture treats
VERIFICATION (a complete proposition to confirm) and EXTRACTION (an unknown slot
to fill) as the same task.  Open "Which register …? / What value …? / Which
terminal …?" queries therefore massively false-refuse.

This module adds a *second, additive* path without touching the frozen V3.25
verifier.  Queries are routed to one of:

* VERIFICATION — the proposition is already complete.  The existing rule
  contract + selective NLI judge run unchanged.
* EXTRACTION — an unknown slot must be filled from candidate-local text using
  deterministic, candidate-grounded extraction (never free-text generation).

Discipline: no LLM, no generation, no ground-truth labels inside the router or
extractor.  This module imports the V3.25 rule/judge; it never modifies them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from .evidence import Decision, analyze_retrieval_evidence
from .evidence_contract import (
    CandidateClaim,
    TypedEvidenceRequirement,
    build_typed_requirement,
    extract_candidate_claim,
)
from .semantic_judge import (
    JudgeDecision,
    route_ambiguity,
)
from .semantic_judge_localnli import (
    LocalNliJudge,
    build_hypothesis,
    build_premise,
)
from .technical import (
    extract_parameter_references,
    normalize_technical_text,
)

# Experimental candidate identity.  This is a development experiment only.
QUERY_TYPE_CANDIDATE_VERSION = "evidence-v327-querytype-candidate"
QUERY_TYPE_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
# This study uses deterministic candidate-grounded extraction only.
LLM_EXTRACTION_USED = "NO"

# V3.29 boundary disposition.  VERIFIER_ONLY is the reference decision path and
# the only mode with decision authority.  EXTRACT_ONLY / EXTRACT_THEN_VERIFY are
# experimental and MUST NOT become production eligibility dependencies.
MODE_DISPOSITION = {
    "VERIFIER_ONLY": "REFERENCE",
    "EXTRACT_ONLY": "EXPERIMENTAL_NOT_RECOMMENDED",
    "EXTRACT_THEN_VERIFY": "EXPERIMENTAL_NOT_RECOMMENDED",
}


class EvidenceQueryType(str, Enum):
    VERIFICATION = "VERIFICATION"
    EXTRACTION = "EXTRACTION"
    UNKNOWN = "UNKNOWN"


class ExtractionSlotType(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    REGISTER = "REGISTER"
    VALUE = "VALUE"
    UNIT = "UNIT"
    ATTRIBUTE = "ATTRIBUTE"
    ACTION = "ACTION"
    SETTING = "SETTING"
    TERMINAL = "TERMINAL"
    CHANNEL = "CHANNEL"
    CONDITION = "CONDITION"
    LOCATION = "LOCATION"
    UNKNOWN = "UNKNOWN"


class ExtractionStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


class ExtractionMultiplicity(str, Enum):
    UNIQUE_SUPPORTED = "UNIQUE_SUPPORTED"
    MULTIPLE_SUPPORTED = "MULTIPLE_SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    NONE_SUPPORTED = "NONE_SUPPORTED"


@dataclass(frozen=True)
class QueryTypeRoute:
    query_type: str
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionRequirement:
    target_entity: str
    slot_type: str
    protocols: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    condition: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionEvidenceResult:
    status: str
    multiplicity: str
    slot_type: str
    value: str
    source_candidate_ids: tuple[str, ...]
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceDecision:
    query_type: str
    decision: str
    reason: str
    extraction: ExtractionEvidenceResult | None = None
    evidence: dict = field(default_factory=dict)
    query_type_route: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "decision": self.decision,
            "reason": self.reason,
            "extraction": self.extraction.as_dict() if self.extraction else None,
            "evidence": self.evidence,
            "query_type_route": self.query_type_route,
        }


# ---------------------------------------------------------------------------
# Query-type router (deterministic)
# ---------------------------------------------------------------------------

_OPEN_LEAD_RE = re.compile(
    r"^\s*(?:what|which|where|when|how|why|under\s+what|name|list|give|tell\s+me|identify|specify)\b",
    re.IGNORECASE,
)
_POLAR_LEAD_RE = re.compile(
    r"^\s*(?:does|is|are|do|can|will|should|must|was|were|has|have|would|could|may)\b",
    re.IGNORECASE,
)

_SLOT_PATTERNS: tuple[tuple[ExtractionSlotType, str], ...] = (
    (ExtractionSlotType.REGISTER, r"\b(?:register|parameter|param|modbus\s+register|object|index)\b"),
    (ExtractionSlotType.TERMINAL, r"\b(?:terminal|terminals?|pin|pins?)\b"),
    (ExtractionSlotType.CHANNEL, r"\b(?:channel|channels?|port|ports?)\b"),
    (ExtractionSlotType.CONDITION, r"\bunder\s+what\s+condition|\b(?:when|condition)\b.{0,30}\b(?:can|could|may|allowed|permitted)\b"),
    (ExtractionSlotType.ACTION, r"\b(?:what\s+(?:action|step|command|happens)|what\s+does\b.{0,60}\bdo\b|how\s+(?:to|does)\b|\b(?:action|procedure)\b)"),
    (ExtractionSlotType.ATTRIBUTE, r"\b(?:what\s+does|what\s+do)\b.{0,80}\b(?:indicate|mean|represent|signal|stand\s+for)\b|\bmeaning\s+of\b"),
    (ExtractionSlotType.SETTING, r"\b(?:setting|option|mode|configuration)\b"),
    (ExtractionSlotType.VALUE, r"\b(?:value|reading|level|limit|range|number|count|rating|address|speed|time|voltage|current)\b|\bdefault\b.{0,30}\b(?:value|ip|address|string|mask)\b|\bwhat\s+is\s+the\s+default\b"),
    (ExtractionSlotType.UNIT, r"\b(?:unit|units)\b"),
    (ExtractionSlotType.LOCATION, r"\bwhere\b|\b(?:located|location)\b"),
    (ExtractionSlotType.IDENTIFIER, r"\b(?:error|alarm|message|fault|diagnostic|code|flag|status|state)\b"),
)


def detect_extraction_slot(query: str) -> ExtractionSlotType | None:
    """Return the requested slot type, or None when no open slot is requested."""
    text = normalize_technical_text(query)
    for slot_type, pattern in _SLOT_PATTERNS:
        if re.search(pattern, text):
            return slot_type
    return None


def _slot_requested(query: str) -> bool:
    """Whether the query explicitly asks to fill an unknown slot."""
    text = normalize_technical_text(query)
    return _OPEN_LEAD_RE.match(text) is not None or bool(
        re.search(r"\b(?:which|what|where|how\s+(?:many|much|to)|under\s+what)\b", text, re.IGNORECASE)
    )


def route_query_type(query: str, requirement: TypedEvidenceRequirement | None = None) -> QueryTypeRoute:
    """Route a query to VERIFICATION / EXTRACTION / UNKNOWN.

    The lead word alone is never sufficient: routing combines the question's
    intent form with a detected open slot and (when available) the typed
    requirement so a query such as "under what condition …" or "what does X
    indicate" is recognized as extraction rather than a yes/no check.
    """
    text = normalize_technical_text(query)
    polar = _POLAR_LEAD_RE.match(text) is not None
    slot = detect_extraction_slot(query)
    slot_requested = _slot_requested(query)

    if polar and slot is None and not slot_requested:
        return QueryTypeRoute(EvidenceQueryType.VERIFICATION.value, 0.9, "polar_proposition")
    if slot is None and not slot_requested:
        if requirement is not None:
            kinds = {item.kind for item in requirement.items if item.criticality == "CRITICAL"}
            if kinds <= {"identifier", "value", "attribute", "action", "detail", "protocol", "qualifier"}:
                return QueryTypeRoute(EvidenceQueryType.VERIFICATION.value, 0.5, "typed_verification")
        return QueryTypeRoute(EvidenceQueryType.UNKNOWN.value, 0.0, "no_slot_no_polar")
    if slot_requested and slot is not None:
        return QueryTypeRoute(EvidenceQueryType.EXTRACTION.value, 0.9, f"open_slot:{slot.value}")
    if slot_requested:
        return QueryTypeRoute(EvidenceQueryType.EXTRACTION.value, 0.6, "open_lead_no_slot")
    if slot is not None:
        # Polar form that still names a slot (e.g. "does register A control X")
        # is a completed proposition, not an extraction.
        return QueryTypeRoute(EvidenceQueryType.VERIFICATION.value, 0.55, "polar_with_slot")
    return QueryTypeRoute(EvidenceQueryType.UNKNOWN.value, 0.0, "unroutable")


class EvidenceQueryTypeRouter:
    """Thin DI wrapper around :func:`route_query_type`."""

    def route(self, query: str, requirement: TypedEvidenceRequirement | None = None) -> QueryTypeRoute:
        return route_query_type(query, requirement)


# ---------------------------------------------------------------------------
# Extraction requirement construction
# ---------------------------------------------------------------------------

_SLOT_NOUNS: dict[str, str] = {
    ExtractionSlotType.REGISTER.value: "register",
    ExtractionSlotType.IDENTIFIER.value: "identifier",
    ExtractionSlotType.VALUE.value: "value",
    ExtractionSlotType.UNIT.value: "unit",
    ExtractionSlotType.ATTRIBUTE.value: "meaning",
    ExtractionSlotType.ACTION.value: "action",
    ExtractionSlotType.SETTING.value: "setting",
    ExtractionSlotType.TERMINAL.value: "terminal",
    ExtractionSlotType.CHANNEL.value: "channel",
    ExtractionSlotType.CONDITION.value: "condition",
    ExtractionSlotType.LOCATION.value: "location",
    ExtractionSlotType.UNKNOWN.value: "value",
}

_TARGET_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "for", "to", "is", "are", "does", "do", "what",
    "which", "when", "where", "how", "under", "on", "with", "by", "that", "this",
    "and", "or", "can", "could", "may", "will", "should", "must",
    "register", "parameter", "param", "terminal", "pin", "channel", "port",
    "value", "setting", "option", "mode", "action", "unit", "code", "error",
    "address", "number", "when", "indicate", "mean", "indicates", "means",
    "default", "factory",
})


def _target_tokens(target_entity: str) -> tuple[str, ...]:
    tokens = tuple(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9][a-z0-9.\-/]*", normalize_technical_text(target_entity))
        if token not in _TARGET_STOPWORDS
    ))
    return tokens


def _value_repeats_target(value: str, target_entity: str) -> bool:
    """Whether an extracted object merely repeats the query's own target language."""
    normalized = normalize_technical_text(value)
    target = normalize_technical_text(target_entity)
    if not normalized or not target:
        return False
    return normalized in target or re.search(
        rf"(?<![\w]){re.escape(normalized)}(?![\w])", target, re.IGNORECASE,
    ) is not None


def build_extraction_requirement(query: str, requirement: TypedEvidenceRequirement | None = None) -> ExtractionRequirement:
    """Derive a best-effort extraction requirement from raw query text.

    The target entity is the phrase left after removing the slot question lead;
    callers with an authored target may construct :class:`ExtractionRequirement`
    directly for evaluation fidelity.
    """
    slot = detect_extraction_slot(query) or ExtractionSlotType.UNKNOWN
    text = normalize_technical_text(query)
    target = text
    for pattern in (
        r"^(?:what|which|where|when|how(?:\s+many|\s+much|\s+to)?)\s+",
        r"^(?:under\s+what\s+condition)\s+",
        r"^(?:what\s+(?:is|are)\s+the\s+)",
    ):
        target = re.sub(pattern, "", target, count=1, flags=re.IGNORECASE).strip()
    noun = _SLOT_NOUNS.get(slot.value, "value")
    target = re.sub(rf"\b{re.escape(noun)}\b", "", target, count=1, flags=re.IGNORECASE).strip()
    target = re.sub(r"^(?:the|a|an)\s+", "", target).strip()
    target = target.rstrip("?？. ").strip()
    protocols: tuple[str, ...] = ()
    condition = ""
    constraint_tokens: list[str] = []
    if requirement is not None:
        for item in requirement.items:
            if item.kind == "protocol":
                protocols += (item.value,)
            elif item.kind in ("qualifier", "scope") and item.criticality == "CRITICAL":
                constraint_tokens.append(item.value)
    return ExtractionRequirement(
        target_entity=target,
        slot_type=slot.value,
        protocols=tuple(dict.fromkeys(protocols)),
        constraints=tuple(dict.fromkeys(constraint_tokens)),
        condition=condition,
    )


# ---------------------------------------------------------------------------
# Candidate-local object extraction (deterministic, slot-typed)
# ---------------------------------------------------------------------------

_VALUE_TOKEN_RE = re.compile(
    r"(?<![0-9a-z])(?:\d{1,3}(?:\.\d{1,3}){3}|\d+(?:\.\d+)?|0x[0-9a-f]+)"
    r"(?:\s*(?:%|percent|µj|uj|j|mm|cm|m|ms|s|seconds?|minutes?|"
    r"v|vac|vdc|hz|khz|mhz|kbyte|byte|bits?|ohm|ω|amp|a))?\b",
    re.IGNORECASE,
)
# Terminal/channel labels: "Terminal A1", "pin 5", "Channel 3", "Port 2".
_TERMINAL_RE = re.compile(
    r"(?<![\w])(?:terminals?|pins?|channels?|ports?|ch)\b(?:\s+number)?\s*:?\s*([a-z]{0,2}\d{1,3}|\d{1,3}[a-z]?)(?![\w])",
    re.IGNORECASE,
)
_ACTION_VERBS = (
    "restart", "reset", "restore", "delete", "remove", "configure", "recover",
    "replace", "initialize", "initialise", "commission", "start", "stop",
    "enable", "disable", "select", "set", "reprogram", "press", "hold",
    "write", "cycle", "open", "close", "wait", "switch",
)
_ATTRIBUTE_VERBS = (
    "indicates", "means", "represents", "signal", "indicate", "signals",
    "defined as", "refers to", "corresponds to", "stands for",
)
_BOOLEAN_STATES = ("enabled", "disabled", "on", "off", "active", "inactive")
_SETTING_TOKEN_RE = re.compile(
    r"(?<![\w])(?:whitelist|blacklist|allowlist|blocklist|enabled|disabled|"
    r"read-only|write/read|read|write|public|private|static|dynamic|"
    r"automatic|manual|slave|master|authenticator|supplicant)(?![\w])",
    re.IGNORECASE,
)


def _extract_register_objects(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(?<![\w])0x([0-9a-f]{1,6})(?![\w])", text, re.IGNORECASE):
        values.append("0x" + match.group(1).lower())
    # EtherCAT object indices / subindices: "1A00", "1C00:0", "6072 hex",
    # "index 3021", "3021 – …".  Target-association upstream keeps these local.
    for match in re.finditer(r"(?<![\w])([0-9a-f]{4,5})(?::[0-9a-f]{1,2})?(?![\w])", text, re.IGNORECASE):
        values.append(match.group(0).upper())
    for match in re.finditer(r"(?<![\w])(?:index|object)\s*(?:number\s*)?([0-9a-f]{4,5})", text, re.IGNORECASE):
        values.append(match.group(1).upper())
    for reference in extract_parameter_references(text):
        values.append(reference.identifier.upper())
    values.extend(match.group(0).upper() for match in re.finditer(
        r"(?<![\w])(?:mw\s?\d{1,5}|[faceps]\d{2,5}|4\d{4})(?![\w])", text, re.IGNORECASE))
    return values


def _extract_value_objects(text: str) -> list[str]:
    return [match.group(0).strip() for match in _VALUE_TOKEN_RE.finditer(text)]


def _extract_terminal_objects(text: str) -> list[str]:
    values: list[str] = []
    for match in _TERMINAL_RE.finditer(text):
        label = match.group(1).strip()
        if label:
            values.append(label)
    # Explicit "Terminal A1: ..." (alpha-numeric labels, colon-separated).
    for match in re.finditer(r"(?<![\w])(?:terminal|pin|channel|port)\s+([a-z]{1,2}\d{1,2})(?![\w])", text, re.IGNORECASE):
        values.append(match.group(1))
    return values


def _extract_action_objects(text: str) -> list[str]:
    return [verb for verb in _ACTION_VERBS if re.search(rf"\b{re.escape(verb)}\w*", text, re.IGNORECASE) is not None]


def _extract_attribute_objects(text: str) -> list[str]:
    """Extract the meaning attributed to a subject (text after an attribute verb)."""
    values: list[str] = []
    for verb in _ATTRIBUTE_VERBS:
        for match in re.finditer(rf"\b{re.escape(verb)}\b", text, re.IGNORECASE):
            tail = text[match.end():match.end() + 160]
            tail = re.split(r"[.;\n]", tail, maxsplit=1)[0]
            tail = re.sub(r"^(?:is|as|a|an|the|that|to|by)\s+", "", tail.strip(), flags=re.IGNORECASE).strip()
            if tail:
                values.append(tail)
    return values


def _extract_setting_objects(text: str) -> list[str]:
    values = [match.group(0) for match in _SETTING_TOKEN_RE.finditer(text)]
    values.extend(state for state in _BOOLEAN_STATES if re.search(rf"\b{state}\b", text, re.IGNORECASE))
    return values


def _extract_condition_objects(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\b(?:if|when|only when|if and only if|in a)\s+(.{0,120}?)(?=,|\.|;|\n|$)", text, re.IGNORECASE):
        clause = match.group(1).strip()
        if clause:
            values.append(clause)
    return values


_EXTRACTORS = {
    ExtractionSlotType.REGISTER.value: _extract_register_objects,
    ExtractionSlotType.IDENTIFIER.value: _extract_register_objects,
    ExtractionSlotType.VALUE.value: _extract_value_objects,
    ExtractionSlotType.UNIT.value: _extract_value_objects,
    ExtractionSlotType.TERMINAL.value: _extract_terminal_objects,
    ExtractionSlotType.CHANNEL.value: _extract_terminal_objects,
    ExtractionSlotType.ACTION.value: _extract_action_objects,
    ExtractionSlotType.ATTRIBUTE.value: _extract_attribute_objects,
    ExtractionSlotType.SETTING.value: _extract_setting_objects,
    ExtractionSlotType.CONDITION.value: _extract_condition_objects,
    ExtractionSlotType.LOCATION.value: _extract_value_objects,
    ExtractionSlotType.UNKNOWN.value: _extract_value_objects,
}


def extract_candidate_objects(text: str, slot_type: str) -> list[str]:
    extractor = _EXTRACTORS.get(slot_type, _extract_value_objects)
    return extractor(text or "")


# ---------------------------------------------------------------------------
# Locality association
# ---------------------------------------------------------------------------

SAME_SEGMENT = 3
ADJACENT_SEGMENT = 2
SAME_SUBSECTION = 1
NO_LOCALITY = 0


def _target_present(target_entity: str, window: str) -> bool:
    if not target_entity:
        return True
    normalized_target = normalize_technical_text(target_entity)
    normalized_window = normalize_technical_text(window)
    if normalized_target and normalized_target in normalized_window:
        return True
    tokens = _target_tokens(target_entity)
    if not tokens:
        return True

    def mentioned(token: str) -> bool:
        return re.search(
            rf"(?<![\w]){re.escape(token)}(?![\w])", normalized_window, re.IGNORECASE,
        ) is not None

    # Short tokens (letters/digits like "B" or "A1") are load-bearing and must be
    # present verbatim; longer tokens only need a majority so synonym/detail gaps
    # do not over-abstain.
    short = [token for token in tokens if len(token) <= 2]
    long = [token for token in tokens if len(token) > 2]
    if short and not all(mentioned(token) for token in short):
        return False
    if not long:
        return True
    long_matches = sum(1 for token in long if mentioned(token))
    return long_matches >= (len(long) + 1) // 2


def _candidate_extractions(
    claim: CandidateClaim, slot_type: str, target_entity: str,
) -> list[tuple[str, int]]:
    """Return ``(value, locality_rank)`` items for one candidate.

    Locality strengthens from the whole chunk down to the single line that
    co-occurs with the target, so a register mentioned elsewhere in a long page
    is never silently attributed to the requested target.
    """
    segments = list(claim.segments) or ([claim.text] if claim.text else [])
    found: list[tuple[str, int]] = []
    for index, segment in enumerate(segments):
        for value in extract_candidate_objects(segment, slot_type):
            if _value_repeats_target(value, target_entity):
                continue
            if _target_present(target_entity, segment):
                found.append((value, SAME_SEGMENT))
                continue
            window = " ".join(segments[max(0, index - 1):index + 2])
            if _target_present(target_entity, window):
                found.append((value, ADJACENT_SEGMENT))
                continue
            subsection_evidence = normalize_technical_text(" ".join((claim.subsection, claim.text)))
            if claim.subsection and _target_present(target_entity, subsection_evidence):
                found.append((value, SAME_SUBSECTION))
    if not found and _target_present(target_entity, claim.text):
        for value in extract_candidate_objects(claim.text, slot_type):
            if _value_repeats_target(value, target_entity):
                continue
            found.append((value, SAME_SUBSECTION))
    return found


def _normalize_object(value: str) -> str:
    return normalize_technical_text(value).strip()


def extract_slot_value(
    query: str,
    candidates: list,
    documents: list,
    analysis: Any,
    *,
    extraction_requirement: ExtractionRequirement | None = None,
) -> ExtractionEvidenceResult:
    """Deterministically extract the unique target-associated slot value, or abstain.

    Returns SUPPORTED only for a single distinct value (UNIQUE_SUPPORTED); a value
    repeated across several candidate chunks still counts as one value.  Multiple
    distinct values are disambiguated only when one clearly wins on locality.
    """
    requirement = extraction_requirement or build_extraction_requirement(query)
    slot_type = requirement.slot_type
    target_entity = requirement.target_entity
    claims = tuple(extract_candidate_claim(candidate, _empty_requirement(target_entity)) for candidate in candidates)

    scored: dict[str, tuple[int, int, list[str]]] = {}
    for claim, candidate in zip(claims, candidates):
        chunk_id = str(candidate.chunk_id or "")
        for value, rank in _candidate_extractions(claim, slot_type, target_entity):
            if not value.strip():
                continue
            key = _normalize_object(value)
            best_rank, count, sources = scored.get(key, (NO_LOCALITY, 0, []))
            scored[key] = (max(best_rank, rank), count + 1, sources + [chunk_id] if chunk_id not in sources else sources)

    if not scored:
        return ExtractionEvidenceResult(
            ExtractionStatus.INSUFFICIENT.value,
            ExtractionMultiplicity.NONE_SUPPORTED.value,
            slot_type,
            "",
            (),
            0.0,
            "EXTRACTION_NONE_SUPPORTED",
        )

    distinct = list(scored.items())
    if len(distinct) == 1:
        key, (rank, count, sources) = distinct[0]
        return ExtractionEvidenceResult(
            ExtractionStatus.SUPPORTED.value,
            ExtractionMultiplicity.UNIQUE_SUPPORTED.value,
            slot_type,
            key,
            tuple(sources),
            _confidence(rank, count),
            "EXTRACTION_UNIQUE_SUPPORTED",
        )

    # Multiple distinct values: prefer the one with the strongest locality; ties are ambiguous.
    ordered = sorted(distinct, key=lambda item: (item[1][0], item[1][1]), reverse=True)
    (best_key, (best_rank, best_count, best_sources)), rest = ordered[0], ordered[1:]
    second_rank, second_count = rest[0][1][0], rest[0][1][1]
    if (best_rank, best_count) > (second_rank, second_count):
        return ExtractionEvidenceResult(
            ExtractionStatus.SUPPORTED.value,
            ExtractionMultiplicity.UNIQUE_SUPPORTED.value,
            slot_type,
            best_key,
            tuple(best_sources),
            _confidence(best_rank, best_count) * 0.85,
            "EXTRACTION_DISAMBIGUATED_BY_LOCALITY",
        )
    return ExtractionEvidenceResult(
        ExtractionStatus.AMBIGUOUS.value,
        ExtractionMultiplicity.AMBIGUOUS.value,
        slot_type,
        "",
        (),
        0.0,
        "EXTRACTION_AMBIGUOUS",
    )


def _empty_requirement(target_entity: str) -> TypedEvidenceRequirement:
    """Minimal requirement so candidate claims attach without typed matching."""
    from .evidence_contract import TypedRequirementItem

    return TypedEvidenceRequirement(
        identity={},
        intent="extraction",
        specificity="open",
        location=False,
        items=(TypedRequirementItem("target", target_entity, "OPTIONAL", "OPTIONAL"),),
    )


def _confidence(rank: int, count: int) -> float:
    base = {NO_LOCALITY: 0.2, SAME_SUBSECTION: 0.55, ADJACENT_SEGMENT: 0.75, SAME_SEGMENT: 0.9}[rank]
    if count > 1:
        base = min(1.0, base + 0.05)
    return round(base, 4)


# ---------------------------------------------------------------------------
# Extraction -> ANSWER mapping
# ---------------------------------------------------------------------------

def extraction_decision(result: ExtractionEvidenceResult) -> tuple[str, str]:
    if result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value:
        return Decision.ANSWER.value, "EXTRACTION_UNIQUE_SUPPORTED"
    if result.multiplicity == ExtractionMultiplicity.NONE_SUPPORTED.value:
        return Decision.ABSTAIN.value, "EXTRACTION_NONE_SUPPORTED"
    return Decision.ABSTAIN.value, "EXTRACTION_AMBIGUOUS"


# ---------------------------------------------------------------------------
# Extract-then-verify (optional; never default)
# ---------------------------------------------------------------------------

def build_extraction_proposition(target_entity: str, slot_type: str, value: str) -> str:
    """Build the directional proposition to verify a candidate slot value.

    The template encodes the *direction* (value belongs to target) so a reversed
    role is CONTRADICTED rather than ENTAILED.
    """
    noun = _SLOT_NOUNS.get(slot_type, "value")
    if slot_type in (ExtractionSlotType.REGISTER.value, ExtractionSlotType.IDENTIFIER.value):
        return f"{value} is the {noun} for {target_entity}"
    if slot_type in (ExtractionSlotType.TERMINAL.value, ExtractionSlotType.CHANNEL.value):
        return f"{value} is the {noun} for {target_entity}"
    if slot_type == ExtractionSlotType.ATTRIBUTE.value:
        return f"{target_entity} indicates {value}"
    if slot_type == ExtractionSlotType.ACTION.value:
        return f"the action for {target_entity} is to {value}"
    return f"the {noun} for {target_entity} is {value}"


SlotVerifier = Any  # any callable -> (decision, confidence)


def verify_extracted_proposition(
    premise: str,
    proposition: str,
    judge: LocalNliJudge,
) -> tuple[str, float]:
    """Return (decision, confidence) for an extracted->propositional NLI check."""
    if judge is None or judge.model is None:
        return JudgeDecision.UNKNOWN.value, 0.0
    probs = judge.predict_probs(premise, proposition)
    decision, confidence = judge.decide_from_probs(probs)
    return decision, confidence


# ---------------------------------------------------------------------------
# Unified decision entry point
# ---------------------------------------------------------------------------

def analyze_querytype_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    mode: str = "EXTRACT_ONLY",
    judge: LocalNliJudge | None = None,
    extraction_requirement: ExtractionRequirement | None = None,
    policy: Any = None,
    identity_matching: bool = True,
) -> EvidenceDecision:
    """Unified query-type-aware evidence decision.

    Modes
    -----
    * ``VERIFIER_ONLY``  — the frozen V3.25 rule + selective NLI path (baseline).
    * ``EXTRACT_ONLY``   — extraction queries take the deterministic path, others the rule+NLI.
    * ``EXTRACT_THEN_VERIFY`` — extraction queries additionally verify the extracted proposition.
    """
    analysis = getattr(result, "query_analysis", None) or _analysis_of(query, documents)
    requirement = build_typed_requirement(query, documents, analysis)
    route = route_query_type(query, requirement)
    rule = analyze_retrieval_evidence(query, result, documents, retrieval_mode, policy=policy, identity_matching=identity_matching)

    # Verifier-only baseline: always the frozen V3.25 path (query type reported as routed).
    if mode == "VERIFIER_ONLY":
        final_decision, reason = _apply_verification_judge(query, rule.decision, requirement, result, judge, policy)
        return EvidenceDecision(
            route.query_type,
            final_decision,
            reason,
            evidence=rule.as_dict(),
            query_type_route=route.as_dict(),
        )

    if route.query_type == EvidenceQueryType.EXTRACTION.value:
        extracted = extract_slot_value(query, list(getattr(result, "candidates", []) or []), documents, analysis, extraction_requirement=extraction_requirement)
        decision, reason = extraction_decision(extracted)
        if mode == "EXTRACT_THEN_VERIFY" and decision == Decision.ANSWER.value and judge is not None and judge.model is not None:
            candidates = list(getattr(result, "candidates", []) or [])
            requirement_for_extract = extraction_requirement or build_extraction_requirement(query)
            premise = build_premise([
                extract_candidate_claim(candidate, _empty_requirement(requirement_for_extract.target_entity))
                for candidate in candidates
            ])
            proposition = build_extraction_proposition(
                requirement_for_extract.target_entity, extracted.slot_type, extracted.value,
            )
            judge_decision, _confidence = verify_extracted_proposition(premise, proposition, judge)
            if judge_decision == JudgeDecision.CONTRADICTS.value:
                # Only a relation reversal vetoes the grounded extraction.
                decision, reason = Decision.ABSTAIN.value, "EXTRACT_THEN_VERIFY_CONTRADICTION"
            elif judge_decision == JudgeDecision.ENTAILS.value:
                decision, reason = Decision.ANSWER.value, "EXTRACT_THEN_VERIFY_ENTAILS"
            else:
                # Neutral / unknown NLI is not a veto: keep the grounded extraction.
                decision, reason = Decision.ANSWER.value, "EXTRACT_THEN_VERIFY_NEUTRAL_FALLBACK"
        return EvidenceDecision(
            route.query_type,
            decision,
            reason,
            extraction=extracted,
            evidence=rule.as_dict(),
            query_type_route=route.as_dict(),
        )

    # VERIFICATION / UNKNOWN -> frozen rule + selective NLI path.
    final_decision, reason = _apply_verification_judge(query, rule.decision, requirement, result, judge, policy)
    return EvidenceDecision(
        route.query_type,
        final_decision,
        reason,
        evidence=rule.as_dict(),
        query_type_route=route.as_dict(),
    )


def _analysis_of(query: str, documents: list) -> Any:
    from .filters import analyze_query
    return analyze_query(query, documents)


def _apply_verification_judge(query: str, rule_decision: str, requirement: TypedEvidenceRequirement, result, judge: LocalNliJudge | None, policy: Any) -> tuple[str, str]:
    """Preserve the V3.25 selective-NLI behavior for the verification path."""
    ambiguity = route_ambiguity(query, requirement)
    if ambiguity is None or judge is None or judge.model is None:
        return rule_decision, "RULE"
    claims = [extract_candidate_claim(candidate, requirement) for candidate in list(getattr(result, "candidates", []) or [])]
    premise = build_premise(claims)
    hypothesis = build_hypothesis(query, requirement)
    probs = judge.predict_probs(premise, hypothesis)
    judge_decision, _ = judge.decide_from_probs(probs)
    if judge_decision == JudgeDecision.ENTAILS.value:
        return rule_decision, "NLI_ENTAILS"
    if judge_decision in (JudgeDecision.CONTRADICTS.value, JudgeDecision.INSUFFICIENT.value):
        return Decision.ABSTAIN.value, "NLI_" + judge_decision
    return rule_decision, "NLI_UNKNOWN_FALLBACK"