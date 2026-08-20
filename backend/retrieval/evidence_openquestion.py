"""Open-question evidence sufficiency (V3.30).

V3.29 fixed the responsibility boundary: Evidence owns *sufficiency* (answerable /
abstain) and must NOT own final answer-object extraction.  This module studies the
open-question half of that boundary: for ``which / what / where / how`` queries,
can Evidence prove that an answer-bearing *relation* exists in the approved
candidate (target + requested relation + some locally-bound object) without
committing to the object's final wording ("Functional Grounding", "a diode",
"DHCP", "Register 247", "cycle power")?

Two distinct concerns are kept apart:

* **Existential sufficiency** — ``relation supported -> sufficient``. Belongs to
  Evidence.
* **Answer extraction** — the final textual/structured value. Belongs downstream;
  its failure never gates eligibility here.

Deterministic only: ``GENERATION_USED == "NO"``.  Grounding normalization has
``GROUNDING_DECISION_AUTHORITY = "NONE"`` (V3.29 contract unchanged).  The frozen
V3.25 rule + selective NLI verifier is imported unchanged for polar verification
and for the hard (identity / scope / safety) abstentions.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .evidence import Decision
from .evidence_contract import extract_candidate_claim
from .evidence_querytype import (
    EvidenceQueryType,
    ExtractionSlotType,
    SAME_SEGMENT,
    _empty_requirement,
    _value_repeats_target,
    analyze_querytype_evidence,
    build_extraction_requirement,
    detect_extraction_slot,
    extract_candidate_objects,
    normalize_technical_text,
)

# Candidate identity.
OPEN_SUFFICIENCY_CANDIDATE_VERSION = "evidence-v330-open-sufficiency-candidate"
OPEN_SUFFICIENCY_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
OPEN_QUESTION_SUFFICIENCY_DEFAULT = "OFF"
GENERATION_USED = "NO"
# V3.29 boundary invariants (re-asserted, never modified here).
GROUNDING_DECISION_AUTHORITY = "NONE"
NORMALIZATION_DECISION_AUTHORITY = "NONE"


class RelationType(str, Enum):
    HAS_IDENTIFIER = "HAS_IDENTIFIER"
    HAS_VALUE = "HAS_VALUE"
    HAS_DEFAULT_VALUE = "HAS_DEFAULT_VALUE"
    HAS_RANGE = "HAS_RANGE"
    HAS_SETTING = "HAS_SETTING"
    USES_TERMINAL = "USES_TERMINAL"
    USES_CHANNEL = "USES_CHANNEL"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    REQUIRES_ACTION = "REQUIRES_ACTION"
    LOCATED_AT = "LOCATED_AT"
    USES_PROTOCOL = "USES_PROTOCOL"
    HAS_PROCEDURE = "HAS_PROCEDURE"
    UNKNOWN = "UNKNOWN"


class ExistentialObject(str, Enum):
    EXISTS_VALUE = "EXISTS_VALUE"
    EXISTS_IDENTIFIER = "EXISTS_IDENTIFIER"
    EXISTS_ACTION = "EXISTS_ACTION"
    EXISTS_NOUN = "EXISTS_NOUN"
    UNKNOWN_OBJECT = "UNKNOWN_OBJECT"


class OpenSufficiencyStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class OpenQuestionRequirement:
    """Requested relation for an open question.

    ``requested_slot_type`` is the *unknown object type*, not a critical value
    that must already match in the query.  A missing VALUE in the query is not a
    missing-evidence requirement (§10).
    """
    target: str
    relation: str = RelationType.UNKNOWN.value
    requested_slot_type: str = ExtractionSlotType.UNKNOWN.value
    attribute: str = ""
    action: str = ""
    identifier: str = ""
    protocol: str = ""
    condition: str = ""
    qualifiers: tuple[str, ...] = ()
    scope_constraints: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRelationClaim:
    """A candidate-local relation claim (existential, not a final answer object)."""
    subject: str
    relation: str
    object_type: str
    object_present: bool
    scope: str
    candidate_id: str
    association: str = ""


@dataclass(frozen=True)
class OpenQuestionSufficiencyResult:
    status: str
    relation: str
    requested_slot: str
    supporting_candidate_ids: tuple[str, ...]
    claims: tuple[CandidateRelationClaim, ...]
    reason: str

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "relation": self.relation,
            "requested_slot": self.requested_slot,
            "supporting_candidate_ids": list(self.supporting_candidate_ids),
            "claims": [asdict(claim) for claim in self.claims],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OpenQuestionEvidenceDecision:
    query: str
    query_type: str
    decision: str
    reason: str
    open_sufficiency: dict | None = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "decision": self.decision,
            "reason": self.reason,
            "open_sufficiency": self.open_sufficiency,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Relation detection
# ---------------------------------------------------------------------------

_RELATION_VERBS: dict[str, tuple[str, ...]] = {
    RelationType.HAS_DEFAULT_VALUE.value: (
        "default", "factory default", "factory", "initial value", "initial",
        "reset value", "restore value", "delivery state", "preset", "out-of-the",
    ),
    RelationType.HAS_RANGE.value: (
        "maximum", "maximum value", "max", "minimum", "min", "range",
        "upper limit", "lower limit", "upper", "lower", "limit",
    ),
    RelationType.HAS_VALUE.value: (
        "value", "reading", "level", "length", "time", "delay", "voltage",
        "current", "count", "number", "rated", "=", ":",
    ),
    RelationType.HAS_IDENTIFIER.value: (
        "register", "index", "object", "address", "code", "byte", "identifier",
        "parameter", "id", "number",
    ),
    RelationType.USES_TERMINAL.value: (
        "terminal", "pin", "contact", "connector", "wire", "color", "colour",
    ),
    RelationType.USES_CHANNEL.value: ("channel", "port", "ports"),
    RelationType.HAS_ATTRIBUTE.value: (
        "indicate", "indicates", "mean", "means", "represent", "state", "status",
        "color", "signal", "led",
    ),
    RelationType.REQUIRES_ACTION.value: (
        "action", "step", "procedure", "press", "cycle", "replace", "connect",
        "install", "component", "diode", "must", "should",
    ),
    RelationType.LOCATED_AT.value: ("located", "location", "mounted", "position", "on", "in"),
    RelationType.HAS_SETTING.value: (
        "setting", "option", "mode", "configuration", "parameter", "state",
        "enabled", "disabled",
    ),
    RelationType.USES_PROTOCOL.value: (
        "protocol", "tcp", "udp", "ethernet", "modbus", "ethercat", "dnp3",
    ),
    RelationType.HAS_PROCEDURE.value: ("procedure", "procedure for", "step"),
}

_DEFAULT_RE = re.compile(r"\bdefault\b|\bfactory\b|\binitial\b|\breset value\b|\bpreset\b|\bdelivery state\b")
_RANGE_RE = re.compile(r"\b(?:maximum|max|minimum|min|range|upper limit|lower limit|limit)\b")
_COUNT_RE = re.compile(r"\bhow many\b|\bnumber of\b|\bhow many\b|\bcount\b|\bhow much\b")
_COMPONENT_RE = re.compile(r"\b(?:component|part|countermeasure|absorber|diode|resistor|surge)\b")

# A sub-module / device identifier (e.g. "di581-s", "nx1p2", "ep1111-0000") that
# proves an open query names hardware documented inside a parent manual.  The V3.25
# model regex sometimes lifts a non-identifier phrase ("start-up", "before 2007",
# "safe-operational") into the equipment slot; those are NOT treated as a
# sub-module and are never used to relax a MODEL_MISMATCH.
_SUB_MODULE_RE = re.compile(r"(?<![a-z0-9])[a-z]{1,12}-?\d{1,4}[a-z0-9-]*", re.IGNORECASE)


def _sub_module_identifiers(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        match.group(0).rstrip("-")
        for match in _SUB_MODULE_RE.finditer(normalize_technical_text(query))
    ))


def detect_relation(query: str, slot_type: str = "") -> str:
    """Classify the requested relation from query wording (default wins over range
    wins over count, then the slot maps to a relation)."""
    text = normalize_technical_text(query)
    if _DEFAULT_RE.search(text):
        return RelationType.HAS_DEFAULT_VALUE.value
    if _RANGE_RE.search(text):
        return RelationType.HAS_RANGE.value
    slot = slot_type or (
        detect_extraction_slot(query).value if detect_extraction_slot(query) else ExtractionSlotType.UNKNOWN.value
    )
    if _COUNT_RE.search(text):
        return RelationType.HAS_VALUE.value
    if _COMPONENT_RE.search(text):
        return RelationType.REQUIRES_ACTION.value
    return {
        ExtractionSlotType.REGISTER.value: RelationType.HAS_IDENTIFIER.value,
        ExtractionSlotType.IDENTIFIER.value: RelationType.HAS_IDENTIFIER.value,
        ExtractionSlotType.TERMINAL.value: RelationType.USES_TERMINAL.value,
        ExtractionSlotType.CHANNEL.value: RelationType.USES_CHANNEL.value,
        ExtractionSlotType.ACTION.value: RelationType.REQUIRES_ACTION.value,
        ExtractionSlotType.ATTRIBUTE.value: RelationType.HAS_ATTRIBUTE.value,
        ExtractionSlotType.LOCATION.value: RelationType.LOCATED_AT.value,
        ExtractionSlotType.CONDITION.value: RelationType.HAS_ATTRIBUTE.value,
        ExtractionSlotType.SETTING.value: RelationType.HAS_SETTING.value,
        ExtractionSlotType.VALUE.value: RelationType.HAS_VALUE.value,
        ExtractionSlotType.UNIT.value: RelationType.HAS_VALUE.value,
    }.get(slot, RelationType.UNKNOWN.value)


def build_open_requirement(query: str) -> OpenQuestionRequirement:
    slot = detect_extraction_slot(query)
    slot_type = slot.value if slot else ExtractionSlotType.UNKNOWN.value
    target = build_extraction_requirement(query).target_entity
    relation = detect_relation(query, slot_type)
    return OpenQuestionRequirement(
        target=target, relation=relation, requested_slot_type=slot_type,
        qualifiers=(), scope_constraints=(),
    )


# ---------------------------------------------------------------------------
# Strict target matching (near-miss discrimination)
# ---------------------------------------------------------------------------

_OPEN_TARGET_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "for", "to", "is", "are", "does", "do", "what",
    "which", "where", "how", "when", "with", "by", "on", "that", "this", "and",
    "or", "can", "could", "may", "will", "should", "must", "be", "been", "being",
    "value", "setting", "option", "mode", "parameter", "register", "terminal",
    "channel", "port", "code", "byte", "word", "error", "address", "number",
    "function", "name", "meaning", "type", "id", "identifier", "index", "object",
    "hex", "decimal", "factory", "default", "state", "level", "status", "unit",
    "time", "range", "count", "what's", "there", "its", "has", "have", "also",
    # question/auxiliary verbs extracted as target by the slot parser
    "did", "was", "were", "has", "had", "might", "would", "shall", "if", "you",
    "your", "from", "into", "out", "during", "near", "against",
    "via", "enter", "entered", "use", "used", "using", "report", "reported", "reports",
    "hold", "holds", "holding", "perform", "performs", "performed", "happen",
    "happens", "happened", "changed", "change", "changes", "done", "contains",
    "contain", "contained", "deletes", "delete", "deleted", "marks", "mark",
    "marked", "assigned", "assign", "provide", "provides", "provided", "indicate",
    "indicates", "indicated", "detected", "detects", "detect", "take", "takes",
    "took", "fail", "fails", "failed", "connected", "connect", "absorb", "screwed",
    "screw", "tightened", "tighten", "belongs", "belong", "specified", "specify",
    "describes", "describe", "shows", "show", "showing", "means", "represent",
    "represents", "stand", "stands", "based", "related", "rated", "configured",
    "configure", "configuration", "module", "modules", "device", "devices",
    "component", "part",
})


def _target_tokens_strict(target: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*", normalize_technical_text(target))
        if token and token not in _OPEN_TARGET_STOPWORDS
    ))


def _word_forms(token: str) -> frozenset[str]:
    forms = {token, token + "s", token + "es", token + "ed", token + "d", token + "ing"}
    if token.endswith("e"):
        stem = token[:-1]
        forms |= {stem + "es", stem + "ed", stem + "d", stem + "ing"}
    if token.endswith("y"):
        stem = token[:-1]
        forms |= {stem + "ies", stem + "ied"}
    return frozenset(forms)


def _target_present_strict(target: str, window: str) -> bool:
    normalized_target = normalize_technical_text(target)
    normalized_window = normalize_technical_text(window)
    if not normalized_target:
        return True
    if normalized_target in normalized_window:
        return True
    tokens = _target_tokens_strict(target)
    if not tokens:
        return True

    def mentioned(token: str) -> bool:
        for form in _word_forms(token):
            if re.search(rf"(?<![\w]){re.escape(form)}(?![\w])", normalized_window, re.IGNORECASE):
                return True
        return False

    return all(mentioned(token) for token in tokens)


def _relation_verb_present(relation: str, window: str) -> bool:
    signals = _RELATION_VERBS.get(relation)
    if not signals:
        return True
    normalized_window = normalize_technical_text(window)
    return any(signal in normalized_window for signal in signals)


# ---------------------------------------------------------------------------
# Relation existence discovery
# ---------------------------------------------------------------------------

def _discover_relation_objects(query, candidates, requirement: OpenQuestionRequirement):
    """Discover relation-supported answer-bearing spans (existential objects).

    The *target* (entity/subject) may span the whole chunk; the *relation verb*
    and its locally-bound object must co-occur in a segment.  Returns
    ``(spans, any_target, any_relation)`` bearing
    ``(candidate_id, surface, rank, index)`` per span.
    """
    spans: list[tuple[str, str, int, int]] = []
    any_target = False
    any_relation = False
    for candidate in candidates:
        chunk_id = str(getattr(candidate, "chunk_id", "") or "")
        claim = extract_candidate_claim(candidate, _empty_requirement(requirement.target))
        whole = claim.text or ""
        target_ok = _target_present_strict(requirement.target, whole)
        if target_ok:
            any_target = True
        if not target_ok:
            continue
        segments = list(claim.segments) or ([whole] if whole else [])
        for index, segment in enumerate(segments):
            relation_ok = _relation_verb_present(requirement.relation, segment)
            if relation_ok:
                any_relation = True
            if not relation_ok:
                continue
            objects = [
                normalize_technical_text(value)
                for value in extract_candidate_objects(segment, requirement.requested_slot_type)
                if not _value_repeats_target(value, requirement.target)
            ]
            objects = [value for value in dict.fromkeys(objects) if value]
            if objects:
                for value in objects:
                    spans.append((chunk_id, value, SAME_SEGMENT, index))
            else:
                region = _answer_region_for(segment, requirement.target)
                if region and not _value_repeats_target(region, requirement.target):
                    spans.append((chunk_id, normalize_technical_text(region), SAME_SEGMENT, index))
    return spans, any_target, any_relation


def _answer_region_for(segment: str, target_entity: str) -> str:
    for marker in (":", "=", "->", "\u2013", "-"):
        idx = segment.rfind(marker)
        if idx >= 0 and idx + 1 < len(segment):
            tail = segment[idx + 1:].strip()
            if tail:
                return tail
    target = normalize_technical_text(target_entity)
    if target and segment.casefold().startswith(target):
        return segment[len(target):].strip(" :=\u2013-")
    return segment


# ---------------------------------------------------------------------------
# Sufficiency decision
# ---------------------------------------------------------------------------

def analyze_open_sufficiency(
    query: str,
    candidates,
    *,
    requirement: OpenQuestionRequirement | None = None,
) -> OpenQuestionSufficiencyResult:
    """Existential relation sufficiency for an open question.

    SUPPORTED  — target present + requested relation verb present + >=1 locally
                 bound object (the object is an *unknown placeholder*; its final
                 wording is downstream's concern, never Evidence eligibility).
    AMBIGUOUS  — relation exists but multiple *distinct* objects conflict (kept as
                 a diagnostic status; the decision layer may still consult the
                 Evidence safety contract).
    INSUFFICIENT — target absent, relation verb absent, or no object.
    """
    requirement = requirement or build_open_requirement(query)
    candidate_list = list(candidates)
    spans, any_target, any_relation = _discover_relation_objects(query, candidate_list, requirement)

    claims: list[CandidateRelationClaim] = []
    for candidate in candidate_list:
        chunk_id = str(getattr(candidate, "chunk_id", "") or "")
        present = any(s[0] == chunk_id for s in spans)
        claims.append(CandidateRelationClaim(
            subject=requirement.target, relation=requirement.relation,
            object_type=ExistentialObject.UNKNOWN_OBJECT.value,
            object_present=present, scope="CANDIDATE", candidate_id=chunk_id,
        ))

    if not spans:
        reason = "TARGET_NOT_PRESENT" if not any_target else (
            "RELATION_UNSUPPORTED" if not any_relation else "NO_OBJECT"
        )
        return OpenQuestionSufficiencyResult(
            OpenSufficiencyStatus.INSUFFICIENT.value, requirement.relation,
            requirement.requested_slot_type, (), tuple(claims), reason,
        )

    supporting = tuple(dict.fromkeys(chunk_id for chunk_id, _, _, _ in spans))
    return OpenQuestionSufficiencyResult(
        OpenSufficiencyStatus.SUPPORTED.value, requirement.relation,
        requirement.requested_slot_type, supporting, tuple(claims), "RELATION_SUPPORTED",
    )


# ---------------------------------------------------------------------------
# Combined decision (rule + selective NLI for hard/polar, open sufficiency for
# soft open-question abstentions)
# ---------------------------------------------------------------------------

# Abstention reasons the open-question path may *re-examine* via relation-existence
# sufficiency.  A relaxed reason never directly becomes ANSWER; it only lets the
# relation check decide.  Hard safety/identity reasons stay guarded and are never
# relaxed: NO_CANDIDATE, UNKNOWN_IDENTIFIER, CROSS_EQUIPMENT, UNKNOWN_PARAMETER,
# UNSUPPORTED_PROCEDURE (credential/security bypass).
_RELAXABLE_REASONS = frozenset({
    "MISSING_VALUE_EVIDENCE",
    "MISSING_ATTRIBUTE_EVIDENCE",
    "MISSING_ACTION_EVIDENCE",
    "MISSING_REQUIREMENT_EVIDENCE",
    "PARTIAL_EVIDENCE_ONLY",
    "INSUFFICIENT_EVIDENCE",
    "PROTOCOL_MISMATCH",
    # Open questions name a sub-module / parameter (e.g. "DI581-S", "AI581-S")
    # that the V3.25 model regex lifted into the equipment slot; the parent-document
    # candidate (e.g. "AC500-S") then looks like a mismatch.  For open questions the
    # relation check re-proves the target is locally present, which is the real
    # cross-equipment guard.
    "MODEL_MISMATCH",
})

# Preserve the V3.29 boundary: grounding/normalization never move the decision.


def analyze_open_question_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    judge=None,
    policy=None,
    identity_matching: bool = True,
    requirement: OpenQuestionRequirement | None = None,
    apply_open_sufficiency: bool = True,
) -> OpenQuestionEvidenceDecision:
    """Evidence decision for an open question.

    The frozen V3.25 rule + selective NLI decides polar/hard cases verbatim.
    Open-question sufficiency only *relaxes* a soft missing-detail abstention into
    ANSWER when the answer-bearing relation is supported; it never overrides a hard
    abstention (identity/scope/safety) and never downgrades an ANSWER.
    """
    base = analyze_querytype_evidence(
        query, result, documents, retrieval_mode,
        mode="VERIFIER_ONLY", judge=judge, policy=policy, identity_matching=identity_matching,
    )
    decision, reason = base.decision, base.reason
    open_result: dict | None = None

    rule_reason = (base.evidence or {}).get("reason", "")
    relaxable = (
        apply_open_sufficiency
        and base.query_type == EvidenceQueryType.EXTRACTION.value
        and base.decision == Decision.ABSTAIN.value
        and rule_reason in _RELAXABLE_REASONS
    )
    if relaxable and rule_reason == "MODEL_MISMATCH":
        # Only relax a model mismatch when the open query names a sub-module
        # identifier that is literally documented in the retrieved text.  A
        # non-identifier phrase lifted by the model regex is not sub-module
        # evidence and keeps the identity abstention.
        candidate_text = normalize_technical_text(
            " ".join(
                str(getattr(getattr(candidate, "document", None), "page_content", "") or "")
                for candidate in list(getattr(result, "candidates", []) or [])
            )
        )
        identifiers = _sub_module_identifiers(query)
        relaxable = any(identifier in candidate_text for identifier in identifiers)
    if relaxable:
        suff = analyze_open_sufficiency(
            query, list(getattr(result, "candidates", []) or []), requirement=requirement,
        )
        open_result = suff.as_dict()
        if suff.status == OpenSufficiencyStatus.SUPPORTED.value:
            decision, reason = Decision.ANSWER.value, "OPEN_RELATION_SUPPORTED"
        # AMBIGUOUS / INSUFFICIENT keep the frozen ABSTAIN (safety contract).

    return OpenQuestionEvidenceDecision(
        query=query, query_type=base.query_type, decision=decision, reason=reason,
        open_sufficiency=open_result, evidence=base.evidence,
    )