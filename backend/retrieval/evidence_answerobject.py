"""Grounded Answer Span & Evidence Sufficiency (V3.28).

V3.27's hard extraction gate answered an open question *only* when a single
canonical slot token (a register, a number, a terminal label) could be
extracted; otherwise it abstained.  The V3.27 development run showed the
failure was not grounding but *answer shape*: many legitimate answers are
natural-language objects (``"Functional Grounding"``, ``"a diode"``,
``"DHCP (dynamic IP address)"``) that never map to a single token.

This module separates two things that V3.27 collapsed:

* **Evidence sufficiency** — ``answerable / abstain``.  Sufficiency is decided
  by *target recognition + a locally-bound answer-bearing surface span + scope +
  conflict resolution*, never by whether a canonical token was produced.
* **Answer normalization** — an *optional, second-stage* reduction of a
  grounded surface span to a canonical token.  A normalization failure does
  **not** imply evidence is insufficient.

Deterministic only: ``LLM_EXTRACTION_USED == "NO"``.  The frozen V3.25 rule +
selective NLI verifier is imported unchanged; it is never modified here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from .evidence import Decision, analyze_retrieval_evidence
from .evidence_contract import CandidateClaim, build_typed_requirement, extract_candidate_claim
from .evidence_querytype import (
    EvidenceQueryType,
    ExtractionRequirement,
    build_extraction_requirement,
    detect_extraction_slot,
    extract_candidate_objects,
    normalize_technical_text,
    route_query_type,
    _empty_requirement,
    _target_present,
    _value_repeats_target,
    _SLOT_NOUNS,
    SAME_SEGMENT,
    ADJACENT_SEGMENT,
    NO_LOCALITY,
)
from .semantic_judge import route_ambiguity
from .semantic_judge_localnli import build_hypothesis, build_premise
from . import semantic_judge_localnli

# Experimental candidate identity.
GROUNDED_SPAN_CANDIDATE_VERSION = "evidence-v328-grounded-span-candidate"
GROUNDED_SPAN_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"
LLM_EXTRACTION_USED = "NO"

# V3.29 boundary disposition.  Grounded answer spans are optional enrichment for
# the downstream answer layer; they have no decision authority over ANSWER/ABSTAIN.
GROUNDED_SPAN_DISPOSITION = "OPTIONAL_EVIDENCE_PAYLOAD"


class AnswerType(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    SCALAR_VALUE = "SCALAR_VALUE"
    ENUM_VALUE = "ENUM_VALUE"
    SHORT_NOUN_PHRASE = "SHORT_NOUN_PHRASE"
    ACTION_PHRASE = "ACTION_PHRASE"
    TABLE_CELL = "TABLE_CELL"
    RELATION_SPAN = "RELATION_SPAN"
    LOCATION_SPAN = "LOCATION_SPAN"


class NormalizationStatus(str, Enum):
    NORMALIZED = "NORMALIZED"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


class SpanMultiplicity(str, Enum):
    UNIQUE = "UNIQUE"
    MULTIPLE_EQUIVALENT = "MULTIPLE_EQUIVALENT"
    MULTIPLE_CONFLICTING = "MULTIPLE_CONFLICTING"
    MULTIPLE_SCOPE_RESOLVABLE = "MULTIPLE_SCOPE_RESOLVABLE"
    NONE = "NONE"


class FailureCause(str, Enum):
    """Root-cause bucket for a wrong evidence decision (evaluation attribution only)."""
    EVIDENCE_ASSOCIATION_FAILURE = "EVIDENCE_ASSOCIATION_FAILURE"
    ANSWER_GRANULARITY_FAILURE = "ANSWER_GRANULARITY_FAILURE"
    NORMALIZATION_FAILURE = "NORMALIZATION_FAILURE"
    PARSER_TABLE_STRUCTURE_LOSS = "PARSER_TABLE_STRUCTURE_LOSS"
    AMBIGUOUS_MULTIPLE_OBJECTS = "AMBIGUOUS_MULTIPLE_OBJECTS"
    GROUNDING_MISSING = "GROUNDING_MISSING"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Answer shape annotation (development overlay; never part of ground truth)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerSpan:
    """A traceable source span for a grounded answer object."""
    chunk_id: str
    page: Any = None
    section: str = ""
    text: str = ""                 # normalized surface text of the span
    start: int = -1                # char offset into the normalized segment
    end: int = -1
    segment_index: int = -1
    locality_rank: int = NO_LOCALITY

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GroundedAnswerObject:
    """A candidate-grounded answer object (surface span first).

    ``surface_text`` is the span verbatim from the source.  ``normalized_value``
    is optional and non-authoritative: it never gates the sufficiency decision.
    """
    answer_type: str
    surface_text: str
    normalized_value: str = ""
    source_candidate_ids: tuple[str, ...] = ()
    source_span: AnswerSpan | None = None
    target: str = ""
    relation: str = ""
    qualifiers: tuple[str, ...] = ()
    scope: str = ""
    association_confidence: float = 0.0
    normalization_status: str = NormalizationStatus.NOT_ATTEMPTED.value

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["source_span"] = self.source_span.as_dict() if self.source_span else None
        return payload


@dataclass(frozen=True)
class SufficiencyResult:
    status: str
    target_recognized: bool
    answer_span_present: bool
    relation_supported: bool
    scope_compatible: bool
    multiplicity: str
    grounded_answers: tuple[GroundedAnswerObject, ...]
    failure_cause: str
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["grounded_answers"] = [a.as_dict() for a in self.grounded_answers]
        return payload


@dataclass(frozen=True)
class GroundedEvidenceDecision:
    query: str
    decision: str
    reason: str
    sufficiency: SufficiencyResult | None = None
    answer: GroundedAnswerObject | None = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "sufficiency": self.sufficiency.as_dict() if self.sufficiency else None,
            "answer": self.answer.as_dict() if self.answer else None,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Answer-type mapping (slot -> token/phrase answer type)
# ---------------------------------------------------------------------------

_TOKEN_ANSWER_TYPE: dict[str, str] = {
    "IDENTIFIER": AnswerType.IDENTIFIER.value,
    "REGISTER": AnswerType.IDENTIFIER.value,
    "VALUE": AnswerType.SCALAR_VALUE.value,
    "UNIT": AnswerType.SCALAR_VALUE.value,
    "SETTING": AnswerType.ENUM_VALUE.value,
    "TERMINAL": AnswerType.IDENTIFIER.value,
    "CHANNEL": AnswerType.IDENTIFIER.value,
    "LOCATION": AnswerType.LOCATION_SPAN.value,
    "ACTION": AnswerType.ACTION_PHRASE.value,
    "ATTRIBUTE": AnswerType.RELATION_SPAN.value,
    "CONDITION": AnswerType.RELATION_SPAN.value,
    "UNKNOWN": AnswerType.SHORT_NOUN_PHRASE.value,
}

_PHRASE_ANSWER_TYPE: dict[str, str] = {
    "ACTION": AnswerType.ACTION_PHRASE.value,
    "ATTRIBUTE": AnswerType.RELATION_SPAN.value,
    "CONDITION": AnswerType.RELATION_SPAN.value,
    "SETTING": AnswerType.SHORT_NOUN_PHRASE.value,
    "TERMINAL": AnswerType.SHORT_NOUN_PHRASE.value,
    "CHANNEL": AnswerType.SHORT_NOUN_PHRASE.value,
    "LOCATION": AnswerType.LOCATION_SPAN.value,
    "UNKNOWN": AnswerType.SHORT_NOUN_PHRASE.value,
    "IDENTIFIER": AnswerType.SHORT_NOUN_PHRASE.value,
    "REGISTER": AnswerType.SHORT_NOUN_PHRASE.value,
    "VALUE": AnswerType.SHORT_NOUN_PHRASE.value,
    "UNIT": AnswerType.SHORT_NOUN_PHRASE.value,
}

# Relation keyword *types* detected in an answer-bearing span.  A token span is
# self-verifying (the token IS the requested object type); a surface/segment span
# must carry at least one relation keyword so a wrong-target/wrong-mode candidate
# is not released.
_RELATION_SIGNALS: dict[str, tuple[str, ...]] = {
    "REGISTER": ("register", "parameter", "index", "object", "address", "code", "byte"),
    "IDENTIFIER": ("register", "parameter", "index", "object", "address", "code", "identifier", "id", "name"),
    "VALUE": ("value", "default", "range", "reading", "level", "time", "voltage", "current", "=", ":"),
    "UNIT": ("unit", "=", ":"),
    "SETTING": ("setting", "option", "mode", "default", "factory", "state", "=", ":"),
    "TERMINAL": ("terminal", "pin", "contact", "grounding", "ground", "=", ":"),
    "CHANNEL": ("channel", "port", "=", ":"),
    "ACTION": ("action", "step", "do", "press", "hold", "set", "cycle", "switch", "connect", "use", "=", ":"),
    "ATTRIBUTE": ("indicates", "means", "represents", "signal", "=", ":"),
    "CONDITION": ("if", "when", "condition", "=", ":"),
    "LOCATION": ("located", "location", "on", "in", "=", ":"),
    "UNKNOWN": ("=", ":"),
}

def _relation_present(span_text: str, slot_type: str) -> bool:
    signals = _RELATION_SIGNALS.get(slot_type, ("=", ":"))
    normalized = normalize_technical_text(span_text)
    return any(signal in normalized for signal in signals)


def _find_offsets(segment: str, needle: str) -> list[tuple[int, int]]:
    """Return start/end offsets of every occurrence of ``needle`` in ``segment``."""
    offsets: list[tuple[int, int]] = []
    start = 0
    segment_l = segment.casefold()
    needle_l = needle.casefold()
    while True:
        index = segment_l.find(needle_l, start)
        if index < 0:
            return offsets
        offsets.append((index, index + len(needle)))
        start = index + max(1, len(needle))


def _answer_region(segment: str, target_entity: str) -> str:
    """The answer-bearing region of a target-bearing segment.

    Prefer the text after a relation separator / the target occurrence; otherwise
    return the whole segment as the surface span.
    """
    if not segment:
        return ""
    # Text after an explicit assignment separator is the most likely answer region.
    for marker in (":", "=", "->", "\u2013", "-"):
        idx = segment.rfind(marker)
        if idx >= 0 and idx + 1 < len(segment):
            tail = segment[idx + 1:].strip()
            if tail:
                return tail
    # Otherwise strip the target phrase itself when it is a clear prefix.
    target = normalize_technical_text(target_entity)
    if target and segment.casefold().startswith(target):
        return segment[len(target):].strip(" :=\u2013-")
    return segment


def _discover_candidate_spans(
    claim: CandidateClaim, slot_type: str, target_entity: str,
) -> list[tuple[str, AnswerSpan, int]]:
    """Return ``(surface_text, span, locality_rank)`` for answer-bearing spans.

    Token spans are only taken from a segment where the target co-occurs in the
    *same* line (SAME_SEGMENT), so a neighbouring "wrong-entity" row (READ byte
    vs WRITE byte) is not released.  When no canonical token exists anywhere in
    the candidate, a single surface-span fallback carries a noun/action/table
    phrase so that a natural-language answer is still grounded.
    """
    segments = list(claim.segments) or ([claim.text] if claim.text else [])
    token_spans: list[tuple[str, AnswerSpan, int]] = []
    for index, segment in enumerate(segments):
        if not _target_present(target_entity, segment):
            continue
        for value in extract_candidate_objects(segment, slot_type):
            if _value_repeats_target(value, target_entity):
                continue
            value = normalize_technical_text(value)
            if not value:
                continue
            for start, end in _find_offsets(segment, value):
                span = AnswerSpan(
                    chunk_id=claim.chunk_id, page=claim.page, section=claim.section,
                    text=segment[start:end], start=start, end=end,
                    segment_index=index, locality_rank=SAME_SEGMENT,
                )
                token_spans.append((value, span, SAME_SEGMENT))
    if token_spans:
        return token_spans

    # Surface-span fallback: the single best target-bound segment's answer region.
    best: tuple[str, AnswerSpan] | None = None
    best_rank = NO_LOCALITY
    for index, segment in enumerate(segments):
        if _target_present(target_entity, segment):
            rank = SAME_SEGMENT
        elif _target_present(target_entity, " ".join(segments[max(0, index - 1):index + 2])):
            rank = ADJACENT_SEGMENT
        else:
            continue
        region = normalize_technical_text(_answer_region(segment, target_entity))
        if not region or _value_repeats_target(region, target_entity):
            continue
        if rank > best_rank:
            best = (region, AnswerSpan(
                chunk_id=claim.chunk_id, page=claim.page, section=claim.section,
                text=normalize_technical_text(segment), start=0, end=len(segment),
                segment_index=index, locality_rank=rank,
            ))
            best_rank = rank
    return [(best[0], best[1], best_rank)] if best else []


def discover_answer_objects(
    query: str,
    candidates: Iterable,
    *,
    slot_type: str,
    target_entity: str,
    answer_type: str | None = None,
    relation: str = "",
) -> tuple[GroundedAnswerObject, ...]:
    """Discover candidate-grounded answer objects for a target/slot.

    Aggregates spans across candidates by distinct normalized surface text,
    resolving multiplicity (UNIQUE / MULTIPLE_EQUIVALENT / MULTIPLE_CONFLICTING /
    MULTIPLE_SCOPE_RESOLVABLE / NONE).
    """
    candidate_list = list(candidates)
    claims = tuple(
        extract_candidate_claim(candidate, _empty_requirement(target_entity))
        for candidate in candidate_list
    )

    scored: dict[str, tuple[int, int, list[str], AnswerSpan]] = {}
    for claim, candidate in zip(claims, candidate_list):
        chunk_id = str(candidate.chunk_id or "")
        for surface, span, rank in _discover_candidate_spans(claim, slot_type, target_entity):
            if not surface.strip():
                continue
            key = normalize_technical_text(surface)
            best_rank, count, sources, _ = scored.get(key, (NO_LOCALITY, 0, [], span))
            sources = sources + ([chunk_id] if chunk_id not in sources else [])
            scored[key] = (max(best_rank, rank), count + 1, sources, span)

    if not scored:
        return ()

    objects: list[GroundedAnswerObject] = []
    token_type = answer_type or _TOKEN_ANSWER_TYPE.get(slot_type, AnswerType.SHORT_NOUN_PHRASE.value)
    for key, (rank, count, sources, span) in scored.items():
        objects.append(GroundedAnswerObject(
            answer_type=token_type,
            surface_text=key,
            source_candidate_ids=tuple(sources),
            source_span=span,
            target=target_entity,
            relation=relation or _SLOT_NOUNS.get(slot_type, "value"),
            scope=claim.scope if hasattr(claim, "scope") else "",
            association_confidence=round(0.55 + 0.15 * min(rank, 3) / 3.0, 4),
            normalization_status=NormalizationStatus.NOT_ATTEMPTED.value,
        ))
    # Keep the object with the strongest locality as the primary answer.
    objects.sort(key=lambda obj: (obj.association_confidence, len(obj.source_candidate_ids)), reverse=True)
    return tuple(objects)


def _multiplicity(objects: tuple[GroundedAnswerObject, ...]) -> str:
    if not objects:
        return SpanMultiplicity.NONE.value
    if len(objects) == 1:
        return SpanMultiplicity.UNIQUE.value
    surfaces = {normalize_technical_text(obj.surface_text) for obj in objects}
    if len(surfaces) == 1:
        return SpanMultiplicity.MULTIPLE_EQUIVALENT.value
    ranks = {obj.source_span.locality_rank if obj.source_span else NO_LOCALITY for obj in objects}
    counts = {len(obj.source_candidate_ids) for obj in objects}
    if len(ranks) > 1 or len(counts) > 1:
        # A single object strictly dominates on confidence.
        if objects[0].association_confidence > objects[1].association_confidence + 0.05:
            return SpanMultiplicity.MULTIPLE_SCOPE_RESOLVABLE.value
    return SpanMultiplicity.MULTIPLE_CONFLICTING.value


def normalize_answer_object(obj: GroundedAnswerObject, slot_type: str) -> GroundedAnswerObject:
    """Optional second-stage normalization; failure is never insufficiency."""
    if not obj.surface_text:
        return replace(obj, normalized_value="", normalization_status=NormalizationStatus.NORMALIZATION_FAILED.value)
    tokens = [normalize_technical_text(t) for t in extract_candidate_objects(obj.surface_text, slot_type)]
    tokens = [t for t in dict.fromkeys(tokens) if not _value_repeats_target(t, obj.target)]
    if len(tokens) == 1:
        return replace(obj, normalized_value=tokens[0], normalization_status=NormalizationStatus.NORMALIZED.value)
    return replace(obj, normalized_value="", normalization_status=NormalizationStatus.NORMALIZATION_FAILED.value)


_CONFLICTING = SpanMultiplicity.MULTIPLE_CONFLICTING.value
_NONE = SpanMultiplicity.NONE.value


def decide_sufficiency(
    query: str,
    candidates: Iterable,
    slot_type: str,
    target_entity: str,
    *,
    answer_type: str | None = None,
    normalize: bool = False,
) -> SufficiencyResult:
    """Run the evidence-sufficiency contract and (optionally) normalization."""
    objects = discover_answer_objects(
        query, candidates, slot_type=slot_type, target_entity=target_entity,
        answer_type=answer_type,
    )
    if normalize:
        objects = tuple(normalize_answer_object(obj, slot_type) for obj in objects)

    multiplicity = _multiplicity(objects)
    target_recognized = bool(objects)
    answer_span_present = bool(objects)
    relation_supported = any(
        _relation_present(obj.surface_text, slot_type)
        or (obj.source_span is not None and _relation_present(obj.source_span.text, slot_type))
        for obj in objects
    ) or any(
        # A single-token span is self-verifying even without a relation keyword.
        obj.surface_text and len(obj.surface_text.split()) == 1 for obj in objects
    )

    if not objects:
        return SufficiencyResult(
            SufficiencyStatus.INSUFFICIENT.value, False, False, False, True, _NONE, (),
            FailureCause.GROUNDING_MISSING.value, 0.0, "GROUNDING_MISSING",
        )
    top = objects[0]
    if multiplicity == _CONFLICTING:
        return SufficiencyResult(
            SufficiencyStatus.AMBIGUOUS.value, True, True, relation_supported, True,
            multiplicity, objects, FailureCause.AMBIGUOUS_MULTIPLE_OBJECTS.value, top.association_confidence,
            "AMBIGUOUS_MULTIPLE_OBJECTS",
        )
    if not relation_supported:
        return SufficiencyResult(
            SufficiencyStatus.INSUFFICIENT.value, True, True, False, True,
            multiplicity, objects, FailureCause.EVIDENCE_ASSOCIATION_FAILURE.value, top.association_confidence,
            "EVIDENCE_ASSOCIATION_FAILURE",
        )
    return SufficiencyResult(
        SufficiencyStatus.SUFFICIENT.value, True, True, True, True,
        multiplicity, objects, "", top.association_confidence, "SUFFICIENT",
    )


# ---------------------------------------------------------------------------
# Unified V3.28 entry point (strategy C/D; A/B reuse the V3.27 module)
# ---------------------------------------------------------------------------

def analyze_sufficiency_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    strategy: str = "GROUNDED_SPAN",
    judge: Any = None,
    target_entity: str = "",
    slot_type: str = "",
    answer_type: str | None = None,
    policy: Any = None,
    identity_matching: bool = True,
) -> GroundedEvidenceDecision:
    """Evidence sufficiency + optional grounded-answer normalization.

    Extraction-routed queries take the grounded-span path; VERIFICATION / UNKNOWN
    queries fall through to the frozen V3.25 rule + selective NLI path unchanged.
    """
    analysis = getattr(result, "query_analysis", None)
    requirement = build_typed_requirement(query, documents, analysis)
    route = route_query_type(query, requirement)
    rule = analyze_retrieval_evidence(
        query, result, documents, retrieval_mode, policy=policy, identity_matching=identity_matching
    )

    if route.query_type != EvidenceQueryType.EXTRACTION.value:
        decision, reason = _frozen_verifier(query, rule.decision, requirement, result, judge, policy)
        return GroundedEvidenceDecision(query, decision, reason, evidence=rule.as_dict())

    ext_requirement = ExtractionRequirement(
        target_entity=target_entity or build_extraction_requirement(query).target_entity,
        slot_type=slot_type or detect_extraction_slot(query).value if detect_extraction_slot(query) else "VALUE",
    )
    normalize = strategy == "GROUNDED_SPAN_NORMALIZED"
    suff = decide_sufficiency(
        query, list(getattr(result, "candidates", []) or []),
        ext_requirement.slot_type, ext_requirement.target_entity,
        answer_type=answer_type, normalize=normalize,
    )
    if suff.status == SufficiencyStatus.SUFFICIENT.value:
        decision, reason = Decision.ANSWER.value, "SUFFICIENT_GROUNDED_SPAN"
    else:
        decision, reason = Decision.ABSTAIN.value, suff.reason
    return GroundedEvidenceDecision(
        query, decision, reason, sufficiency=suff, answer=suff.grounded_answers[0] if suff.grounded_answers else None,
        evidence=rule.as_dict(),
    )


def _frozen_verifier(query, rule_decision, requirement, result, judge, policy):
    """The exact V3.25 selective-NLI behavior for non-extraction queries."""
    polarity_route = route_ambiguity(query, requirement)
    if polarity_route is None or judge is None or getattr(judge, "model", None) is None:
        return rule_decision, "RULE"
    claims = [extract_candidate_claim(candidate, requirement) for candidate in list(getattr(result, "candidates", []) or [])]
    premise = build_premise(claims)
    hypothesis = build_hypothesis(query, requirement)
    probs = judge.predict_probs(premise, hypothesis)
    judge_decision, _ = judge.decide_from_probs(probs)
    if judge_decision == semantic_judge_localnli.JudgeDecision.ENTAILS.value:
        return rule_decision, "NLI_ENTAILS"
    if judge_decision in (semantic_judge_localnli.JudgeDecision.CONTRADICTS.value, semantic_judge_localnli.JudgeDecision.INSUFFICIENT.value):
        return Decision.ABSTAIN.value, "NLI_" + judge_decision
    return rule_decision, "NLI_UNKNOWN_FALLBACK"