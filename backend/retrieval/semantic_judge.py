"""Experimental semantic-judge contract and ambiguity router (V3.24 study).

This module defines HOW a relation-aware judge would be layered on top of the
rule-based Evidence contract (Typed Requirement -> Candidate Claim -> Scope ->
Coverage).  It is a feasibility artifact and is **OFF by default**: the shipped
Evidence pipeline must keep ``SEMANTIC_JUDGE_DEFAULT == "OFF"`` until a judge
candidate is deliberately frozen behind its own gate.

Ground-truth discipline: no judge in this module receives the expected label,
the annotation rationale, or the failure class.  Judges only ever see the query,
the typed requirements, candidate-local claims, supporting candidate text and
(non-labelling) scope metadata.  A real judge must therefore be evaluated on its
own merits, never fitted to private DEV case IDs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Protocol


class JudgeDecision(str, Enum):
    """Relation-level verdict; the Evidence decision maps this to ANSWER/ABSTAIN."""

    ENTAILS = "ENTAILS"
    CONTRADICTS = "CONTRADICTS"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"


class RelationFailure(str, Enum):
    """Abstract relation-failure taxonomy observed on the DEV identifier slice."""

    ROLE_REVERSAL = "ROLE_REVERSAL"
    PREDICATE_REVERSAL = "PREDICATE_REVERSAL"
    CONDITION_MISMATCH = "CONDITION_MISMATCH"
    ACTION_MISMATCH = "ACTION_MISMATCH"
    DEFAULT_TARGET_MISMATCH = "DEFAULT_TARGET_MISMATCH"
    ENTITY_RELATION_MISMATCH = "ENTITY_RELATION_MISMATCH"
    OTHER = "OTHER"


class AmbiguityType(str, Enum):
    """Why a query is routed to the judge (labels the tripped relation ambiguity)."""

    RELATION_AMBIGUITY = "RELATION_AMBIGUITY"
    ROLE_AMBIGUITY = "ROLE_AMBIGUITY"
    PREDICATE_AMBIGUITY = "PREDICATE_AMBIGUITY"
    CONDITION_AMBIGUITY = "CONDITION_AMBIGUITY"
    ACTION_AMBIGUITY = "ACTION_AMBIGUITY"


@dataclass(frozen=True)
class SemanticJudgeResult:
    """Structured judge output; never a free-text ANSWER/ABSTAIN."""

    decision: str
    confidence: float
    reason_code: str
    subject_match: bool = False
    predicate_match: bool = False
    object_match: bool = False
    condition_match: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_JUDGE_DECISIONS = frozenset(item.value for item in JudgeDecision)


class SemanticEvidenceJudge(Protocol):
    """Minimal judge interface consumed by the experimental decision mapping."""

    def judge(
        self,
        query: str,
        requirements: Any,
        claims: Any = None,
        scope: Any = None,
    ) -> SemanticJudgeResult:
        ...


class RuleOnlyJudge:
    """Non-judge: always UNKNOWN so the decision mapping falls back to the rule."""

    def judge(
        self,
        query: str,
        requirements: Any,
        claims: Any = None,
        scope: Any = None,
    ) -> SemanticJudgeResult:
        return SemanticJudgeResult(
            decision=JudgeDecision.UNKNOWN.value,
            confidence=0.0,
            reason_code="JUDGE_DISABLED",
        )


# ---------------------------------------------------------------------------
# Ambiguity router
# ---------------------------------------------------------------------------

_POLAR_LEAD_RE = re.compile(r"^\s*(?:does|is|are|can|do|will|should|must|was|were|has|have)\b", re.IGNORECASE)

_CONTRAST_MARKERS = ("rather than", "instead of", "first ")
_CONDITIONAL_MARKERS = ("whenever", " if ", "unless", " when ", "exceeds", "exceed")
_DEFAULT_MARKERS = ("default", "required")
_ROLE_MARKERS = (" while ", " only ", " without ", " all ", " versus ", " vs ")
_ACTION_VERBS = (
    "restart", "reset", "restore", "delete", "delet", "remove", "configure",
    "recover", "replace", "initialize", "initialise", "commission", "start",
    "stop", "enable", "disable", "select", "set", "reprogram",
)
# ``default``/``required`` alone do not imply a relation ambiguity: plain
# default-value lookups stay on the rule path; only their co-occurrence with a
# conditional marker routes to CONDITION_AMBIGUITY.
_RELATION_MARKERS = _CONTRAST_MARKERS + _CONDITIONAL_MARKERS + _ROLE_MARKERS


def _critical_kinds(requirement: Any) -> Counter:
    counter: Counter = Counter()
    for item in getattr(requirement, "items", ()) or ():
        if getattr(item, "criticality", None) == "CRITICAL":
            counter[getattr(item, "kind", "")] += 1
    return counter


def route_ambiguity(query: str, requirement: Any) -> AmbiguityType | None:
    """Return the relation-ambiguity label when a judge is warranted, else None.

    Rationale (grounded in the DEV identifier near-miss pattern, not in query
    IDs): lexical coverage answers correctly for open locator questions and for
    clear supported/unsupported cases.  It fails specifically on polar
    verification questions that bind two entities to a reversed role or that
    assert a contrast/condition the candidate cannot entail.  Only those are
    routed, so the judge is never asked to re-answer the stable slices.
    """
    text = (query or "").lower()
    if _POLAR_LEAD_RE.match(text) is None:
        return None

    kinds = _critical_kinds(requirement)
    has_contrast = any(marker in text for marker in _CONTRAST_MARKERS)
    has_default = any(marker in text for marker in _DEFAULT_MARKERS)
    has_conditional = any(marker in text for marker in _CONDITIONAL_MARKERS)
    has_role_pair = any(marker in text for marker in _ROLE_MARKERS)

    if has_contrast and any(verb in text for verb in _ACTION_VERBS):
        return AmbiguityType.ACTION_AMBIGUITY
    if has_contrast:
        return AmbiguityType.PREDICATE_AMBIGUITY

    if has_default and has_conditional and (
        kinds.get("attribute", 0) or kinds.get("value_kind", 0) or kinds.get("value", 0)
    ):
        return AmbiguityType.CONDITION_AMBIGUITY

    if has_role_pair and (kinds.get("value", 0) >= 2 or kinds.get("qualifier", 0) >= 2):
        return AmbiguityType.ROLE_AMBIGUITY

    if any(marker in text for marker in _RELATION_MARKERS):
        return AmbiguityType.RELATION_AMBIGUITY
    return None


class EvidenceAmbiguityRouter:
    """Thin object wrapper around :func:`route_ambiguity` for DI/testing."""

    def route(self, query: str, requirement: Any) -> AmbiguityType | None:
        return route_ambiguity(query, requirement)


# ---------------------------------------------------------------------------
# Decision mapping
# ---------------------------------------------------------------------------

def resolve_decision(rule_decision: str, judge: SemanticJudgeResult) -> str:
    """Map a judge verdict onto a final ANSWER/ABSTAIN, preserving rule wins.

    ENTAILS      -> keep the rule decision (a routed positive stays ANSWER).
    CONTRADICTS  -> ABSTAIN (the claim's relation is inverted).
    INSUFFICIENT -> ABSTAIN (the claim's relation is unsupported).
    UNKNOWN      -> conservative fallback to the rule decision (never over-abstain
                    on a judge that cannot call the relation; evaluated separately).
    """
    decision = getattr(judge, "decision", JudgeDecision.UNKNOWN.value)
    if decision == JudgeDecision.ENTAILS.value:
        return rule_decision
    if decision in (JudgeDecision.CONTRADICTS.value, JudgeDecision.INSUFFICIENT.value):
        return "ABSTAIN"
    return rule_decision  # UNKNOWN (or malformed) -> trust rule


# ---------------------------------------------------------------------------
# Schema validation (structured output, not free text)
# ---------------------------------------------------------------------------

def validate_judge_schema(result: SemanticJudgeResult) -> list[str]:
    """Return a list of schema violations (empty list means valid)."""
    violations: list[str] = []
    if result.decision not in VALID_JUDGE_DECISIONS:
        violations.append(f"INVALID_DECISION:{result.decision!r}")
    if not isinstance(result.confidence, (int, float)) or not 0.0 <= float(result.confidence) <= 1.0:
        violations.append(f"INVALID_CONFIDENCE:{result.confidence!r}")
    if not isinstance(result.reason_code, str) or not result.reason_code.strip():
        violations.append("MISSING_REASON_CODE")
    for field in ("subject_match", "predicate_match", "object_match", "condition_match"):
        if not isinstance(getattr(result, field), bool):
            violations.append(f"NON_BOOL_{field}")
    return violations


# ---------------------------------------------------------------------------
# LLM judge adapter (structured JSON, NOT enabled by default, never auto-sent)
# ---------------------------------------------------------------------------

_JUDGE_JSON_SCHEMA = (
    '{"decision": "ENTAILS|CONTRADICTS|INSUFFICIENT|UNKNOWN", '
    '"confidence": 0.0, "reason_code": "...", "subject_match": false, '
    '"predicate_match": false, "object_match": false, "condition_match": false}'
)


def build_judge_prompt(query: str, requirement: Any, claims_text: Iterable[str] | None = None) -> str:
    """Build a structured-output prompt for an LLM{NLI judge.  No private data is
    sent unless a caller explicitly opts in; this adapter only formats input."""
    items = "\n".join(
        f"- {getattr(item, 'kind', '?')}:{getattr(item, 'value', '')} "
        f"({getattr(item, 'criticality', '?')}/{getattr(item, 'match_mode', '?')})"
        for item in getattr(requirement, "items", ()) or ()
    ) or "(none)"
    evidence = "\n---\n".join(claims_text or ()) or "(none)"
    return (
        "You are an entailment-only relation judge for an industrial RAG evidence contract.\n"
        "Decide whether the candidate evidence ENTAILS, CONTRADICTS, or is INSUFFICIENT "
        "for the query's asserted relation (role/predicate/condition/action). "
        "Return UNKNOWN if the relation cannot be determined.\n"
        f"Query: {query}\nRequirements:\n{items}\nCandidate evidence:\n{evidence}\n"
        f"Respond with exactly one JSON object matching: {_JUDGE_JSON_SCHEMA}"
    )


def parse_judge_response(raw: str) -> SemanticJudgeResult:
    """Parse the structured judge JSON; raises ValueError on malformed output."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("JUDGE_RESPONSE_NO_JSON")
    payload = json.loads(match.group(0))
    result = SemanticJudgeResult(
        decision=str(payload.get("decision", JudgeDecision.UNKNOWN.value)),
        confidence=float(payload.get("confidence", 0.0)),
        reason_code=str(payload.get("reason_code", "")),
        subject_match=bool(payload.get("subject_match", False)),
        predicate_match=bool(payload.get("predicate_match", False)),
        object_match=bool(payload.get("object_match", False)),
        condition_match=bool(payload.get("condition_match", False)),
    )
    violations = validate_judge_schema(result)
    if violations:
        raise ValueError("JUDGE_SCHEMA_VIOLATION:" + ";".join(violations))
    return result


# ---------------------------------------------------------------------------
# Configuration / feasibility flags
# ---------------------------------------------------------------------------

# Production default: the judge must stay OFF until a candidate is frozen.
SEMANTIC_JUDGE_DEFAULT = "OFF"

# Whether a real external LLM judge was evaluated on private benchmark data.
# This study is offline: private chunks were never sent to a third-party API.
LLM_JUDGE_REAL_PRIVATE_EVAL = "NO"

# No true local NLI/entailment model is cached in the current environment.
# A local entailment model (e.g. a cross-encoder NLI / bart-mnli checkpoint) is
# the dependency required before LOCAL_NLI can be exercised; do not auto-download.
REQUIRED_LOCAL_NLI_DEPENDENCY = "cross-encoder/nli-* (or facebook/bart-large-mnli family)"