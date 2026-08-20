"""Evidence / Answer boundary (V3.29).

V3.27/V3.28 showed that letting answer-object *extraction* (a hard extraction
gate, or a grounded-span gate) co-own the ``ANSWER``/``ABSTAIN`` decision trades
answerable recall away for false-answer safety.  The decisive finding is
``EVIDENCE_SHOULD_NOT_OWN_ANSWER_EXTRACTION``.

This module therefore formalizes the responsibility boundary as a *candidate*:

* **Evidence owns sufficiency only** — the frozen V3.25 rule contract plus the
  selective local-NLI verifier decide ``ANSWER``/``ABSTAIN`` (exactly
  ``VERIFIER_ONLY`` from V3.27, for every query type including open "which/what"
  questions).  Nothing else may change that decision.

* **Grounding is optional enrichment** — answer-bearing spans / objects are
  discovered *best-effort* and attached as a non-authoritative payload for the
  downstream answer layer.  They never gate the decision.

* **Normalization is optional and non-authoritative** — a failed normalization
  keeps the surface span; it never implies insufficiency.

Constants make the invariants machine-checkable:

* ``EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"``
* ``NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"``
* ``GROUNDING_DECISION_AUTHORITY = "NONE"``
* ``GENERATION_USED = "NO"``, ``LLM_ANSWER_USED = "NO"``
* ``GROUNDING_ENRICHMENT_DEFAULT = "OFF"``

Deterministic only; no generator, no LLM answer, no prompt design, no citation
rendering here.  ``AnswerContext`` is the pure-data downstream interface; the
generation layer itself is out of scope for this phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .evidence import Decision
from .evidence_answerobject import (
    AnswerSpan,
    GroundedAnswerObject,
    NormalizationStatus,
    SpanMultiplicity,
    _multiplicity,
    discover_answer_objects,
    normalize_answer_object,
)
from .evidence_contract import build_typed_requirement, extract_candidate_claim
from .evidence_querytype import (
    EvidenceQueryType,
    analyze_querytype_evidence,
    build_extraction_requirement,
    detect_extraction_slot,
)
from .filters import analyze_query

# Candidate identity.
EVIDENCE_BOUNDARY_CANDIDATE_VERSION = "evidence-v329-boundary-candidate"
EVIDENCE_BOUNDARY_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

# Formal non-gating flags (machine-checkable boundary).
EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"
NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"
GROUNDING_DECISION_AUTHORITY = "NONE"
GROUNDING_ENRICHMENT_DEFAULT = "OFF"
GENERATION_USED = "NO"
LLM_ANSWER_USED = "NO"


class GroundingStatus(str, Enum):
    """Metadata status of the optional grounding pass (never the decision)."""
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    FOUND = "FOUND"
    MULTIPLE = "MULTIPLE"
    AMBIGUOUS = "AMBIGUOUS"
    NONE = "NONE"


@dataclass(frozen=True)
class Citation:
    """Traceable source provenance for an approved candidate."""
    chunk_id: str
    document_id: str = ""
    page: Any = None
    section: str = ""
    subsection: str = ""
    snippet: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GroundingPayload:
    """Best-effort grounding enrichment; no decision authority."""
    status: str
    objects: tuple[GroundedAnswerObject, ...]
    spans: tuple[AnswerSpan, ...]
    normalization: dict

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "objects": [obj.as_dict() for obj in self.objects],
            "spans": [span.as_dict() for span in self.spans],
            "normalization": self.normalization,
        }


@dataclass(frozen=True)
class EvidenceDecisionV2:
    """Evidence decision contract: required vs optional fields.

    Only ``decision`` and ``reason`` are authoritative.  Every other field is
    non-authoritative metadata that must never change ``decision``.
    """
    query: str
    decision: str
    reason: str
    query_type: str = ""
    supporting_candidate_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    relation_judge_result: dict = field(default_factory=dict)
    grounding_status: str = GroundingStatus.NOT_ATTEMPTED.value
    grounded_objects: tuple[GroundedAnswerObject, ...] = ()
    grounded_spans: tuple[AnswerSpan, ...] = ()
    normalization: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    query_type_route: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "query_type": self.query_type,
            "supporting_candidate_ids": list(self.supporting_candidate_ids),
            "citations": [citation.as_dict() for citation in self.citations],
            "relation_judge_result": self.relation_judge_result,
            "grounding": {
                "status": self.grounding_status,
                "objects": [obj.as_dict() for obj in self.grounded_objects],
                "spans": [span.as_dict() for span in self.grounded_spans],
                "normalization": self.normalization,
            },
            "metadata": self.metadata,
            "evidence": self.evidence,
            "query_type_route": self.query_type_route,
        }


@dataclass(frozen=True)
class AnswerContext:
    """Downstream-facing pure-data interface (no generator here).

    The only context a future answer/generation layer may consume; it carries
    **approved** evidence only.  On ``ABSTAIN`` there are no approved candidates.
    """
    query: str
    evidence_decision: str
    reason: str
    approved_candidate_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    grounded_objects: tuple[GroundedAnswerObject, ...]
    relation_judge: dict
    metadata: dict

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "evidence_decision": self.evidence_decision,
            "reason": self.reason,
            "approved_candidate_ids": list(self.approved_candidate_ids),
            "citations": [citation.as_dict() for citation in self.citations],
            "grounded_objects": [obj.as_dict() for obj in self.grounded_objects],
            "relation_judge": self.relation_judge,
            "metadata": self.metadata,
        }


_STATUS_BY_MULTIPLICITY = {
    SpanMultiplicity.NONE.value: GroundingStatus.NONE.value,
    SpanMultiplicity.UNIQUE.value: GroundingStatus.FOUND.value,
    SpanMultiplicity.MULTIPLE_EQUIVALENT.value: GroundingStatus.FOUND.value,
    SpanMultiplicity.MULTIPLE_SCOPE_RESOLVABLE.value: GroundingStatus.MULTIPLE.value,
    SpanMultiplicity.MULTIPLE_CONFLICTING.value: GroundingStatus.AMBIGUOUS.value,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_boundary_evidence(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    enrich_grounding: bool = False,
    normalize: bool = False,
    policy: Any = None,
    identity_matching: bool = True,
) -> EvidenceDecisionV2:
    """Evidence decision + optional non-gating grounding enrichment.

    The ``ANSWER``/``ABSTAIN`` decision is taken by the frozen V3.25 rule +
    selective NLI path (``VERIFIER_ONLY`` for *every* query type).  Grounding /
    normalization are computed only when ``enrich_grounding`` and attached as
    metadata; they are bidirectionally incapable of changing the decision.
    """
    base = analyze_querytype_evidence(
        query, result, documents, retrieval_mode,
        mode="VERIFIER_ONLY", judge=judge, policy=policy, identity_matching=identity_matching,
    )
    decision = base.decision
    reason = base.reason

    candidates = list(getattr(result, "candidates", []) or [])
    requirement = _typed_requirement(query, documents, result)
    citations = tuple(_citation(candidate, requirement) for candidate in candidates)
    # Approved candidates exist only when evidence ANSWERs; downstream may use
    # exactly these (§32 boundary).
    supporting_ids = (
        tuple(str(candidate.chunk_id) for candidate in candidates)
        if decision == Decision.ANSWER.value else ()
    )

    grounding = GroundingPayload(GroundingStatus.NOT_ATTEMPTED.value, (), (), {})
    if enrich_grounding:
        grounding = _enrich_grounding(query, candidates, base.query_type, normalize=normalize)

    return EvidenceDecisionV2(
        query=query,
        decision=decision,
        reason=reason,
        query_type=base.query_type,
        supporting_candidate_ids=supporting_ids,
        citations=citations,
        relation_judge_result=_relation_judge_meta(reason, base.evidence),
        grounding_status=grounding.status,
        grounded_objects=grounding.objects,
        grounded_spans=grounding.spans,
        normalization=grounding.normalization,
        metadata={
            "EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER": EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER,
            "NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER": NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER,
            "GROUNDING_DECISION_AUTHORITY": GROUNDING_DECISION_AUTHORITY,
            "GROUNDING_ENRICHMENT_DEFAULT": GROUNDING_ENRICHMENT_DEFAULT,
            "GENERATION_USED": GENERATION_USED,
            "LLM_ANSWER_USED": LLM_ANSWER_USED,
        },
        evidence=base.evidence,
        query_type_route=base.query_type_route,
    )


def build_answer_context(
    decision: EvidenceDecisionV2,
    *,
    extra_metadata: dict | None = None,
) -> AnswerContext:
    """Project a decision into the downstream pure-data contract.

    Approved candidates == ``supporting_candidate_ids``; on ABSTAIN the context
    carries empty citations/objects so a downstream layer cannot leak non-approved
    evidence into an answer.
    """
    answered = decision.decision == Decision.ANSWER.value
    return AnswerContext(
        query=decision.query,
        evidence_decision=decision.decision,
        reason=decision.reason,
        approved_candidate_ids=decision.supporting_candidate_ids,
        citations=decision.citations if answered else (),
        grounded_objects=decision.grounded_objects if answered else (),
        relation_judge=decision.relation_judge_result,
        metadata={**decision.metadata, **(extra_metadata or {})},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _typed_requirement(query: str, documents: list, result):
    analysis = getattr(result, "query_analysis", None)
    if analysis is None:
        analysis = analyze_query(query, documents)
    return build_typed_requirement(query, documents, analysis)


def _citation(candidate, requirement) -> Citation:
    claim = extract_candidate_claim(candidate, requirement)
    return Citation(
        chunk_id=claim.chunk_id,
        document_id=claim.document_id,
        page=claim.page,
        section=claim.section,
        subsection=claim.subsection,
        snippet=(claim.text[:200] if claim.text else ""),
    )


def _relation_judge_meta(reason: str, evidence: dict) -> dict:
    rule_reason = (evidence or {}).get("reason", "")
    if reason.startswith("NLI_"):
        return {"judge_used": True, "judge_decision": reason, "rule_reason": rule_reason}
    return {"judge_used": False, "judge_decision": "NOT_APPLIED", "rule_reason": rule_reason}


def _enrich_grounding(query: str, candidates: list, query_type: str, *, normalize: bool) -> GroundingPayload:
    if query_type != EvidenceQueryType.EXTRACTION.value or not candidates:
        return GroundingPayload(GroundingStatus.NOT_ATTEMPTED.value, (), (), {})

    slot = detect_extraction_slot(query)
    slot_type = slot.value if slot else "VALUE"
    target_entity = build_extraction_requirement(query).target_entity
    objects = discover_answer_objects(query, candidates, slot_type=slot_type, target_entity=target_entity)
    if normalize:
        objects = tuple(normalize_answer_object(obj, slot_type) for obj in objects)

    status = _STATUS_BY_MULTIPLICITY.get(_multiplicity(objects), GroundingStatus.NONE.value)
    spans = tuple(obj.source_span for obj in objects if obj.source_span is not None)
    return GroundingPayload(status, objects, spans, _normalization_summary(objects))


def _normalization_summary(objects: tuple[GroundedAnswerObject, ...]) -> dict:
    return {
        "normalized": sum(1 for obj in objects if obj.normalization_status == NormalizationStatus.NORMALIZED.value),
        "failed": sum(1 for obj in objects if obj.normalization_status == NormalizationStatus.NORMALIZATION_FAILED.value),
        "not_attempted": sum(1 for obj in objects if obj.normalization_status == NormalizationStatus.NOT_ATTEMPTED.value),
        "values": [obj.normalized_value for obj in objects if obj.normalization_status == NormalizationStatus.NORMALIZED.value],
    }