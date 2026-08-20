"""Tests for V3.29 Evidence / Answer boundary.

These tests pin the formal responsibility boundary:

* The ``ANSWER``/``ABSTAIN`` decision comes only from the frozen V3.25 rule +
  selective NLI (``VERIFIER_ONLY``); grounding/normalization are optional
  enrichment with ``GROUNDING_DECISION_AUTHORITY = "NONE"``.
* Grounding metadata is bidirectionally incapable of changing the decision:
  FOUND does not upgrade ABSTAIN, AMBIGUOUS does not downgrade ANSWER, NONE /
  normalization-failure do not change ANSWER.
* The downstream ``AnswerContext`` carries approved candidates only.

Deterministic only: no model weights, no network, no generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.evidence_boundary import (  # noqa: E402
    EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER,
    GENERATION_USED,
    GROUNDING_DECISION_AUTHORITY,
    GROUNDING_ENRICHMENT_DEFAULT,
    LLM_ANSWER_USED,
    NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER,
    EVIDENCE_BOUNDARY_CANDIDATE_STATUS,
    EVIDENCE_BOUNDARY_CANDIDATE_VERSION,
    GroundingStatus,
    analyze_boundary_evidence,
    build_answer_context,
)
from backend.retrieval.evidence_querytype import analyze_querytype_evidence  # noqa: E402


class _Doc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _candidate(text: str, chunk_id: str = "chunk-1", model: str = "ACME-1"):
    meta = {"chunk_id": chunk_id, "document_id": "doc-1", "manufacturer": "Acme",
            "equipment_model": model, "equipment_type": "plc_controller", "section": "S1",
            "page": 3}
    return RetrievalCandidate(document=_Doc(text, meta), retrieval_source="test")


def _result(*candidates) -> RetrievalResult:
    return RetrievalResult(list(candidates))


def _base(query, result):
    return analyze_querytype_evidence(
        query, result, [c.document for c in result.candidates], "test", mode="VERIFIER_ONLY",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_boundary_constants():
    assert EVIDENCE_BOUNDARY_CANDIDATE_VERSION == "evidence-v329-boundary-candidate"
    assert EVIDENCE_BOUNDARY_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"
    assert EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER == "NO"
    assert NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER == "NO"
    assert GROUNDING_DECISION_AUTHORITY == "NONE"
    assert GROUNDING_ENRICHMENT_DEFAULT == "OFF"
    assert GENERATION_USED == "NO"
    assert LLM_ANSWER_USED == "NO"


# ---------------------------------------------------------------------------
# Decision invariance: grounding/normalization never change the decision
# ---------------------------------------------------------------------------

_FIXTURES = (
    # extraction ANSWER (grounding FOUND)
    ("Which register resets the watchdog timer?", "Register 0x1007 resets the watchdog timer."),
    # extraction ABSTAIN (unknown identifier; grounding NONE)
    ("Which register is 0x9999?", "Register 0x1007 resets the watchdog timer."),
    # extraction ABSTAIN (model mismatch; grounding FOUND -> must NOT upgrade)
    ("Which register resets the watchdog timer on the ACME-2?",
     "Register 0x1007 resets the watchdog timer."),
    # verification (grounding NOT_ATTEMPTED)
    ("Does the RUN LED indicate the device is powered?", "The RUN LED is green when powered."),
)


def test_decision_invariance_grounding_on_off():
    for query, text in _FIXTURES:
        result = _result(_candidate(text))
        base = _base(query, result)
        off = analyze_boundary_evidence(query, result, [result.candidates[0].document], "test", enrich_grounding=False)
        on = analyze_boundary_evidence(query, result, [result.candidates[0].document], "test", enrich_grounding=True, normalize=True)
        assert off.decision == base.decision, query
        assert off.reason == base.reason, query
        assert on.decision == base.decision, query
        assert on.reason == base.reason, query


# ---------------------------------------------------------------------------
# Directional invariants (§30)
# ---------------------------------------------------------------------------

def test_answer_not_downgraded_by_grounding_ambiguity():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate(
        "Register 0x1007 resets the watchdog timer.\nRegister 0x1008 resets the watchdog timer."
    )
    result = _result(candidate)
    base = _base(query, result)
    assert base.decision == "ANSWER"
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    # Grounding discovered two conflicting values, but must NOT flip ANSWER -> ABSTAIN.
    assert decision.decision == "ANSWER"
    assert decision.grounding_status == GroundingStatus.AMBIGUOUS.value
    assert decision.grounding_status != decision.decision  # metadata is not the decision


def test_abstain_not_upgraded_by_grounding_found():
    query = "Which register resets the watchdog timer on the ACME-2?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.", model="ACME-1")
    result = _result(candidate)
    base = _base(query, result)
    assert base.decision == "ABSTAIN"
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    # Grounding found "0x1007", but must NOT flip ABSTAIN -> ANSWER.
    assert decision.decision == "ABSTAIN"
    assert decision.grounding_status == GroundingStatus.FOUND.value
    assert decision.grounded_objects
    assert decision.supporting_candidate_ids == ()


def test_answer_unchanged_when_grounding_found_and_normalization_fails():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.")
    result = _result(candidate)
    base = _base(query, result)
    assert base.decision == "ANSWER"
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    assert decision.decision == "ANSWER"
    assert decision.grounding_status == GroundingStatus.FOUND.value


def test_answer_unchanged_when_grounding_absent():
    query = "Does the RUN LED indicate the device is powered?"
    candidate = _candidate("The RUN LED is green when powered.")
    result = _result(candidate)
    base = _base(query, result)
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    assert decision.decision == base.decision
    assert decision.grounding_status == GroundingStatus.NOT_ATTEMPTED.value  # polar query: no grounding attempted


# ---------------------------------------------------------------------------
# Payload serialization + provenance
# ---------------------------------------------------------------------------

def test_optional_grounded_object_serialized():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.")
    result = _result(candidate)
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    payload = decision.as_dict()
    assert payload["grounding"]["status"] == GroundingStatus.FOUND.value
    assert payload["grounding"]["objects"]
    assert decision.grounded_objects[0].surface_text


def test_source_provenance_retained():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.", chunk_id="chunk-wd-1")
    result = _result(candidate)
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    # Citation provenance.
    citation = decision.citations[0]
    assert citation.chunk_id == "chunk-wd-1"
    assert citation.document_id == "doc-1"
    assert citation.section == "S1"
    assert citation.page == 3
    # Grounded object provenance.
    obj = decision.grounded_objects[0]
    assert obj.source_candidate_ids
    assert obj.source_span is not None
    assert obj.source_span.chunk_id == "chunk-wd-1"
    assert obj.source_span.start >= 0 and obj.source_span.end > obj.source_span.start


# ---------------------------------------------------------------------------
# Downstream approved-candidate boundary (§32)
# ---------------------------------------------------------------------------

def test_answer_context_only_receives_approved_candidates():
    query = "Which register resets the watchdog timer on the ACME-2?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.", model="ACME-1")
    result = _result(candidate)
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    assert decision.decision == "ABSTAIN"
    context = build_answer_context(decision)
    assert context.evidence_decision == "ABSTAIN"
    assert context.approved_candidate_ids == ()
    assert context.citations == ()
    assert context.grounded_objects == ()


def test_answer_context_passes_approved_when_answered():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.", chunk_id="chunk-wd-1")
    result = _result(candidate)
    decision = analyze_boundary_evidence(query, result, [candidate.document], "test", enrich_grounding=True, normalize=True)
    assert decision.decision == "ANSWER"
    context = build_answer_context(decision)
    assert context.evidence_decision == "ANSWER"
    assert context.approved_candidate_ids == ("chunk-wd-1",)
    assert context.citations
    assert context.grounded_objects


# ---------------------------------------------------------------------------
# V3.25 exact-equivalence
# ---------------------------------------------------------------------------

def test_v325_exact_equivalence():
    for query, text in _FIXTURES:
        result = _result(_candidate(text))
        base = _base(query, result)
        decision = analyze_boundary_evidence(query, result, [result.candidates[0].document], "test", enrich_grounding=False)
        assert decision.decision == base.decision, query
        assert decision.reason == base.reason, query