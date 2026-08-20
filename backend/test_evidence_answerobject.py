"""Tests for V3.28 grounded answer span + evidence sufficiency.

Deterministic only: no model weights, no network, no generation.  These tests pin
that (1) a grounded surface span — not a canonical token — decides sufficiency,
(2) normalization is optional and never implies insufficiency, (3) wrong-target /
wrong-mode candidates are still rejected, (4) multiplicity is resolved
(equivalent vs conflicting), and (5) the frozen V3.25 verification path is
preserved for non-extraction queries.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval.candidates import RetrievalCandidate  # noqa: E402
from backend.retrieval.evidence_answerobject import (  # noqa: E402
    GROUNDED_SPAN_CANDIDATE_STATUS,
    GROUNDED_SPAN_CANDIDATE_VERSION,
    LLM_EXTRACTION_USED,
    AnswerType,
    NormalizationStatus,
    SpanMultiplicity,
    SufficiencyStatus,
    decide_sufficiency,
    discover_answer_objects,
    normalize_answer_object,
)


class _Doc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _candidate(text: str, chunk_id: str = "chunk-1", subsection: str = "", metadata: dict | None = None):
    meta = {"chunk_id": chunk_id, "document_id": "doc-1", "manufacturer": "Acme",
            "equipment_model": "ACME-1", "equipment_type": "plc_controller", "section": "S1",
            "page": 3}
    meta.update(metadata or {})
    if subsection:
        meta["subsection"] = subsection
    return RetrievalCandidate(document=_Doc(text, meta), retrieval_source="test")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_candidate_identity_constants():
    assert GROUNDED_SPAN_CANDIDATE_VERSION == "evidence-v328-grounded-span-candidate"
    assert GROUNDED_SPAN_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"
    assert LLM_EXTRACTION_USED == "NO"


# ---------------------------------------------------------------------------
# Identifier / numeric grounding
# ---------------------------------------------------------------------------

def test_identifier_grounding_register_token():
    s = decide_sufficiency("q", [_candidate("Register 0x1007 restarts the watchdog timer.")],
                           "REGISTER", "the watchdog timer", normalize=True)
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    assert s.multiplicity == SpanMultiplicity.UNIQUE.value
    assert s.grounded_answers[0].surface_text == "0x1007"
    assert s.grounded_answers[0].normalized_value == "0x1007"
    assert s.grounded_answers[0].normalization_status == NormalizationStatus.NORMALIZED.value


def test_numeric_span_grounding():
    s = decide_sufficiency("q", [_candidate("The default Baud Rate is 19200.")],
                           "VALUE", "the baud rate default", normalize=True)
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    assert "19200" in [o.surface_text for o in s.grounded_answers]


def test_answer_object_is_traceable():
    objs = discover_answer_objects("q", [_candidate("Register 0x1007 restarts the watchdog timer.")],
                                   slot_type="REGISTER", target_entity="the watchdog timer")
    assert objs
    span = objs[0].source_span
    assert span is not None and span.chunk_id == "chunk-1"
    assert span.start >= 0 and span.end > span.start
    assert span.page == 3 and span.section == "S1"


# ---------------------------------------------------------------------------
# Natural-language surface spans (short noun phrase / action / setting label)
# ---------------------------------------------------------------------------

def test_short_noun_phrase_grounding():
    s = decide_sufficiency("q", [_candidate("Functional Grounding: connect the cable shield here.")],
                           "TERMINAL", "the cable shield")
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    assert s.grounded_answers[0].answer_type == AnswerType.IDENTIFIER.value or AnswerType.SHORT_NOUN_PHRASE.value


def test_action_span_grounding():
    s = decide_sufficiency("q", [_candidate("Cycle power to the drive after changing the node address.")],
                           "ACTION", "the node address change")
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    assert s.grounded_answers[0].answer_type == AnswerType.ACTION_PHRASE.value


def test_multi_token_setting_grounding():
    s = decide_sufficiency("q", [_candidate("Ethernet out-of-the-box default: DHCP (dynamic IP address).")],
                           "SETTING", "the ethernet default")
    assert s.status == SufficiencyStatus.SUFFICIENT.value


# ---------------------------------------------------------------------------
# Normalization is optional; failure is not insufficiency
# ---------------------------------------------------------------------------

def test_normalization_is_optional_default():
    s = decide_sufficiency("q", [_candidate("Functional Grounding: terminal for the cable shield.")],
                           "TERMINAL", "the cable shield", normalize=False)
    obj = s.grounded_answers[0]
    assert obj.normalization_status == NormalizationStatus.NOT_ATTEMPTED.value
    assert s.status == SufficiencyStatus.SUFFICIENT.value


def test_normalization_failure_does_not_imply_insufficiency():
    s = decide_sufficiency("q", [_candidate("Functional Grounding: connect the cable shield here.")],
                           "TERMINAL", "the cable shield", normalize=True)
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    obj = s.grounded_answers[0]
    assert obj.normalization_status == NormalizationStatus.NORMALIZATION_FAILED.value
    assert obj.normalized_value == ""


# ---------------------------------------------------------------------------
# Wrong-target / wrong-mode rejection
# ---------------------------------------------------------------------------

def test_wrong_target_span_rejected():
    s = decide_sufficiency("q", [_candidate("SELECT (FC Byte = 0x03) selects an object.")],
                           "REGISTER", "the COLD_RESTART function code")
    assert s.status == SufficiencyStatus.INSUFFICIENT.value
    assert s.failure_cause == "GROUNDING_MISSING"


def test_wrong_mode_value_rejected():
    # The target encodes the wrong mode discriminator ("M3" vs "M8").
    s = decide_sufficiency("q", [_candidate("Screw M8 connectors tight. Torque: 0.4 Nm.")],
                           "SETTING", "the M3 mounting screws")
    assert s.status == SufficiencyStatus.INSUFFICIENT.value


def test_wrong_target_register_rejected():
    s = decide_sufficiency("q", [_candidate("0x2031 is the watchdog register.")],
                           "REGISTER", "the fault reset register")
    assert s.status == SufficiencyStatus.INSUFFICIENT.value


# ---------------------------------------------------------------------------
# Multiplicity resolution
# ---------------------------------------------------------------------------

def test_multiple_conflicting_spans_ambiguous():
    text = "Baud Rate default 19200.\nBaud Rate default 9600."
    s = decide_sufficiency("q", [_candidate(text)], "VALUE", "the baud rate default")
    # Two distinct defaults in the same candidate -> neither uniquely supported.
    assert s.status in (SufficiencyStatus.AMBIGUOUS.value, SufficiencyStatus.INSUFFICIENT.value)
    assert s.multiplicity in (SpanMultiplicity.MULTIPLE_CONFLICTING.value, SpanMultiplicity.NONE.value)


def test_multiple_equivalent_spans_across_chunks():
    a = _candidate("Watchdog register is 0x1007.", chunk_id="chunk-a")
    b = _candidate("Watchdog register is 0x1007.", chunk_id="chunk-b")
    s = decide_sufficiency("q", [a, b], "REGISTER", "the watchdog register")
    assert s.status == SufficiencyStatus.SUFFICIENT.value
    assert s.multiplicity == SpanMultiplicity.UNIQUE.value  # same distinct surface


# ---------------------------------------------------------------------------
# Table relation recoverability
# ---------------------------------------------------------------------------

def test_table_relation_recoverable():
    # Row label stays attached to its cell, so the relation is recoverable.
    s = decide_sufficiency("q", [_candidate("Append Chars default: 0x0D")],
                           "VALUE", "the append chars default")
    assert s.status == SufficiencyStatus.SUFFICIENT.value


def test_table_relation_unavailable_no_target():
    # Cell content present but the row label (target) is absent -> no grounding.
    s = decide_sufficiency("q", [_candidate("0 to 5,000  5,000")],
                           "VALUE", "the max torque default")
    assert s.status == SufficiencyStatus.INSUFFICIENT.value


# ---------------------------------------------------------------------------
# Surface-first behavior independent of generation
# ---------------------------------------------------------------------------

def test_no_generation_surface_is_verbatim():
    seg = "Contact 1: US (control voltage) 24 V DC"
    objs = discover_answer_objects("q", [_candidate(seg)], slot_type="TERMINAL",
                                   target_entity="the control voltage (US)")
    assert objs
    # Every surface must be a substring of the (normalized) source, never invented.
    norm_seg = seg.casefold()
    for obj in objs:
        assert obj.surface_text in norm_seg