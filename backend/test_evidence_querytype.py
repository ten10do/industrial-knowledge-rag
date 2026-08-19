"""Tests for the V3.27 query-type-aware evidence architecture.

Deterministic only: no model weights, no network, no generation.  These tests
pin the router, slot detection, association, extraction status mapping,
extract-then-verify mapping, and confirm the verification (rule + selective NLI)
path is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval.candidates import RetrievalCandidate  # noqa: E402
from backend.retrieval.evidence_querytype import (  # noqa: E402
    EvidenceQueryType,
    ExtractionMultiplicity,
    ExtractionRequirement,
    ExtractionSlotType,
    ExtractionStatus,
    JudgeDecision,
    LLM_EXTRACTION_USED,
    QUERY_TYPE_CANDIDATE_STATUS,
    QUERY_TYPE_CANDIDATE_VERSION,
    build_extraction_proposition,
    build_extraction_requirement,
    detect_extraction_slot,
    extract_slot_value,
    extraction_decision,
    route_query_type,
    verify_extracted_proposition,
)


class _Doc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _candidate(text: str, chunk_id: str = "chunk-1", subsection: str = "", metadata: dict | None = None):
    meta = {"chunk_id": chunk_id, "document_id": "doc-1", "manufacturer": "Acme",
            "equipment_model": "ACME-1", "equipment_type": "plc_controller", "section": "S1"}
    meta.update(metadata or {})
    if subsection:
        meta["subsection"] = subsection
    return RetrievalCandidate(document=_Doc(text, meta), retrieval_source="test")


def _fake_judge(decision: str, confidence: float = 0.9):
    class FakeJudge:
        model = object()

        def __init__(self):
            self._decision = decision
            self._confidence = confidence
            self.entailment_threshold = 0.5
            self.contradiction_threshold = 0.5
            self.unknown_floor = 0.33

        def predict_probs(self, premise, hypothesis):
            return (0.01, 0.98, 0.01) if decision == JudgeDecision.ENTAILS.value else (0.98, 0.01, 0.01)

        def decide_from_probs(self, probs):
            return self._decision, self._confidence

    return FakeJudge()


def _req(target: str, slot: str) -> ExtractionRequirement:
    return ExtractionRequirement(target_entity=target, slot_type=slot)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_routes_polar_proposition_to_verification():
    route = route_query_type("Does the switch support MODBUS TCP?")
    assert route.query_type == EvidenceQueryType.VERIFICATION.value
    assert route.confidence > 0.5


def test_routes_open_slot_to_extraction():
    route = route_query_type("Which register restarts the watchdog timer?")
    assert route.query_type == EvidenceQueryType.EXTRACTION.value


def test_routes_default_value_lookup_to_extraction():
    route = route_query_type("What is the factory default IP address?")
    assert route.query_type == EvidenceQueryType.EXTRACTION.value


def test_routes_polar_with_slot_to_verification():
    # A complete proposition ("default subnet mask IS 192.168...") is verification.
    route = route_query_type("Is the default subnet mask 192.168.127.253?")
    assert route.query_type == EvidenceQueryType.VERIFICATION.value


def test_unroutable_is_unknown():
    route = route_query_type("Please continue.")
    assert route.query_type == EvidenceQueryType.UNKNOWN.value


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------

def test_detect_slot_which_register():
    assert detect_extraction_slot("Which register restarts the watchdog timer?") == ExtractionSlotType.REGISTER


def test_detect_slot_what_value():
    assert detect_extraction_slot("What is the maximum number of restart attempts?") == ExtractionSlotType.VALUE


def test_detect_slot_which_terminal():
    assert detect_extraction_slot("Which terminal is the +24 VDC supply?") == ExtractionSlotType.TERMINAL


def test_detect_slot_what_action():
    assert detect_extraction_slot("What action restarts the device?") == ExtractionSlotType.ACTION


def test_detect_slot_under_what_condition():
    assert detect_extraction_slot("Under what condition can the outputs be switched?") == ExtractionSlotType.CONDITION


def test_detect_slot_attribute_meaning():
    assert detect_extraction_slot("What does SUPPLY LOW indicate?") == ExtractionSlotType.ATTRIBUTE


# ---------------------------------------------------------------------------
# Extraction association
# ---------------------------------------------------------------------------

def test_extract_register_unique():
    candidate = _candidate("Register 0x1007 restarts the watchdog timer.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("watchdog timer", "REGISTER"))
    assert result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value
    assert result.status == ExtractionStatus.SUPPORTED.value
    assert result.value == "0x1007"


def test_extract_value_unique():
    candidate = _candidate("The factory default IP address is 192.168.127.253.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("factory default ip address", "VALUE"))
    assert result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value
    assert "192.168.127.253" in result.value


def test_extract_terminal_unique():
    candidate = _candidate("Terminal A1: + 24 VDC supply for the control system.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("control system supply", "TERMINAL"))
    assert result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value
    assert result.value.upper() in {"A1", "A1:", "A1:"}


def test_extract_action_unique():
    candidate = _candidate("Press the rotary knob for 3 to 8 seconds to restart the device.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("restart the device", "ACTION"))
    assert result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value
    assert "press" in result.value


def test_wrong_target_register_is_none():
    # The register belongs to a different parameter than the requested target.
    candidate = _candidate("Register 0x2041 formats the file system Flash.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("restart the watchdog", "REGISTER"))
    assert result.multiplicity == ExtractionMultiplicity.NONE_SUPPORTED.value
    assert result.status == ExtractionStatus.INSUFFICIENT.value


def test_wrong_mode_value_is_none():
    candidate = _candidate("In mode A the default gain is 100.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("mode B", "VALUE"))
    assert result.multiplicity == ExtractionMultiplicity.NONE_SUPPORTED.value


def test_wrong_action_is_none():
    candidate = _candidate("Press the knob for 3 to 8 seconds to reset the device.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("shut down the device", "ACTION"))
    assert result.multiplicity == ExtractionMultiplicity.NONE_SUPPORTED.value


def test_ambiguous_two_values():
    candidate = _candidate("Parameter X: 100\nParameter X: 200")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("parameter x", "VALUE"))
    assert result.multiplicity == ExtractionMultiplicity.AMBIGUOUS.value
    assert result.status == ExtractionStatus.AMBIGUOUS.value


def test_none_extraction_abstains():
    candidate = _candidate("This chunk has no relevant value.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("unrelated target", "VALUE"))
    assert result.multiplicity == ExtractionMultiplicity.NONE_SUPPORTED.value
    decision, _ = extraction_decision(result)
    assert decision == "ABSTAIN"


def test_unique_extraction_answers():
    candidate = _candidate("The default SNMP community string is Public.")
    result = extract_slot_value("q", [candidate], [], None, extraction_requirement=_req("snmp community string", "SETTING"))
    assert result.multiplicity == ExtractionMultiplicity.UNIQUE_SUPPORTED.value
    decision, _ = extraction_decision(result)
    assert decision == "ANSWER"


# ---------------------------------------------------------------------------
# Extract-then-verify mapping
# ---------------------------------------------------------------------------

def test_extract_then_verify_entails():
    judge = _fake_judge(JudgeDecision.ENTAILS.value)
    decision, confidence = verify_extracted_proposition("Register 0x1007 restarts the watchdog.", "0x1007 is the register for the watchdog", judge)
    assert decision == JudgeDecision.ENTAILS.value
    assert confidence > 0.5


def test_extract_then_verify_contradiction():
    judge = _fake_judge(JudgeDecision.CONTRADICTS.value)
    decision, _ = verify_extracted_proposition("Register 0x1007 restarts the watchdog.", "0x1005 is the register for the watchdog", judge)
    assert decision == JudgeDecision.CONTRADICTS.value


def test_proposition_encodes_direction():
    proposition = build_extraction_proposition("the watchdog timer", "REGISTER", "0x1007")
    assert "0x1007" in proposition and "watchdog timer" in proposition


# ---------------------------------------------------------------------------
# Verification path preserved
# ---------------------------------------------------------------------------

def test_extraction_requirement_builds_target():
    requirement = build_extraction_requirement("Which register restarts the watchdog timer?")
    assert requirement.slot_type == ExtractionSlotType.REGISTER.value
    assert "watchdog" in requirement.target_entity


def test_candidate_constants():
    assert QUERY_TYPE_CANDIDATE_VERSION == "evidence-v327-querytype-candidate"
    assert QUERY_TYPE_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"
    assert LLM_EXTRACTION_USED == "NO"