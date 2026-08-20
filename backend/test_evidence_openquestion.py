"""Tests for V3.30 open-question evidence sufficiency.

Pins the existential relation contract:

* ``relation supported == target + requested relation verb + a locally-bound
  object`` -> ANSWER, without committing to the object's final wording.
* Wrong attribute / wrong target / wrong fault discriminators are rejected.
* Grounding and normalization have ``NONE`` decision authority (V3.29 boundary
  unchanged); the frozen V3.25 rule + selective NLI polar path is left untouched.

Deterministic only: no model weights, no network, no generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.evidence_openquestion import (  # noqa: E402
    GENERATION_USED,
    GROUNDING_DECISION_AUTHORITY,
    NORMALIZATION_DECISION_AUTHORITY,
    OPEN_QUESTION_SUFFICIENCY_DEFAULT,
    OPEN_SUFFICIENCY_CANDIDATE_STATUS,
    OPEN_SUFFICIENCY_CANDIDATE_VERSION,
    OpenQuestionRequirement,
    OpenSufficiencyStatus,
    RelationType,
    analyze_open_question_evidence,
    analyze_open_sufficiency,
    build_open_requirement,
    detect_relation,
    _sub_module_identifiers,
)


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


def _suff(query, candidates, requirement):
    return analyze_open_sufficiency(query, list(candidates), requirement=requirement)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    assert OPEN_SUFFICIENCY_CANDIDATE_VERSION == "evidence-v330-open-sufficiency-candidate"
    assert OPEN_SUFFICIENCY_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"
    assert OPEN_QUESTION_SUFFICIENCY_DEFAULT == "OFF"
    assert GROUNDING_DECISION_AUTHORITY == "NONE"
    assert NORMALIZATION_DECISION_AUTHORITY == "NONE"
    assert GENERATION_USED == "NO"


# ---------------------------------------------------------------------------
# Relation detection + requirement construction
# ---------------------------------------------------------------------------

def test_detect_relation_maps_verbs():
    assert detect_relation("What is the default Baud Rate of P100?", "") == RelationType.HAS_DEFAULT_VALUE.value
    assert detect_relation("What is the maximum value of P100?", "") == RelationType.HAS_RANGE.value
    assert detect_relation("How many ports does the EL6652 provide?", "") == RelationType.HAS_VALUE.value
    assert detect_relation("What component must be connected near the load?", "") == RelationType.REQUIRES_ACTION.value


def test_sub_module_identifiers():
    assert _sub_module_identifiers("default value of the DI581-S module") == ("di581-s",)
    assert _sub_module_identifiers("older terminals before 2007") == ()
    assert _sub_module_identifiers("Safe-Operational state") == ()


# ---------------------------------------------------------------------------
# Existential relation sufficiency (positive)
# ---------------------------------------------------------------------------

def test_relation_default_value_supported():
    req = OpenQuestionRequirement(target="baud rate p100", relation=RelationType.HAS_DEFAULT_VALUE.value,
                                  requested_slot_type="VALUE")
    result = _suff("What is the default Baud Rate of P100?",
                   [_candidate("P100 default baud rate is 9600. The unit is baud.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


def test_relation_register_supported():
    req = OpenQuestionRequirement(target="reset watchdog", relation=RelationType.HAS_IDENTIFIER.value,
                                  requested_slot_type="REGISTER")
    result = _suff("Which register resets the watchdog timer?",
                   [_candidate("Register 0x1007 resets the watchdog timer.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


def test_relation_terminal_supported():
    req = OpenQuestionRequirement(target="control voltage", relation=RelationType.USES_TERMINAL.value,
                                  requested_slot_type="TERMINAL")
    result = _suff("Which terminal carries the control voltage?",
                   [_candidate("Contact 1: control voltage input terminal.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


def test_relation_action_short_phrase_supported():
    # "a diode" is a natural-language object; sufficiency does NOT require
    # normalizing it into a structured value.
    req = OpenQuestionRequirement(target="inductive load", relation=RelationType.REQUIRES_ACTION.value,
                                  requested_slot_type="ACTION")
    result = _suff("What must be connected near an inductive load?",
                   [_candidate("A diode is connected near the inductive load.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


# ---------------------------------------------------------------------------
# Existential relation sufficiency (near-miss negatives)
# ---------------------------------------------------------------------------

def test_relation_wrong_attribute_max_vs_default_rejected():
    # "maximum" is not "default": the requested relation verb is absent.
    req = OpenQuestionRequirement(target="torque p100", relation=RelationType.HAS_DEFAULT_VALUE.value,
                                  requested_slot_type="VALUE")
    result = _suff("What is the default value of P100?",
                   [_candidate("P100 maximum value is 500.")], req)
    assert result.status == OpenSufficiencyStatus.INSUFFICIENT.value


def test_relation_wrong_target_register_rejected():
    # SELECT vs READ function code: the discriminator target is absent.
    req = OpenQuestionRequirement(target="select function code", relation=RelationType.HAS_IDENTIFIER.value,
                                  requested_slot_type="REGISTER")
    result = _suff("Which register encodes the SELECT function code?",
                   [_candidate("READ (FC Byte = 0x01) selectable function code.")], req)
    assert result.status == OpenSufficiencyStatus.INSUFFICIENT.value


def test_relation_wrong_function_terminal_rejected():
    # "control voltage" vs "EtherCAT Tx+": the target discriminator is absent.
    req = OpenQuestionRequirement(target="control voltage connector", relation=RelationType.USES_TERMINAL.value,
                                  requested_slot_type="TERMINAL")
    result = _suff("Which core wire is the control voltage?",
                   [_candidate("EtherCAT Tx+ signal core.")], req)
    assert result.status == OpenSufficiencyStatus.INSUFFICIENT.value


# ---------------------------------------------------------------------------
# Grounding / normalization have no decision authority
# ---------------------------------------------------------------------------

def test_grounding_and_normalization_never_decision_authority():
    # Relation supported but no grounding/normalization available -> still ANSWER.
    req = OpenQuestionRequirement(target="baud rate p100", relation=RelationType.HAS_DEFAULT_VALUE.value,
                                  requested_slot_type="VALUE")
    result = _suff("What is the default Baud Rate of P100?",
                   [_candidate("P100 default baud rate is USE THE TABLE.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


def test_supported_relation_is_not_downgraded_by_ambiguity():
    # Multiple objects is still existential sufficiency (object selection is
    # downstream's job, not Evidence eligibility).
    req = OpenQuestionRequirement(target="function code", relation=RelationType.HAS_IDENTIFIER.value,
                                  requested_slot_type="REGISTER")
    result = _suff("Which register encodes the read function code?",
                   [_candidate("READ 0x01 / WRITE 0x02 / SELECT 0x03 function codes.")], req)
    assert result.status == OpenSufficiencyStatus.SUPPORTED.value


# ---------------------------------------------------------------------------
# Decision layer: polar verification untouched, sub-module gate, no downgrade
# ---------------------------------------------------------------------------

def test_verification_decision_untouched():
    query = "Does the RUN LED indicate the device is powered?"
    result = _result(_candidate("The RUN LED is green when the device is powered."))
    docs = [candidate.document for candidate in result.candidates]
    decision = analyze_open_question_evidence(query, result, docs, "test")
    assert decision.query_type == "VERIFICATION"
    assert decision.open_sufficiency is None


def test_open_decision_never_downgrades_answer():
    # An already-ANSWER extraction query is untouched by the open path.
    query = "Which register resets the watchdog timer?"
    result = _result(_candidate("Register 0x1007 resets the watchdog timer."))
    docs = [candidate.document for candidate in result.candidates]
    decision = analyze_open_question_evidence(query, result, docs, "test")
    assert decision.decision == "ANSWER"


def test_model_mismatch_without_sub_module_identifier_not_relaxed():
    # A model-mismatch abstention backed only by a phrase (no sub-module
    # identifier) is never relaxed into ANSWER.
    query = "Which switch address deletes the boot project from Flash memory?"
    result = _result(_candidate("A switch address is used at start-up."))
    docs = [candidate.document for candidate in result.candidates]
    decision = analyze_open_question_evidence(query, result, docs, "test")
    assert decision.decision == "ABSTAIN"