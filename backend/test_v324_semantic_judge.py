"""Public tests for the V3.24 semantic-judge contract and ambiguity router.

These cover the abstract relation taxonomy and the judge decision mapping
without reproducing any private DEV query or annotation text.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.retrieval.evidence_contract import TypedRequirementItem
from backend.retrieval.semantic_judge import (
    AmbiguityType,
    JudgeDecision,
    RuleOnlyJudge,
    SemanticJudgeResult,
    build_judge_prompt,
    parse_judge_response,
    resolve_decision,
    route_ambiguity,
    validate_judge_schema,
)


def _item(kind, value, criticality="CRITICAL", mode="NORMALIZED"):
    return TypedRequirementItem(kind=kind, value=value, criticality=criticality, match_mode=mode)


def _req(*items):
    return SimpleNamespace(items=items)


def _judge(decision, confidence=0.9, **flags):
    return SemanticJudgeResult(decision=decision, confidence=confidence, reason_code="synthetic", **flags)


# ---------------------------------------------------------------------------
# Ambiguity router: trigger taxonomy
# ---------------------------------------------------------------------------

def test_router_role_reversal_triggers():
    ambiguity = route_ambiguity(
        "Does address 247 broadcast to all slaves while address 0 targets one?",
        _req(_item("value", "247"), _item("value", "0"), _item("protocol", "bus")),
    )
    assert ambiguity == AmbiguityType.ROLE_AMBIGUITY


def test_router_role_reversal_qualifier_triggers():
    # feature<->model role swap (ERS3/ERS4 shaped, expressed abstractly)
    ambiguity = route_ambiguity(
        "Do all X3 units provide timed braking while X4 units provide only plain stop?",
        _req(_item("qualifier", "X3"), _item("qualifier", "X4"), _item("qualifier", "braking")),
    )
    assert ambiguity == AmbiguityType.ROLE_AMBIGUITY


def test_router_predicate_reversal_triggers():
    ambiguity = route_ambiguity(
        "Does unit A precede unit B rather than unit B preceding unit A?",
        _req(_item("attribute", "order")),
    )
    assert ambiguity == AmbiguityType.PREDICATE_AMBIGUITY


def test_router_condition_mismatch_triggers():
    ambiguity = route_ambiguity(
        "Is scalar mode the default and required whenever current exceeds the limit?",
        _req(_item("attribute", "default_value"), _item("value_kind", "default")),
    )
    assert ambiguity == AmbiguityType.CONDITION_AMBIGUITY


def test_router_action_mismatch_triggers():
    ambiguity = route_ambiguity(
        "Does recovery first require deleting the configuration rather than restarting the unit?",
        _req(_item("attribute", "status")),
    )
    assert ambiguity == AmbiguityType.ACTION_AMBIGUITY


# ---------------------------------------------------------------------------
# Ambiguity router: non-trigger guards (protect stable slices)
# ---------------------------------------------------------------------------

def test_router_clear_supported_open_question_does_not_trigger():
    assert route_ambiguity(
        "Which parameter selects the reset behaviour?", _req(_item("attribute", "selection"))
    ) is None


def test_router_clear_unsupported_polar_value_check_does_not_trigger():
    # A plain wrong-value polar lookup is handled by rule value association, not relation reasoning.
    assert route_ambiguity(
        "Is parameter P30 default 5 seconds?", _req(_item("value", "5"))
    ) is None


def test_router_stable_l4_positive_does_not_trigger():
    assert route_ambiguity(
        "Which parameters define restart attempts and the delay between attempts?",
        _req(_item("action", "restart"), _item("attribute", "delay")),
    ) is None


def test_router_stable_semantic_positive_does_not_trigger():
    assert route_ambiguity(
        "What channel mode is displayed when Class B is enabled?", _req(_item("attribute", "mode"))
    ) is None


# ---------------------------------------------------------------------------
# Decision mapping
# ---------------------------------------------------------------------------

def test_entails_keeps_rule_answer():
    assert resolve_decision("ANSWER", _judge(JudgeDecision.ENTAILS.value)) == "ANSWER"
    assert resolve_decision("ABSTAIN", _judge(JudgeDecision.ENTAILS.value)) == "ABSTAIN"


def test_contradicts_maps_to_abstain():
    assert resolve_decision("ANSWER", _judge(JudgeDecision.CONTRADICTS.value)) == "ABSTAIN"


def test_insufficient_maps_to_abstain():
    assert resolve_decision("ANSWER", _judge(JudgeDecision.INSUFFICIENT.value)) == "ABSTAIN"


def test_unknown_conservatively_falls_back_to_rule():
    unknown = _judge(JudgeDecision.UNKNOWN.value, confidence=0.0)
    assert resolve_decision("ANSWER", unknown) == "ANSWER"
    assert resolve_decision("ABSTAIN", unknown) == "ABSTAIN"


# ---------------------------------------------------------------------------
# Schema validation and structured LLM adapter output
# ---------------------------------------------------------------------------

def test_judge_schema_valid():
    assert validate_judge_schema(_judge(JudgeDecision.CONTRADICTS.value, subject_match=False, predicate_match=True)) == []


def test_judge_schema_rejects_invalid_decision_and_confidence():
    violations = validate_judge_schema(SemanticJudgeResult(decision="ANSWER", confidence=1.7, reason_code="x"))
    assert any("INVALID_DECISION" in violation for violation in violations)
    assert any("INVALID_CONFIDENCE" in violation for violation in violations)


def test_judge_schema_rejects_missing_reason_and_non_bool_match():
    violations = validate_judge_schema(SemanticJudgeResult(decision=JudgeDecision.UNKNOWN.value, confidence=0.5, reason_code="", subject_match="yes"))
    assert any("MISSING_REASON_CODE" in violation for violation in violations)
    assert any("NON_BOOL_subject_match" in violation for violation in violations)


def test_llm_judge_parse_roundtrip():
    raw = ('{"decision": "CONTRADICTS", "confidence": 0.97, "reason_code": "ROLE_REVERSAL", '
           '"subject_match": true, "predicate_match": false, '
           '"object_match": true, "condition_match": false}')
    result = parse_judge_response(raw)
    assert result.decision == JudgeDecision.CONTRADICTS.value
    assert result.confidence == 0.97
    assert result.subject_match is True
    assert result.object_match is True


def test_llm_judge_parse_rejects_malformed_output():
    with pytest.raises(ValueError, match="JUDGE_RESPONSE_NO_JSON"):
        parse_judge_response("not an object at all")


def test_judge_prompt_contains_only_query_requirements_evidence():
    prompt = build_judge_prompt(
        "Does X rather than Y?", _req(_item("action", "restart")), ["evidence chunk text"]
    )
    assert "Does X rather than Y?" in prompt
    assert "restart" in prompt
    assert "evidence chunk text" in prompt
    assert "ENTAILS" in prompt


def test_rule_only_judge_returns_unknown():
    result = RuleOnlyJudge().judge("any query?", _req())
    assert result.decision == JudgeDecision.UNKNOWN.value
    assert result.reason_code == "JUDGE_DISABLED"