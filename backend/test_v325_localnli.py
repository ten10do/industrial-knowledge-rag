"""Public tests for the V3.25 local-NLI judge adapter (fake model, no weights)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.retrieval.evidence_contract import TypedRequirementItem
from backend.retrieval.semantic_judge import (
    JudgeDecision,
    SemanticJudgeResult,
    resolve_decision,
    route_ambiguity,
)
from backend.retrieval.semantic_judge_localnli import (
    CONTRADICTION_INDEX,
    ENTAILMENT_INDEX,
    NEUTRAL_INDEX,
    LocalNliJudge,
    build_hypothesis,
    build_premise,
    to_declarative,
)


def _item(kind, value, criticality="CRITICAL", mode="NORMALIZED"):
    return TypedRequirementItem(kind=kind, value=value, criticality=criticality, match_mode=mode)


def _req(*items):
    return SimpleNamespace(items=items)


def _claim(text):
    return SimpleNamespace(text=text)


class FakeModel:
    """CrossEncoder look-alike returning fixed 3-way logits per pair."""

    def __init__(self, logits=(0.0, 0.0, 0.0)):
        self.logits = list(logits)
        self.calls = 0
        self.last_pairs = []

    def predict(self, pairs):
        self.calls += 1
        self.last_pairs = list(pairs)
        return [self.logits for _ in pairs]


def _judge_with_logits(logits, **kwargs):
    return LocalNliJudge(FakeModel(logits), **kwargs)


# ---------------------------------------------------------------------------
# Relation proposition construction
# ---------------------------------------------------------------------------

def test_to_declarative_strips_polar_prefix_and_question_mark():
    assert to_declarative("Does address 247 broadcast to all slaves?") == "Address 247 broadcast to all slaves"
    assert to_declarative("Is scalar mode the default?") == "Scalar mode the default"


def test_build_premise_joins_candidate_texts():
    premise = build_premise([_claim("slave address 1 to 247"), _claim("only the individual address")])
    assert premise == "slave address 1 to 247 only the individual address"


def test_build_hypothesis_is_declarative_query():
    assert build_hypothesis("Does A precede B rather than B preceding A?") == "A precede B rather than B preceding A"


# ---------------------------------------------------------------------------
# Label mapping + confidence
# ---------------------------------------------------------------------------

def test_entailment_maps_to_entails():
    result = _judge_with_logits([0.0, 10.0, 0.0]).judge("Does A support X?", _req(), [_claim("A supports X")])
    assert result.decision == JudgeDecision.ENTAILS.value
    assert result.confidence > 0.99


def test_contradiction_maps_to_contradicts():
    result = _judge_with_logits([10.0, 0.0, 0.0]).judge("Does A support X?", _req(), [_claim("B supports X")])
    assert result.decision == JudgeDecision.CONTRADICTS.value


def test_neutral_maps_to_insufficient():
    result = _judge_with_logits([0.0, 0.0, 10.0]).judge("Does A support X?", _req(), [_claim("A has model id 2")])
    assert result.decision == JudgeDecision.INSUFFICIENT.value


def test_probabilities_sum_to_one():
    # logits order = [contradiction, entailment, neutral]; entailment highest here
    probabilities = LocalNliJudge(FakeModel([1.0, 3.0, 2.0])).predict_probs("p", "h")
    assert abs(sum(probabilities) - 1.0) < 1e-3
    assert probabilities[ENTAILMENT_INDEX] > probabilities[NEUTRAL_INDEX] > probabilities[CONTRADICTION_INDEX]


# ---------------------------------------------------------------------------
# Thresholds + UNKNOWN
# ---------------------------------------------------------------------------

def test_entailment_below_threshold_becomes_unknown():
    # softmax([0, 2, 1]) -> entail ~0.67 but floor/threshold interplay
    judge = _judge_with_logits([0.0, 1.0, 1.0], entailment_threshold=0.9, contradiction_threshold=0.5, unknown_floor=0.2)
    result = judge.judge("Does A support X?", _req(), [_claim("x")])
    assert result.decision == JudgeDecision.UNKNOWN.value


def test_low_confidence_becomes_unknown():
    # nearly uniform logits -> argmax prob just above 1/3, below floor
    judge = _judge_with_logits([1.0, 1.1, 1.0], unknown_floor=0.4)
    result = judge.judge("Does A support X?", _req(), [_claim("x")])
    assert result.decision == JudgeDecision.UNKNOWN.value


def test_model_missing_returns_unknown_not_loaded():
    result = LocalNliJudge(None).judge("Does A support X?", _req(), [_claim("x")])
    assert result.decision == JudgeDecision.UNKNOWN.value
    assert result.reason_code == "NLI_MODEL_NOT_LOADED"


# ---------------------------------------------------------------------------
# Selective invocation + fallback parity with rule baseline
# ---------------------------------------------------------------------------

def test_router_non_trigger_does_not_invoke_model():
    model = FakeModel()
    judge = LocalNliJudge(model)
    ambiguity = route_ambiguity(
        "Which parameter selects the reset behaviour?", _req(_item("attribute", "selection"))
    )
    assert ambiguity is None  # non-trigger -> judge never called
    assert model.calls == 0

    triggered = route_ambiguity(
        "Does A broadcast while B targets one?", _req(_item("value", "247"), _item("value", "0"))
    )
    assert triggered is not None
    judge.judge("Does A broadcast while B targets one?", _req(), [_claim("x")])
    assert model.calls == 1


def test_unknown_fallback_keeps_rule_decision():
    unknown = SemanticJudgeResult(decision=JudgeDecision.UNKNOWN.value, confidence=0.0, reason_code="low")
    assert resolve_decision("ANSWER", unknown) == "ANSWER"
    assert resolve_decision("ABSTAIN", unknown) == "ABSTAIN"


def test_judge_output_passes_schema_validation():
    from backend.retrieval.semantic_judge import validate_judge_schema

    result = _judge_with_logits([0.0, 10.0, 0.0]).judge("Does A support X?", _req(), [_claim("A supports X")])
    assert validate_judge_schema(result) == []