"""Integration tests for the explicit mixed Evidence candidate orchestrator (V3.32).

Deterministic only: no model weights, no network, no generation.  Verifies

* dispatch (VERIFICATION / OPEN / FALLBACK),
* that a router-triggered verification actually calls the local-NLI judge,
* that the open path actually invokes V3.30 open sufficiency,
* hard safety/identity abstention is never relaxed,
* grounding authority NONE, production defaults OFF, activation explicit,
* capability preflight rejects an unwired judge,
* diagnostic path trace is complete,
* exact-equivalence with the frozen V3.25 / V3.30 entrypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.evidence_boundary import GROUNDING_ENRICHMENT_DEFAULT  # noqa: E402
from backend.retrieval.evidence_mixed import (  # noqa: E402
    GROUNDING_DECISION_AUTHORITY,
    MIXED_CANDIDATE_STATUS,
    MIXED_CANDIDATE_VERSION,
    NORMALIZATION_DECISION_AUTHORITY,
    analyze_mixed_evidence,
    assert_mixed_candidate_capabilities,
)
from backend.retrieval.evidence_openquestion import (  # noqa: E402
    OPEN_QUESTION_SUFFICIENCY_DEFAULT,
    analyze_open_question_evidence,
    build_open_requirement,
)
from backend.retrieval.evidence_querytype import (  # noqa: E402
    EvidenceQueryType,
    QueryTypeRoute,
    analyze_querytype_evidence,
)
import backend.retrieval.evidence_mixed as em  # noqa: E402
from backend.retrieval.semantic_judge import SEMANTIC_JUDGE_DEFAULT  # noqa: E402


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


def _docs(*candidates) -> list:
    return [candidate.document for candidate in candidates]


class _FakeNliJudge:
    """Deterministic local-NLI double exposing the duck-typed interface the
    frozen verification path actually calls (.model / .predict_probs /
    .decide_from_probs)."""

    def __init__(self, decision: str = "CONTRADICTS", probs=(0.9, 0.05, 0.05)):
        self.model = object()
        self.decision = decision
        self.probs = probs
        self.calls = 0

    def predict_probs(self, premise, hypothesis):
        self.calls += 1
        return self.probs

    def decide_from_probs(self, probs):
        return (self.decision, max(float(p) for p in probs))


class _NonForwardingJudge:
    """Replica of the V3.31 sealed runner's recording wrapper (judge()-only)."""

    def __init__(self, judge):
        self._judge = judge

    def judge(self, query, requirements, claims=None, scope=None):
        return self._judge.judge(query, requirements, claims, scope)


# ---------------------------------------------------------------------------
# Constants / composition / defaults
# ---------------------------------------------------------------------------

def test_constants_and_defaults():
    assert MIXED_CANDIDATE_VERSION == "evidence-v332-integrated-candidate"
    assert MIXED_CANDIDATE_STATUS == "EXPERIMENTAL_CANDIDATE"
    assert GROUNDING_DECISION_AUTHORITY == "NONE"
    assert NORMALIZATION_DECISION_AUTHORITY == "NONE"
    assert SEMANTIC_JUDGE_DEFAULT == "OFF"
    assert OPEN_QUESTION_SUFFICIENCY_DEFAULT == "OFF"
    assert GROUNDING_ENRICHMENT_DEFAULT == "OFF"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_verification_dispatch():
    query = "Does unit A precede unit B rather than unit B preceding unit A?"
    candidate = _candidate("Unit A precedes unit B.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "VERIFICATION"


def test_open_dispatch():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "OPEN"


def test_fallback_dispatch(monkeypatch):
    query = "Please continue."
    candidate = _candidate("generic safety text.")
    monkeypatch.setattr(
        em, "route_query_type",
        lambda q, r: QueryTypeRoute(EvidenceQueryType.UNKNOWN.value, 0.0, "no_slot_no_polar"),
    )
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "FALLBACK"
    # Conservative frozen baseline still produces a deterministic decision.
    assert decision.decision in ("ANSWER", "ABSTAIN")


# ---------------------------------------------------------------------------
# Verification path + NLI wiring
# ---------------------------------------------------------------------------

def test_verification_clear_supported_no_nli():
    query = "Does the RUN LED indicate the device is powered?"
    candidate = _candidate("The RUN LED is green when the device is powered.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "VERIFICATION"
    assert decision.nli_router_triggered is False
    assert decision.nli_decision is None
    assert decision.final_decision_source == "RULE"


def test_verification_hard_reversal_triggers_nli():
    query = "Does unit A precede unit B rather than unit B preceding unit A?"
    candidate = _candidate("Unit A precedes unit B.")
    judge = _FakeNliJudge(decision="CONTRADICTS")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test", judge=judge)
    assert decision.query_path == "VERIFICATION"
    assert decision.nli_router_triggered is True
    assert judge.calls == 1
    assert decision.nli_decision == "CONTRADICTS"
    assert decision.final_decision_source == "NLI"
    assert decision.decision == "ABSTAIN"


def test_verification_clear_unsupported_no_nli():
    query = "Please continue."
    candidate = _candidate("generic safety text.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "VERIFICATION"
    assert decision.decision == "ABSTAIN"
    assert decision.nli_router_triggered is False
    assert decision.final_decision_source == "RULE"


# ---------------------------------------------------------------------------
# Open path + V3.30 open sufficiency
# ---------------------------------------------------------------------------

def test_open_supported_through_sufficiency():
    query = "What is the default value of the DI581-S module?"
    candidate = _candidate("The DI581-S module default value is 500.", model="AC500-S")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "OPEN"
    assert decision.decision == "ANSWER"
    assert decision.reason == "OPEN_RELATION_SUPPORTED"
    assert decision.open_sufficiency_invoked is True
    assert decision.open_sufficiency_status == "SUPPORTED"
    assert decision.final_decision_source == "OPEN_SUFFICIENCY"


def test_open_hard_safety_gate_not_relaxed():
    query = "What is the default IP address of the unit?"
    decision = analyze_mixed_evidence(query, _result(), [], "test")
    assert decision.query_path == "OPEN"
    assert decision.decision == "ABSTAIN"
    assert decision.base_rule_reason == "NO_CANDIDATE"
    assert decision.open_sufficiency_invoked is False
    assert decision.final_decision_source == "HARD_GATE"


def test_open_near_miss_negative_rejected():
    query = "What is the default value of P100?"
    candidate = _candidate("P100 maximum value is 500.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.query_path == "OPEN"
    assert decision.decision == "ABSTAIN"
    assert decision.open_sufficiency_invoked is True
    assert decision.open_sufficiency_status == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Grounding / activation / preflight / diagnostics
# ---------------------------------------------------------------------------

def test_grounding_authority_none():
    query = "Which register resets the watchdog timer?"
    candidate = _candidate("Register 0x1007 resets the watchdog timer.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    assert decision.grounding_status == "NONE"


def test_capability_preflight_accepts_wired_judge():
    judge = _FakeNliJudge()
    assert assert_mixed_candidate_capabilities(judge) == []


def test_capability_preflight_rejects_unwired_judge():
    wrapped = _NonForwardingJudge(_FakeNliJudge())
    violations = assert_mixed_candidate_capabilities(wrapped)
    assert "NLI_MODEL_NOT_LOADED" in violations
    assert "NLI_PREDICT_PROBS_NOT_CALLABLE" in violations
    assert "NLI_DECIDE_FROM_PROBS_NOT_CALLABLE" in violations


def test_capability_preflight_rejects_missing_judge():
    violations = assert_mixed_candidate_capabilities(None)
    assert "NLI_JUDGE_NOT_PROVIDED" in violations


def test_diagnostic_path_trace_complete():
    query = "Does the RUN LED indicate the device is powered?"
    candidate = _candidate("The RUN LED is green when the device is powered.")
    decision = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    payload = decision.as_dict()
    for key in (
        "query_path", "decision", "reason", "base_rule_decision", "base_rule_reason",
        "nli_router_triggered", "nli_decision", "open_sufficiency_invoked",
        "open_sufficiency_status", "final_decision_source", "grounding_status",
    ):
        assert key in payload


# ---------------------------------------------------------------------------
# Historical exact-equivalence
# ---------------------------------------------------------------------------

def test_historical_v325_exact_equivalence_verification():
    query = "Does unit A precede unit B rather than unit B preceding unit A?"
    candidate = _candidate("Unit A precedes unit B.")
    judge = _FakeNliJudge(decision="CONTRADICTS")
    mixed = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test", judge=judge)
    direct = analyze_querytype_evidence(
        query, _result(candidate), _docs(candidate), "test", mode="VERIFIER_ONLY", judge=judge,
    )
    assert mixed.decision == direct.decision
    assert mixed.reason == direct.reason
    # base_rule_decision == rule-only (judge=None) decision
    rule = analyze_querytype_evidence(
        query, _result(candidate), _docs(candidate), "test", mode="VERIFIER_ONLY", judge=None,
    )
    assert mixed.base_rule_decision == rule.decision


def test_historical_v330_exact_equivalence_open():
    query = "What is the default value of the DI581-S module?"
    candidate = _candidate("The DI581-S module default value is 500.", model="AC500-S")
    mixed = analyze_mixed_evidence(query, _result(candidate), _docs(candidate), "test")
    direct = analyze_open_question_evidence(
        query, _result(candidate), _docs(candidate), "test",
        requirement=build_open_requirement(query), apply_open_sufficiency=True,
    )
    assert mixed.decision == direct.decision
    assert mixed.reason == direct.reason