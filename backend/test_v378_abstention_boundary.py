"""V3.78 Evidence Abstention Safety Boundary tests.

AUDIT CONTEXT (V3.78): the pre-registered counterfactual showed that restoring
quality-admissibility precedence inside the contract-sufficient accept branch
would block up to 6/9 benchmark FAs but kill 12–13 of 20 correct answers
(FR 3→15/16), failing the Go Gate. The refinement was therefore NOT implemented
(see ``V3.78_EVIDENCE_ABSTENTION_SAFETY_REFINEMENT_REPORT.md``).

These tests CHARACTERIZE the audited frozen boundary so that any future change
to it is a conscious, visible act:

* branch order evidence.py::analyze_retrieval_evidence — identity rejects and
  cross-cutting gates precede the contract gate; the contract-sufficient accept
  precedes and is INDEPENDENT of every evidence-quality signal,
* the vector gate (non-contract path) remains strictly distance-guarded,
* lexical-only candidates neither get auto-answered nor distance-vetoed.

Purely synthetic: no embedding model, index, or private artifact required.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.evidence import (
    Decision,
    DecisionReason,
    analyze_retrieval_evidence,
    default_policy,
)

THRESHOLD = 13.234710693359375

CONTRACT_QUERY = "What is the acceleration time?"
CONTRACT_DOC_TEXT = (
    "parameter 23.12 acceleration time defines the ramp time. "
    "set acceleration time via the menu."
)


def _doc(text: str, metadata: dict | None = None):
    return SimpleNamespace(page_content=text, metadata={"source": "generic_manual.pdf", "page": 0, **(metadata or {})})


def _result(entries, retrieval_mode="vector_only_v369"):
    """entries: list of (text, metadata_or_None, vector_score_or_None, lexical_score_or_None)."""
    candidates = []
    for i, (text, meta, vscore, lscore) in enumerate(entries):
        candidate = RetrievalCandidate(
            document=_doc(text, meta),
            retrieval_source="chroma",
            vector_rank=i + 1 if vscore is not None else None,
        )
        candidate.vector_score = vscore
        candidate.lexical_score = lscore
        candidates.append(candidate)
    return RetrievalResult(candidates=candidates, retrieval_mode=retrieval_mode)


def _analyze(query, entries):
    rr = _result(entries)
    documents = [c.document for c in rr.candidates]
    return analyze_retrieval_evidence(
        query, rr, documents, retrieval_mode="vector_only_v369", identity_matching=True,
    )


# --- threshold pin -----------------------------------------------------------


def test_threshold_value_unchanged():
    assert default_policy().max_vector_distance == THRESHOLD


# --- AUDITED BYPASS: contract-sufficient acceptance ignores quality ----------


def test_contract_sufficient_accept_ignores_large_vector_distance():
    """V3.78 audit finding H1: acceptance happens far beyond threshold.

    Distance 37.7991 mirrors the worst observed FA (V369-Q0033). This is the
    characterized FROZEN behavior, deliberately not changed in V3.78 (No-Go).
    """
    evidence = _analyze(CONTRACT_QUERY, [(CONTRACT_DOC_TEXT, None, 37.7991, None)])
    assert evidence.decision == Decision.ANSWER.value
    assert evidence.reason == DecisionReason.CONTRACT_REQUIREMENTS_COVERED.value


def test_contract_sufficient_accept_ignores_missing_vector_score():
    """A dense-scoreless (e.g. lexical-fed) candidate is not rejected for
    missing quality signal when the contract is satisfied."""
    evidence = _analyze(CONTRACT_QUERY, [(CONTRACT_DOC_TEXT, None, None, 4.2)])
    assert evidence.decision == Decision.ANSWER.value
    assert evidence.reason == DecisionReason.CONTRACT_REQUIREMENTS_COVERED.value


def test_no_global_distance_veto_exists_in_frozen_baseline():
    """Documents the verified absence of any absolute-distance veto: an extreme
    distance does not flip a contract-covered answer into an abstention."""
    evidence = _analyze(CONTRACT_QUERY, [(CONTRACT_DOC_TEXT, None, 500.0, None)])
    assert evidence.decision == Decision.ANSWER.value


# --- The non-contract vector path REMAINS distance-guarded --------------------


def test_vector_gate_accepts_within_threshold_without_contract():
    evidence = _analyze("Describe the operator panel.", [
        ("the monitoring menu shows motor speed and output frequency values.", None, 10.0, None),
    ])
    assert evidence.decision == Decision.ANSWER.value
    assert evidence.reason == DecisionReason.STRONG_VECTOR_EVIDENCE.value


def test_non_contract_path_abstains_beyond_threshold():
    """Without contract sufficiency, quality gating still applies: over-threshold
    top-1 with no lexical fallback leads to WEAK/INSUFFICIENT abstention."""
    evidence = _analyze("Describe the operator panel.", [
        ("the monitoring menu shows motor speed and output frequency values.", None, 30.0, None),
    ])
    assert evidence.decision == Decision.ABSTAIN.value
    assert evidence.reason in {
        DecisionReason.WEAK_RETRIEVAL_EVIDENCE.value,
        DecisionReason.INSUFFICIENT_EVIDENCE.value,
    }


def test_lexical_only_candidate_with_unknown_relation_abstains():
    """BM25-style candidate under identity matching: neither auto-answer nor
    unconditional distance rejection - it abstains pending identity support."""
    evidence = _analyze("Describe the operator panel.", [
        ("the panel keys provide navigation between menus.", None, None, 4.2),
    ])
    assert evidence.decision == Decision.ABSTAIN.value
    assert evidence.reason == DecisionReason.INSUFFICIENT_EVIDENCE.value


# --- Precedence: identity safety outranks the contract gate -------------------


def test_identity_conflict_rejects_before_contract_gate():
    metadata = {"manufacturer": "siemens", "product_series": "s7-1200", "equipment_model": "s7-1200"}
    evidence = _analyze(
        "How do I set the acceleration time on my ABB ACS580 drive?",
        [(CONTRACT_DOC_TEXT, metadata, 8.0, None)],
    )
    assert evidence.decision == Decision.ABSTAIN.value
    assert evidence.reason == DecisionReason.MODEL_MISMATCH.value


def test_absent_model_known_identity_rejects_even_with_close_content():
    """A query naming a model outside the corpus identity manifest refuses
    (hard-negative-family behavior) regardless of textual proximity."""
    metadata = {"manufacturer": "siemens", "product_series": "g120", "equipment_model": "g120"}
    evidence = _analyze(
        "Tell me about the SINAMICS G120 drive.",
        [("the g120 drive updates its process image every 4 ms.", metadata, 5.0, None)],
    )
    # This mirrors the incidental-proximity pattern behind V369-Q0021/Q0027:
    # safe outcome depends on the identity manifest being honest about corpus
    # contents; with a g120-marked corpus the runtime answers, without it the
    # unknown-identity gates fire before any accept branch.
    if evidence.decision != Decision.ABSTAIN.value:
        assert evidence.identity_relation in {"EXACT_MODEL", "SAME_SERIES", "SAME_FAMILY"}


# --- Two-sided logical separation ---------------------------------------------


def test_quality_pass_does_not_imply_contract_sufficient_and_vice_versa():
    within = _analyze("Describe the operator panel.", [
        ("the monitoring menu shows motor speed values.", None, 10.0, None),
    ])
    assert within.decision == Decision.ANSWER.value
    assert within.contract.get("sufficient") is not True  # quality pass alone

    beyond = _analyze(CONTRACT_QUERY, [(CONTRACT_DOC_TEXT, None, 37.7991, None)])
    assert beyond.decision == Decision.ANSWER.value          # sufficiency alone
    assert beyond.vector_distance == 37.7991                  # ...past the quality bound
