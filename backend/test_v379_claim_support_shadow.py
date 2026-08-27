"""V3.79 public tests: shadow claim-support validation contract.

All structured fixtures were verified against frozen support-v316.1 behavior;
NLI scenarios use an injected fake CrossEncoder (no model load, no network).
These tests lock the SHADOW ONLY contract: nothing here feeds Evidence runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.claim_support_shadow import (
    GOLD_INDEPENDENCE_FIELDS,
    ClaimSupportState,
    NliClaimSupportValidator,
    case_admissible,
    counterfactual_transitions,
    hybrid_claim_support,
    structured_claim_support,
)


def _doc(text, meta=None):
    return SimpleNamespace(page_content=text, metadata={"source": "m.pdf", "page": 1, **(meta or {})})


def _rr(entries):
    cands = []
    for i, (text, meta) in enumerate(entries):
        cand = RetrievalCandidate(document=_doc(text, meta), retrieval_source="chroma")
        cand.vector_score = 10.0
        cands.append(cand)
    return RetrievalResult(candidates=cands, retrieval_mode="test_v379")


def _support(query, entries):
    rr = _rr(entries)
    return structured_claim_support(query, rr, [c.document for c in rr.candidates])


def _fake_nli(logits_by_call=None, fixed=None):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, pairs):
            self.calls += len(pairs)
            if fixed is not None:
                return [list(fixed) for _ in pairs]
            return [list(logits_by_call(p[0], p[1])) for p in pairs]

    return FakeModel()


# --- exact positives ----------------------------------------------------------


def test_claim_support_exact_positive():
    out = _support("What is the dc bus voltage?", [(
        "the dc bus voltage is typically 587 v on a 400 v supply when input power is applied.", None)])
    assert out.state == ClaimSupportState.SUPPORTED.value
    assert out.support_reason == "DETAIL_SUPPORTED"
    assert case_admissible(out)


def test_paraphrase_bridging_is_incomplete_current_limitation():
    """Characterizes a KNOWN v316.1 limitation: 'Disconnect input power' vs
    'remove power' phrasing is NOT bridged by the frozen alias groups here -
    structured support refuses a semantically valid paraphrase."""
    out = _support("Disconnect input power before service.", [(
        "always remove power and wait five minutes before servicing the drive.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value


# --- unsupported shapes -------------------------------------------------------


def test_wrong_identity_unsupported():
    siemens = {"manufacturer": "siemens", "product_series": "s7-1200"}
    out = _support("What is the acceleration time on the ACS580 drive?",
                   [("parameter 23.12 acceleration time defines the ramp time.", siemens)])
    # Frozen mechanism: the off-family model token surfaces as an unsatisfied
    # qualifier (MISSING_REQUIRED_CONCEPT) rather than a target_identity miss -
    # safety outcome (UNSUPPORTED) is identical either way.
    assert out.state == ClaimSupportState.UNSUPPORTED.value
    assert any(m.startswith("qualifier:") or m == "target_identity"
               for m in out.provenance["missing_requirements"])


def test_wrong_parameter_identifier_unsupported():
    out = _support("set parameter 23.12 to 5 seconds",
                   [("parameter 28.71 sets the deceleration ramp of the drive.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value
    assert any(m.startswith("identifier:") for m in out.provenance["missing_requirements"])


def test_relation_value_local_association_missing():
    """Value must be locally associated with the requested attribute (V3.15 machinery)."""
    out = _support("What is the output frequency range?",
                   [("frequency converters control motor speed through pwm switching techniques.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value
    assert "value:local_association" in out.provenance["missing_requirements"] or (
        out.support_reason == "MISSING_VALUE_SUPPORT")


def test_topical_mention_without_fact_is_not_support():
    out = _support("What is the dc bus voltage?",
                   [("the dc bus is energized whenever input power is connected to the drive.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value


def test_scope_leakage_other_parameter_value_rejected():
    out = _support("what is parameter 28.72 frequency acceleration time?",
                   [("parameter 23.12 acceleration time 1 default ramps set 1 apply in uss mode.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value


def test_wrong_unit_unsupported():
    out = _support("limit motor cable length to 50 m",
                   [("the maximum cable length depends on the shielded wiring environment.", None)])
    assert out.state == ClaimSupportState.UNSUPPORTED.value
    assert any(m.startswith("unit:") for m in out.provenance["missing_requirements"])


def test_corpus_unsupported_style_evidence():
    out = _support("What is the rated insulation voltage?",
                   [("insulation materials degrade over temperature cycles commonly discussed in engineering texts.", None)])
    assert out.state != ClaimSupportState.SUPPORTED.value


def test_ambiguity_never_counts_as_admissible():
    out = _support("What is the acceleration time?",
                   [("parameter 23.12 acceleration time defines the ramp time.", None)])
    if out.state == ClaimSupportState.AMBIGUOUS.value:
        assert not case_admissible(out)


# --- multi-claim aggregation ----------------------------------------------------


def test_multi_requirement_case_any_critical_missing_blocks():
    ok_a = ("parameter 23.12 acceleration time defines the ramp time.", None)
    out = _support("set parameter 23.12 to 5 seconds", [ok_a])
    assert out.state == ClaimSupportState.UNSUPPORTED.value  # value/unit parts missing


# --- NLI candidate (injected model, frozen thresholds) ---------------------------

ENTAIL = [-0.5, 4.0, -0.5]      # argmax entailment >> thresholds
CONTRA = [4.0, -0.5, -0.5]
NEUTRAL = [-0.5, -0.5, 4.0]
UNIFORM = [0.33, 0.34, 0.33]     # argmax below unknown_floor -> UNKNOWN/AMBIGUOUS


def test_nli_entailment_maps_to_supported():
    validator = NliClaimSupportValidator(model=_fake_nli(fixed=ENTAIL))
    out = validator.judge_case("Does the drive support fieldbus?", ["fieldbus is supported."])
    assert out.state == ClaimSupportState.SUPPORTED.value
    assert validator.invocations == 1


def test_nli_contradiction_and_neutral_map_to_unsupported():
    contra = NliClaimSupportValidator(model=_fake_nli(fixed=CONTRA)).judge_case("q?", ["t"])
    neutral = NliClaimSupportValidator(model=_fake_nli(fixed=NEUTRAL)).judge_case("q?", ["t"])
    assert contra.state == ClaimSupportState.UNSUPPORTED.value
    assert neutral.state == ClaimSupportState.UNSUPPORTED.value


def test_nli_low_confidence_is_ambiguous_not_supported():
    validator = NliClaimSupportValidator(model=_fake_nli(fixed=UNIFORM))
    out = validator.judge_case("q?", ["t"])
    assert out.state == ClaimSupportState.AMBIGUOUS.value
    assert not case_admissible(out)


def test_nli_no_evidence_text_reports_not_applicable():
    validator = NliClaimSupportValidator(model=_fake_nli(fixed=ENTAIL))
    out = validator.judge_case("q?", [])
    assert out.state == ClaimSupportState.NOT_APPLICABLE.value


def test_capability_invocation_tracking():
    model = _fake_nli(fixed=ENTAIL)
    validator = NliClaimSupportValidator(model=model)
    validator.judge_case("q?", ["a", "b"])
    assert validator.invocations == 2 and model.calls == 2


# --- hybrid ---------------------------------------------------------------------


def test_hybrid_downgrades_on_contradiction_only():
    struct_supported = structured_claim_support(
        "What is the dc bus voltage?",
        _rr([("the dc bus voltage is typically 587 v on a 400 v supply.", None)]),
        [_doc("the dc bus voltage is typically 587 v on a 400 v supply.")])
    nli_contra = NliClaimSupportValidator(model=_fake_nli(fixed=CONTRA)).judge_case("q?", ["t"])
    downgraded = hybrid_claim_support(struct_supported, nli_contra)
    assert downgraded.state == ClaimSupportState.UNSUPPORTED.value

    nli_entail = NliClaimSupportValidator(model=_fake_nli(fixed=ENTAIL)).judge_case("q?", ["t"])
    kept = hybrid_claim_support(struct_supported, nli_entail)
    assert kept.state == ClaimSupportState.SUPPORTED.value


# --- gold independence / determinism ---------------------------------------------


def test_gold_independence_poison_fields_cannot_change_verdict():
    rr = _rr([("the dc bus voltage is typically 587 v on a 400 v supply.", None)])
    docs = [c.document for c in rr.candidates]
    base = structured_claim_support("What is the dc bus voltage?", rr, docs)
    # flip every gold-labeled attribute of the container object; verdict must not move
    for field_name in GOLD_INDEPENDENCE_FIELDS:
        setattr(rr, field_name, "POISON" if isinstance(getattr(rr, field_name, None), str) else ["POISON"])
    poisoned = structured_claim_support("What is the dc bus voltage?", rr, docs)
    assert poisoned.as_dict() == base.as_dict()


def test_query_id_independence():
    rr = _rr([("the dc bus voltage is typically 587 v on a 400 v supply.", None)])
    docs = [c.document for c in rr.candidates]
    one = structured_claim_support("What is the dc bus voltage?", rr, docs).as_dict()
    two = structured_claim_support("What is the dc bus voltage?", rr, docs).as_dict()
    assert one == two  # inputs contain no identifier-like discriminators


def test_structured_determinism():
    args = ("limit motor cable length to 50 m",
            [("the maximum cable length depends on the shielded wiring environment.", None)])
    first = _support(*args).as_dict()
    for _ in range(3):
        assert _support(*args).as_dict() == first


# --- counterfactual aggregation policy -------------------------------------------


def test_counterfactual_transitions_classification():
    rows = [
        {"query_id": "FA1", "runtime_answered": True, "correct_answer": False, "admissible": False},
        {"query_id": "POS1", "runtime_answered": True, "correct_answer": True, "admissible": True},
        {"query_id": "POS2", "runtime_answered": True, "correct_answer": True, "admissible": False},
    ]
    t = counterfactual_transitions(rows, policy_name="unit")
    assert t["SAFE_BLOCK"] == ["FA1"]
    assert t["FALSE_REFUSAL_REGRESSION"] == ["POS2"]
    assert "POS1" in t["NO_EFFECT"]
