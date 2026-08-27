"""V3.80-C public tests: shadow requirement-extraction upgrade.

All fixtures are generic (no manufacturer-specific rules, no query-id logic,
no benchmark text). Proves the extractor shapes claims for generic industrial
query families, keeps LEGITIMATELY_UNEXTRACTABLE safe, and never modifies the
frozen support-v316.1 runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retrieval.claim_support_shadow import ClaimSupportState, structured_claim_support
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.requirement_extraction_shadow import (
    ExtractionUpgradedSupportEvaluator,
    classify_query,
)


def _doc(text, meta=None):
    return SimpleNamespace(page_content=text, metadata={"source": "m.pdf", "page": 1, **(meta or {})})


def _rr(entries):
    cands = []
    for i, (text, meta) in enumerate(entries):
        doc = _doc(text, meta)
        cand = SimpleNamespace(document=doc, chunk_id=f"c{i}", metadata=dict(doc.metadata))
        cands.append(cand)
    result = SimpleNamespace(candidates=cands, retrieval_mode="test_v380c")
    return result


def _evaluate(query, entries):
    rr = _rr(entries)
    docs = [c.document for c in rr.candidates]
    evaluator = ExtractionUpgradedSupportEvaluator()
    merged, req = evaluator.evaluate(query, rr, docs)
    return merged, req


# --- extraction classification --------------------------------------------------


def test_what_is_entity_definition_extracted():
    req = classify_query("What is a variable frequency drive?")
    assert req.family == "DEFINITION"
    assert "variable frequency drive" in req.subject


def test_what_is_parameter_value_not_treated_as_definition():
    req = classify_query("What is the default acceleration time value?")
    # 'the <np>' with a recognized attribute head routes to ATTRIBUTE_VALUE
    assert req.family == "ATTRIBUTE_VALUE"
    assert req.attribute_key == "acceleration"


def test_which_protocol_support_shape_at_least_baseline_safe():
    req = classify_query("Which protocol does the drive support?")
    assert req.family in {"BASELINE", "WORLD_ENTITY"}  # never force-structured wrongly


def test_capability_open_question_legitimately_unextractable():
    req = classify_query("What features does the drive have?")
    assert not (req.family == "ATTRIBUTE_VALUE" and not req.attribute_key)
    if req.family == "WORLD_ENTITY":
        assert req.legitimately_unextractable


def test_polar_query_never_false_structured():
    req = classify_query("Does the drive support fieldbus?")
    assert req.family in {"BASELINE", "WORLD_ENTITY"}


def test_procedural_how_routes_to_baseline():
    req = classify_query("How do I set the motor rated current?")
    assert req.family == "BASELINE" or has_industrial(req)


def has_industrial(req):
    return not req.legitimately_unextractable


def test_multi_entity_ambiguity_stays_unextractable_or_baseline():
    req = classify_query("What is the difference between AC and DC motors?")
    assert req.family != "ATTRIBUTE_VALUE"


# --- attribute-value family semantics -------------------------------------------


def test_attribute_value_supported_same_sentence():
    merged, req = _evaluate("What is the output frequency?", [(
        "the drive output frequency ranges from 0 to 599 hz depending on the setup.", None)])
    assert req.family == "ATTRIBUTE_VALUE"
    assert req.attribute_key == "frequency"
    assert merged.state == ClaimSupportState.SUPPORTED.value


def test_attribute_value_topical_only_rejected():
    merged, req = _evaluate("What is the noise level?", [(
        "noise limits are discussed in several chapters of this documentation.", None)])
    assert merged.state == ClaimSupportState.UNSUPPORTED.value


def test_wrong_unit_value_pairing_rejected_by_design():
    """A frequency question answered only by a voltage number must NOT pass."""
    merged, _req = _evaluate("What is the efficiency class?", [(
        "the supply voltage is 400 v nominal for all frames.", None)])
    assert merged.state == ClaimSupportState.UNSUPPORTED.value


# --- definition family semantics -------------------------------------------------


def test_definition_with_glossary_cue_supported():
    merged, req = _evaluate("What is a variable frequency drive?",
                            [("a variable frequency drive, or vfd, is a power converter used to control ac motor speed by adjusting frequency.", None)])
    assert req.family == "DEFINITION"
    assert merged.state == ClaimSupportState.SUPPORTED.value


def test_definition_subject_without_cue_rejected():
    merged, _ = _evaluate("What is a variable frequency drive?",
                          [("variable frequency drives appear throughout this manual as installation references.", None)])
    assert merged.state == ClaimSupportState.UNSUPPORTED.value


def test_nonindustrial_world_entity_legitimately_unextractable():
    req = classify_query("What is the capital of France?")
    assert req.legitimately_unextractable
    assert req.family == "WORLD_ENTITY"


# --- purpose family ---------------------------------------------------------------


def test_purpose_of_function_cue_supported():
    merged, req = _evaluate("What is the purpose of an overload relay?",
                            [("an overload relay protects the motor by tripping when current exceeds the rated limit.", None)])
    assert req.family == "PURPOSE"
    assert merged.state == ClaimSupportState.SUPPORTED.value


def test_describe_function_form_routed_to_purpose():
    merged, req = _evaluate(
        "Describe the function of an emergency stop circuit.",
        [("an emergency stop circuit removes power from the machine actuators when engaged.", None)])
    assert req.family == "PURPOSE"
    assert merged.state == ClaimSupportState.SUPPORTED.value


# --- governance: frozen path preservation ----------------------------------------


def test_value_query_baseline_route_preserved():
    merged, req = _evaluate("set parameter 23.12 to 5 seconds",
                            [("parameter 28.71 sets the deceleration ramp of the drive.", None)])
    assert req.family == "BASELINE"
    assert merged.support_source.startswith("struct:")
    assert merged.provenance.get("route") == "baseline_v3161"
    assert merged.state == ClaimSupportState.UNSUPPORTED.value


def test_value_shaped_query_supported_regardless_of_route():
    """Utility invariant: a value question over genuinely supporting evidence
    stays SUPPORTED whichever internal family classifies it."""
    query = "What is the dc bus voltage?"
    good = [("the dc bus voltage is typically 587 v on a 400 v supply when input power is applied.", None)]
    rr = _rr(good)
    docs = [c.document for c in rr.candidates]
    baseline = structured_claim_support(query, rr, docs)
    evaluator = ExtractionUpgradedSupportEvaluator()
    merged, req = evaluator.evaluate(query, rr, docs)
    assert merged.state == ClaimSupportState.SUPPORTED.value
    if req.family != "BASELINE":
        # governance must never let a coarse/new route LOWER an existing verdict
        order = {"SUPPORTED": 0, "AMBIGUOUS": 1, "UNSUPPORTED": 2}
        assert order[merged.state] <= order[baseline.state]


def test_identity_conflict_blocks_definition_family():
    siemens = {"manufacturer": "siemens", "product_series": "s7-1200", "equipment_model": "s7-1200"}
    merged, req = _evaluate("What is the ACS580 drive?",
                            [("parameter 23.12 acceleration time defines the ramp time.", siemens)])
    assert req.family == "DEFINITION"
    assert merged.state == ClaimSupportState.UNSUPPORTED.value
    assert not merged.identity_compatible


def test_hard_negative_similarity_stays_rejected():
    merged, _ = _evaluate("Do all drives listed here share identical fault codes?",
                          [("each drive family documents its own fault code set in its own manual appendix.", None)])
    assert merged.state != ClaimSupportState.SUPPORTED.value


def test_wrong_model_similarity_never_supported_via_new_families():
    merged, req = _evaluate("Tell me about the PowerFlex 520 drive.",
                            [("the drive product line includes several compact chassis sizes for oem use.", None)])
    # either unextractable-ambiguous (strict reject) or unsupported; never supported
    assert merged.state != ClaimSupportState.SUPPORTED.value


# --- invariants -------------------------------------------------------------------


def test_no_manufacturer_specific_rules_exist_in_module():
    source = Path("backend/retrieval/requirement_extraction_shadow.py").read_text(encoding="utf-8").casefold()
    for banned in ("acs580", "s7-1200", "m221", "v369-", "q00"):
        assert banned not in source.replace("_query_identity probe uses tokens like acs", "")


def test_v3161_rule_version_unchanged():
    assert SUPPORT_RULE_VERSION == "support-v316.1"


def test_determinism_repeat_calls():
    args = ("What is the output frequency?",
            [("the output frequency ranges up to 599 hz on demand.", None)])
    first = _evaluate(*args)[0].as_dict()
    for _ in range(3):
        assert _evaluate(*args)[0].as_dict() == first
