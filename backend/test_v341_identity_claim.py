"""Tests for the V3.41 identity claim framework and candidate."""

from types import SimpleNamespace

from langchain_core.documents import Document

from backend.evaluation import v341_identity_claim as v
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.identity_claim_v341 import (
    IDENTITY_CLAIM_CANDIDATE_VERSION,
    IdentityClaim,
    analyze_identity_claim_evidence,
    bind_query_to_claim,
    build_document_claims,
)


def _doc(content: str, **meta):
    base = {"chunk_id": "c1", "document_id": "doc-1", "manufacturer": "Acme Industrial",
            "product_family": "ZetaDrive", "product_series": "Zeta-100",
            "equipment_model": "ZD-100", "model_aliases": ["ZetaDrive|ZD-100|Zeta 100"]}
    base.update(meta)
    return Document(page_content=content, metadata=base)


def _row(case, expected, index):
    return {
        "query_id": f"V341-{index:03d}", "query": f"Identity {case} question {index}?",
        "document_id": "doc-1", "expected": expected, "identity_case": case,
        "relevant_chunk_ids": ["chunk-1"] if expected == "ANSWER" else [],
        "confidence": "HIGH", "new_document": True, "identity_expected": "COMPATIBLE",
        "scope_ambiguous": False, "scope_reason": "", "parser_recoverable": True,
        "parser_reason": "", "identity_hard_negative": False,
    }


def _dataset():
    rows, index = [], 0
    for case in v.IDENTITY_CASES:
        for slot in range(10):
            expected = "ANSWER" if slot < v.IDENTITY_ANSWER_QUOTA[case] else "ABSTAIN"
            index += 1
            rows.append(_row(case, expected, index))
    return {"benchmark_version": v.BENCHMARK_VERSION, "uses_v335_data": False,
            "uses_v337_data": False, "queries": rows}


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

def test_constants_and_quotas():
    assert v.BENCHMARK_VERSION == "v341-identity-claim-dev-v1"
    assert len(v.IDENTITY_CASES) == 6
    assert sum(v.IDENTITY_ANSWER_QUOTA.values()) == 30
    assert {r.value for r in v.IdentityEvidenceRelation} == {
        "EXPLICIT", "SECTION_INHERITED", "DOCUMENT_INHERITED", "FAMILY_INHERITED", "UNSUPPORTED"}


def test_validate_dataset_paths():
    assert v.validate_dataset(_dataset()).ok
    payload = _dataset()
    payload["queries"][0]["expected"] = "ABSTAIN"
    payload["queries"][0]["relevant_chunk_ids"] = []
    assert any(e.startswith("ANSWER_COUNT") for e in v.validate_dataset(payload).errors)
    payload = _dataset()
    next(r for r in payload["queries"] if r["identity_case"] == "PRONOUN_IDENTITY")["identity_case"] = "BANANA"
    assert any(e.startswith("IDENTITY_CASE") for e in v.validate_dataset(payload).errors)


def test_classify_baseline_and_metrics():
    assert v.classify_baseline({"expected": "ANSWER", "decision": "ABSTAIN",
                                "identity_result": "INCOMPATIBLE"})[0] == "SAFE_CLAIM_EXPANSION"
    assert v.classify_baseline({"expected": "ABSTAIN", "decision": "ABSTAIN"})[1] == "BASELINE_CORRECT_REFUSAL"
    records = [{"expected": "ANSWER", "decision": "ANSWER"},
               {"expected": "ANSWER", "decision": "ABSTAIN"},
               {"expected": "ABSTAIN", "decision": "ABSTAIN"},
               {"expected": "ABSTAIN", "decision": "ANSWER"}]
    m = v.metrics(records)
    assert m["accuracy"] == 0.5 and m["false_refusals"] == 1 and m["false_answers"] == 1


def test_case_slices_and_acceptance():
    records = [{"identity_case": "PRONOUN_IDENTITY", "expected": "ANSWER", "decision": "ANSWER",
                "baseline_decision": "ABSTAIN"}]
    slices = v.case_slices(records)
    assert slices["PRONOUN_IDENTITY"]["modeled_relation"] == "DOCUMENT_INHERITED"
    assert slices["PRONOUN_IDENTITY"]["expanded_from_baseline"] == 1
    baseline = {"false_refusals": 5, "false_answer_rate": 0.5}
    ok = v.acceptance(baseline, {"false_refusals": 2, "false_answer_rate": 0.5},
                      unsafe_expansion=0, baseline_hard_negative_fa=20,
                      candidate_hard_negative_fa=20, runtime_integrity=True)
    assert ok["status"] == "DEV_READY" and ok["fr_reduction"] == 0.6


# ---------------------------------------------------------------------------
# Candidate: claim building and binding
# ---------------------------------------------------------------------------

def _claim():
    claims = build_document_claims([_doc("ZetaDrive ZD-100 content.")])
    return claims["doc-1"]


def test_candidate_version():
    assert IDENTITY_CLAIM_CANDIDATE_VERSION == "identity-v341-claim-expansion-candidate"


def test_single_product_corpus_builds_claim_and_multi_product_refuses():
    claim = _claim()
    assert claim.family == "zetadrive"
    assert claim.manufacturer == "acme industrial"
    assert "zd 100" in claim.aliases
    mixed = [
        _doc("ZetaDrive content.", document_id="doc-1"),
        _doc("OtherDrive content.", manufacturer="Beta Corp", product_family="OtherDrive",
             equipment_model="OD-1", model_aliases="OtherDrive|OD-1", document_id="doc-2"),
    ]
    claims = build_document_claims(mixed)
    assert set(claims) == {"doc-1", "doc-2"}  # per-document claims, both well-formed
    assert claims["doc-2"].family == "otherdrive"


def test_explicit_alias_and_family_bindings():
    claim = _claim()
    assert bind_query_to_claim("Does the ZD-100 support PID control?", claim, [_doc("x")])[0] == "EXPLICIT"
    assert bind_query_to_claim("Is the ZetaDrive family rated for 480 V?", claim, [_doc("x")])[0] == "FAMILY_INHERITED"


def test_pronoun_and_section_bindings():
    claim = _claim()
    assert bind_query_to_claim("Must this drive be grounded before operation?", claim, [_doc("x")])[0] == "DOCUMENT_INHERITED"
    relation, code = bind_query_to_claim("In this section, is the stall prevention enabled?", claim, [_doc("x")])
    assert relation == "SECTION_INHERITED" and code == "SECTION_CONTEXT_INHERITED"


def test_cross_manufacturer_and_foreign_model_rejected():
    claim = _claim()
    assert bind_query_to_claim("Can the Siemens S7-1200 CPU connect to this drive?", claim, [_doc("x")]) is None
    assert bind_query_to_claim("Is the ODE-3-320180 frame rating 24 A on the ZetaDrive?", claim, [_doc("x")]) is None


def test_parameter_owner_mismatch_rejected_but_owned_parameter_binds():
    docs = [_doc("ZetaDrive ZD-100 content. Parameter P-12 selects the control mode.")]
    claim = build_document_claims(docs)["doc-1"]
    assert bind_query_to_claim("Must parameter P-99 be set for the ZetaDrive?", claim, docs) is None
    relation, _code = bind_query_to_claim(
        "Does parameter P-12 select the control mode on this drive?", claim, docs)
    assert relation in ("EXPLICIT", "FAMILY_INHERITED", "DOCUMENT_INHERITED")


def _result(*docs):
    return RetrievalResult([
        RetrievalCandidate(document=d, retrieval_source="hybrid") for d in docs
    ])


# ---------------------------------------------------------------------------
# Decision wrapper
# ---------------------------------------------------------------------------

def _baseline(*, identity: str = "INCOMPATIBLE", reason: str = "NO_COMPATIBLE_IDENTITY_CLAIM",
              decision: str = "ABSTAIN"):
    value = SimpleNamespace(
        query_path="VERIFICATION", decision=decision, reason="RULE",
        final_decision_source="RULE", identity_boundary={"status": identity, "reason": reason},
        delegated_to_existing_evidence=False,
        existing_evidence={"base_rule_reason": "MISSING_VALUE_EVIDENCE"},
    )
    value.as_dict = lambda: {"decision": decision, "identity_boundary": value.identity_boundary,
                             "delegated_to_existing_evidence": False}
    return value


def test_identity_blocked_query_is_expanded(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.identity_claim_v341.analyze_identity_aware_evidence",
        lambda *a, **k: _baseline())
    monkeypatch.setattr(
        "backend.retrieval.identity_claim_v341.analyze_mixed_evidence",
        lambda *a, **k: SimpleNamespace(decision="ANSWER", query_path="VERIFICATION"))
    docs = [_doc("ZetaDrive ZD-100 content.")]
    decision = analyze_identity_claim_evidence("Must this drive be grounded before operation?", _result(*docs), docs, "hybrid")
    assert decision.decision == "ANSWER"
    assert decision.expanded is True
    assert decision.final_decision_source == "V341_IDENTITY_CLAIM"
    assert decision.claim_relation == "DOCUMENT_INHERITED"


def test_cross_manufacturer_query_stays_refused(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.identity_claim_v341.analyze_identity_aware_evidence",
        lambda *a, **k: _baseline())
    docs = [_doc("ZetaDrive ZD-100 content.")]
    decision = analyze_identity_claim_evidence(
        "Can the Siemens S7-1200 CPU connect to this drive?", _result(*docs), docs, "hybrid")
    assert decision.decision == "ABSTAIN"
    assert decision.expanded is False
    assert decision.claim_reason_code == "MANUFACTURER_MISMATCH_REJECTED"


def test_non_identity_block_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.identity_claim_v341.analyze_identity_aware_evidence",
        lambda *a, **k: _baseline(reason="SOME_OTHER_BLOCK"))
    decision = analyze_identity_claim_evidence(
        "Any question at all?", RetrievalResult([]), [_doc("x")], "hybrid")
    assert decision.decision == "ABSTAIN"
    assert decision.reason == "NON_IDENTITY_BLOCK_PRESERVED"
    assert decision.claim_relation == "UNSUPPORTED"


def test_already_answered_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "backend.retrieval.identity_claim_v341.analyze_identity_aware_evidence",
        lambda *a, **k: _baseline(decision="ANSWER", identity="COMPATIBLE",
                                  reason="COMPATIBLE_CLAIM_PRESENT"))
    decision = analyze_identity_claim_evidence(
        "Is the ZD-100 rating 2.3 A?", RetrievalResult([]), [_doc("x")], "hybrid")
    assert decision.decision == "ANSWER"
    assert decision.expanded is False
    assert decision.claim_reason_code == "ALREADY_ANSWERED"
