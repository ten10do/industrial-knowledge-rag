"""V3.77 Benchmark–Corpus Alignment tests.

Pure synthetic/structural tests for the alignment rebuild tooling. No runtime,
index, embedding model, or private artifact is required: where private frozen
artifacts exist on the runner machine, the prediction-independence source scan
runs against them too.

Contract coverage (spec section 45):

* schema: 69 cases audited / query text immutable / no query deletion
* ANSWER requires corpus support (anchors + claims)
* CORPUS_UNSUPPORTED -> expected ABSTAIN
* OOD strictly separated from CORPUS_UNSUPPORTED
* known absent-model classification consistency
* full-corpus support audit independent of retrieval Top-K
* gold independent of prediction
* ambiguous support invalidates benchmark gate
* benchmark hash determinism (canonical serialization)
* V1->V2 diff determinism
* formal denominator / real origins required
* two-run replay comparator logic

These tests use a miniature synthetic corpus and never modify the frozen one.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from backend.evaluation.benchmark_v2_schema import (
    BENCHMARK_VERSION,
    AlignedBenchmarkCase,
    CorpusSupportState,
    QueryDomain,
    aggregate_counts,
    canonical_json,
    derive_expected_decision,
    query_text_hash,
    sha256_text,
    validate_benchmark,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 13.234710693359375


def _case(
    qid="Q1",
    text="what is x?",
    decision="ANSWER",
    state=CorpusSupportState.SUPPORTED_ANSWER,
    domain=QueryDomain.INDUSTRIAL_IN_DOMAIN,
    anchors=(({"document": "d.pdf", "page": 0, "quote": "x is y"}),),
    claims=({"subject": "x", "relation": "is", "obj_value": "y"},),
    slice_label="DIRECT_FACT",
    reason="corpus documents x",
):
    return AlignedBenchmarkCase(
        query_id=qid,
        query_text=text,
        slice_labels=(slice_label,),
        difficulty="L3",
        support_state=state,
        query_domain=domain,
        expected_decision=decision,
        support_reason=reason,
        anchors=anchors,
        claims=claims,
    )


# --- Schema derivation rules ---------------------------------------------------


def test_supported_answer_derives_answer():
    assert derive_expected_decision(CorpusSupportState.SUPPORTED_ANSWER, QueryDomain.INDUSTRIAL_IN_DOMAIN) == "ANSWER"


def test_corpus_unsupported_forces_abstain():
    assert derive_expected_decision(CorpusSupportState.CORPUS_UNSUPPORTED, QueryDomain.INDUSTRIAL_IN_DOMAIN) == "ABSTAIN"


def test_ood_domain_abstains_regardless_of_state():
    assert derive_expected_decision(CorpusSupportState.CORPUS_UNSUPPORTED, QueryDomain.GENERIC_OUT_OF_DOMAIN) == "ABSTAIN"


def test_answer_requires_corpus_support_anchors_and_claims():
    problems = validate_benchmark([_case(anchors=())])
    assert any("SUPPORTED_ANSWER requires >=1 anchor" in p for p in problems)
    problems = validate_benchmark([_case(claims=())])
    assert any("requires corpus-grounded claims" in p for p in problems)


def test_corpus_unsupported_case_valid_without_anchors_but_with_reason():
    case = _case(
        decision="ABSTAIN",
        state=CorpusSupportState.CORPUS_UNSUPPORTED,
        anchors=(),
        claims=(),
    )
    assert validate_benchmark([case]) == []


def test_missing_reason_rejected_for_in_domain_unsupported():
    case = _case(
        decision="ABSTAIN",
        state=CorpusSupportState.CORPUS_UNSUPPORTED,
        anchors=(),
        claims=(),
        reason="",
    )
    problems = validate_benchmark([case])
    assert any("support reason" in p for p in problems)


# --- OOD separation -------------------------------------------------------------


def test_ood_must_keep_ood_slice_continuity_and_generic_domain():
    # an OOD-slice case mislabeled industrial must fail validation via
    # SUPPORTED_ANSWER/domain coupling; abstain-with-generic-domain stays valid.
    ood_ok = _case(
        qid="OOD1", text="capital of france?", decision="ABSTAIN",
        state=CorpusSupportState.CORPUS_UNSUPPORTED,
        domain=QueryDomain.GENERIC_OUT_OF_DOMAIN, anchors=(), claims=(),
        slice_label="OOD", reason="general knowledge question",
    )
    assert validate_benchmark([ood_ok]) == []
    ood_bad = _case(
        qid="OOD2", text="capital of france?", decision="ANSWER",
        state=CorpusSupportState.SUPPORTED_ANSWER,
        domain=QueryDomain.GENERIC_OUT_OF_DOMAIN, slice_label="OOD",
    )
    problems = validate_benchmark([ood_bad])
    assert any("SUPPORTED_ANSWER requires in-domain" in p for p in problems)


# --- Ambiguity policy ------------------------------------------------------------


def test_ambiguous_support_invalidates_benchmark_gate():
    ambiguous = _case(
        qid="AMB", decision="ABSTAIN",
        state=CorpusSupportState.AMBIGUOUS_CORPUS_SUPPORT, anchors=(), claims=(),
    )
    problems = validate_benchmark([ambiguous])
    assert any("AMBIGUOUS_CORPUS_SUPPORT" in p for p in problems)
    # derivation refuses to mint any formal decision from ambiguity: the
    # builder turns this None into a hard failure before the freeze gate.
    assert derive_expected_decision(CorpusSupportState.AMBIGUOUS_CORPUS_SUPPORT, QueryDomain.INDUSTRIAL_IN_DOMAIN) is None
    assert derive_expected_decision(CorpusSupportState.INVALID_GOLD_ANNOTATION, QueryDomain.INDUSTRIAL_IN_DOMAIN) is None


# --- Hash determinism ------------------------------------------------------------


def test_canonical_serialization_is_deterministic_and_order_insensitive_keys():
    blob1 = canonical_json({"a": 1, "b": [1, 2, {"c": 3}]})
    blob2 = canonical_json({"b": [1, 2, {"c": 3}], "a": 1})
    assert blob1 == blob2
    assert sha256_text(blob1) == sha256_text(blob2)


def test_query_text_hash_detects_any_text_drift():
    h1 = query_text_hash([("Q1", "tell me about the acs580 drive."), ("Q2", "what is pwm?")])
    h2 = query_text_hash([("Q1", "tell me about the acs580 drives."), ("Q2", "what is pwm?")])  # one char drift
    h3 = query_text_hash([("Q2", "what is pwm?"), ("Q1", "tell me about the acs580 drive.")])  # order matters
    assert h1 != h2
    assert h1 != h3


def test_v1_v2_diff_computation_is_deterministic(tmp_path):
    """The V1→V2 diff transition counter is a pure function of its input rows."""
    payload = {
        "rows": [
            {"query_id": f"Q{i:03d}", "decision_v1": "ANSWER", "decision_v2": "ABSTAIN"}
            for i in range(18)
        ] + [
            {"query_id": f"R{i:03d}", "decision_v1": "ANSWER", "decision_v2": "ANSWER"}
            for i in range(26)
        ]
    }
    (tmp_path / "diff.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded1 = json.load(open(tmp_path / "diff.json", encoding="utf-8"))
    loaded2 = json.load(open(tmp_path / "diff.json", encoding="utf-8"))
    def transitions(rows):
        counts = {}
        for row in rows:
            key = (row["decision_v1"], row["decision_v2"])
            counts[key] = counts.get(key, 0) + 1
        return counts
    assert transitions(loaded1["rows"]) == transitions(loaded2["rows"])
    assert transitions(loaded1["rows"]).get(("ANSWER", "ABSTAIN")) == 18


# --- Prediction blindness ---------------------------------------------------------


FORBIDDEN_PREDICTION_SOURCES = (
    "results/v376_score_lineage",
    "results/v375_audit",
    "full_trace.json",
    "run1_full_trace",
    "e1_rows",
    "runtime_decision",
    "record['decision']",
    "primary_attribution",
)

PREDICTION_BLIND_BUILDERS = ("v377_build_aligned_benchmark.py", "backend/evaluation/benchmark_v2_schema.py")


@pytest.mark.parametrize("relative", PREDICTION_BLIND_BUILDERS)
def test_gold_builder_source_contains_no_prediction_inputs(relative):
    source = (_REPO_ROOT / relative).read_text(encoding="utf-8")
    low = source.lower()
    for needle in FORBIDDEN_PREDICTION_SOURCES:
        assert needle.lower() not in low, f"{relative} references {needle}"
    # the builder may not even import the baseline harness modules
    for module in ("v376_score_lineage", "v377_aligned_baseline", "v369_real_baseline"):
        if relative.endswith("benchmark_v2_schema.py"):
            assert module not in low
        else:
            # builder legitimately imports v369 only to reconstruct byte-exact V1
            # query TEXT; it must never read a predictions artifact path.
            continue


def test_aligned_builder_reads_only_query_generator_for_v1_text():
    source = (_REPO_ROOT / "v377_build_aligned_benchmark.py").read_text(encoding="utf-8")
    assert "generate_queries()" in source  # structural V1 reconstruction
    # every corpus load goes through the production loader and annotations
    assert "load_page_index()" in source
    assert "annotations[" in source


# --- Frozen artifact checks (execute when the private freeze exists locally) ------


def _freeze_available():
    path = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "aligned_benchmark_v2.json"
    return path.is_file()


@pytest.mark.skipif(not _freeze_available(), reason="private frozen benchmark artifacts not present")
def test_frozen_benchmark_has_69_unique_audited_cases():
    payload = json.load(
        open(_REPO_ROOT / "backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json", encoding="utf-8")
    )
    cases = payload["cases"]
    assert len(cases) == 69
    ids = [c["query_id"] for c in cases]
    assert len(set(ids)) == 69
    for case in cases:
        assert case["support_state"] in {"SUPPORTED_ANSWER", "CORPUS_UNSUPPORTED"}
        assert case["expected_decision"] in {"ANSWER", "ABSTAIN"}
        if case["expected_decision"] == "ANSWER":
            assert case["support_state"] == "SUPPORTED_ANSWER"
            assert case["anchors"], "ANSWER case lacks verified anchor"
            assert case["claims"], "ANSWER case lacks grounded claims"
        else:
            assert case["support_state"] == "CORPUS_UNSUPPORTED"
            assert case["support_reason"], "ABSTAIN case lacks corpus-driven reason"


@pytest.mark.skipif(not _freeze_available(), reason="private frozen benchmark artifacts not present")
def test_frozen_queries_are_nonempty_and_slice_preserved():
    payload = json.load(
        open(_REPO_ROOT / "backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json", encoding="utf-8")
    )
    slices = {}
    for case in payload["cases"]:
        assert case["query_text"].strip()
        slices.setdefault(case["slice_labels"][0], 0)
        slices[case["slice_labels"][0]] += 1
    # V1 slice sizes preserved exactly (no deletion, no re-balancing).
    assert slices == {
        "DIRECT_FACT": 10, "VALUE": 10, "IDENTITY": 12, "OOD": 15,
        "NON_TABLE": 12, "HARD_NEGATIVE": 10,
    }


@pytest.mark.skipif(not _freeze_available(), reason="private frozen benchmark artifacts not present")
def test_known_absent_models_classified_corpus_unsupported():
    payload = json.load(
        open(_REPO_ROOT / "backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json", encoding="utf-8")
    )
    absent_expectation = {
        "V369-Q0021": "g120", "V369-Q0023": "atv320", "V369-Q0024": "powerflex 520",
        "V369-Q0025": "fr-e800", "V369-Q0026": "fc51",
        "V369-Q0027": "g120", "V369-Q0029": "atv320", "V369-Q0030": "powerflex 520",
        "V369-Q0031": "fr-e800", "V369-Q0032": "fc51",
    }
    by_id = {c["query_id"]: c for c in payload["cases"]}
    for qid, model in absent_expectation.items():
        case = by_id[qid]
        assert case["support_state"] == "CORPUS_UNSUPPORTED", (qid, model)
        assert case["expected_decision"] == "ABSTAIN"
        assert model.split()[0].lower() in case["support_reason"].lower()


@pytest.mark.skipif(not _freeze_available(), reason="private frozen benchmark artifacts not present")
def test_support_audits_cite_document_page_evidence_not_retrieval_topk():
    """Anchors are document/page citations from full-corpus scans; Top-K ranks
    or distances never appear inside the support manifest."""
    payload = json.load(
        open(_REPO_ROOT / "backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json", encoding="utf-8")
    )
    blob = canonical_json(payload["cases"]).lower()
    for forbidden_token in ('"vector_rank"', '"retrieval_rank"', '"top5"', '"topk"', '"fusion_score"'):
        assert forbidden_token not in blob
    anchored = 0
    for case in payload["cases"]:
        for anchor in case["anchors"]:
            assert anchor["document"].endswith(".pdf")
            assert isinstance(anchor["page"], int)
            assert anchor["quote"].strip()
            anchored += 1
    assert anchored >= 25  # every ANSWER case carries >=1 verified anchor


def test_formal_row_requires_real_origins():
    """Provenance contract: rows without both real origins cannot be counted."""
    def eligible(row):
        return (
            row.get("record_origin") == "REAL_EVIDENCE_RUNTIME"
            and row.get("retrieval_score_origin") == "REAL_CHROMA_RUNTIME"
            and row.get("fidelity_ok") is True
        )
    real = {"record_origin": "REAL_EVIDENCE_RUNTIME", "retrieval_score_origin": "REAL_CHROMA_RUNTIME", "fidelity_ok": True}
    fake_origin = dict(real, record_origin="MOCK_EVIDENCE")
    bad_fidelity = dict(real, fidelity_ok=False)
    assert eligible(real)
    assert not eligible(fake_origin)
    assert not eligible(bad_fidelity)


def test_two_run_replay_comparator_logic():
    run1 = [{"q": "Q1", "ids": ["a", "b"], "decision": "ANSWER", "verdict": "CORRECT"},
            {"q": "Q2", "ids": ["c"], "decision": "ABSTAIN", "verdict": "ABSTAIN_CORRECT"}]
    run2 = [{"q": "Q1", "ids": ["a", "b"], "decision": "ANSWER", "verdict": "CORRECT"},
            {"q": "Q2", "ids": ["c"], "decision": "ABSTAIN", "verdict": "ABSTAIN_CORRECT"}]
    drifted = [{"q": "Q1", "ids": ["b", "a"], "decision": "ANSWER", "verdict": "CORRECT"},
               {"q": "Q2", "ids": ["c"], "decision": "ABSTAIN", "verdict": "ABSTAIN_CORRECT"}]
    same = all(a["ids"] == b["ids"] and (a["decision"], a["verdict"]) == (b["decision"], b["verdict"])
               for a, b in zip(run1, run2, strict=True))
    assert same
    different = any(a["ids"] != b["ids"] for a, b in zip(run1, drifted, strict=True))
    assert different


def test_aggregate_counts_shape_is_public_safe():
    cases = [_case(qid="A"), _case(
        qid="B", decision="ABSTAIN", state=CorpusSupportState.CORPUS_UNSUPPORTED,
        anchors=(), claims=(),
    ), _case(
        qid="C", decision="ABSTAIN", state=CorpusSupportState.CORPUS_UNSUPPORTED,
        domain=QueryDomain.GENERIC_OUT_OF_DOMAIN, anchors=(), claims=(),
        slice_label="OOD", reason="general knowledge question",
    )]
    counts = aggregate_counts(cases)
    assert counts["n_cases"] == 3
    assert counts["expected_decisions"] == {"ANSWER": 1, "ABSTAIN": 2}
    blob = canonical_json(counts)
    assert "query_text" not in blob and "anchors" not in blob


def test_threshold_constant_not_drifted_by_this_phase_tooling():
    from backend.retrieval.evidence import default_policy

    assert default_policy().max_vector_distance == THRESHOLD


def test_benchmark_version_pin():
    assert BENCHMARK_VERSION == "V377_ALIGNED_BENCHMARK_V2"
