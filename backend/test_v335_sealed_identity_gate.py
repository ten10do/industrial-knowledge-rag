"""Tests for the frozen V3.35 sealed identity validation policy."""

from __future__ import annotations

from backend.evaluation.v335_sealed_identity_gate import (
    GATE_VERSION, IDENTITY_SLICES, decide, score, validate_corpus, validate_queries,
)


def _manifests():
    return [{
        "document_id": f"l-{index}", "file": f"documents/l-{index}.pdf",
        "source_name": f"Manual {index}", "document_type": "User Manual",
        "official_url": f"https://vendor{index}.example/manual.pdf",
        "source_type": "official_vendor_publication",
        "manufacturer": f"Vendor {index}", "equipment_type": "controller",
        "equipment_model": f"M{index}", "product_family": f"F{index}",
        "product_series": f"S{index}", "language": "English",
        "sha256": f"{index:064x}", "pages": 10,
        "download_timestamp": "2026-08-21T00:00:00Z",
    } for index in range(1, 7)]


def _queries():
    rows = []
    for index in range(60):
        rows.append({
            "query_id": f"L-{index:03d}", "query": f"Unique sealed query {index}",
            "expected": "ANSWER" if index % 2 == 0 else "ABSTAIN",
            "difficulty": "L4" if index % 5 else "L5", "confidence": "HIGH",
            "document_id": f"l-{index % 6 + 1}",
            "identity_slice": IDENTITY_SLICES[index % len(IDENTITY_SLICES)],
            "evidence_chunk_id": f"chunk-{index:03d}",
            "expected_identity": "UNKNOWN" if index % 2 == 0 else "INCOMPATIBLE",
        })
    return rows


def test_gate_version_is_frozen():
    assert GATE_VERSION == "v335-sealed-identity-gate-v1"


def test_valid_corpus_requires_six_distinct_manufacturers():
    assert validate_corpus(_manifests(), []).ok
    bad = _manifests()
    bad[-1]["manufacturer"] = bad[0]["manufacturer"]
    assert "MANUFACTURER_COUNT:5" in validate_corpus(bad, []).errors


def test_forbidden_document_overlap_is_rejected():
    manifests = _manifests()
    forbidden = [{"official_url": manifests[0]["official_url"], "sha256": "x", "source_name": "x"}]
    assert any(item.startswith("FORBIDDEN_URL") for item in validate_corpus(manifests, forbidden).errors)


def test_valid_query_distribution_passes():
    assert validate_queries(_queries(), {f"l-{index}" for index in range(1, 7)}).ok


def test_query_balance_and_slice_coverage_are_mandatory():
    queries = _queries()
    queries[1]["expected"] = "ANSWER"
    assert any(item.startswith("ANSWER_BALANCE") for item in validate_queries(
        queries, {f"l-{index}" for index in range(1, 7)},
    ).errors)
    missing = [item for item in _queries() if item["identity_slice"] != IDENTITY_SLICES[-1]]
    assert f"MISSING_SLICE:{IDENTITY_SLICES[-1]}" in validate_queries(
        missing, {f"l-{index}" for index in range(1, 7)},
    ).errors


def test_score_uses_class_specific_denominators():
    metrics = score([
        {"expected": "ANSWER", "predicted": "ANSWER"},
        {"expected": "ANSWER", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ABSTAIN"},
        {"expected": "ABSTAIN", "predicted": "ANSWER"},
    ])
    assert metrics["accuracy"] == 0.5
    assert metrics["answerable_recall"] == 0.5
    assert metrics["abstain_recall"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5


def test_ready_requires_all_four_policy_conditions():
    baseline = {"false_answer_rate": 0.5, "false_refusal_rate": 0.1}
    candidate = {"false_answer_rate": 0.2, "false_refusal_rate": 0.15}
    slices = {name: {"false_answer_rate": 0.2} for name in IDENTITY_SLICES}
    assert decide(baseline, candidate, slices, runtime_valid=True)["decision"] == "SEALED_IDENTITY_READY"
    slices[IDENTITY_SLICES[0]]["false_answer_rate"] = 0.21
    assert decide(baseline, candidate, slices, runtime_valid=True)["decision"] == "SEALED_IDENTITY_FAIL"
    assert decide(baseline, candidate, {}, runtime_valid=True)["decision"] == "SEALED_IDENTITY_FAIL"
    assert decide(baseline, candidate, {}, runtime_valid=False)["decision"] == "SEALED_IDENTITY_FAIL"
