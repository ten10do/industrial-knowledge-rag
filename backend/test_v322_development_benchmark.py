"""Public contract tests for the V3.22 development-only benchmark."""
from __future__ import annotations

from copy import deepcopy

import pytest

from backend.evaluation import v322_development_benchmark as gate
from backend.evaluation.v311_resume import hash_json


CLASSES = tuple(sorted(gate.FAILURE_CLASSES))
STYLES = (
    "PARAMETER_TABLE", "PROCEDURE_STEPS", "NARRATIVE_SPEC", "PROTOCOL_REFERENCE",
)


def _freeze(manifest: dict) -> None:
    queries = manifest["queries"]
    manifest["freeze"] = {
        "query_sha256": hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in queries]),
        "annotation_sha256": hash_json(queries),
        "manifest_sha256": hash_json(manifest),
    }


def _manifest(split: str, prefix: str, families: tuple[str, ...]) -> dict:
    pair_count = 32 if split == "DEV-TRAIN-V2" else 18
    documents = []
    candidates = {}
    for index in range(4):
        document_id = f"{prefix}-doc-{index}"
        documents.append({
            "document_id": document_id,
            "manufacturer": f"Vendor-{index}",
            "manufacturer_slice": "UNSEEN" if split == "DEV-TUNE-V2" and index == 3 else "SEEN",
            "equipment_type": ("plc_controller", "drive", "industrial_communication", "remote_io_fieldbus")[index],
            "product_family": f"{prefix}-family-{index}",
            "file": f"{prefix}/manual-{index}.pdf",
        })
        for part in range(2):
            candidates[f"{prefix}-chunk-{index}-{part}"] = {
                "content": f"fixed real-manual-like candidate part {part}",
                "metadata": {"document_id": document_id, "chunk_id": f"{prefix}-chunk-{index}-{part}"},
            }
    queries = []
    semantic_min = 16 if split == "DEV-TRAIN-V2" else 10
    multi_min = 8 if split == "DEV-TRAIN-V2" else 5
    for index in range(pair_count):
        document = documents[index % 4]
        failure_class = families[index % len(families)]
        for answerable, suffix in ((True, "p"), (False, "n")):
            queries.append({
                "query_id": f"{prefix}-{index:02d}{suffix}",
                "pair_id": f"{prefix}-pair-{index:02d}",
                "query": f"Does requirement {prefix} {index} {suffix} apply to the documented operating condition?",
                "answerable": answerable,
                "document_id": document["document_id"],
                "manufacturer": document["manufacturer"],
                "manufacturer_slice": document["manufacturer_slice"],
                "failure_class": failure_class,
                "focus": ("identifier", "protocol", "value", "action")[index % 4],
                "document_style": STYLES[index % len(STYLES)],
                "difficulty": ("L3_SEMANTIC", "L4_SCOPE_COMPOSITION", "L5_HARD_NEAR_MISS")[index % 3],
                "surface_form_type": "semantic_paraphrase" if answerable else "sentence",
                "negative_hardness": None if answerable else "N4",
                "critical_requirements": [f"documented condition {index}"],
                "non_critical_cues": [prefix],
                "expected_scope": "SAME_SECTION",
                "forbidden_scope_reason": None if answerable else "The nearby statement belongs to a different condition.",
                "annotation_rationale": "The fixed candidate establishes the positive or the documented contrast for the negative.",
                "confidence": "HIGH",
                "core": True,
                "claim_type": "SEMANTIC_EQUIVALENT" if answerable else "RELATED_ONLY",
                "candidate_chunk_ids": (
                    [f"{prefix}-chunk-{index % 4}-0", f"{prefix}-chunk-{index % 4}-1"]
                    if index < multi_min else [f"{prefix}-chunk-{index % 4}-0"]
                ),
                "semantic_positive": answerable and index < semantic_min,
                "multi_chunk_positive": answerable and index < multi_min,
                "unsafe_multi_chunk_negative": (not answerable) and index < multi_min,
            })
    manifest = {
        "development_set_id": f"synthetic-{prefix}",
        "benchmark_version": gate.BENCHMARK_VERSION,
        "split": split,
        "documents": documents,
        "queries": queries,
        "candidates": candidates,
    }
    _freeze(manifest)
    return manifest


def _sets() -> tuple[dict, dict]:
    return (
        _manifest("DEV-TRAIN-V2", "train", CLASSES),
        _manifest("DEV-TUNE-V2", "tune", CLASSES),
    )


def test_split_counts_balance_categories_difficulty_styles_and_quotas():
    train, tune = _sets()
    report = gate.validate_benchmark(train, tune)
    assert report["validity"] == "VALID"
    assert report["sealed_gate"] == "NO"
    assert report["train"]["queries"] == 64
    assert report["tune"]["queries"] == 36
    assert report["train"]["answerable"] == report["train"]["abstain"]
    assert set(report["train"]["failure_classes"]) == gate.FAILURE_CLASSES
    assert set(report["tune"]["failure_classes"]) == gate.FAILURE_CLASSES
    assert report["train"]["semantic_positive"] >= 16
    assert report["tune"]["semantic_positive"] >= 10
    assert report["train"]["safe_multi_chunk_positive"] >= 8
    assert report["tune"]["unsafe_multi_chunk_negative"] >= 5
    assert len(report["train"]["document_style"]) >= 4
    assert report["train"]["hard_difficulty_share"] >= .60
    assert report["tune"]["hard_negative_share"] >= .60


def test_document_disjointness_and_product_line_overlap_reporting():
    train, tune = _sets()
    report = gate.validate_independence(train, tune)
    assert report["train_tune_document_leakage"] == 0
    assert report["product_line_disjoint"]

    tune["documents"][0]["document_id"] = train["documents"][0]["document_id"]
    with pytest.raises(ValueError, match="TRAIN_TUNE_DOCUMENT_LEAKAGE"):
        gate.validate_independence(train, tune)

    train, tune = _sets()
    tune["documents"][0]["product_family"] = train["documents"][0]["product_family"]
    report = gate.validate_independence(train, tune)
    assert report["product_line_overlap"] == ["train-family-0"]


def test_d_e_exclusion_and_unseen_manufacturer_slice_are_enforced():
    train, tune = _sets()
    train["documents"][0]["product_family"] = "S7-1200"
    with pytest.raises(ValueError, match="D_E_PRODUCT_LEAK"):
        gate.validate_manifest(train)

    train, tune = _sets()
    for document in tune["documents"]:
        document["manufacturer_slice"] = "SEEN"
    with pytest.raises(ValueError, match="UNSEEN_MANUFACTURER_REQUIRED"):
        gate.validate_manifest(tune)

    train, tune = _sets()
    holdout = {"documents": [{"document_id": "d-x", "source_path": train["documents"][0]["file"]}]}
    train["documents"][0]["source_path"] = train["documents"][0]["file"]
    with pytest.raises(ValueError, match="D_E_DOCUMENT_LEAKAGE"):
        gate.holdout_document_audit(train, tune, [holdout])


def test_critical_optional_scope_and_negative_hardness_are_enforced():
    train, _ = _sets()
    broken = deepcopy(train)
    broken["queries"][0]["critical_requirements"] = []
    with pytest.raises(ValueError, match="CRITICAL_REQUIREMENTS_REQUIRED"):
        gate.validate_manifest(broken)

    broken = deepcopy(train)
    negative = next(row for row in broken["queries"] if not row["answerable"])
    negative["forbidden_scope_reason"] = ""
    with pytest.raises(ValueError, match="FORBIDDEN_SCOPE_REASON_REQUIRED"):
        gate.validate_manifest(broken)

    broken = deepcopy(train)
    broken["queries"][0]["expected_scope"] = "WHOLE_DOCUMENT"
    with pytest.raises(ValueError, match="EXPECTED_SCOPE_INVALID"):
        gate.validate_manifest(broken)


def test_versioning_and_all_three_freeze_hashes_are_enforced():
    train, _ = _sets()
    broken = deepcopy(train)
    broken["benchmark_version"] = "v1"
    with pytest.raises(ValueError, match="BENCHMARK_VERSION_INVALID"):
        gate.validate_manifest(broken)

    for key, match in (
        ("query_sha256", "QUERY_HASH"),
        ("annotation_sha256", "ANNOTATION_HASH"),
        ("manifest_sha256", "MANIFEST_HASH"),
    ):
        broken = deepcopy(train)
        broken["freeze"][key] = "0" * 64
        with pytest.raises(ValueError, match=match):
            gate.validate_manifest(broken)


def test_historical_query_exact_normalized_and_high_overlap_are_rejected():
    train, tune = _sets()
    query = train["queries"][0]["query"]
    with pytest.raises(ValueError, match="HISTORICAL_QUERY_LEAKAGE"):
        gate.query_similarity_audit(train, tune, [{"queries": [{"query_id": "old", "query": query}]}])

    normalized_variant = f"  {query.upper()} !!!"
    with pytest.raises(ValueError, match="HISTORICAL_QUERY_LEAKAGE"):
        gate.query_similarity_audit(train, tune, [{"queries": [{"query_id": "old", "query": normalized_variant}]}])

    high_overlap = query.replace("documented", "specified")
    with pytest.raises(ValueError, match="HISTORICAL_QUERY_LEAKAGE"):
        gate.query_similarity_audit(train, tune, [{"queries": [{"query_id": "old", "query": high_overlap}]}])


def test_ambiguous_annotations_cannot_enter_the_core_set():
    train, _ = _sets()
    train["queries"][0]["confidence"] = "AMBIGUOUS"
    with pytest.raises(ValueError, match="AMBIGUOUS_CORE_CASE"):
        gate.validate_manifest(train)


def test_comparison_requires_same_frozen_manifest_and_reports_deltas():
    metrics = {
        "decision_accuracy": .5, "answerable_recall": .4, "abstain_recall": .6,
        "false_answer_rate": .4, "false_refusal_rate": .6,
    }
    baseline = {
        "benchmark_version": gate.BENCHMARK_VERSION, "split": "DEV-TUNE-V2",
        "evidence_rule_version": "evidence-v320.1", "support_rule_version": "support-v316.1",
        "freeze": {"manifest_sha256": "a" * 64}, "metrics": metrics,
        "by_difficulty": {"L5": metrics}, "by_document_style": {"MIXED": metrics},
        "by_manufacturer_slice": {"UNSEEN": metrics}, "by_failure_class": {"value": metrics},
    }
    candidate = deepcopy(baseline)
    candidate["evidence_rule_version"] = "evidence-v321.1"
    candidate["metrics"] = {**metrics, "answerable_recall": .5}
    report = gate.compare_reports(baseline, candidate)
    assert report["metric_delta_candidate_minus_baseline"]["answerable_recall"] == pytest.approx(.1)

    candidate["freeze"]["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="COMPARISON_MANIFEST_MISMATCH"):
        gate.compare_reports(baseline, candidate)
