from backend.evaluation import v353_vertical_merged_cell_coverage_contract_reassessment as v


def _manifest() -> dict:
    return {
        "benchmark_version": v.BENCHMARK_VERSION,
        "source_benchmark_version": v.SOURCE_BENCHMARK_VERSION,
        "source_fixture_count": v.SOURCE_FIXTURE_COUNT,
        "source_file_sha256": v.SOURCE_FILE_SHA256,
        "contract_sha256": v.CONTRACT_SHA256,
        "read_only_reassessment": True,
        "reran_v352": False,
        "modified_v351_contract": False,
        "modified_parser": False,
    }


def test_manifest_requires_frozen_read_only_v352_source():
    assert v.validate_source_manifest(_manifest()) == ()
    changed = _manifest()
    changed["reran_v352"] = True
    assert "V352_RERUN_FORBIDDEN" in v.validate_source_manifest(changed)


def test_option_assessment_selects_one_minimum_safe_representation():
    result = v.assess_representation_options()
    assert result["unique_minimum_safe"]
    assert result["selected"] == ["ANCHOR_PLUS_COVERED_ROW_IDS"]
    assert not result["options"]["SPAN_ARITHMETIC"]["safe"]
    assert not result["options"]["DUPLICATE_MODEL_ROW_ID"]["safe"]
    assert result["options"]["FULL_CELL_OWNERSHIP_GRAPH"]["safe"]
    assert not result["options"]["FULL_CELL_OWNERSHIP_GRAPH"]["selected"]


def test_complete_reassessment_preserves_guards_and_routes_to_candidate_definition():
    summary = {
        "vertical_fixtures": 8, "vertical_valid": 4, "vertical_invalid": 4,
        "false_rejections": 4, "unsafe_acceptances": 0,
        "schema_gap_annotations": 4, "safe_guard_annotations": 4,
    }
    result = v.decide(
        summary, v.assess_representation_options(), integrity=True,
        annotations_complete=True, source_reconciled=True,
    )
    assert result["status"] == "CONTRACT_REASSESSMENT_COMPLETE"
    assert result["minimum_safe_representation"] == "ANCHOR_PLUS_COVERED_ROW_IDS"
    summary["safe_guard_annotations"] = 3
    assert v.decide(
        summary, v.assess_representation_options(), integrity=True,
        annotations_complete=True, source_reconciled=True,
    )["status"] == "RUNTIME_INVALID"
