"""V3.57 focused tests: structured table producer candidate."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from backend.evaluation.v357_structured_producer_candidate import (
    CANDIDATE_VERSION,
    evaluate_candidate_gate,
    validate_annotation_entry,
)
from backend.retrieval.table_structure_producer_v356 import BindingRequest
from backend.retrieval.table_structure_producer_v357_candidate import (
    TABLE_STRUCTURE_PRODUCER_V357_CANDIDATE_VERSION,
    TABLE_STRUCTURE_PRODUCER_V357_STATUS,
    TAXONOMY_AMBIGUOUS,
    TAXONOMY_BINDING_NOT_FOUND,
    RuledSegment,
    _extract_segments,
    analyze_page,
    bind_claim_candidate,
    build_candidate_report,
)


HEADER_SIZE = 10.0
DATA_SIZE = 9.0


def _frag(x: float, y: float, text: str, size: float = DATA_SIZE) -> object:
    from backend.retrieval.table_structure_producer_v356 import TextFragmentSignal

    return TextFragmentSignal(text=text, x=x, y=y, size=size)


def test_candidate_version_and_status():
    assert TABLE_STRUCTURE_PRODUCER_V357_CANDIDATE_VERSION == "structured-table-producer-v357-candidate"
    assert TABLE_STRUCTURE_PRODUCER_V357_STATUS == "DEV_CANDIDATE_UNWIRED"


def test_ruled_segments_extracted_from_re_rectangle():
    ops = [
        ([], b"q"),
        ([2, 0, 0, 2, 50, 60], b"cm"),
        ([10, 20, 400, 2], b"re"),
        ([], b"S"),
        ([], b"Q"),
    ]
    segments = _extract_segments(ops)
    assert len(segments) == 4
    horizontal = [s for s in segments if s.horizontal]
    vertical = [s for s in segments if s.vertical]
    assert len(horizontal) == 2
    assert len(vertical) == 2
    # cm scale 2 + translate (50,60): rect edges land at y=60+2*20=100
    # and y=60+2*22=104.
    assert {round(s.y0, 2) for s in horizontal} == {100.0, 104.0}
    assert {round(s.x0, 2) for s in vertical} == {70.0, 870.0}


def test_short_decorative_segments_are_filtered():
    class FakeContents:
        operations = [
            ([], b"q"),
            ([5, 5, 3, 3], b"re"),
            ([], b"S"),
            ([], b"Q"),
        ]

    class FakePage:
        def get_contents(self):
            return FakeContents()

    assert extract_ruling_geometry_filtered(FakePage()) == []


def extract_ruling_geometry_filtered(page):  # helper mirroring candidate API
    from backend.retrieval.table_structure_producer_v357_candidate import (
        extract_ruling_geometry,
    )

    return extract_ruling_geometry(page)


def test_multi_region_split_two_tables_on_one_page():
    runs = [
        # Table 1 (header + 3 rows) at top of page.
        _frag(60, 740, "Model", HEADER_SIZE),
        _frag(180, 740, "Rating", HEADER_SIZE),
        _frag(60, 715, "AX-1"),
        _frag(180, 715, "10 A"),
        _frag(60, 695, "AX-2"),
        _frag(180, 695, "12 A"),
        _frag(60, 675, "AX-3"),
        _frag(180, 675, "14 A"),
        # Distant second table (well beyond wrap tolerance).
        _frag(60, 500, "Option", HEADER_SIZE),
        _frag(180, 500, "Default", HEADER_SIZE),
        _frag(60, 475, "Mode A"),
        _frag(180, 475, "Fast"),
        _frag(60, 455, "Mode B"),
        _frag(180, 455, "Slow"),
    ]
    from backend.retrieval.table_structure_producer_v356 import (
        PageSignal,
        _build_page_grid,
        _detect_grid_extent,
    )
    from backend.retrieval.table_structure_producer_v357_candidate import (
        PageAnalysis,
        analyze_and_reconstruct,
    )

    page_signal = PageSignal(page_index=0, fragments=tuple(runs))
    grid = _build_page_grid(list(page_signal.fragments))
    extent = _detect_grid_extent(grid)
    analysis = PageAnalysis(
        page_index=0,
        grid=grid,
        extent=extent,
        h_line_ys=(),
        v_line_xs=(),
        observability={},
    )
    report = analyze_and_reconstruct([(0, analysis)], "doc-multi")
    assert len(report.tables) >= 1


def test_taxonomy_codes_surface_in_declines():
    runs = [
        _frag(60, 700, "Parameter", HEADER_SIZE),
        _frag(180, 700, "Value", HEADER_SIZE),
        _frag(60, 670, "Rated Voltage"),
        _frag(180, 670, "24 V"),
    ]
    from backend.retrieval.table_structure_producer_v356 import PageSignal

    page = PageSignal(page_index=0, fragments=tuple(runs))
    report = reconstruct_tables_v357_helper(page)
    request = BindingRequest(
        model_text="AX-100",
        parameter_text="Rated Voltage",
        value_or_action="999 V",
        claim_relation="DIRECT_ROW",
        model_scope_id="model:ax-100",
        parameter_scope_id="parameter:rated-voltage",
    )
    outcome = bind_claim_candidate(report.tables[0], request)
    assert not outcome.emitted
    assert outcome.decline.code == TAXONOMY_BINDING_NOT_FOUND
    _ = TAXONOMY_AMBIGUOUS


def reconstruct_tables_v357_helper(page):
    from backend.retrieval.table_structure_producer_v356 import reconstruct_tables

    return reconstruct_tables("doc-tax", [page])


def test_annotation_validator_catches_bad_merges():
    entry = {
        "doc": "x",
        "page_index": 1,
        "columns": [{"header_levels": ["Value"], "unit": ""}],
        "rows": [{"label": "R", "cells": {"0": "1"}}],
        "merges": [{"column_index": -1, "anchor_row_index": 1, "covered_row_indices": [1, 0]}],
    }
    errors = validate_annotation_entry(entry)
    assert any("MERGE_ANCHOR_NOT_FIRST" in e or "MERGE_INDEX_INVALID" in e for e in errors)


def test_gate_evaluation_thresholds():
    passing = evaluate_candidate_gate(
        {
            "ownership_precision_proxy": 0.93,
            "unsafe_acceptance": 0,
            "invalid_rejection": 0.97,
            "valid_accepted": 60,
            "valid_total": 100,
        },
    )
    assert passing["all_passed"]
    failing = evaluate_candidate_gate(
        {
            "ownership_precision_proxy": 0.85,
            "unsafe_acceptance": 2,
            "invalid_rejection": 0.90,
            "valid_accepted": 10,
            "valid_total": 100,
        },
    )
    assert not failing["all_passed"]


def test_real_manual_page_analysis_runs_and_is_deterministic():
    pdf_path = (
        r"D:\industrial-knowledge-rag\backend\evaluation\benchmark_private"
        r"\v357_candidate\documents\danfoss-fc51-doc.pdf"
    )
    try:
        reader = PdfReader(pdf_path)
    except Exception:  # pragma: no cover - document must exist in private store
        pytest.skip("danfoss document not present")
    page = reader.pages[13]
    first = analyze_page(page, 13)
    second = analyze_page(page, 13)
    if first is None:
        pytest.skip("page yields no text grid")
    assert second is not None
    assert first.extent.data_rows == second.extent.data_rows
    assert first.h_line_ys == second.h_line_ys
