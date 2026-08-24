"""V3.56 focused tests: experimental table-structure producer feasibility.

The producer is exercised on hand-built real signal shapes (pypdf visitor/
operation equivalents) plus one tiny inline PDF through the genuine pypdf
extraction path. Private benchmark fixtures are not needed for these tests.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from backend.retrieval.evidence_table_ownership import TableOwnershipRelation
from backend.retrieval.table_structure_proof_contract_v351 import (
    FORBIDDEN_CONCLUSION_FIELDS,
    TABLE_STRUCTURE_PROOF_VERSION,
    TableStructureClaim,
    validate_table_structure_proof,
)
from backend.retrieval.table_structure_proof_contract_v354_candidate import (
    MERGED_CELL_COVERAGE_VERSION,
    validate_merged_cell_coverage_proof,
)
from backend.retrieval.table_structure_producer_v356 import (
    DECLINE_AMBIGUOUS_BINDING,
    DECLINE_BINDING_NOT_FOUND,
    MERGED_CELL_COVERAGE_VERSION_COMPAT,
    TABLE_STRUCTURE_PRODUCER_V356_STATUS,
    TABLE_STRUCTURE_PRODUCER_V356_VERSION,
    BindingRequest,
    PageSignal,
    TextFragmentSignal,
    bind_claim_to_table,
    extract_page_signals,
    infer_cell_role,
    reconstruct_tables,
)


HEADER_SIZE = 10.0
DATA_SIZE = 9.0


def _frag(x: float, y: float, text: str, size: float = DATA_SIZE) -> TextFragmentSignal:
    return TextFragmentSignal(text=text, x=x, y=y, size=size)


def _matrix_page_signal() -> PageSignal:
    """Models as row labels; model 25B vertically merged over three rows."""
    runs = [
        _frag(60, 740, "Model Current Ratings", 11),
        _frag(60, 700, "Model", HEADER_SIZE),
        _frag(180, 700, "Rated Current", HEADER_SIZE),
        _frag(300, 700, "Peak Current", HEADER_SIZE),
        _frag(60, 670, "PowerFlex 25A"),
        _frag(180, 670, "12 A"),
        _frag(300, 670, "20 A"),
        _frag(60, 650, "PowerFlex 25B"),
        _frag(180, 650, "18 A"),
        _frag(300, 650, "30 A"),
        _frag(180, 630, "24 A"),
        _frag(300, 630, "40 A"),
        _frag(180, 610, "30 A"),
        _frag(300, 610, "50 A"),
    ]
    return PageSignal(page_index=0, fragments=tuple(runs))


def _request(model="PowerFlex 25B", parameter="Peak Current", value="40 A",
             relation=TableOwnershipRelation.COLUMN_BOUND.value):
    return BindingRequest(
        model_text=model,
        parameter_text=parameter,
        value_or_action=value,
        claim_relation=relation,
        model_scope_id=f"model:{model.lower().replace(' ', '-')}",
        parameter_scope_id=f"parameter:{parameter.lower().replace(' ', '-')}",
    )


def test_producer_version_and_status_are_feasibility_only():
    assert TABLE_STRUCTURE_PRODUCER_V356_VERSION == "table-structure-producer-v356-feasibility"
    assert TABLE_STRUCTURE_PRODUCER_V356_STATUS == "EXPERIMENTAL_FEASIBILITY_ONLY"


def test_producer_is_not_wired_into_retrieval_package():
    # Feasibility producer must stay unwired: the package init must neither
    # import nor export it.
    import inspect

    import backend.retrieval as package

    source = inspect.getsource(package)
    assert "table_structure_producer_v356" not in source
    assert not any(
        getattr(obj, "__module__", "") == "backend.retrieval.table_structure_producer_v356"
        for obj in vars(package).values()
    )


def test_reconstruction_detects_grid_headers_and_vertical_merge():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    assert len(report.tables) == 1
    table = report.tables[0]
    assert len(table.header_row_ids) == 1
    assert table.section_caption == "Model Current Ratings"
    assert len(table.vertical_merges) == 1
    merge = table.vertical_merges[0]
    anchor = table.cells and [
        c for c in table.cells if c.row_id == merge.anchor_row_id and c.column_id == merge.column_id
    ]
    assert anchor and anchor[0].text == "PowerFlex 25B"
    assert len(merge.covered_row_ids) == 3


def test_reconstruction_is_deterministic_and_hashable():
    first = reconstruct_tables("doc-test", [_matrix_page_signal()])
    second = reconstruct_tables("doc-test", [_matrix_page_signal()])
    assert first.digest == second.digest != ""
    assert first.canonical_json() == second.canonical_json()


def test_column_bound_binding_with_coverage_rescue_validates():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    outcome = bind_claim_to_table(report.tables[0], _request())
    assert outcome.emitted and outcome.decline is None
    emission = outcome.emission
    result = validate_merged_cell_coverage_proof(
        emission.proof,
        TableStructureClaim(
            document_id="doc-test",
            ownership_relation=TableOwnershipRelation.COLUMN_BOUND.value,
            model_scope_id=_request().model_scope_id,
            parameter_scope_id=_request().parameter_scope_id,
            value_or_action="40 A",
        ),
        emission.coverage,
    )
    assert result.valid
    assert result.coverage_used
    assert "MODEL_ROW_OWNERSHIP_MISMATCH" in result.base_reason_codes


def test_cross_model_trap_declines_instead_of_guessing():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    # 12 A belongs to 25A's own row; it lies OUTSIDE the 25B merged span,
    # so a precision-first producer declines instead of binding cross-model.
    outcome = bind_claim_to_table(
        report.tables[0],
        _request(parameter="Rated Current", value="12 A"),
    )
    assert not outcome.emitted
    assert outcome.decline.code == DECLINE_BINDING_NOT_FOUND


def test_duplicate_anchor_labels_with_shared_value_decline_ambiguous():
    runs = [
        _frag(60, 700, "Model", HEADER_SIZE),
        _frag(180, 700, "Continuous", HEADER_SIZE),
        _frag(60, 670, "Drive X"),
        _frag(180, 670, "30 A"),
        _frag(180, 650, "31 A"),
        _frag(60, 630, "Drive X"),
        _frag(180, 630, "30 A"),
        _frag(180, 610, "33 A"),
    ]
    page = PageSignal(page_index=0, fragments=tuple(runs))
    report = reconstruct_tables("doc-dup", [page])
    assert len(report.tables[0].vertical_merges) == 2
    request = BindingRequest(
        model_text="Drive X",
        parameter_text="Continuous",
        value_or_action="30 A",
        claim_relation=TableOwnershipRelation.COLUMN_BOUND.value,
        model_scope_id="model:drive-x",
        parameter_scope_id="parameter:continuous",
    )
    outcome = bind_claim_to_table(report.tables[0], request)
    assert not outcome.emitted
    assert outcome.decline.code == DECLINE_AMBIGUOUS_BINDING


def test_absent_value_declines_not_found():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    outcome = bind_claim_to_table(report.tables[0], _request(value="999 A"))
    assert not outcome.emitted
    assert outcome.decline.code == DECLINE_BINDING_NOT_FOUND


def test_direct_row_route_on_parameter_label_table():
    runs = [
        _frag(60, 700, "Parameter", HEADER_SIZE),
        _frag(180, 700, "Value", HEADER_SIZE),
        _frag(60, 670, "Rated Voltage"),
        _frag(180, 670, "24 V"),
    ]
    page = PageSignal(page_index=0, fragments=tuple(runs))
    report = reconstruct_tables("doc-direct", [page])
    request = BindingRequest(
        model_text="AX-100",
        parameter_text="Rated Voltage",
        value_or_action="24 V",
        claim_relation=TableOwnershipRelation.DIRECT_ROW.value,
        model_scope_id="model:ax-100",
        parameter_scope_id="parameter:rated-voltage",
    )
    outcome = bind_claim_to_table(report.tables[0], request)
    assert outcome.emitted
    assert outcome.relation == TableOwnershipRelation.DIRECT_ROW.value
    result = validate_table_structure_proof(
        outcome.emission.proof,
        TableStructureClaim(
            document_id="doc-direct",
            ownership_relation=TableOwnershipRelation.DIRECT_ROW.value,
            model_scope_id=request.model_scope_id,
            parameter_scope_id=request.parameter_scope_id,
            value_or_action="24 V",
        ),
    )
    assert result.valid


def test_cross_reference_route_matches_reference_target():
    runs = [
        _frag(60, 700, "Item", HEADER_SIZE),
        _frag(180, 700, "Requirement", HEADER_SIZE),
        _frag(300, 700, "Reference", HEADER_SIZE),
        _frag(60, 670, "Shield Grounding"),
        _frag(180, 670, "Connect Shield"),
        _frag(300, 670, "See Table 7-2"),
    ]
    page = PageSignal(page_index=0, fragments=tuple(runs))
    report = reconstruct_tables("doc-ref", [page])
    request = BindingRequest(
        model_text="AX-100",
        parameter_text="Shield Grounding",
        value_or_action="Table 7-2",
        claim_relation=TableOwnershipRelation.CROSS_REFERENCE.value,
        model_scope_id="model:ax-100",
        parameter_scope_id="parameter:shield-grounding",
    )
    outcome = bind_claim_to_table(report.tables[0], request)
    assert outcome.emitted
    assert outcome.relation == TableOwnershipRelation.CROSS_REFERENCE.value
    proof = outcome.emission.proof
    assert proof.reference_target == "Table 7-2"
    assert proof.cell_role == "REFERENCE"


def test_emitted_coverage_matches_frozen_v354_candidate_contract():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    outcome = bind_claim_to_table(report.tables[0], _request())
    coverage = outcome.emission.coverage
    assert coverage["coverage_version"] == MERGED_CELL_COVERAGE_VERSION_COMPAT
    assert coverage["coverage_kind"] == "VERTICAL_MERGED_CELL"
    assert coverage["coverage_owner_scope_id"] == _request().model_scope_id


def test_extract_page_signals_reads_real_pypdf_operations():
    content = (
        b"BT /F1 9 Tf 1 0 0 1 60 700 Tm (Model) Tj ET\n"
        b"BT /F1 9 Tf 1 0 0 1 180 700 Tm (Value) Tj ET\n"
    )
    page = PdfReader(io.BytesIO(_minimal_pdf(content))).pages[0]
    signals = extract_page_signals(page, 3)
    assert signals.page_index == 3
    assert [(f.text, f.x, f.y, f.size) for f in signals.fragments] == [
        ("Model", 60.0, 700.0, 9.0),
        ("Value", 180.0, 700.0, 9.0),
    ]


def _minimal_pdf(content: bytes) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_cell_roles_are_deterministic_heuristics():
    assert infer_cell_role("See Table 7-2") == "REFERENCE"
    assert infer_cell_role("(factory default)") == "QUALIFIER"
    assert infer_cell_role("Press Stop") == "ACTION"
    assert infer_cell_role("24 V") == "VALUE"


def test_forbidden_conclusion_fields_still_rejected_by_frozen_contract():
    report = reconstruct_tables("doc-test", [_matrix_page_signal()])
    outcome = bind_claim_to_table(report.tables[0], _request())
    payload = outcome.emission.proof.as_dict()
    assert not (FORBIDDEN_CONCLUSION_FIELDS & payload.keys())
    payload["answer"] = "40 A"
    result = validate_table_structure_proof(
        payload,
        TableStructureClaim(
            document_id="doc-test",
            ownership_relation=TableOwnershipRelation.COLUMN_BOUND.value,
            model_scope_id=_request().model_scope_id,
            parameter_scope_id=_request().parameter_scope_id,
            value_or_action="40 A",
        ),
    )
    assert not result.valid


def test_public_protocol_constants_are_preregistered():
    from backend.evaluation.v356_producer_feasibility import (
        BENCHMARK_VERSION,
        FIXTURE_COUNT,
        GATE,
        PRODUCER_VERSION,
        V351_CONTRACT_SHA256,
        V354_CANDIDATE_SHA256,
        evaluate_feasibility_gate,
    )

    assert BENCHMARK_VERSION == "v356-producer-feasibility-fixtures-v1"
    assert PRODUCER_VERSION == TABLE_STRUCTURE_PRODUCER_V356_VERSION
    assert FIXTURE_COUNT >= 64
    assert GATE["max_unsafe_structure_acceptance"] == 0
    gate = evaluate_feasibility_gate(
        {
            "condition_B_producer": {
                "valid_acceptance": 1.0,
                "invalid_rejection": 1.0,
                "unsafe_structure_acceptance": 0,
                "ownership_precision": 1.0,
            },
            "structure_means": {"merge_precision": 0.95},
            "condition_C_flat_control": {"invalid_accepted_rate": 0.5},
            "replication_stability": {"applicable": 7, "stable": 7},
        },
    )
    assert gate["all_passed"]
    failing = evaluate_feasibility_gate(
        {
            "condition_B_producer": {
                "valid_acceptance": 0.9,
                "invalid_rejection": 1.0,
                "unsafe_structure_acceptance": 1,
                "ownership_precision": 1.0,
            },
            "structure_means": {"merge_precision": 0.95},
            "condition_C_flat_control": {"invalid_accepted_rate": 0.5},
            "replication_stability": {"applicable": 7, "stable": 7},
        },
    )
    assert not failing["all_passed"]
