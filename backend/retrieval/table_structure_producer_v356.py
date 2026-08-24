"""V3.56 experimental merged-cell coverage structure producer (feasibility).

FEASIBILITY PROTOTYPE only. Not wired into retrieval, ingestion, Evidence
runtime, Identity, Support, or any API. It reconstructs table structure
exclusively from really obtainable parser signals (text fragments with Tm
start coordinates delivered by pypdf visitor callbacks) and emits objects
compatible with the frozen V3.51 TableStructureProof contract and the frozen
V3.54 MergedCellCoverage candidate. It makes no ANSWER/ABSTAIN decision,
performs no query reasoning, and judges no evidence sufficiency.

Signal limitations honored (V3.56 input-signal audit):
- no HTML/XML table objects and no explicit merged-cell hints exist;
- regions, rows, and columns derive from coordinate clustering of fragment
  START positions only (no glyph-width estimation);
- rotated-text pages degrade extraction and stay out of scope;
- precision-first: the producer declines rather than guesses.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

from .table_structure_proof_contract_v351 import (
    TABLE_STRUCTURE_PROOF_VERSION,
    TableStructureProof,
)


TABLE_STRUCTURE_PRODUCER_V356_VERSION = "table-structure-producer-v356-feasibility"
TABLE_STRUCTURE_PRODUCER_V356_STATUS = "EXPERIMENTAL_FEASIBILITY_ONLY"

X_CLUSTER_TOLERANCE = 3.0
Y_CLUSTER_TOLERANCE = 2.5
MIN_GRID_ROWS = 2
MIN_GRID_COLUMNS = 2
MIN_ROW_OCCUPIED_COLUMNS = 2
MAX_WRAP_GAP_STEPS = 2          # sparse cluster steps tolerated inside the grid
HEADER_MAX_PITCH_FACTOR = 1.9   # header rows sit within regular pitch of the grid
CAPTION_MIN_GAP_FACTOR = 1.45   # captions start beyond this many pitches above
CAPTION_MAX_GAP_FACTOR = 5.0

_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:see|refer\s+to|cf\.?|per)\s+(.+?)\s*$",
    re.IGNORECASE,
)
_ACTION_VERBS = frozenset({
    "press", "hold", "check", "install", "remove", "set", "select", "turn",
    "verify", "connect", "disconnect", "ground", "cycle", "replace", "clean",
    "tighten", "adjust", "insert", "rotate", "slide", "lift", "store",
})
_QUALIFIER_PATTERN = re.compile(r"^\(.*\)$")

DECLINE_NO_TABLE = "NO_TABLE_RECONSTRUCTED"
DECLINE_BINDING_NOT_FOUND = "BINDING_NOT_FOUND"
DECLINE_AMBIGUOUS_BINDING = "AMBIGUOUS_BINDING"


def norm_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9.%+/-]+", " ", str(value or "").casefold()).split())


def slug(value: Any) -> str:
    return norm_text(value).replace(" ", "-")


def stable_suffix(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def infer_cell_role(text: str) -> str:
    normalized = text.strip()
    if _REFERENCE_PATTERN.match(normalized):
        return "REFERENCE"
    if _QUALIFIER_PATTERN.match(normalized):
        return "QUALIFIER"
    first_word = re.split(r"[\s/]+", normalized, maxsplit=1)[0].casefold().strip(".,;:")
    if first_word in _ACTION_VERBS:
        return "ACTION"
    return "VALUE"


def reference_target(text: str) -> str:
    match = _REFERENCE_PATTERN.match(text.strip())
    return match.group(1) if match else ""


# --------------------------------------------------------------------------
# Input signal types (exactly what pypdf visitor callbacks really deliver)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextFragmentSignal:
    """One text-showing operation: text, Tm start point, requested Tf size."""

    text: str
    x: float
    y: float
    size: float = 0.0


@dataclass(frozen=True)
class PageSignal:
    page_index: int
    fragments: tuple[TextFragmentSignal, ...]


def extract_page_signals(page: Any, page_index: int) -> PageSignal:
    """Read real signals from a pypdf page object (experimental reader).

    Reads pypdf's own parsed content-stream operations (the production
    parser library) and reports each text-showing operation together with
    its effective Tm translation. Visitor-callback granularity proved
    unreliable in pypdf 6.14 for same-line horizontal advances (memoized
    positions reset to (0,0) with phantom leading spaces), so the
    operation level is used instead; both are real parser output.
    """
    fragments: list[TextFragmentSignal] = []
    contents = page.get_contents()
    operations = getattr(contents, "operations", None) or ()
    tx = ty = 0.0
    size = 0.0
    in_text = False
    for operands, operator in operations:
        name = operator.decode("ascii", "replace") if isinstance(operator, bytes) else str(operator)
        if name == "BT":
            in_text = True
            continue
        if name == "ET":
            in_text = False
            continue
        if name == "Tf" and len(operands) >= 2:
            try:
                size = float(operands[1])
            except (TypeError, ValueError):
                pass
            continue
        if name == "Tm" and len(operands) >= 6:
            try:
                tx, ty = float(operands[4]), float(operands[5])
            except (TypeError, ValueError):
                pass
            continue
        if name == "Td" and len(operands) >= 2:
            try:
                tx += float(operands[0])
                ty += float(operands[1])
            except (TypeError, ValueError):
                pass
            continue
        if name in {"Tj", "TJ", "'", '"'} and in_text:
            pieces: list[str] = []
            for operand in operands:
                if isinstance(operand, str):
                    pieces.append(operand)
                elif isinstance(operand, bytes):
                    pieces.append(operand.decode("latin-1", "replace"))
                elif isinstance(operand, (list, tuple)):
                    for item in operand:
                        if isinstance(item, (str, bytes)):
                            pieces.append(
                                item if isinstance(item, str)
                                else item.decode("latin-1", "replace"),
                            )
            text = "".join(pieces)
            if text and text.strip():
                fragments.append(TextFragmentSignal(text.strip(), tx, ty, size))
    return PageSignal(page_index, tuple(fragments))


# --------------------------------------------------------------------------
# Reconstructed structure types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerticalMergeRegion:
    column_id: str
    anchor_row_id: str
    covered_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProducerCell:
    row_id: str
    column_id: str
    text: str
    cell_role: str


@dataclass(frozen=True)
class ProducerTable:
    document_id: str
    table_region_id: str
    page_indices: tuple[int, ...]
    column_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    header_row_ids: tuple[str, ...]
    cells: tuple[ProducerCell, ...]
    header_paths: tuple[tuple[str, tuple[str, ...]], ...]
    vertical_merges: tuple[VerticalMergeRegion, ...] = ()
    section_caption: str = ""
    section_caption_page: int = -1

    def cell_map(self) -> dict[tuple[str, str], ProducerCell]:
        return {(cell.row_id, cell.column_id): cell for cell in self.cells}

    def header_path_for(self, column_id: str) -> tuple[str, ...]:
        return dict(self.header_paths).get(column_id, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "table_region_id": self.table_region_id,
            "page_indices": list(self.page_indices),
            "column_ids": list(self.column_ids),
            "row_ids": list(self.row_ids),
            "header_row_ids": list(self.header_row_ids),
            "cells": [
                {
                    "row_id": cell.row_id,
                    "column_id": cell.column_id,
                    "text": cell.text,
                    "cell_role": cell.cell_role,
                }
                for cell in self.cells
            ],
            "header_paths": [
                {"column_id": column, "path": list(path)}
                for column, path in self.header_paths
            ],
            "vertical_merges": [
                {
                    "column_id": merge.column_id,
                    "anchor_row_id": merge.anchor_row_id,
                    "covered_row_ids": list(merge.covered_row_ids),
                }
                for merge in self.vertical_merges
            ],
            "section_caption": self.section_caption,
            "section_caption_page": self.section_caption_page,
        }


@dataclass(frozen=True)
class ReconstructionReport:
    tables: tuple[ProducerTable, ...]
    digest: str = ""

    def canonical_json(self) -> str:
        return json.dumps(
            [table.as_dict() for table in self.tables],
            sort_keys=True,
            ensure_ascii=False,
        )

    def result_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProducerDecline:
    code: str
    detail: str = ""


@dataclass(frozen=True)
class ProducerEmission:
    proof: TableStructureProof
    coverage: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proof": self.proof.as_dict(),
            "coverage": dict(self.coverage) if self.coverage else None,
        }


@dataclass(frozen=True)
class ProducerOutcome:
    emission: ProducerEmission | None
    decline: ProducerDecline | None
    relation: str = ""

    @property
    def emitted(self) -> bool:
        return self.emission is not None


MERGED_CELL_COVERAGE_VERSION_COMPAT = "merged-cell-coverage-v354-candidate"


# --------------------------------------------------------------------------
# Grid reconstruction internals
# --------------------------------------------------------------------------


def _cluster(values: list[float], tolerance: float) -> list[list[float]]:
    ordered = sorted(values)
    groups: list[list[float]] = []
    for value in ordered:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


@dataclass(frozen=True)
class _PageGrid:
    row_center_y: dict[int, float]          # cluster index -> mean y (ascending index = higher y)
    col_center_x: dict[int, float]
    cells: dict[tuple[int, int], list[tuple[str, float]]]   # (row, col) -> [(text, size)]


def _build_page_grid(frags: list[TextFragmentSignal]) -> _PageGrid | None:
    x_groups = _cluster([f.x for f in frags], X_CLUSTER_TOLERANCE)
    y_groups = _cluster([f.y for f in frags], Y_CLUSTER_TOLERANCE)
    if len(x_groups) < MIN_GRID_COLUMNS or len(y_groups) < MIN_GRID_ROWS:
        return None
    col_centers = [sum(g) / len(g) for g in x_groups]
    row_centers = [sum(g) / len(g) for g in y_groups]

    def nearest(centers: list[float], point: float, tolerance: float) -> int | None:
        best_index, best_distance = None, tolerance
        for index, center in enumerate(centers):
            distance = abs(center - point)
            if distance <= best_distance:
                best_index, best_distance = index, distance
        return best_index

    cells: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for fragment in frags:
        column = nearest(col_centers, fragment.x, X_CLUSTER_TOLERANCE)
        row = nearest(row_centers, fragment.y, Y_CLUSTER_TOLERANCE)
        if column is None or row is None:
            continue
        cells.setdefault((row, column), []).append((fragment.text, fragment.size))

    return _PageGrid(
        row_center_y={i: c for i, c in enumerate(row_centers)},
        col_center_x={i: c for i, c in enumerate(col_centers)},
        cells=cells,
    )


@dataclass(frozen=True)
class _GridExtent:
    data_rows: list[int]     # cluster indices, TOP to BOTTOM (descending y)
    header_rows: list[int]   # cluster indices above the core, TOP to BOTTOM
    columns: list[int]       # cluster indices, LEFT to RIGHT
    pitch_points: float
    caption: str = ""


def _cell_text(grid: _PageGrid, row: int, column: int) -> str:
    entries = grid.cells.get((row, column))
    if not entries:
        return ""
    return " ".join(text for text, _size in entries).strip()


def _row_sizes(grid: _PageGrid, row: int) -> list[float]:
    sizes: list[float] = []
    for (r, _c), entries in grid.cells.items():
        if r == row:
            sizes.extend(size for _text, size in entries)
    return sizes


def _detect_grid_extent(grid: _PageGrid) -> _GridExtent | None:
    row_columns: dict[int, set[int]] = {}
    for (row, column) in grid.cells:
        row_columns.setdefault(row, set()).add(column)
    dense = [
        row for row, cols in row_columns.items()
        if len(cols) >= MIN_ROW_OCCUPIED_COLUMNS
    ]
    if len(dense) < MIN_GRID_ROWS:
        return None

    # Cluster indices ascend with y; the topmost row has the largest index.
    dense_top_first = sorted(dense, key=lambda r: -grid.row_center_y[r])
    runs: list[list[int]] = [[dense_top_first[0]]]
    for row in dense_top_first[1:]:
        if runs[-1][-1] - row <= MAX_WRAP_GAP_STEPS:
            runs[-1].append(row)
        else:
            runs.append([row])
    seed = max(runs, key=len)
    if len(seed) < MIN_GRID_ROWS:
        return None

    def _mode_size(rows: list[int]) -> float:
        counts: dict[float, int] = {}
        for row in rows:
            for size in _row_sizes(grid, row):
                key = round(size, 1)
                counts[key] = counts.get(key, 0) + 1
        return max(counts, key=lambda k: (counts[k], k)) if counts else 0.0

    provisional_mode = _mode_size(seed)

    # Iterative dilation: absorb occupied clusters within wrap-gap reach of
    # the core. Vertically merged spans leave single-cell interior rows that
    # must rejoin the grid; captions (distinct type size) stay outside.
    core_set = set(seed)
    changed = True
    while changed:
        changed = False
        for row in list(row_columns):
            if row in core_set:
                continue
            if not any(abs(row - member) <= MAX_WRAP_GAP_STEPS for member in core_set):
                continue
            sizes = _row_sizes(grid, row)
            looks_like_grid = (
                len(row_columns[row]) >= MIN_ROW_OCCUPIED_COLUMNS
                or bool(sizes)
                and (
                    abs(round(max(sizes), 1) - provisional_mode) <= 0.25
                    or abs(round(min(sizes), 1) - provisional_mode) <= 0.25
                )
            )
            if looks_like_grid:
                core_set.add(row)
                changed = True
    core = sorted(core_set, reverse=True)  # top-first
    if len(core) < MIN_GRID_ROWS:
        return None

    columns_used: set[int] = set()
    for row in core:
        columns_used |= row_columns[row]
    columns = sorted(columns_used)
    if len(columns) < MIN_GRID_COLUMNS:
        return None

    gaps = [
        abs(grid.row_center_y[b] - grid.row_center_y[a])
        for a, b in zip(core, core[1:])
    ]
    pitch = statistics.median(gaps) if gaps else 14.0

    # Data-size mode: most common rounded Tf size inside the final core.
    data_mode_size = _mode_size(core)

    # Leading core rows set in a clearly larger type are the header band.
    # Contiguity with the grid is guaranteed by run construction.
    header_rows: list[int] = []
    remaining = list(core)
    while remaining:
        candidate = remaining[0]
        sizes = _row_sizes(grid, candidate)
        if sizes and min(sizes) >= data_mode_size + 0.25:
            header_rows.append(candidate)
            remaining.pop(0)
        else:
            break
    data_rows = remaining

    # Section caption: nearest standalone line above, beyond caption gap.
    caption = ""
    top_row = core[0]
    candidates_above = [
        row for row in row_columns
        if row not in core and row not in header_rows
    ]
    for row in sorted(candidates_above, reverse=True):
        gap = grid.row_center_y[row] - grid.row_center_y[top_row]
        if gap < CAPTION_MIN_GAP_FACTOR * pitch or gap > CAPTION_MAX_GAP_FACTOR * pitch:
            continue
        texts = [_cell_text(grid, row, column) for column in columns]
        caption = " ".join(t for t in texts if t).strip()
        break

    return _GridExtent(
        data_rows=data_rows,
        header_rows=header_rows,
        columns=columns,
        pitch_points=pitch,
        caption=caption,
    )


def _detect_vertical_merges(
    extent: _GridExtent,
    grid: _PageGrid,
) -> list[tuple[int, int, list[int]]]:
    """Return (col_cluster, anchor_row_cluster, covered_row_clusters) triples."""
    merges: list[tuple[int, int, list[int]]] = []
    data_rows = extent.data_rows
    for column in extent.columns:
        index = 0
        while index < len(data_rows):
            row = data_rows[index]
            if not _cell_text(grid, row, column):
                index += 1
                continue
            covered = [row]
            cursor = index + 1
            while cursor < len(data_rows):
                below = data_rows[cursor]
                if _cell_text(grid, below, column):
                    break
                populated_elsewhere = any(
                    _cell_text(grid, below, other)
                    for other in extent.columns
                    if other != column
                )
                if not populated_elsewhere:
                    break
                covered.append(below)
                cursor += 1
            if len(covered) >= 2:
                merges.append((column, covered[0], list(covered)))
            index = max(cursor, index + 1)
    return merges


def reconstruct_tables(document_id: str, pages: Iterable[PageSignal]) -> ReconstructionReport:
    """Deterministically reconstruct tables from real page signals."""
    tables: list[ProducerTable] = []
    for page in pages:
        page_frags = [f for f in page.fragments if f.text and f.text.strip()]
        if len(page_frags) < 3:
            continue
        grid = _build_page_grid(page_frags)
        if grid is None:
            continue
        extent = _detect_grid_extent(grid)
        if extent is None:
            continue

        all_rows = extent.header_rows + extent.data_rows  # top to bottom
        region_key = stable_suffix(document_id, str(page.page_index), ",".join(map(str, all_rows)))
        region_id = f"{document_id}:tbl:p{page.page_index}:{region_key}"
        column_ids = {c: f"{region_id}:c{i:02d}" for i, c in enumerate(extent.columns)}
        row_ids = {r: f"{region_id}:r{i:03d}" for i, r in enumerate(all_rows)}
        header_set = set(extent.header_rows)

        cells: list[ProducerCell] = []
        for row in all_rows:
            for column in extent.columns:
                text = _cell_text(grid, row, column)
                if not text:
                    continue
                role = "HEADER" if row in header_set else infer_cell_role(text)
                cells.append(ProducerCell(row_ids[row], column_ids[column], text, role))

        header_paths = _build_header_paths(extent, grid, region_id, column_ids)
        merges = tuple(
            VerticalMergeRegion(
                column_id=column_ids[column],
                anchor_row_id=row_ids[anchor],
                covered_row_ids=tuple(row_ids[r] for r in covered),
            )
            for column, anchor, covered in _detect_vertical_merges(extent, grid)
        )
        caption_row = extent.caption

        tables.append(
            ProducerTable(
                document_id=document_id,
                table_region_id=region_id,
                page_indices=(page.page_index,),
                column_ids=tuple(column_ids[c] for c in extent.columns),
                row_ids=tuple(row_ids[r] for r in all_rows),
                header_row_ids=tuple(row_ids[r] for r in extent.header_rows),
                cells=tuple(cells),
                header_paths=tuple(header_paths),
                vertical_merges=merges,
                section_caption=caption_row,
                section_caption_page=page.page_index if caption_row else -1,
            ),
        )
    report = ReconstructionReport(tables=tuple(tables))
    return ReconstructionReport(tables=report.tables, digest=report.result_digest())


def _build_header_paths(
    extent: _GridExtent,
    grid: _PageGrid,
    region_id: str,
    column_ids: dict[int, str],
) -> list[tuple[str, tuple[str, ...]]]:
    if not extent.header_rows:
        return []
    leaf_row = extent.header_rows[-1]
    upper_rows = extent.header_rows[:-1]
    paths: list[tuple[str, tuple[str, ...]]] = []
    for column in extent.columns:
        leaf_text = _cell_text(grid, leaf_row, column)
        if not leaf_text:
            continue
        ancestors: list[str] = []
        for upper in upper_rows:
            keys = sorted(
                (c for c in extent.columns if _cell_text(grid, upper, c)),
            )
            if not keys:
                continue
            at_or_left = [c for c in keys if c <= column]
            chosen = at_or_left[-1] if at_or_left else keys[0]
            ancestors.append(_cell_text(grid, upper, chosen))
        path_texts = [*ancestors, leaf_text]
        # Header-path entries carry the rendered header texts verbatim; the
        # frozen contract treats them as opaque strings and only requires
        # the claim's parameter scope id to appear among them.
        paths.append((column_ids[column], tuple(path_texts)))
    return paths


# --------------------------------------------------------------------------
# Claim binding (structure location only; no answer/decision semantics)
# --------------------------------------------------------------------------


def _strip_unit_suffix(text: str) -> str:
    """'Voltage (V)' and 'Voltage' compare equal after normalization."""
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return norm_text(stripped) if stripped else norm_text(text)


def header_scope_id(parameter_scope_id: str, header_text: str, parameter_text: str) -> str:
    """Path entries carrying the parameter text reuse the claim's scope id."""
    if (
        norm_text(header_text) == norm_text(parameter_text)
        or _strip_unit_suffix(header_text) == norm_text(parameter_text)
    ) and parameter_scope_id:
        return parameter_scope_id
    return f"header:{slug(header_text)}"


@dataclass(frozen=True)
class BindingRequest:
    model_text: str
    parameter_text: str
    value_or_action: str
    claim_relation: str
    model_scope_id: str
    parameter_scope_id: str
    qualifier_scope_id: str = ""
    section_scope_id: str = ""


def bind_claim_to_table(
    table: ProducerTable,
    request: BindingRequest,
) -> ProducerOutcome:
    cell_map = {(cell.row_id, cell.column_id): cell for cell in table.cells}
    label_column = table.column_ids[0] if table.column_ids else ""
    data_rows = [r for r in table.row_ids if r not in set(table.header_row_ids)]

    model_label_cells = [
        cell for cell in table.cells
        if cell.column_id == label_column and norm_text(cell.text) == norm_text(request.model_text)
    ]
    model_header_cells = [
        cell for cell in table.cells
        if cell.cell_role == "HEADER" and norm_text(cell.text) == norm_text(request.model_text)
    ]
    # Only LEAF header cells (bottom header row) can anchor a
    # parameter-column; ancestor cells belong to lineage inheritance.
    leaf_header_row = table.header_row_ids[-1] if table.header_row_ids else ""
    param_header_cells = {
        cell.column_id
        for cell in table.cells
        if cell.cell_role == "HEADER"
        and cell.row_id == leaf_header_row
        and norm_text(cell.text) == norm_text(request.parameter_text)
    }
    param_label_cells = [
        cell for cell in table.cells
        if cell.column_id == label_column
        and cell.cell_role != "HEADER"
        and norm_text(cell.text) == norm_text(request.parameter_text)
    ]

    candidates: list[ProducerEmission] = []

    def add(relation: str, *, model_row: str, value_row: str, parameter_column: str,
            value_column: str, header_path: tuple[str, ...], cell_role: str,
            model_column: str = "", reference_target_text: str = "") -> None:
        proof = _compose_proof(
            table, request,
            relation=relation,
            model_row=model_row,
            value_row=value_row,
            parameter_column=parameter_column,
            value_column=value_column,
            header_path=header_path,
            cell_role=cell_role,
            model_column=model_column,
            reference_target_text=reference_target_text,
        )
        coverage = None
        if relation == "COLUMN_BOUND" and model_row and value_row and model_row != value_row:
            merge = _find_merged_span(table, model_row, value_row)
            if merge is not None:
                coverage = _compose_coverage(table, request, merge, proof)
            else:
                return  # cross-row binding without proven coverage is not emitted
        candidates.append(ProducerEmission(proof, coverage))

    # ---- Route A: COLUMN_BOUND — models as row labels ------------------
    if model_label_cells and param_header_cells:
        for model_cell in model_label_cells:
            for value_column in sorted(param_header_cells):
                for row in data_rows:
                    value_cell = cell_map.get((row, value_column))
                    if value_cell is None or value_cell.cell_role == "REFERENCE":
                        continue
                    if norm_text(value_cell.text) != norm_text(request.value_or_action):
                        continue
                    raw_path = table.header_path_for(value_column)
                    header_path = tuple(
                        header_scope_id(
                            request.parameter_scope_id,
                            entry,
                            request.parameter_text,
                        )
                        for entry in raw_path
                    )
                    if request.parameter_scope_id not in header_path:
                        continue
                    add(
                        "COLUMN_BOUND",
                        model_row=model_cell.row_id,
                        value_row=row,
                        parameter_column=value_column,
                        value_column=value_column,
                        header_path=header_path,
                        cell_role=value_cell.cell_role,
                    )

    # ---- Route B: parameter as row label --------------------------------
    for param_cell in param_label_cells:
        row = param_cell.row_id
        model_column_restriction: str | None = None
        if len(model_header_cells) > 1:
            continue
        if len(model_header_cells) == 1:
            model_column_restriction = model_header_cells[0].column_id
        elif model_label_cells:
            continue
        for cell in table.cells:
            if cell.row_id != row or cell.column_id == label_column:
                continue
            if cell.cell_role in {"HEADER", "REFERENCE"}:
                continue
            if model_column_restriction and cell.column_id != model_column_restriction:
                continue
            if norm_text(cell.text) != norm_text(request.value_or_action):
                continue
            if request.section_scope_id:
                caption_slug = f"section:{slug(table.section_caption)}" if table.section_caption else ""
                if caption_slug != request.section_scope_id:
                    continue
                add(
                    "SECTION_INHERITED",
                    model_row=row,
                    value_row=row,
                    parameter_column=label_column,
                    value_column=cell.column_id,
                    header_path=(),
                    cell_role=cell.cell_role,
                )
            else:
                add(
                    "DIRECT_ROW",
                    model_row=row,
                    value_row=row,
                    parameter_column=label_column,
                    value_column=cell.column_id,
                    header_path=(),
                    cell_role=cell.cell_role,
                    model_column=model_column_restriction or "",
                )

    # ---- Route C: HEADER_INHERITED — parameter only in header lineage ---
    # Suppressed when the table already uses model-row x parameter-column
    # topology (Route A territory): otherwise the same binding would be
    # produced under two relations.
    leaf_header_texts = {
        norm_text(cell.text)
        for cell in table.cells
        if cell.cell_role == "HEADER"
        and cell.row_id == leaf_header_row
        and cell.column_id != label_column
    }
    column_bound_topology = bool(model_label_cells) and (
        norm_text(request.parameter_text) in leaf_header_texts
    )
    if not param_label_cells and not column_bound_topology:
        for cell in table.cells:
            if cell.cell_role in {"HEADER", "REFERENCE"}:
                continue
            if cell.column_id == label_column:
                continue
            if norm_text(cell.text) != norm_text(request.value_or_action):
                continue
            raw_path = table.header_path_for(cell.column_id)
            header_path = tuple(
                header_scope_id(
                    request.parameter_scope_id,
                    entry,
                    request.parameter_text,
                )
                for entry in raw_path
            )
            if request.parameter_scope_id not in header_path:
                continue
            add(
                "HEADER_INHERITED",
                model_row="",
                value_row=cell.row_id,
                parameter_column=label_column,
                value_column=cell.column_id,
                header_path=header_path,
                cell_role=cell.cell_role,
            )

    # ---- Route D: CROSS_REFERENCE — the cell points at a target ---------
    for cell in table.cells:
        if cell.cell_role != "REFERENCE":
            continue
        target = reference_target(cell.text)
        if norm_text(target) != norm_text(request.value_or_action):
            continue
        source_row = next(
            (p.row_id for p in param_label_cells if p.row_id == cell.row_id),
            None,
        )
        if source_row is None and len(model_label_cells) == 1:
            source_row = model_label_cells[0].row_id
        if source_row is None:
            continue
        add(
            "CROSS_REFERENCE",
            model_row=source_row,
            value_row=cell.row_id,
            parameter_column=label_column,
            value_column=cell.column_id,
            header_path=(),
            cell_role="REFERENCE",
            reference_target_text=target,
        )

    distinct = {
        (
            emission.proof.relation,
            emission.proof.value_row_id,
            emission.proof.value_column_id,
            emission.proof.reference_target,
        )
        for emission in candidates
    }
    if len(distinct) > 1:
        return ProducerOutcome(None, ProducerDecline(DECLINE_AMBIGUOUS_BINDING, f"{len(distinct)} locations"))
    if not candidates:
        return ProducerOutcome(None, ProducerDecline(DECLINE_BINDING_NOT_FOUND))
    emission = candidates[0]
    return ProducerOutcome(emission, None, emission.proof.relation)





def _find_merged_span(
    table: ProducerTable, model_row_id: str, value_row_id: str,
) -> VerticalMergeRegion | None:
    for merge in table.vertical_merges:
        if merge.anchor_row_id == model_row_id and value_row_id in merge.covered_row_ids:
            return merge
    return None


def _compose_proof(
    table: ProducerTable,
    request: BindingRequest,
    *,
    relation: str,
    model_row: str,
    value_row: str,
    parameter_column: str,
    value_column: str,
    header_path: tuple[str, ...],
    cell_role: str,
    model_column: str = "",
    reference_target_text: str = "",
) -> TableStructureProof:
    binding_key = "|".join(filter(None, (
        request.model_scope_id,
        request.parameter_scope_id,
        norm_text(request.value_or_action),
        relation,
        model_row,
        value_row,
        value_column,
        reference_target_text,
    )))
    proof_id = f"{table.table_region_id}:proof:{stable_suffix(binding_key)}"
    chunk_ids = tuple(f"{table.document_id}:p{page}" for page in table.page_indices)
    span = (1, 1)
    if model_row and value_row and model_row in table.row_ids and value_row in table.row_ids:
        low, high = sorted((table.row_ids.index(model_row), table.row_ids.index(value_row)))
        if high > low:
            span = (high - low + 1, 1)
    safe_role = cell_role if cell_role in {"VALUE", "ACTION", "QUALIFIER", "REFERENCE"} else "VALUE"
    return TableStructureProof(
        proof_id=proof_id,
        document_id=table.document_id,
        table_region_id=table.table_region_id,
        relation=relation,
        model_scope_id=request.model_scope_id,
        parameter_scope_id=request.parameter_scope_id,
        value_scope_id=f"value:{slug(request.value_or_action)}",
        model_text=request.model_text,
        parameter_text=request.parameter_text,
        value_text=request.value_or_action,
        cell_role=safe_role,
        chunk_ids=chunk_ids,
        proof_version=TABLE_STRUCTURE_PROOF_VERSION,
        model_row_id=model_row,
        parameter_row_id=value_row if relation in {"DIRECT_ROW", "SECTION_INHERITED", "CROSS_REFERENCE"} else "",
        value_row_id=value_row,
        model_column_id=model_column,
        parameter_column_id=parameter_column,
        value_column_id=value_column,
        header_path=header_path,
        qualifier_scope_id=request.qualifier_scope_id,
        section_scope_id=request.section_scope_id,
        merged_cell_span=span,
        reference_target=reference_target_text,
    )


def _compose_coverage(
    table: ProducerTable,
    request: BindingRequest,
    merge: VerticalMergeRegion,
    proof: TableStructureProof,
) -> dict[str, Any]:
    return {
        "coverage_id": (
            f"{table.table_region_id}:coverage:"
            f"{stable_suffix(merge.column_id, '|'.join(merge.covered_row_ids))}"
        ),
        "document_id": table.document_id,
        "table_region_id": table.table_region_id,
        "coverage_owner_scope_id": request.model_scope_id,
        "coverage_anchor_cell_id": f"{merge.anchor_row_id}@{merge.column_id}",
        "covered_row_ids": tuple(merge.covered_row_ids),
        "chunk_ids": tuple(proof.chunk_ids),
        "coverage_version": MERGED_CELL_COVERAGE_VERSION_COMPAT,
        "coverage_kind": "VERTICAL_MERGED_CELL",
        "conflicting_coverage_ids": (),
    }
