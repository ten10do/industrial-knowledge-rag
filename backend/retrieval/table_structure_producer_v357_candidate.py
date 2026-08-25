"""V3.57 structured table producer candidate.

Hardening of the V3.56 feasibility prototype into a freezable candidate:

- adds REAL ruled-line geometry extracted from pypdf's parsed content
  operations (re/m/l/c path operators with graphics-state tracking);
- multi-region segmentation (several tables per page);
- header-band detection with three deterministic strategies in priority
  order: ruled separation, type-size contrast, numeric-density rule;
- failure taxonomy codes and observability counters;
- same frozen contract outputs (V3.51 proofs / V3.54 coverage).

Still EXPERIMENTAL DEV-ONLY: not wired into any runtime path, no LLM,
no OCR, deterministic and hashable.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable

from .table_structure_producer_v356 import (
    BindingRequest,
    CAPTION_MAX_GAP_FACTOR,
    CAPTION_MIN_GAP_FACTOR,
    MIN_ROW_OCCUPIED_COLUMNS,
    MAX_WRAP_GAP_STEPS,
    PageSignal,
    ProducerCell,
    ProducerDecline,
    ProducerOutcome,
    ProducerTable,
    ReconstructionReport,
    TextFragmentSignal,
    VerticalMergeRegion,
    _GridExtent,
    _PageGrid,
    _build_header_paths,
    _build_page_grid,
    _cell_text,
    _cluster,
    _detect_grid_extent,
    _detect_vertical_merges,
    _row_sizes,
    infer_cell_role,
    bind_claim_to_table as _v356_bind_claim_to_table,
    extract_page_signals,
    norm_text,
    stable_suffix,
)


TABLE_STRUCTURE_PRODUCER_V357_CANDIDATE_VERSION = "structured-table-producer-v357-candidate"
TABLE_STRUCTURE_PRODUCER_V357_STATUS = "DEV_CANDIDATE_UNWIRED"

MIN_DATA_ROWS = 2

# Failure taxonomy codes (observability).
TAXONOMY_NO_TEXT_GRID = "NO_TEXT_GRID"
TAXONOMY_HEADER_UNSPLIT = "HEADER_BAND_UNSPLIT"
TAXONOMY_MERGE_FOUND = "VERTICAL_MERGE_FOUND"
TAXONOMY_BINDING_NOT_FOUND = "BINDING_NOT_FOUND"
TAXONOMY_AMBIGUOUS = "AMBIGUOUS_LOCATION"


# --------------------------------------------------------------------------
# Ruling-line geometry from parsed content operations
# --------------------------------------------------------------------------


def _mat_mult(a: list[float], b: list[float]) -> list[float]:
    return [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]


def _apply(ctm: list[float], x: float, y: float) -> tuple[float, float]:
    return (
        ctm[0] * x + ctm[2] * y + ctm[4],
        ctm[1] * x + ctm[3] * y + ctm[5],
    )


@dataclass(frozen=True)
class RuledSegment:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def horizontal(self) -> bool:
        return abs(self.y1 - self.y0) <= 0.75

    @property
    def vertical(self) -> bool:
        return abs(self.x1 - self.x0) <= 0.75

    @property
    def length(self) -> float:
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5


_STROKE_OPS = {"S", "s", "B", "B*", "b", "b*", "f", "F", "f*", "n"}


def _extract_segments(operations: list) -> list[RuledSegment]:
    """Collect stroked path segments in device space from parsed ops."""
    segments: list[RuledSegment] = []
    ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    stack: list[list[float]] = []
    current: list[tuple[float, float]] = []
    pos = (0.0, 0.0)

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            for a, b in zip(current, current[1:]):
                segments.append(RuledSegment(a[0], a[1], b[0], b[1]))
        current = []

    for operands, operator in operations:
        name = (
            operator.decode("ascii", "replace")
            if isinstance(operator, bytes)
            else str(operator)
        )
        try:
            nums = [float(v) for v in operands]
        except (TypeError, ValueError):
            nums = []
        if name == "q":
            stack.append(ctm.copy())
        elif name == "Q":
            ctm = stack.pop() if stack else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif name == "cm" and len(nums) >= 6:
            ctm = _mat_mult(nums[:6], ctm)
        elif name == "m" and len(nums) >= 2:
            flush()
            pos = _apply(ctm, nums[0], nums[1])
            current = [pos]
        elif name == "l" and len(nums) >= 2:
            pos = _apply(ctm, nums[0], nums[1])
            current.append(pos)
        elif name == "c" and len(nums) >= 6:
            pos = _apply(ctm, nums[4], nums[5])
            current.append(pos)
        elif name == "re" and len(nums) >= 4:
            x, y, w, h = nums[:4]
            corners = [
                _apply(ctm, x, y),
                _apply(ctm, x + w, y),
                _apply(ctm, x + w, y + h),
                _apply(ctm, x, y + h),
                _apply(ctm, x, y),
            ]
            for a, b in zip(corners, corners[1:]):
                segments.append(RuledSegment(a[0], a[1], b[0], b[1]))
            pos = corners[-1]
        elif name in _STROKE_OPS:
            flush()
    flush()
    return segments


def extract_ruling_geometry(page: Any) -> list[RuledSegment]:
    contents = page.get_contents()
    operations = getattr(contents, "operations", None) or ()
    segments = _extract_segments(list(operations))
    return [
        s for s in segments
        if s.length >= 12.0 and (s.horizontal or s.vertical)
    ]


# --------------------------------------------------------------------------
# Geometry-aware page analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageAnalysis:
    page_index: int
    grid: _PageGrid
    extent: _GridExtent
    h_line_ys: tuple[float, ...]
    v_line_xs: tuple[float, ...]
    observability: dict[str, int]


def _visitor_fragments(page: Any, page_index: int) -> tuple[TextFragmentSignal, ...]:
    fragments: list[TextFragmentSignal] = []

    def visitor_text(text, cm, tm, font_dict, font_size):  # noqa: ANN001
        if text and str(text).strip():
            fragments.append(
                TextFragmentSignal(str(text).strip(), float(tm[4]), float(tm[5])),
            )

    try:
        page.extract_text(visitor_text=visitor_text)
    except Exception:  # noqa: BLE001
        return ()
    return tuple(fragments)


def _fragment_quality(fragments: tuple[TextFragmentSignal, ...]) -> float:
    """Score by position diversity and printable-text cleanliness."""
    if not fragments:
        return 0.0
    positions = {(f.x, f.y) for f in fragments}
    diversity = len(positions) / len(fragments)
    clean_chars = total = 0
    for f in fragments:
        for ch in f.text:
            total += 1
            if ch.isprintable() or ch.isspace():
                clean_chars += 1
    cleanliness = clean_chars / max(total, 1)
    return diversity * 0.5 + cleanliness * 0.5


def analyze_page(page: Any, page_index: int) -> PageAnalysis | None:
    from .table_structure_producer_v357_walk import walk_page_content

    raw_frags, raw_segments = walk_page_content(page, _reader_of(page))
    walker_frags = tuple(
        TextFragmentSignal(f["text"], f["x"], f["y"], f["size"])
        for f in raw_frags
        if f["text"].strip()
    )
    visitor_frags = _visitor_fragments(page, page_index)

    candidates = []
    if walker_frags:
        candidates.append((_fragment_quality(walker_frags), walker_frags))
    if visitor_frags:
        candidates.append((_fragment_quality(visitor_frags), visitor_frags))
    if not candidates:
        return None
    # Prefer the reader whose output has both diverse coordinates and
    # cleanly decodable text (ops-walker wins on synthetic direct-Tm
    # pages; pypdf's decoded visitor wins on real CMap-encoded manuals).
    _score, frags = max(candidates, key=lambda item: item[0])
    if len(frags) < 6:
        return None
    grid = _build_page_grid(list(frags))
    if grid is None:
        return None
    extent = _detect_grid_extent(grid)
    if extent is None:
        return None
    segments = [
        s for s in raw_segments
        if s.length >= 12.0 and (s.horizontal or s.vertical)
    ]
    ys: list[float] = []
    xs: list[float] = []
    for s in segments:
        if s.horizontal:
            ys.append(round((s.y0 + s.y1) / 2, 1))
        elif s.vertical:
            xs.append(round((s.x0 + s.x1) / 2, 1))
    h_clusters = _cluster(ys, 2.0)
    v_clusters = _cluster(xs, 2.0)
    observability = {
        "fragments": len(frags),
        "segments": len(segments),
        "h_line_clusters": len(h_clusters),
        "v_line_clusters": len(v_clusters),
    }
    h_centers = sorted(sum(g) / len(g) for g in h_clusters)
    v_centers = sorted(sum(g) / len(g) for g in v_clusters)
    return PageAnalysis(
        page_index=page_index,
        grid=grid,
        extent=extent,
        h_line_ys=tuple(h_centers),
        v_line_xs=tuple(v_centers),
        observability=observability,
    )


def _reader_of(page: Any) -> Any:
    indirect = getattr(page, "indirect_reference", None)
    if indirect is not None:
        return getattr(indirect, "pdf", None)
    return getattr(page, "pdf", None)


# --------------------------------------------------------------------------
# Header-band strategies
# --------------------------------------------------------------------------

_NUMERIC_PATTERN = re.compile(r"\d")


def _cells_of_row(grid: _PageGrid, row: int, columns: list[int]) -> list[str]:
    return [_cell_text(grid, row, c) for c in columns]


def _header_band_by_geometry(
    analysis: PageAnalysis,
) -> tuple[list[int], list[int]] | None:
    """Split headers using a strong horizontal ruling under the top band."""
    core = analysis.extent.data_rows
    grid = analysis.grid
    if not analysis.h_line_ys or len(core) < 3:
        return None
    top_y = grid.row_center_y[core[0]]
    second_y = grid.row_center_y[core[1]]
    separators = [
        y for y in analysis.h_line_ys
        if second_y - 1.5 <= y <= second_y + 1.5
    ]
    if not separators:
        return None
    separator = separators[0]
    header_rows = [r for r in core if grid.row_center_y[r] > separator + 1.0]
    data_rows = [r for r in core if r not in set(header_rows)]
    if header_rows and data_rows:
        return sorted(header_rows, reverse=True), data_rows
    return None


def _header_band_by_numeric_density(
    grid: _PageGrid,
    core: list[int],
) -> tuple[list[int], list[int]] | None:
    """Pop leading all-textual rows when later rows carry numeric values."""
    columns = sorted(grid.col_center_x)
    if len(core) < 3:
        return None

    def textual_fraction(row: int) -> float:
        cells = _cells_of_row(grid, row, columns)
        cells = [c for c in cells if c]
        if not cells:
            return 1.0
        numeric = sum(1 for t in cells if _NUMERIC_PATTERN.search(t))
        return 1.0 - numeric / len(cells)

    lower_half = core[len(core) // 2:]
    if max(textual_fraction(r) for r in lower_half) < 0.34:
        return None
    header_rows: list[int] = []
    remaining = list(core)
    while remaining and textual_fraction(remaining[0]) >= 0.8:
        header_rows.append(remaining.pop(0))
    if header_rows and len(remaining) >= MIN_DATA_ROWS:
        return header_rows, remaining
    return None


def _header_band_by_size(
    grid: _PageGrid,
    core: list[int],
    mode: float | None = None,
) -> tuple[list[int], list[int]] | None:
    if mode is None:
        mode_counts: dict[float, int] = {}
        for row in core:
            for size in _row_sizes(grid, row):
                key = round(size, 1)
                mode_counts[key] = mode_counts.get(key, 0) + 1
        mode = max(mode_counts, key=lambda k: (mode_counts[k], k)) if mode_counts else 0.0
    header_rows: list[int] = []
    remaining = list(core)
    while remaining:
        sizes = _row_sizes(grid, remaining[0])
        if sizes and min(sizes) >= mode + 0.25:
            header_rows.append(remaining.pop(0))
        else:
            break
    if header_rows and remaining:
        return header_rows, remaining
    return None


# --------------------------------------------------------------------------
# Multi-region reconstruction
# --------------------------------------------------------------------------


def _extent_for_run(
    grid: _PageGrid,
    run: list[int],
    row_columns: dict[int, set[int]],
    analysis: PageAnalysis,
    taxonomy: dict[str, int],
) -> _GridExtent | None:
    core = sorted({r for r in run if row_columns.get(r)}, reverse=True)
    if len(core) < MIN_DATA_ROWS + 1 and len(core) < 2:
        return None
    columns_used: set[int] = set()
    for row in core:
        columns_used |= row_columns[row]
    columns = sorted(columns_used)
    if len(columns) < 2:
        return None
    gaps = [
        abs(grid.row_center_y[b] - grid.row_center_y[a])
        for a, b in zip(core, core[1:])
    ]
    pitch = statistics.median(gaps) if gaps else 14.0

    header_rows: list[int] = []
    data_rows = list(core)
    strategy = "none"
    geo = _header_band_by_geometry(analysis)
    if geo is not None:
        candidate_core_set = set(core)
        header_rows = [r for r in geo[0] if r in candidate_core_set]
        data_rows = [r for r in geo[1] if r in candidate_core_set]
        if header_rows and data_rows:
            strategy = "ruled"
        else:
            header_rows, data_rows = [], list(core)
    if strategy == "none":
        alt = _header_band_by_size(grid, core)
        if alt is not None:
            header_rows, data_rows = alt
            strategy = "size"
    if strategy == "none":
        alt = _header_band_by_numeric_density(grid, core)
        if alt is not None:
            header_rows, data_rows = alt
            strategy = "numeric"
    if strategy == "none":
        taxonomy[TAXONOMY_HEADER_UNSPLIT] = taxonomy.get(TAXONOMY_HEADER_UNSPLIT, 0) + 1

    # Caption separation: a lone oversized line sitting well above the
    # header band is a caption, not part of the grid.
    caption = ""
    if header_rows:
        grid_row_indices = sorted(set(header_rows) | set(data_rows), reverse=True)
        if len(grid_row_indices) >= 2:
            top = grid_row_indices[0]
            next_y = grid.row_center_y[grid_row_indices[1]]
            top_gap = grid.row_center_y[top] - next_y
            row_cells = [
                _cell_text(grid, top, c) for c in columns
            ]
            non_empty = [t for t in row_cells if t]
            pitch_guard = max(pitch, 1.0)
            if (
                len(non_empty) <= 2
                and top_gap > CAPTION_MIN_GAP_FACTOR * pitch_guard * 1.3
            ):
                caption = " ".join(non_empty).strip()
                header_rows = header_rows[1:]
        # Recompute caption from rows ABOVE the band (unclaimed territory).
        if not caption:
            candidates_above = [
                r for r in row_columns
                if r not in core and r not in set(header_rows)
            ]
            top_row = core[0]
            for row in sorted(candidates_above, reverse=True):
                gap = grid.row_center_y[row] - grid.row_center_y[top_row]
                if gap < CAPTION_MIN_GAP_FACTOR * pitch or gap > CAPTION_MAX_GAP_FACTOR * pitch:
                    continue
                texts = [_cell_text(grid, row, c) for c in columns]
                caption = " ".join(t for t in texts if t).strip()
                break
    else:
        candidates_above = [
            r for r in row_columns
            if r not in core and r not in set(header_rows)
        ]
        top_row = core[0]
        for row in sorted(candidates_above, reverse=True):
            gap = grid.row_center_y[row] - grid.row_center_y[top_row]
            if gap < CAPTION_MIN_GAP_FACTOR * pitch or gap > CAPTION_MAX_GAP_FACTOR * pitch:
                continue
            texts = [_cell_text(grid, row, c) for c in columns]
            caption = " ".join(t for t in texts if t).strip()
            break
    return _GridExtent(
        data_rows=data_rows,
        header_rows=sorted(header_rows, reverse=True),
        columns=columns,
        pitch_points=pitch,
        caption=caption,
    )


def analyze_and_reconstruct(
    analyses: list[tuple[int, PageAnalysis]],
    document_id: str,
) -> ReconstructionReport:
    tables: list[ProducerTable] = []
    taxonomy: dict[str, int] = {}
    observability: dict[str, int] = {"pages": len(analyses)}
    regions = 0
    for page_index, analysis in analyses:
        grid = analysis.grid
        row_columns: dict[int, set[int]] = {}
        for (row, column) in grid.cells:
            row_columns.setdefault(row, set()).add(column)
        dense = {
            r for r, cols in row_columns.items()
            if len(cols) >= MIN_ROW_OCCUPIED_COLUMNS
        }
        ordered = sorted(dense, key=lambda r: -grid.row_center_y[r])
        runs: list[set[int]] = []
        for row in ordered:
            merged_into = None
            for run in runs:
                if any(abs(row - member) <= 2 for member in run):
                    run.add(row)
                    merged_into = run
                    break
            if merged_into is None:
                runs.append({row})
        # Iteratively grow each dense seed into adjacent occupied rows
        # (span-interior single-cell rows rejoin; captions stay outside).
        grown_runs: list[set[int]] = []
        for run in runs:
            lo, hi = min(run), max(run)
            for direction in (+1, -1):
                steps = 0
                while steps < MAX_WRAP_GAP_STEPS:
                    candidate = hi + 1 if direction == 1 else lo - 1
                    if candidate in row_columns:
                        if direction == 1:
                            hi = candidate
                        else:
                            lo = candidate
                        steps += 1
                    else:
                        break
            grown_runs.append({r for r in range(lo, hi + 1) if r in row_columns})
        # Merge overlapping grown ranges.
        runs = grown_runs
        changed = True
        while changed:
            changed = False
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    if runs[i] & runs[j]:
                        runs[i] |= runs[j]
                        runs.pop(j)
                        changed = True
                        break
                if changed:
                    break
        # Table-likeness via column recurrence: real grid columns recur
        # across many rows; prose scatters. Require >=2 recurrent columns
        # and that most rows touch at least one recurrent column.
        kept_runs: list[set[int]] = []
        for run in runs:
            n = len(run)
            if n < MIN_DATA_ROWS + 1:
                continue
            col_rows: dict[int, int] = {}
            for r in run:
                for c in row_columns[r]:
                    col_rows[c] = col_rows.get(c, 0) + 1
            recurrent = {
                c for c, k in col_rows.items()
                if k >= max(2, int(0.35 * n))
            }
            if len(recurrent) < 2:
                continue
            touching = sum(1 for r in run if row_columns[r] & recurrent)
            if touching / n >= 0.5:
                kept_runs.append(run)
        runs = kept_runs
        regions = 0
        for run in runs:
            if len(run) < 2:
                continue
            extent = _extent_for_run(grid, sorted(run), row_columns, analysis, taxonomy)
            if extent is None:
                continue
            # Region qualification: prefer ruled grids; accept unruled
            # regions only when a header band was confidently split.
            band_rows = extent.header_rows + extent.data_rows
            ys = [grid.row_center_y[r] for r in band_rows]
            xs = [grid.col_center_x[c] for c in extent.columns]
            y_lo, y_hi = min(ys) - 2.0, max(ys) + 2.0
            x_lo, x_hi = min(xs) - 2.0, max(xs) + 2.0
            h_inside = sum(1 for y in analysis.h_line_ys if y_lo <= y <= y_hi)
            v_inside = sum(1 for x in analysis.v_line_xs if x_lo <= x <= x_hi)
            # Open-sided tables (no vertical rules) are common in
            # industrial manuals: require several row separators and at
            # least one vertical rule.
            ruled = h_inside >= 3 and v_inside >= 1
            header_split = bool(extent.header_rows)
            if not ruled and not header_split:
                taxonomy["REGION_REJECTED_PROSE"] = taxonomy.get("REGION_REJECTED_PROSE", 0) + 1
                continue
            all_rows = extent.header_rows + extent.data_rows
            region_key = stable_suffix(document_id, str(page_index), ",".join(map(str, all_rows)))
            region_id = f"{document_id}:tbl:p{page_index}:{region_key}"
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
            paths = _build_header_paths(extent, grid, region_id, column_ids)
            merge_triples = _detect_vertical_merges(extent, grid)
            merges = tuple(
                VerticalMergeRegion(
                    column_id=column_ids[c],
                    anchor_row_id=row_ids[a],
                    covered_row_ids=tuple(row_ids[r] for r in covered),
                )
                for c, a, covered in merge_triples
            )
            if merges:
                taxonomy[TAXONOMY_MERGE_FOUND] = taxonomy.get(TAXONOMY_MERGE_FOUND, 0) + len(merges)
            tables.append(
                ProducerTable(
                    document_id=document_id,
                    table_region_id=region_id,
                    page_indices=(page_index,),
                    column_ids=tuple(column_ids[c] for c in extent.columns),
                    row_ids=tuple(row_ids[r] for r in all_rows),
                    header_row_ids=tuple(row_ids[r] for r in extent.header_rows),
                    cells=tuple(cells),
                    header_paths=tuple(paths),
                    vertical_merges=merges,
                    section_caption=extent.caption,
                    section_caption_page=page_index if extent.caption else -1,
                ),
            )
            regions += 1
    observability["regions"] = regions
    report = ReconstructionReport(tables=tuple(tables))
    return ReconstructionReport(tables=report.tables, digest=report.result_digest())


def build_candidate_report(
    document_id: str,
    pages: Iterable[Any],
    strategy: str = "coordinate",
    row_policy: str = "r3_guard",
    y_tolerance: float = 2.5,
) -> tuple[ReconstructionReport, dict[str, int], dict[str, int]]:
    """Full pipeline over pypdf pages.

    strategy:
      - "coordinate": text fragments + ruling geometry (V3.56 lineage)
      - "layout":     public layout-mode text blocks (no CMap wall)
    row_policy (layout strategy only):
      - "r3_guard":                 R3 wrap-absorption guard only
      - "ruling_strong_split":      full-width rulings force new rows
      - "ruling_full_plus_partial": + qualified partial rulings
    """
    if strategy == "layout":
        from collections import Counter

        from .table_structure_producer_v357_layout import (
            cluster_h_rulings,
            match_line_y_positions,
            reconstruct_from_layout,
        )
        from .table_structure_producer_v357_walk import walk_page_content

        tables: list[ProducerTable] = []
        taxonomy: dict[str, int] = {}
        observability: dict[str, int] = {"pages": len(pages)}
        boundary_log: list[dict] = []
        reader = None
        for index, page in enumerate(pages):
            if reader is None:
                indirect = getattr(page, "indirect_reference", None)
                reader = getattr(indirect, "pdf", None) if indirect else None
            try:
                layout_text = page.extract_text(extraction_mode="layout") or ""
            except Exception:  # noqa: BLE001
                layout_text = ""
            raw_frags, raw_segments = walk_page_content(page, _reader_of(page))
            visitor_frags = [
                (f.x, f.y, f.text)
                for f in _visitor_fragments(page, index)
            ]
            h_rulings = cluster_h_rulings(
                [s for s in raw_segments if s.horizontal],
            )
            lines = layout_text.splitlines()
            line_positions = match_line_y_positions(lines, visitor_frags)
            xs = [x for x, _y in line_positions if x is not None]
            global_span = (
                min(xs) if xs else 0.0,
                max((f.x + 40.0 for f in _visitor_fragments(page, index)), default=612.0),
            )
            page_tables = reconstruct_from_layout(
                document_id,
                index,
                layout_text,
                h_rulings=h_rulings if row_policy != "r3_guard" else None,
                line_positions=line_positions if row_policy != "r3_guard" else None,
                policy=row_policy,
                y_tolerance=y_tolerance,
                global_x_span=global_span,
                boundary_log=boundary_log,
            )
            if not page_tables:
                taxonomy["PAGE_SKIPPED"] = taxonomy.get("PAGE_SKIPPED", 0) + 1
            tables.extend(page_tables)
        observability["regions"] = len(tables)
        verdicts = Counter(b["verdict"] for b in boundary_log)
        for verdict, count in verdicts.items():
            taxonomy[f"BOUNDARY_{verdict}"] = count
        report = ReconstructionReport(tables=tuple(tables))
        return (
            ReconstructionReport(tables=report.tables, digest=report.result_digest()),
            taxonomy,
            observability,
        )

    analyses: list[tuple[int, PageAnalysis]] = []
    taxonomy: dict[str, int] = {}
    for index, page in enumerate(pages):
        analysis = analyze_page(page, index)
        if analysis is not None:
            analyses.append((index, analysis))
        else:
            taxonomy["PAGE_SKIPPED"] = taxonomy.get("PAGE_SKIPPED", 0) + 1
    report = analyze_and_reconstruct(analyses, document_id)
    return report, taxonomy, {k: v for _, a in analyses for k, v in a.observability.items()}


# --------------------------------------------------------------------------
# LabelColumnScore — multi-feature label-column inference (R3)
# --------------------------------------------------------------------------

LABEL_SCORE_WEIGHTS: dict[str, float] = {
    "left_prior": 1.0,
    "text_ratio": 1.2,
    "coverage": 0.4,
    "numeric_penalty": -1.6,
    "unit_penalty": -0.8,
    "param_id_bonus": 0.5,
}
LABEL_AMBIGUOUS_MARGIN = 0.15

_UNIT_PATTERN = re.compile(
    r"^[-+]?[\d.,]+\s?(?:a|v|v ac|v dc|w|kw|hz|ma|rpm|mm|n(?:m)?|s|ms|%|bar|psi)?\b\.?$",
    re.IGNORECASE,
)
_PARAM_ID_PATTERN = re.compile(
    r"^(?:[a-z]{1,3}[.\-]?\d{1,4}[a-z]?|\d{1,3}-\d{1,3})$",
    re.IGNORECASE,
)


def _alpha_ratio(text: str) -> float:
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / max(len(text), 1)


def _is_value_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    has_digit = any(ch.isdigit() for ch in stripped)
    return _UNIT_PATTERN.match(stripped) is not None or (
        has_digit and _digit_ratio(stripped) >= 0.34
        and not _PARAM_ID_PATTERN.match(stripped)
    )


def score_label_columns(table: ProducerTable) -> tuple[list[dict], bool]:
    """Score every column as a candidate label column.

    Returns (ranked candidates, ambiguous_flag). Each candidate carries the
    feature vector and its reason code so ablations can attribute wins.
    """
    data_rows = [
        r for r in table.row_ids if r not in set(table.header_row_ids)
    ]
    n_rows = len(data_rows)
    n_cols = len(table.column_ids)
    if not data_rows or n_cols < 2:
        return [], False
    cell_map = {(c.row_id, c.column_id): c for c in table.cells}

    candidates: list[dict] = []
    for index, cid in enumerate(table.column_ids):
        texts = [
            cell_map[(r, cid)].text
            for r in data_rows
            if (r, cid) in cell_map
        ]
        coverage = len(texts) / n_rows
        non_empty = [t for t in texts if t.strip()]
        text_ratio = (
            sum(_alpha_ratio(t) for t in non_empty) / max(len(non_empty), 1)
        )
        value_like = sum(1 for t in non_empty if _is_value_like(t))
        param_ids = sum(1 for t in non_empty if _PARAM_ID_PATTERN.match(t.strip()))
        numeric_ratio = value_like / max(len(non_empty), 1)
        param_id_ratio = param_ids / max(len(non_empty), 1)
        unit_ratio = (
            sum(1 for t in non_empty if _UNIT_PATTERN.match(t.strip()))
            / max(len(non_empty), 1)
        )
        left_prior = 1.0 - index / max(n_cols - 1, 1)

        features = {
            "left_prior": round(left_prior, 3),
            "text_ratio": round(text_ratio, 3),
            "coverage": round(coverage, 3),
            "numeric_ratio": round(numeric_ratio, 3),
            "unit_ratio": round(unit_ratio, 3),
            "param_id_ratio": round(param_id_ratio, 3),
        }
        score = (
            LABEL_SCORE_WEIGHTS["left_prior"] * left_prior
            + LABEL_SCORE_WEIGHTS["text_ratio"] * text_ratio
            + LABEL_SCORE_WEIGHTS["coverage"] * coverage
            + LABEL_SCORE_WEIGHTS["numeric_penalty"] * numeric_ratio
            + LABEL_SCORE_WEIGHTS["unit_penalty"] * unit_ratio
            + LABEL_SCORE_WEIGHTS["param_id_bonus"] * param_id_ratio
        )
        reason = "label_candidate"
        if numeric_ratio >= 0.6:
            reason = "value_column_numeric_dominant"
        elif unit_ratio >= 0.5:
            reason = "value_column_unit_dominant"
        candidates.append(
            {
                "column_id": cid,
                "score": round(score, 4),
                **features,
                "reason_code": reason,
            },
        )
    candidates.sort(key=lambda c: -c["score"])
    ambiguous = (
        len(candidates) >= 2
        and (candidates[0]["score"] - candidates[1]["score"]) < LABEL_AMBIGUOUS_MARGIN
    )
    return candidates, ambiguous


def with_effective_label_column(table: ProducerTable) -> ProducerTable:
    """Rotate column order so the scored best label column leads.

    When the top two candidates are within LABEL_AMBIGUOUS_MARGIN the
    original order is kept (ambiguous abstention at rotation level).
    """
    scored, _ambiguous = score_label_columns(table)
    if not scored:
        return table
    effective = scored[0]["column_id"]
    if effective == table.column_ids[0]:
        return table
    reordered = [effective] + [
        cid for cid in table.column_ids if cid != effective
    ]
    return ProducerTable(
        document_id=table.document_id,
        table_region_id=table.table_region_id,
        page_indices=table.page_indices,
        column_ids=tuple(reordered),
        row_ids=table.row_ids,
        header_row_ids=table.header_row_ids,
        cells=table.cells,
        header_paths=table.header_paths,
        vertical_merges=table.vertical_merges,
        section_caption=table.section_caption,
        section_caption_page=table.section_caption_page,
    )


def without_section_caption_cell(table: ProducerTable) -> ProducerTable:
    """Drop the caption cell from the grid before binding.

    Layout regions keep the printed caption line inside the block; as a
    cell it sits in the label column and falsely registers as a model-row
    label when the query model text derives from the same caption.
    """
    if not table.section_caption:
        return table
    caption_norm = norm_text(table.section_caption)
    filtered = tuple(
        c for c in table.cells if norm_text(c.text) != caption_norm
    )
    if len(filtered) == len(table.cells):
        return table
    return ProducerTable(
        document_id=table.document_id,
        table_region_id=table.table_region_id,
        page_indices=table.page_indices,
        column_ids=table.column_ids,
        row_ids=table.row_ids,
        header_row_ids=table.header_row_ids,
        cells=filtered,
        header_paths=table.header_paths,
        vertical_merges=table.vertical_merges,
        section_caption="",
        section_caption_page=-1,
    )


def prepare_table_for_binding(table: ProducerTable) -> ProducerTable:
    """Binding-time region cleanup:

    1. drop single-cell rows (captions, footnotes) — they sit in the label
       column but never constitute model rows, and their texts collide
       with document-scoped model strings;
    2. rotate so the most-recurrent leftmost column leads.
    """
    label_col0 = table.column_ids[0]
    cells_by_row: dict[str, list[ProducerCell]] = {}
    for c in table.cells:
        cells_by_row.setdefault(c.row_id, []).append(c)
    single_cell_rows = {
        rid
        for rid, cells in cells_by_row.items()
        if len(cells) == 1 and cells[0].column_id == label_col0
    }
    filtered = tuple(
        c for c in table.cells if c.row_id not in single_cell_rows
    )
    stripped = (
        table
        if len(filtered) == len(table.cells)
        else ProducerTable(
            document_id=table.document_id,
            table_region_id=table.table_region_id,
            page_indices=table.page_indices,
            column_ids=table.column_ids,
            row_ids=table.row_ids,
            header_row_ids=table.header_row_ids,
            cells=filtered,
            header_paths=table.header_paths,
            vertical_merges=table.vertical_merges,
            section_caption=table.section_caption,
            section_caption_page=table.section_caption_page,
        )
    )
    return with_effective_label_column(stripped)


def bind_claim_candidate(table: ProducerTable, request: BindingRequest) -> ProducerOutcome:
    adjusted = prepare_table_for_binding(table)
    outcome = _v356_bind_claim_to_table(adjusted, request)
    if not outcome.emitted and outcome.decline is not None:
        mapped = (
            TAXONOMY_BINDING_NOT_FOUND
            if outcome.decline.code == "BINDING_NOT_FOUND"
            else TAXONOMY_AMBIGUOUS
        )
        return ProducerOutcome(None, ProducerDecline(mapped, outcome.decline.detail), "")
    return outcome


_ = TextFragmentSignal  # re-export hygiene
