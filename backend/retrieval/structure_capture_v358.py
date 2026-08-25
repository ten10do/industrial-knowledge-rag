"""V3.58 fine-grained structure-capture prototype (feasibility).

Pipeline (directive §11/§14/§17/§29/§31):

    pdfplumber chars/words + rect/line edges
        -> CapturedTextAtom list
        -> TableGeometryGraph (row bands x column bands, edge-derived)
        -> LogicalCell regions (fragment assignment by geometry)
        -> query-agnostic normalized cell texts

Query/GT semantics are never consulted during reconstruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pdfplumber

from .table_structure_producer_v356 import stable_suffix


_LETTERSPACE = re.compile(r"\s+")


def _fix(text: str) -> str:
    collapsed = _LETTERSPACE.sub(" ", text).strip()
    parts = collapsed.split(" ")
    repaired: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) == 1 and part.isalpha():
            buffer += part
        else:
            if buffer:
                repaired.append(buffer + part)
                buffer = ""
            else:
                repaired.append(part)
    if buffer:
        repaired.append(buffer)
    return " ".join(repaired)


@dataclass(frozen=True)
class CapturedTextAtom:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    page: int
    source_id: int


@dataclass(frozen=True)
class GeometryEdge:
    orientation: str          # "h" | "v"
    position: float           # y for h, x for v
    span_start: float
    span_end: float


@dataclass(frozen=True)
class LogicalCell:
    cell_id: str
    page: int
    row_band: tuple[float, float]
    column_band: tuple[float, float]
    fragments: tuple[str, ...]
    normalized_text: str
    merge_proofs: tuple[str, ...] = ()
    boundary_confidence: str = "INFERRED_BOUNDARY"


def capture_page_atoms(page) -> tuple[list[CapturedTextAtom], list[GeometryEdge]]:
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    atoms = [
        CapturedTextAtom(
            text=w["text"],
            x0=w["x0"], x1=w["x1"], y0=w["top"], y1=w["bottom"],
            page=page.page_number,
            source_id=i,
        )
        for i, w in enumerate(words)
    ]
    edges: list[GeometryEdge] = []
    for e in page.edges:
        if e["orientation"] == "h":
            edges.append(GeometryEdge("h", round((e["top"] + e["bottom"]) / 2, 2),
                                      min(e["x0"], e["x1"]), max(e["x0"], e["x1"])))
        else:
            edges.append(GeometryEdge("v", round((e["x0"] + e["x1"]) / 2, 2),
                                      min(e["top"], e["bottom"]), max(e["top"], e["bottom"])))
    return atoms, edges


def cluster_positions(values: list[float], tolerance: float) -> list[tuple[float, float]]:
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(min(g), max(g)) for g in groups]


def derive_row_bands(
    atoms: list[CapturedTextAtom],
    h_edges: list[GeometryEdge],
) -> list[tuple[float, float]]:
    """Row bands: strong h-edge midpoints act as boundaries when they cover
    the content span; otherwise fall back to atom-y clustering."""
    if not atoms:
        return []
    x_min = min(a.x0 for a in atoms)
    x_max = max(a.x1 for a in atoms)
    content_span = max(x_max - x_min, 1.0)
    strong_boundaries = sorted(
        e.position for e in h_edges
        if (min(e.span_end, x_max) - max(e.span_start, x_min)) / content_span >= 0.5
        and x_min - 20 <= e.position <= x_max + 20
    )
    if len(strong_boundaries) >= 2:
        # Deduplicate near-identical boundary positions.
        deduped: list[float] = []
        for b in strong_boundaries:
            if not deduped or b - deduped[-1] > 4.0:
                deduped.append(b)
        tops = [a.y0 for a in atoms]
        bottom = max(a.y1 for a in atoms)
        cuts = [t for t in deduped if min(tops) - 6 <= t <= bottom + 6]
        bands: list[tuple[float, float]] = []
        lo = min(tops) - 3
        for cut in cuts:
            bands.append((lo, cut))
            lo = cut
        bands.append((lo, bottom + 3))
        return bands
    # Fallback: cluster atom vertical centers.
    centers = cluster_positions(
        [(a.y0 + a.y1) / 2 for a in atoms], tolerance=4.0,
    )
    expanded = []
    all_y0 = [a.y0 for a in atoms]
    all_y1 = [a.y1 for a in atoms]
    for lo, hi in centers:
        inner_lo = max((a.y0 for a in atoms if lo <= (a.y0 + a.y1) / 2 <= hi), default=lo)
        inner_hi = min((a.y1 for a in atoms if lo <= (a.y0 + a.y1) / 2 <= hi), default=hi)
        expanded.append((inner_lo - 1, inner_hi + 1))
    _ = all_y0, all_y1
    return expanded


def derive_column_bands(
    atoms: list[CapturedTextAtom],
    v_edges: list[GeometryEdge],
) -> list[tuple[float, float]]:
    """Column bands: strong v-edge positions as separators when available,
    else atom-x clustering."""
    if not atoms:
        return []
    x_min = min(a.x0 for a in atoms)
    x_max = max(a.x1 for a in atoms)
    span = max(x_max - x_min, 1.0)
    tops = [a.y0 for a in atoms]
    bottoms = [a.y1 for a in atoms]
    y_lo, y_hi = min(tops), max(bottoms)
    height = max(y_hi - y_lo, 1.0)
    strong_separators = sorted(
        e.position for e in v_edges
        if (min(e.span_end, y_hi) - max(e.span_start, y_lo)) / height >= 0.45
        and x_min - 20 <= e.position <= x_max + 20
    )
    deduped: list[float] = []
    for s in strong_separators:
        if not deduped or s - deduped[-1] > 4.0:
            deduped.append(s)
    if len(deduped) >= 1 and len(deduped) <= 12:
        bounds = [x_min - 3] + deduped + [x_max + 3]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    clustered = cluster_positions([a.x0 for a in atoms], tolerance=6.0)
    return clustered


def assign_atom_to_cell(
    atom: CapturedTextAtom,
    row_bands: list[tuple[float, float]],
    column_bands: list[tuple[float, float]],
) -> tuple[int, int]:
    center_y = (atom.y0 + atom.y1) / 2
    center_x = (atom.x0 + atom.x1) / 2

    def pick(bands: list[tuple[float, float]], point: float) -> int:
        for i, (lo, hi) in enumerate(bands):
            if lo <= point <= hi:
                return i
        best_index, best_distance = -1, float("inf")
        for i, (lo, hi) in enumerate(bands):
            distance = min(abs(point - lo), abs(point - hi))
            if distance < best_distance:
                best_index, best_distance = i, distance
        return best_index

    return (
        pick(row_bands, center_y),
        pick(column_bands, center_x),
    )


# --------------------------------------------------------------------------
# V3 final refinement: rectangle-first capture, conservative fragments
# --------------------------------------------------------------------------


def _cluster_bands(values: list[float], tol: float) -> list[tuple[float, float, float]]:
    ordered = sorted(set(values))
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(min(g), max(g), sum(g) / len(g)) for g in groups]


def _edge_covers(
    position: float,
    s0: float,
    s1: float,
    edge_list: list,
    min_cov: float,
) -> bool:
    span = s1 - s0
    if span <= 0:
        return False
    for e in edge_list:
        if abs(e.position - position) > 3.0:
            continue
        overlap = min(e.span_end, s1) - max(e.span_start, s0)
        if overlap / span >= min_cov:
            return True
    return False


def reconstruct_logical_cells_v3(
    document_id: str,
    page_index: int,
    page,
    use_edges: bool = True,
) -> dict:
    """Rectangle-first reconstruction (final refinement).

    1. Explicit cell rectangles from clustered h/v ruling bands; all four
       sides must be covered by real edges. Atoms merge inside a rectangle
       with proof SAME_EXPLICIT_CELL_RECT.
    2. Atoms outside rectangles stay per-visual-line fragments split by
       x-gaps — same-column stacking is ambiguous and NOT auto-merged.
    """
    atoms, edges = capture_page_atoms(page)
    if not atoms:
        return {"cells": [], "atoms": [], "row_bands": [], "column_bands": []}
    h_edges = [e for e in edges if e.orientation == "h"] if use_edges else []
    v_edges = [e for e in edges if e.orientation == "v"] if use_edges else []

    h_bands = _cluster_bands([e.position for e in h_edges], 2.5)
    v_bands = _cluster_bands([e.position for e in v_edges], 2.5)

    rects: list[tuple[float, float, float, float]] = []
    if len(h_bands) >= 2 and len(v_bands) >= 2:
        for i in range(len(h_bands) - 1):
            hy_a, hy_b = h_bands[i][2], h_bands[i + 1][2]
            for j in range(len(v_bands) - 1):
                vx_a, vx_b = v_bands[j][2], v_bands[j + 1][2]
                if (
                    _edge_covers(hy_a, vx_a, vx_b, h_edges, 0.7)
                    and _edge_covers(hy_b, vx_a, vx_b, h_edges, 0.7)
                    and _edge_covers(vx_a, hy_a, hy_b, v_edges, 0.7)
                    and _edge_covers(vx_b, hy_a, hy_b, v_edges, 0.7)
                ):
                    rects.append((vx_a, hy_b, vx_b, hy_a))

    rect_groups: dict[tuple[float, float, float, float], list] = {}
    loose: list = []
    for atom in atoms:
        cx, cy = (atom.x0 + atom.x1) / 2, (atom.y0 + atom.y1) / 2
        placed = False
        for rect in rects:
            x0, yb, x1, yt = rect
            if x0 <= cx <= x1 and yb <= cy <= yt:
                rect_groups.setdefault(rect, []).append(atom)
                placed = True
                break
        if not placed:
            loose.append(atom)

    cells: list[LogicalCell] = []
    for rect, group in sorted(rect_groups.items()):
        x0, yb, x1, yt = rect
        frags = tuple(_fix(a.text) for a in sorted(group, key=lambda a: (-a.y0, a.x0)))
        cells.append(
            LogicalCell(
                cell_id=f"{document_id}:p{page_index}:rect:{stable_suffix(str(rect))}",
                page=page_index,
                row_band=(yb, yt),
                column_band=(x0, x1),
                fragments=frags,
                normalized_text=" ".join(frags),
                merge_proofs=("SAME_EXPLICIT_CELL_RECT",),
                boundary_confidence="EXPLICIT_BOUNDARY",
            ),
        )

    by_line: dict[float, list] = {}
    for atom in loose:
        key = round(atom.y0)
        matched = next((k for k in by_line if abs(k - key) <= 2), None)
        if matched is None:
            by_line[key] = [atom]
        else:
            by_line[matched].append(atom)

    ambiguous_pairs = 0
    tops_sorted = sorted(by_line)
    for ti, top in enumerate(tops_sorted):
        line_atoms = sorted(by_line[top], key=lambda a: a.x0)
        groups: list[list] = [[line_atoms[0]]]
        for prev, atom in zip(line_atoms, line_atoms[1:]):
            if atom.x0 - prev.x1 > 12.0:
                groups.append([atom])
            else:
                groups[-1].append(atom)
        for gi, group in enumerate(groups):
            x0g = min(a.x0 for a in group)
            x1g = max(a.x1 for a in group)
            frags = tuple(_fix(a.text) for a in group)
            if gi > 0 or ti > 0:
                ambiguous_pairs += 1
            cells.append(
                LogicalCell(
                    cell_id=f"{document_id}:p{page_index}:loose:{stable_suffix(str(top), str(gi))}",
                    page=page_index,
                    row_band=(top - 1.0, top + 9.0),
                    column_band=(x0g, x1g),
                    fragments=frags,
                    normalized_text=" ".join(frags),
                    merge_proofs=(),
                    boundary_confidence="AMBIGUOUS",
                ),
            )
    return {
        "cells": cells,
        "atoms": atoms,
        "row_bands": [(b[0], b[1]) for b in h_bands],
        "column_bands": [(b[0], b[1]) for b in v_bands],
        "ambiguous_pairs": ambiguous_pairs,
        "region_key": f"{document_id}:p{page_index}",
    }


def reconstruct_logical_cells(
    document_id: str,
    page_index: int,
    page,
    use_edges: bool = True,
    mode: str = "v2",
) -> dict:
    """Return atoms, edges, row/column bands and logical cells.

    R4-refinement: per-band x-projection columns plus h-edge-aware wrap
    merging. Row bands are derived first; each band gets its own column
    spans from the x-projection of its atoms (falling back to strong
    v-edges). A band whose label region is empty while value columns carry
    text is treated as a wrapped continuation of the previous band when NO
    strong h-edge separates them; otherwise it stays a distinct row.

    mode="v3" selects the final-refinement rectangle-first reconstruction.
    """
    if mode == "v3":
        return reconstruct_logical_cells_v3(
            document_id, page_index, page, use_edges=use_edges,
        )
    atoms, edges = capture_page_atoms(page)
    if not atoms:
        return {"cells": [], "atoms": [], "row_bands": [], "column_bands": []}
    if use_edges:
        h_edges = [e for e in edges if e.orientation == "h"]
        v_edges = [e for e in edges if e.orientation == "v"]
    else:
        h_edges = []
        v_edges = []
    row_bands = derive_row_bands(atoms, h_edges)

    def h_edge_between(y_lo: float, y_hi: float) -> bool:
        return any(
            y_lo + 1.5 <= e.position <= y_hi - 1.5 for e in h_edges
        )

    # Column spans per band via x-projection with gap threshold.
    def column_spans_for(band_atoms: list[CapturedTextAtom]) -> list[tuple[float, float]]:
        if not band_atoms:
            return []
        intervals = sorted((a.x0, a.x1) for a in band_atoms)
        merged_spans: list[tuple[float, float]] = [intervals[0]]
        gap_threshold = 14.0
        for start, end in intervals[1:]:
            last_start, last_end = merged_spans[-1]
            if start - last_end <= gap_threshold:
                merged_spans[-1] = (last_start, max(last_end, end))
            else:
                merged_spans.append((start, end))
        return merged_spans

    logical_rows: list[dict] = []
    for lo, hi in row_bands:
        band_atoms = [
            a for a in atoms if lo - 0.5 <= (a.y0 + a.y1) / 2 <= hi + 0.5
        ]
        if not band_atoms:
            continue
        spans = column_spans_for(band_atoms)
        assigned: dict[int, list[str]] = {}
        for atom in band_atoms:
            center = (atom.x0 + atom.x1) / 2
            col_index = next(
                (i for i, (s, e) in enumerate(spans) if s <= center <= e),
                None,
            )
            if col_index is None:
                best_i, best_d = -1, float("inf")
                for i, (s, e) in enumerate(spans):
                    d = min(abs(center - s), abs(center - e))
                    if d < best_d:
                        best_i, best_d = i, d
                col_index = max(best_i, 0)
            assigned.setdefault(col_index, []).append(_fix(atom.text))
        logical_rows.append(
            {
                "band": (lo, hi),
                "columns": assigned,
                "label_empty": 0 not in assigned,
            },
        )

    # Wrap merging: a row whose label region is empty while other columns
    # have text continues the previous row ONLY when no strong h-edge
    # separates the two bands (§16-§17 fusion rule).
    merged_rows: list[dict] = []
    merge_proofs_by_row: list[list[str]] = []
    for row in logical_rows:
        if (
            merged_rows
            and row["label_empty"]
            and any(row["columns"].values())
            and not h_edge_between(
                merged_rows[-1]["band"][0], row["band"][1],
            )
        ):
            target = merged_rows[-1]
            for col, texts in row["columns"].items():
                target["columns"].setdefault(col, []).extend(texts)
            merge_proofs_by_row[-1].append("VERTICAL_CONTINUATION_WITHIN_CELL")
            continue
        merged_rows.append(row)
        merge_proofs_by_row.append([])

    cells: list[LogicalCell] = []
    for index, row in enumerate(merged_rows):
        lo, hi = row["band"]
        for col, pieces in sorted(row["columns"].items()):
            normalized = " ".join(pieces)
            cells.append(
                LogicalCell(
                    cell_id=f"{document_id}:p{page_index}:r{index}:c{col}",
                    page=page_index,
                    row_band=(lo, hi),
                    column_band=(col * 1.0, col * 1.0),
                    fragments=tuple(pieces),
                    normalized_text=normalized,
                    merge_proofs=tuple(merge_proofs_by_row[index]),
                    boundary_confidence="GEOMETRIC_BOUNDARY"
                    if h_edges
                    else "INFERRED_BOUNDARY",
                ),
            )
    return {
        "cells": cells,
        "atoms": atoms,
        "row_bands": row_bands,
        "column_bands": [],
        "region_key": f"{document_id}:p{page_index}",
    }
