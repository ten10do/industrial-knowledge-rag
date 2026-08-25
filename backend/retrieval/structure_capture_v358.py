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


def reconstruct_logical_cells(
    document_id: str,
    page_index: int,
    page,
    use_edges: bool = True,
) -> dict:
    """Return atoms, edges, row/column bands and logical cells."""
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
    column_bands = derive_column_bands(atoms, v_edges)

    grid: dict[tuple[int, int], list[CapturedTextAtom]] = {}
    for atom in atoms:
        r, c = assign_atom_to_cell(atom, row_bands, column_bands)
        grid.setdefault((r, c), []).append(atom)

    region_key_source = f"{document_id}:p{page_index}"
    cells: list[LogicalCell] = []
    for (r, c), group in sorted(grid.items()):
        ordered = sorted(group, key=lambda a: (round(a.y0, 1), a.x0))
        fragments = tuple(_fix(a.text) for a in ordered)
        normalized = " ".join(fragments)
        proofs = ["SAME_COLUMN_NO_SEPARATOR"]
        cells.append(
            LogicalCell(
                cell_id=f"{region_key_source}:c:r{r}k{c}",
                page=page_index,
                row_band=row_bands[r],
                column_band=column_bands[c],
                fragments=fragments,
                normalized_text=normalized,
                merge_proofs=tuple(proofs),
                boundary_confidence="GEOMETRIC_BOUNDARY"
                if h_edges or v_edges
                else "INFERRED_BOUNDARY",
            ),
        )
    return {
        "cells": cells,
        "atoms": atoms,
        "row_bands": row_bands,
        "column_bands": column_bands,
        "region_key": region_key_source,
    }
