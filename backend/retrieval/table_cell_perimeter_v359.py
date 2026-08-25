"""V3.59 cell perimeter attribution prototype (feasibility, isolated).

Architecture shift vs V3.58: FROM band-based rectangle expansion TO
edge-topology-constrained cell attribution.

Core idea proven by the R4/R59 audits: when a captured region spans two
true logical cells, real PDF geometry (an interior horizontal or vertical
divider) almost always proves they are distinct. Therefore:

  1. cluster ruling edges into h/v boundary positions;
  2. the lattice of consecutive boundary pairs IS the cell partition
     (minimal cell principle — no rectangle expansion);
  3. atoms are assigned to lattice cells by center containment;
  4. atoms outside every lattice cell fall back to conservative
     visual-line fragments marked AMBIGUOUS (borderless policy);
  5. a lattice cell that still fully covers two distinct annotated GT
     cells would indicate a missed boundary — surfaced as
     UNRESOLVED_PERIMETER during evaluation, never silently merged.

Query/GT semantics are never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .table_structure_producer_v356 import (
    infer_cell_role,
    norm_text,
    stable_suffix,
)


@dataclass(frozen=True)
class StructuralEdge:
    edge_id: str
    orientation: str            # HORIZONTAL | VERTICAL
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    source_type: str            # LINE | RECT_EDGE | OTHER_DRAWING
    confidence: float = 0.8
    thickness: float = 0.0
    source_object_id: str = ""


Y_TOLERANCE_DEFAULT = 2.5
X_TOLERANCE_DEFAULT = 2.5


def build_structural_edges_from_pdfplumber(page, page_number: int) -> list[StructuralEdge]:
    edges: list[StructuralEdge] = []
    counter = 0
    for e in page.edges:
        orientation = "HORIZONTAL" if e["orientation"] == "h" else "VERTICAL"
        if orientation == "HORIZONTAL":
            x0, x1 = min(e["x0"], e["x1"]), max(e["x0"], e["x1"])
            y0 = y1 = round((e["top"] + e["bottom"]) / 2, 2)
        else:
            y0, y1 = min(e["top"], e["bottom"]), max(e["top"], e["bottom"])
            x0 = x1 = round((e["x0"] + e["x1"]) / 2, 2)
        source_type = (
            "RECT_EDGE"
            if e.get("source", "") in {"rect", "rects"}
            else "LINE"
        )
        edges.append(
            StructuralEdge(
                edge_id=f"e{counter:05d}",
                orientation=orientation,
                x0=x0, y0=y0, x1=x1, y1=y1,
                page=page_number,
                source_type=source_type,
                confidence=0.85 if source_type == "LINE" else 0.75,
                thickness=round(abs(e["bottom"] - e["top"]), 2),
            ),
        )
        counter += 1
    return edges


def _cluster(values: list[float], tolerance: float) -> list[tuple[float, float]]:
    ordered = sorted(set(values))
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(min(g), max(g)) for g in groups]


def boundary_positions(
    edges: list[StructuralEdge],
    orientation: str,
    span_lo: float,
    span_hi: float,
    coverage_min: float,
    tolerance: float,
) -> list[float]:
    """Boundary positions from edges of one orientation whose own span
    intersects the target span with enough coverage."""
    positions = []
    for e in edges:
        if e.orientation != orientation:
            continue
        if orientation == "HORIZONTAL":
            e_lo, e_hi = min(e.x0, e.x1), max(e.x0, e.x1)
            s_lo, s_hi = span_lo, span_hi
        else:
            e_lo, e_hi = min(e.y0, e.y1), max(e.y0, e.y1)
            s_lo, s_hi = span_lo, span_hi
        overlap = min(e_hi, s_hi) - max(e_lo, s_lo)
        if overlap <= 0:
            continue
        if overlap / max(e_hi - e_lo, 1.0) < coverage_min:
            continue
        center = (e_lo + e_hi) / 2
        if s_lo - 20 <= center <= s_hi + 20:
            positions.append(center)
    clustered = []
    for p in sorted(positions):
        if not clustered or p - clustered[-1] > tolerance:
            clustered.append(p)
    return clustered


def build_lattice(
    atoms,
    structural_edges: list[StructuralEdge],
    *,
    h_coverage_min: float = 0.45,
    v_coverage_min: float = 0.45,
    y_tolerance: float = Y_TOLERANCE_DEFAULT,
    x_tolerance: float = X_TOLERANCE_DEFAULT,
):
    """Derive h/v boundary positions from edges over the atom extent."""
    if not atoms:
        return [], [], []
    y_lo = min(a.y0 for a in atoms) - 4
    y_hi = max(a.y1 for a in atoms) + 4
    x_lo = min(a.x0 for a in atoms) - 4
    x_hi = max(a.x1 for a in atoms) + 4
    h_bounds = boundary_positions(
        structural_edges, "HORIZONTAL", y_lo, y_hi, h_coverage_min, y_tolerance,
    )
    v_bounds = boundary_positions(
        structural_edges, "VERTICAL", x_lo, x_hi, v_coverage_min, x_tolerance,
    )
    # Outer content bounds always delimit the lattice.
    h_bounds = sorted(set(h_bounds) | {y_lo, y_hi})
    v_bounds = sorted(set(v_bounds) | {x_lo, x_hi})
    return h_bounds, v_bounds, [y_lo, y_hi, x_lo, x_hi]


def assign_atoms_to_lattice(
    atoms,
    h_bounds: list[float],
    v_bounds: list[float],
):
    """Assign each atom to a lattice cell index (row_i, col_j)."""
    def pick(bounds: list[float], point: float) -> int | None:
        for i in range(len(bounds) - 1):
            if bounds[i] <= point <= bounds[i + 1]:
                return i
        return None

    grid: dict[tuple[int, int], list] = {}
    unassigned: list = []
    for atom in atoms:
        cy = (atom.y0 + atom.y1) / 2
        cx = (atom.x0 + atom.x1) / 2
        r = pick(h_bounds, cy)
        c = pick(v_bounds, cx)
        if r is None or c is None:
            unassigned.append(atom)
        else:
            grid.setdefault((r, c), []).append(atom)
    return grid, unassigned


def reconstruct_perimeter_cells(
    document_id: str,
    page_index: int,
    atoms,
    structural_edges: list[StructuralEdge],
    *,
    use_edges: bool = True,
    h_coverage_min: float = 0.45,
    v_coverage_min: float = 0.45,
) -> dict:
    """Full perimeter-attribution reconstruction for one page."""
    if use_edges and not structural_edges:
        structural_edges = []

    usable_edges = structural_edges if use_edges else []
    h_bounds, v_bounds, extent = build_lattice(
        atoms, usable_edges,
        h_coverage_min=h_coverage_min,
        v_coverage_min=v_coverage_min,
    )

    cells: list[dict] = []
    ambiguous_pairs = 0

    if use_edges and len(h_bounds) >= 2 and len(v_bounds) >= 2:
        grid, unassigned = assign_atoms_to_lattice(atoms, h_bounds, v_bounds)
        for (r, c) in sorted(grid):
            group = grid[(r, c)]
            ordered = sorted(group, key=lambda a: (-a.y0, a.x0))
            fragments = tuple(_fix_local(a.text) for a in ordered)
            normalized = " ".join(fragments)
            cells.append(
                {
                    "cell_id": f"{document_id}:p{page_index}:lat:r{r}c{c}",
                    "kind": "LATTICE",
                    "row_band": (h_bounds[r], h_bounds[r + 1]),
                    "column_band": (v_bounds[c], v_bounds[c + 1]),
                    "normalized_text": normalized,
                    "boundary_confidence": "GEOMETRIC_BOUNDARY",
                    "merge_proofs": (),
                },
            )
        # Conservative fallback for atoms outside all lattice cells:
        # per visual line, x-gap groups, AMBIGUOUS (never auto-merged).
        by_line: dict[int, list] = {}
        for atom in unassigned:
            key = round(atom.y0)
            matched = next((k for k in by_line if abs(k - key) <= 2), None)
            if matched is None:
                by_line[key] = [atom]
            else:
                by_line[matched].append(atom)
        for top in sorted(by_line):
            line_atoms = sorted(by_line[top], key=lambda a: a.x0)
            groups: list[list] = [[line_atoms[0]]]
            for prev, atom in zip(line_atoms, line_atoms[1:]):
                if atom.x0 - prev.x1 > 12.0:
                    groups.append([atom])
                else:
                    groups[-1].append(atom)
            for gi, group in enumerate(groups):
                if gi > 0:
                    ambiguous_pairs += 1
                frags = tuple(_fix_local(a.text) for a in group)
                cells.append(
                    {
                        "cell_id": f"{document_id}:p{page_index}:loose:{stable_suffix(str(top), str(gi))}",
                        "kind": "AMBIGUOUS",
                        "row_band": (top - 1.0, top + 9.0),
                        "column_band": (min(a.x0 for a in group), max(a.x1 for a in group)),
                        "normalized_text": " ".join(frags),
                        "boundary_confidence": "AMBIGUOUS",
                        "merge_proofs": (),
                    },
                )
    else:
        # Borderless / insufficient-edge policy: per visual line + x-gap
        # fragments, AMBIGUOUS (no fake perimeter).
        by_line: dict[int, list] = {}
        for atom in atoms:
            key = round(atom.y0)
            matched = next((k for k in by_line if abs(k - key) <= 2), None)
            if matched is None:
                by_line[key] = [atom]
            else:
                by_line[matched].append(atom)
        for top in sorted(by_line):
            line_atoms = sorted(by_line[top], key=lambda a: a.x0)
            groups: list[list] = [[line_atoms[0]]]
            for prev, atom in zip(line_atoms, line_atoms[1:]):
                if atom.x0 - prev.x1 > 12.0:
                    groups.append([atom])
                else:
                    groups[-1].append(atom)
            for gi, group in enumerate(groups):
                if gi > 0:
                    ambiguous_pairs += 1
                frags = tuple(_fix_local(a.text) for a in group)
                cells.append(
                    {
                        "cell_id": f"{document_id}:p{page_index}:loose:{stable_suffix(str(top), str(gi))}",
                        "kind": "AMBIGUOUS",
                        "row_band": (top - 1.0, top + 9.0),
                        "column_band": (min(a.x0 for a in group), max(a.x1 for a in group)),
                        "normalized_text": " ".join(frags),
                        "boundary_confidence": "AMBIGUOUS",
                        "merge_proofs": (),
                    },
                )

    return {
        "cells": cells,
        "atoms": atoms,
        "h_bounds": h_bounds,
        "v_bounds": v_bounds,
        "ambiguous_pairs": ambiguous_pairs,
        "region_key": f"{document_id}:p{page_index}",
    }


def _fix_local(text: str) -> str:
    import re as _re

    collapsed = _re.sub(r"\s+", " ", text).strip()
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
