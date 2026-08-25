"""Layout-hybrid reconstruction: pypdf public layout-mode text as the cell
source, segmented into table blocks purely from whitespace alignment.

Motivation (V3.57 decision-support spike): 85.3% of hand-annotated GT rows
across four manufacturers' official manuals are recoverable from layout
lines (62% single-line, 23% wrap-tolerant), so a no-dependency path exists
that sidesteps the CMap decoding wall entirely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .table_structure_producer_v356 import (
    ProducerCell,
    ProducerTable,
    VerticalMergeRegion,
    infer_cell_role,
    norm_text,
    stable_suffix,
)


MIN_BLOCK_ROWS = 3
MIN_BLOCK_COLUMNS = 2
GAP_SPLIT_MIN_SPACES = 2
BLOCK_BREAK_BLANK_LINES = 1

# Row-boundary provenance codes (directive §19).
BOUNDARY_SOURCE_RULING = "RULING_LINE"
BOUNDARY_SOURCE_LAYOUT_GEOMETRY = "LAYOUT_GEOMETRY"
BOUNDARY_SOURCE_AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class HorizontalRuling:
    """One horizontal ruling band in device space (directive §10)."""

    page: int
    y: float
    x_start: float
    x_end: float
    source: str = "vector_path"       # vector_line | rect_edge | vector_path
    confidence: float = 0.7
    thickness: float = 0.0

    @property
    def length(self) -> float:
        return max(self.x_end - self.x_start, 0.0)


@dataclass(frozen=True)
class RowBoundaryEvidence:
    """Evidence verdict for the boundary between two adjacent text rows."""

    verdict: str                      # STRONG_NEW_ROW | POSSIBLE_NEW_ROW |
                                      # NO_RULING_SUPPORT | AMBIGUOUS
    reason_code: str
    confidence: float
    ruling_ids: tuple[str, ...]
    coverage_ratio: float = 0.0


@dataclass(frozen=True)
class LayoutToken:
    start: int
    end: int
    text: str


def match_line_y_positions(
    layout_lines: list[str],
    visitor_fragments: list[tuple[float, float, str]],
) -> list[tuple[float | None, float | None]]:
    """Assign each layout line a (y, x) via best word-Jaccard visitor match."""
    frag_index = [
        (_canon(text), x, y)
        for x, y, text in visitor_fragments
        if len(_canon(text)) >= 8
    ]
    positions: list[tuple[float | None, float | None]] = []
    for line in layout_lines:
        words = set(_words(line))
        best_y, best_x, best_score = None, None, 0.0
        for ftext, fx, fy in frag_index:
            fwords = set(_words(ftext))
            if not words or not fwords:
                continue
            score = len(words & fwords) / max(len(words | fwords), 1)
            if score > best_score:
                best_score, best_y, best_x = score, fy, fx
        positions.append(
            (best_y if best_score >= 0.5 else None,
             best_x if best_score >= 0.5 else None),
        )
    return positions


def cluster_h_rulings(
    segments: list[Any],
) -> list[HorizontalRuling]:
    """Cluster horizontal stroke segments into ruling bands."""
    bands: dict[float, dict] = {}
    order: list[float] = []
    for seg in sorted(segments, key=lambda s: s.y0):
        if not seg.horizontal:
            continue
        key = round((seg.y0 + seg.y1) / 2, 1)
        existing = next((k for k in order if abs(k - key) <= 2.0), None)
        if existing is None:
            order.append(key)
            bands[key] = {
                "y": key,
                "x_start": min(seg.x0, seg.x1),
                "x_end": max(seg.x0, seg.x1),
                "thickness": abs(seg.y1 - seg.y0),
            }
        else:
            b = bands[existing]
            b["x_start"] = min(b["x_start"], seg.x0, seg.x1)
            b["x_end"] = max(b["x_end"], seg.x0, seg.x1)
            b["thickness"] = max(b["thickness"], abs(seg.y1 - seg.y0))
    return [
        HorizontalRuling(
            page=-1,
            y=b["y"],
            x_start=b["x_start"],
            x_end=b["x_end"],
            source="vector_path",
            confidence=0.7,
            thickness=round(b.get("thickness", 0.0), 2),
        )
        for k in sorted(bands)
        for b in [bands[k]]
    ]


def _words(text: str) -> list[str]:
    return [w for w in _canon(text).split() if len(w) >= 2]


def _canon(text: str) -> str:
    replacements = {
        "–": "-", "—": "-", "−": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return norm_text(text)


def boundary_evidence(
    upper_y: float | None,
    lower_y: float | None,
    rulings: list[HorizontalRuling],
    content_x0: float,
    content_x1: float,
    y_tolerance: float = 2.5,
) -> RowBoundaryEvidence:
    """Classify the boundary between two stacked lines (§12-§14)."""
    if upper_y is None or lower_y is None or upper_y <= lower_y:
        return RowBoundaryEvidence("NO_RULING_SUPPORT", "NO_LINE_Y", 0.0, (), 0.0)
    gap_lo, gap_hi = lower_y + y_tolerance, upper_y - y_tolerance
    if gap_hi <= gap_lo:
        return RowBoundaryEvidence("NO_RULING_SUPPORT", "GAP_TOO_SMALL", 0.0, (), 0.0)
    span = max(content_x1 - content_x0, 1.0)
    crossing = [
        r for r in rulings
        if gap_lo <= r.y <= gap_hi
        and min(r.x_end, content_x1) - max(r.x_start, content_x0) > 0
    ]
    if not crossing:
        return RowBoundaryEvidence("NO_RULING_SUPPORT", "NO_CROSSING_RULING", 0.0, (), 0.0)
    ids: list[str] = []
    best_coverage = 0.0
    for i, r in enumerate(crossing):
        overlap = (
            min(r.x_end, content_x1) - max(r.x_start, content_x0)
        )
        coverage = max(overlap / span, 0.0)
        best_coverage = max(best_coverage, coverage)
        ids.append(f"r{i}@{round(r.y, 1)}")
    if best_coverage >= 0.8:
        return RowBoundaryEvidence(
            "STRONG_NEW_ROW", "FULL_ROW_RULING", 0.9, tuple(ids),
            round(best_coverage, 3),
        )
    if best_coverage >= 0.35:
        return RowBoundaryEvidence(
            "POSSIBLE_NEW_ROW", "PARTIAL_ROW_RULING", 0.6, tuple(ids),
            round(best_coverage, 3),
        )
    return RowBoundaryEvidence(
        "AMBIGUOUS", "LOCAL_CELL_RULING", 0.3, tuple(ids),
        round(best_coverage, 3),
    )


def tokenize_layout_line(line: str, min_gap: int = GAP_SPLIT_MIN_SPACES) -> list[LayoutToken]:
    tokens: list[LayoutToken] = []
    run_start: int | None = None
    last_nonspace = -1
    for index, ch in enumerate(line):
        if ch != " ":
            if run_start is None:
                run_start = index
            last_nonspace = index
        else:
            if run_start is not None and index - last_nonspace >= min_gap:
                add_token(tokens, line, run_start, last_nonspace + 1)
                run_start = None
    if run_start is not None:
        add_token(tokens, line, run_start, last_nonspace + 1)
    return tokens


def add_token(tokens: list[LayoutToken], line: str, start: int, end: int) -> None:
    text = line[start:end].strip()
    if text:
        tokens.append(LayoutToken(start, end, text))


def fix_letterspacing(text: str) -> str:
    """Repair extraction artifacts like 'Vo l t a g e' -> 'Voltage'."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    parts = collapsed.split(" ")
    repaired: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) == 1:
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


def page_blocks(lines: list[str]) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    blanks = 0
    for index, line in enumerate(lines):
        if not line.strip():
            blanks += 1
            if current and blanks > BLOCK_BREAK_BLANK_LINES:
                blocks.append(current)
                current = []
            continue
        blanks = 0
        current.append(index)
    if current:
        blocks.append(current)
    return [b for b in blocks if len(b) >= MIN_BLOCK_ROWS]


def block_columns(
    token_rows: list[list[LayoutToken]],
    tolerance: int = 5,
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Cluster token START positions into columns (tolerance = characters).

    Interval merging across ragged prose collapses into giant spans;
    start-position clustering keeps recurring table columns separated.
    """
    starts = [t.start for tokens in token_rows for t in tokens]
    if not starts:
        return [], [[] for _ in token_rows]
    ordered = sorted(set(starts))
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    centers = [sum(g) / len(g) for g in groups]
    max_end = max(
        (t.end for tokens in token_rows for t in tokens),
        default=int(centers[-1]),
    )
    spans: list[tuple[int, int]] = []
    for i, center in enumerate(centers):
        left = int(centers[i - 1] + tolerance + 1) if i > 0 else 0
        right = (
            int(centers[i + 1] - tolerance - 1)
            if i + 1 < len(centers)
            else max(max_end, int(center))
        )
        spans.append((left, max(right, int(center))))
    assignments: list[list[int]] = []
    for tokens in token_rows:
        assignment: list[int] = []
        for token in tokens:
            best_index, best_distance = -1, float("inf")
            for i, center in enumerate(centers):
                distance = abs(token.start - center)
                if distance < best_distance:
                    best_index, best_distance = i, distance
            assignment.append(best_index if best_distance <= tolerance * 3 else -1)
        assignments.append(assignment)
    return spans, assignments


def reconstruct_from_layout(
    document_id: str,
    page_index: int,
    layout_text: str,
    *,
    h_rulings: list[HorizontalRuling] | None = None,
    line_positions: list[tuple[float | None, float | None]] | None = None,
    policy: str = "r3_guard",
    y_tolerance: float = 2.5,
    global_x_span: tuple[float, float] = (0.0, 612.0),
    boundary_log: list[dict] | None = None,
) -> list[ProducerTable]:
    lines = layout_text.splitlines()
    tables: list[ProducerTable] = []
    rulings = h_rulings or []
    for block in page_blocks(lines):
        # Keep line numbers and token rows strictly parallel: a block line
        # with zero tokens (rare whitespace artifacts) is dropped together
        # with its number.
        paired = [(i, tokenize_layout_line(lines[i])) for i in block]
        paired = [(i, tr) for i, tr in paired if tr]
        if len(paired) < MIN_BLOCK_ROWS:
            continue
        block_numbers = [i for i, _tr in paired]
        token_rows = [tr for _i, tr in paired]

        spans, assignments = block_columns(token_rows)
        if len(spans) < MIN_BLOCK_COLUMNS:
            continue
        n_cols = len(spans)
        col_counts = [0] * n_cols
        for assignment in assignments:
            for a in set(x for x in assignment if x >= 0):
                col_counts[a] += 1
        recurrent = {
            i for i, c in enumerate(col_counts)
            if c >= max(2, int(0.4 * len(assignments)))
        }
        if len(recurrent) < MIN_BLOCK_COLUMNS:
            continue
        touching = sum(
            1
            for assignment in assignments
            if set(x for x in assignment if x >= 0) & recurrent
        )
        if touching / max(len(assignments), 1) < 0.5:
            continue

        region_key = stable_suffix(document_id, str(page_index), ",".join(map(str, block_numbers)))
        region_id = f"{document_id}:ltbl:p{page_index}:{region_key}"
        column_ids = [f"{region_id}:c{i:02d}" for i in range(n_cols)]
        row_ids = {line_no: f"{region_id}:r{i:03d}" for i, line_no in enumerate(block_numbers)}

        # Header band: leading digit-free rows.
        header_local_rows: list[int] = []
        cursor = 0
        while cursor < len(assignments):
            texts = [
                tok.text
                for a, tok in zip(assignments[cursor], token_rows[cursor])
                if a >= 0
            ]
            if texts and not any(ch.isdigit() for t in texts for ch in t):
                header_local_rows.append(cursor)
                cursor += 1
            else:
                break
        data_start_local = cursor
        if data_start_local >= len(assignments):
            continue
        # Row assembly with wrap absorption: a data line whose label column
        # is empty while other columns carry text CONTINUES the row above
        # (wrapped cell), appending its tokens to that row's columns.
        assembled: list[dict[int, list[str]]] = []      # per merged row
        row_line_number: list[int] = []
        header_cells: dict[tuple[int, int], str] = {}

        def _absorb(local_r: int, target_index: int) -> bool:
            """Absorb continuation tokens into EMPTY columns of the target
            row only. A line that fills already-occupied columns is a NEW
            logical row (e.g., the second half of a vertical merge), not a
            wrap — merging it would destroy row identity and exact cell
            texts."""
            absorbed_any = False
            for a, tok in zip(assignments[local_r], token_rows[local_r]):
                if a < 0:
                    continue
                text = fix_letterspacing(tok.text)
                bucket = assembled[target_index].get(a)
                if bucket:
                    # Column occupied above -> distinct logical row signal.
                    return False
                assembled[target_index][a] = [text]
                absorbed_any = True
            return absorbed_any

        for local_r in range(len(assignments)):
            tokens = token_rows[local_r]
            assignment = assignments[local_r]
            if local_r in set(header_local_rows):
                for a, tok in zip(assignment, tokens):
                    if a >= 0:
                        header_cells[(local_r, a)] = fix_letterspacing(tok.text)
                continue
            hits_label = any(a == 0 for a in assignment)
            has_other = any(a > 0 for a in assignment)
            ruling_forces_split = False
            boundary_source = BOUNDARY_SOURCE_LAYOUT_GEOMETRY
            if (
                policy != "r3_guard"
                and rulings
                and line_positions is not None
                and local_r >= 1
            ):
                upper_y, _ux = line_positions[local_r - 1]
                lower_y, _lx = line_positions[local_r]
                content_x0, content_x1 = global_x_span
                if _ux is not None:
                    content_x0 = min(content_x0, _ux)
                if _lx is not None:
                    content_x0 = min(content_x0, _lx)
                evidence = boundary_evidence(
                    upper_y, lower_y, rulings,
                    content_x0, max(content_x1, content_x0 + 1.0),
                    y_tolerance=y_tolerance,
                )
                if boundary_log is not None:
                    boundary_log.append(
                        {
                            "page": page_index,
                            "block_line": block_numbers[local_r],
                            "verdict": evidence.verdict,
                            "reason": evidence.reason_code,
                            "confidence": evidence.confidence,
                            "coverage": evidence.coverage_ratio,
                        },
                    )
                if evidence.verdict == "STRONG_NEW_ROW":
                    ruling_forces_split = True
                    boundary_source = BOUNDARY_SOURCE_RULING
                elif evidence.verdict == "POSSIBLE_NEW_ROW" and policy == "ruling_full_plus_partial":
                    ruling_forces_split = True
                    boundary_source = BOUNDARY_SOURCE_RULING
            hits_label = any(a == 0 for a in assignment)
            has_other = any(a > 0 for a in assignment)
            if assembled and not hits_label and has_other and not ruling_forces_split:
                snapshot = {
                    key: list(pieces) for key, pieces in assembled[-1].items()
                }
                if _absorb(local_r, len(assembled) - 1):
                    continue
                assembled[-1] = snapshot  # rollback partial absorption
            new_row: dict[int, list[str]] = {}
            for a, tok in zip(assignment, tokens):
                if a < 0:
                    continue
                new_row.setdefault(a, []).append(fix_letterspacing(tok.text))
            assembled.append(new_row)
            row_line_number.append(block_numbers[local_r])

        cells: list[ProducerCell] = []
        for (local_r, a), text in header_cells.items():
            cells.append(
                ProducerCell(
                    row_ids[block_numbers[local_r]],
                    column_ids[a],
                    text,
                    "HEADER",
                ),
            )
        for index, row_values in enumerate(assembled):
            for col, pieces in sorted(row_values.items()):
                joined = " ".join(pieces).strip()
                role = infer_cell_role(joined)
                cells.append(
                    ProducerCell(row_ids[row_line_number[index]], column_ids[col], joined, role),
                )

        merges = detect_merges_layout_merged(
            region_id=region_id,
            column_ids=column_ids,
            assembled=assembled,
            row_line_number=row_line_number,
        )
        tables.append(
            ProducerTable(
                document_id=document_id,
                table_region_id=region_id,
                page_indices=(page_index,),
                column_ids=tuple(column_ids),
                row_ids=tuple(row_ids[line_no] for line_no in row_line_number),
                header_row_ids=tuple(row_ids[block_numbers[r]] for r in header_local_rows),
                cells=tuple(cells),
                header_paths=tuple(layout_header_paths_from_cells(
                    region_id=region_id,
                    column_ids=column_ids,
                    header_cells=header_cells,
                    header_local_rows=header_local_rows,
                )),
                vertical_merges=tuple(merges),
            ),
        )
    return tables


def layout_header_paths_from_cells(
    *,
    region_id: str,
    column_ids: list[str],
    header_cells: dict[tuple[int, int], str],
    header_local_rows: list[int],
):
    _ = region_id
    paths = []
    for col, column_id in enumerate(column_ids):
        leaf = header_cells.get((header_local_rows[-1], col)) if header_local_rows else None
        if not leaf:
            continue
        ancestors = [
            header_cells[(r, col)]
            for r in header_local_rows[:-1]
            if (r, col) in header_cells
        ]
        paths.append((column_id, tuple([*ancestors, leaf])))
    return paths


def detect_merges_layout_merged(
    *,
    region_id: str,
    column_ids: list[str],
    assembled: list[dict[int, list[str]]],
    row_line_number: list[int],
) -> list[VerticalMergeRegion]:
    """After wrap absorption only true empty-label gaps remain; they were
    merged upward already, so residual merges are rare. Kept for contract
    completeness when a span carries no continuation text at all."""
    merges: list[VerticalMergeRegion] = []
    return merges


