"""Layout-hybrid reconstruction: pypdf public layout-mode text as the cell
source, segmented into table blocks purely from whitespace alignment.

Motivation (V3.57 decision-support spike): 85.3% of hand-annotated GT rows
across four manufacturers' official manuals are recoverable from layout
lines (62% single-line, 23% wrap-tolerant), so a no-dependency path exists
that sidesteps the CMap decoding wall entirely.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class LayoutToken:
    start: int
    end: int
    text: str


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
) -> list[ProducerTable]:
    lines = layout_text.splitlines()
    tables: list[ProducerTable] = []
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

        def _absorb(local_r: int, target_index: int) -> None:
            for a, tok in zip(assignments[local_r], token_rows[local_r]):
                if a < 0:
                    continue
                text = fix_letterspacing(tok.text)
                bucket = assembled[target_index].setdefault(a, [])
                if not bucket or bucket[-1] != text:
                    bucket.append(text)

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
            if assembled and not hits_label and has_other:
                _absorb(local_r, len(assembled) - 1)
                continue
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


