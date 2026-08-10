from dataclasses import dataclass
import re
from typing import Callable

from .classifier import FAULT_CODE_PATTERN, classify_knowledge
from .models import (
    IndustrialChunk,
    IndustrialDocumentMetadata,
    PageText,
    stable_chunk_id,
)


HEADING_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百\d]+章\s*.+$"),
    re.compile(r"^\d+(?:\.\d+)+\s+\S.+$"),
    re.compile(r"^[一二三四五六七八九十]+、\s*\S.+$"),
    re.compile(r"^[（(][一二三四五六七八九十\d]+[）)]\s*\S.+$"),
)
KEYWORD_HEADINGS = {
    "概述",
    "技术参数",
    "故障代码",
    "故障码",
    "操作步骤",
    "准备工作",
    "启动流程",
    "安全注意事项",
    "异常处理",
    "维护与保养",
    "点检内容",
    "overview",
    "technical specifications",
    "fault codes",
    "alarm codes",
    "operating procedure",
    "maintenance",
}


@dataclass
class _Unit:
    section: str
    subsection: str
    lines: list[tuple[int, str]]


def _heading_level(line: str) -> int | None:
    value = line.strip()
    if value.casefold() in KEYWORD_HEADINGS:
        return 1
    for pattern in HEADING_PATTERNS:
        if pattern.match(value):
            numeric = re.match(r"^(\d+(?:\.\d+)+)", value)
            return numeric.group(1).count(".") + 1 if numeric else 1
    return None


def _section_units(pages: list[PageText], default_title: str) -> list[_Unit]:
    units = []
    section = default_title
    subsection = ""
    current = _Unit(section, subsection, [])
    for page in pages:
        if (
            current.lines
            and current.section == default_title
            and not current.subsection
        ):
            units.append(current)
            current = _Unit(section, subsection, [])
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                if current.lines and current.lines[-1][1]:
                    current.lines.append((page.page, ""))
                continue
            level = _heading_level(line)
            in_procedure = classify_knowledge(section, "") == "procedure"
            if level is not None and not (
                in_procedure and re.match(r"^\d+[.)、]", line)
            ):
                if current.lines:
                    units.append(current)
                if level == 1:
                    section, subsection = line, ""
                else:
                    subsection = line
                current = _Unit(section, subsection, [(page.page, line)])
            else:
                current.lines.append((page.page, line))
    if current.lines:
        units.append(current)
    return units


def _fault_units(unit: _Unit) -> list[_Unit]:
    starts = [
        index
        for index, (_, line) in enumerate(unit.lines)
        if FAULT_CODE_PATTERN.match(line)
    ]
    if not starts:
        return [unit]
    results = []
    if starts[0] > 0:
        results.append(_Unit(unit.section, unit.subsection, unit.lines[: starts[0]]))
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(unit.lines)
        code_line = unit.lines[start][1]
        results.append(_Unit(unit.section, code_line, unit.lines[start:end]))
    return [item for item in results if any(line for _, line in item.lines)]


def _recursive_text_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > chunk_size:
        window = remaining[:chunk_size]
        cut = max(window.rfind(separator) for separator in ("\n\n", "\n", "。", ";", "；", " "))
        if cut < chunk_size // 2:
            cut = chunk_size
        else:
            cut += 1
        value = remaining[:cut].strip()
        if value:
            chunks.append(value)
        next_start = max(0, cut - overlap)
        remaining = remaining[next_start:]
    if remaining.strip():
        chunks.append(remaining.strip())
    return chunks


def chunk_pages(
    pages: list[PageText],
    document: IndustrialDocumentMetadata,
    *,
    chunk_size: int = 800,
    overlap: int = 150,
    fallback_splitter: Callable[[str, int, int], list[str]] | None = None,
) -> list[IndustrialChunk]:
    chunks = []
    atomic_limit = chunk_size * 2
    units = [
        fault_unit
        for section_unit in _section_units(pages, document.title or document.file_name)
        for fault_unit in _fault_units(section_unit)
    ]
    for unit in units:
        content = "\n".join(line for _, line in unit.lines).strip()
        if not content:
            continue
        page_numbers = [page for page, line in unit.lines if line]
        page_start = min(page_numbers)
        page_end = max(page_numbers)
        code_match = FAULT_CODE_PATTERN.match(content)
        error_code = code_match.group(1).upper() if code_match else ""
        knowledge_type = classify_knowledge(unit.section, content, error_code)
        limit = atomic_limit if knowledge_type in {"fault", "procedure", "parameter"} else chunk_size
        parts = (
            fallback_splitter(content, limit, overlap)
            if fallback_splitter and len(content) > limit
            else _recursive_text_split(content, limit, overlap)
        )
        for part in parts:
            metadata = IndustrialDocumentMetadata(
                **{
                    **document.to_flat_dict(),
                    "section": unit.section,
                    "subsection": unit.subsection,
                    "page": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "knowledge_type": knowledge_type,
                    "error_code": error_code,
                    "chunk_id": stable_chunk_id(
                        document.document_id,
                        unit.section,
                        unit.subsection,
                        page_start,
                        page_end,
                        part,
                    ),
                    "chunk_index": len(chunks),
                }
            ).to_flat_dict()
            chunks.append(IndustrialChunk(part, metadata))
    return chunks
