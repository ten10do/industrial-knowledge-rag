from pathlib import Path
from typing import Callable

from .chunker import chunk_pages
from .classifier import (
    classify_document,
    infer_equipment_model,
    infer_equipment_type,
    infer_language,
    infer_manufacturer,
    infer_publish_date,
    infer_title,
    infer_version,
)
from .models import IndustrialDocumentMetadata, PageText, stable_document_id


def _identity_content(path: Path, pages: list[PageText]) -> bytes:
    if path.exists():
        return path.read_bytes()
    normalized = "\n\f\n".join(page.text.strip() for page in pages)
    return normalized.encode("utf-8")


def ingest_pages(
    file_path: str | Path,
    pages: list[PageText],
    *,
    chunk_size: int = 800,
    overlap: int = 150,
    fallback_splitter: Callable[[str, int, int], list[str]] | None = None,
):
    path = Path(file_path)
    valid_pages = [page for page in pages if page.text and page.text.strip()]
    if not valid_pages:
        raise ValueError(
            "PDF 没有读取到有效文字内容。请使用文字版 PDF，不要使用扫描版 PDF。"
        )
    sample_text = "\n".join(page.text for page in valid_pages[:5])
    document = IndustrialDocumentMetadata(
        document_id=stable_document_id(
            _identity_content(path, valid_pages),
        ),
        source=path.name,
        file_name=path.name,
        document_type=classify_document(path.name, sample_text),
        manufacturer=infer_manufacturer(sample_text),
        equipment_type=infer_equipment_type(sample_text),
        equipment_model=infer_equipment_model(sample_text),
        title=infer_title(sample_text, path.name),
        language=infer_language(sample_text),
        document_version=infer_version(sample_text),
        publish_date=infer_publish_date(sample_text),
    )
    chunks = chunk_pages(
        valid_pages,
        document,
        chunk_size=chunk_size,
        overlap=overlap,
        fallback_splitter=fallback_splitter,
    )
    if not chunks:
        raise ValueError("PDF 结构解析后没有得到有效文本块。")
    return chunks
