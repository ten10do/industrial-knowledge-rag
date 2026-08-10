"""Shared industrial document ingestion for light and full RAG modes."""

from .models import IndustrialChunk, IndustrialDocumentMetadata, PageText
from .pipeline import ingest_pages

__all__ = [
    "IndustrialChunk",
    "IndustrialDocumentMetadata",
    "PageText",
    "ingest_pages",
]
