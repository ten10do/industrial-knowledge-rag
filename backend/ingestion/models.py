from dataclasses import asdict, dataclass
import hashlib
import re


DOCUMENT_TYPES = {
    "manual",
    "fault_code",
    "sop",
    "maintenance",
    "technical_spec",
    "general",
}
KNOWLEDGE_TYPES = {
    "overview",
    "specification",
    "parameter",
    "procedure",
    "fault",
    "warning",
    "maintenance",
    "table",
    "general",
}
INGESTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class IndustrialDocumentMetadata:
    document_id: str
    source: str
    file_name: str
    document_type: str = "general"
    manufacturer: str = ""
    equipment_type: str = ""
    equipment_model: str = ""
    title: str = ""
    section: str = ""
    subsection: str = ""
    page: int = 0
    page_start: int = 0
    page_end: int = 0
    language: str = "unknown"
    document_version: str = ""
    publish_date: str = ""
    knowledge_type: str = "general"
    error_code: str = ""
    chunk_id: str = ""
    chunk_index: int = 0

    def to_flat_dict(self) -> dict[str, str | int]:
        values = asdict(self)
        if values["document_type"] not in DOCUMENT_TYPES:
            values["document_type"] = "general"
        if values["knowledge_type"] not in KNOWLEDGE_TYPES:
            values["knowledge_type"] = "general"
        return values


@dataclass(frozen=True)
class IndustrialChunk:
    page_content: str
    metadata: dict[str, str | int]


def normalized_content(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_document_id(content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"doc-{digest[:24]}"


def stable_chunk_id(
    document_id: str,
    section: str,
    subsection: str,
    page_start: int,
    page_end: int,
    content: str,
) -> str:
    identity = "\n".join(
        (
            document_id,
            section,
            subsection,
            str(page_start),
            str(page_end),
            normalized_content(content),
        )
    )
    return f"chunk-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
