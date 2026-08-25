"""V3.62 bounded table-context bundle construction.

Attaches same-table-region sibling chunks to primary retrieval results,
expanding Evidence input completeness WITHOUT altering ranking or decision
semantics. Every auxiliary chunk is explicitly role-marked so observability
can distinguish PRIMARY from TABLE_CONTEXT_AUXILIARY.

Feature flag: TABLE_REGION_CONTEXT_ENABLED (default OFF).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_TABLE_SIBLING_CHUNKS = 4
MAX_TABLE_CONTEXT_TOKENS = 800


@dataclass(frozen=True)
class TableRegionMetadata:
    document_id: str
    page: int
    table_region_id: str
    section: str = ""
    region_type: str = "BORDERLESS"
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ContextBundleEntry:
    chunk_id: str
    page_content: str
    metadata: dict
    role: str               # PRIMARY | TABLE_CONTEXT_AUXILIARY
    provenance_reason: str  # SAME_TABLE_REGION_ID | CAPTION | ...


@dataclass(frozen=True)
class TableContextBundle:
    table_region_id: str
    entries: tuple[ContextBundleEntry, ...]
    reason_codes: tuple[str, ...] = ()
    token_estimate: int = 0


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def extract_table_region_id(metadata: dict) -> str:
    return str(metadata.get("table_region_id", ""))


def build_table_context_bundle(
    primary_candidates: list,
    all_documents: list,
    *,
    max_siblings: int = MAX_TABLE_SIBLING_CHUNKS,
    max_tokens: int = MAX_TABLE_CONTEXT_TOKENS,
    feature_enabled: bool = False,
) -> TableContextBundle | None:
    """Build a bounded same-table context bundle from retrieval results.

    Returns None when feature is disabled, no table region detected,
    or budget exhausted.
    """
    if not feature_enabled:
        return None
    if not primary_candidates:
        return None

    # Find the first primary candidate that belongs to a table region.
    table_region_id = ""
    primary_entry = None
    for cand in primary_candidates:
        rid = extract_table_region_id(getattr(cand, "metadata", {}))
        if rid:
            table_region_id = rid
            primary_entry = ContextBundleEntry(
                chunk_id=cand.chunk_id,
                page_content=getattr(cand.document, "page_content", ""),
                metadata=dict(cand.metadata),
                role="PRIMARY",
                provenance_reason="PRIMARY_RETRIEVAL_HIT",
            )
            break

    if not table_region_id:
        return None

    # Collect same-table-region sibling chunks from corpus.
    entries: list[ContextBundleEntry] = []
    if primary_entry:
        entries.append(primary_entry)

    primary_chunk_ids = {
        getattr(c, "chunk_id", "") for c in primary_candidates
    }
    token_budget = max_tokens - (
        _estimate_tokens(primary_entry.page_content) if primary_entry else 0
    )

    seen_ids = set(primary_chunk_ids)
    for doc in all_documents:
        meta = getattr(doc, "metadata", {}) or {}
        doc_rid = extract_table_region_id(meta)
        if doc_rid != table_region_id:
            continue
        cid = str(meta.get("chunk_id", ""))
        if cid in seen_ids:
            continue
        content = getattr(doc, "page_content", "")
        est = _estimate_tokens(content)
        if token_budget - est < 0:
            break
        token_budget -= est
        seen_ids.add(cid)
        entries.append(
            ContextBundleEntry(
                chunk_id=cid,
                page_content=content,
                metadata=dict(meta),
                role="TABLE_CONTEXT_AUXILIARY",
                provenance_reason="SAME_TABLE_REGION_ID",
            ),
        )
        if len(entries) - 1 >= max_siblings:
            break

    if len(entries) <= 1 and not primary_entry:
        return None

    reason_codes = ["SAME_TABLE_REGION_ENRICHMENT"]
    total_tokens = sum(_estimate_tokens(e.page_content) for e in entries)
    return TableContextBundle(
        table_region_id=table_region_id,
        entries=tuple(entries),
        reason_codes=tuple(reason_codes),
        token_estimate=total_tokens,
    )
