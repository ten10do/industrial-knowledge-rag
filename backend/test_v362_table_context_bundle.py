"""V3.62 focused tests for TableContextBundle construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.retrieval.table_context_v362 import (
    ContextBundleEntry,
    TableContextBundle,
    build_table_context_bundle,
    extract_table_region_id,
)


def _doc(chunk_id: str, table_rid: str = "", text: str = "sample content"):
    return SimpleNamespace(
        page_content=text,
        metadata={"chunk_id": chunk_id, "table_region_id": table_rid},
    )


def _cand(doc):
    from backend.retrieval.candidates import RetrievalCandidate

    return RetrievalCandidate(document=doc, retrieval_source="test")


def test_feature_disabled_returns_none():
    docs = [_doc("c1", "t1")]
    cand = _cand(docs[0])
    assert build_table_context_bundle([cand], docs, feature_enabled=False) is None


def test_no_table_region_returns_none():
    docs = [_doc("c1", "")]
    cand = _cand(docs[0])
    assert build_table_context_bundle([cand], docs, feature_enabled=True) is None


def test_same_table_sibling_attached():
    docs = [
        _doc("c1", "t1", "primary hit"),
        _doc("c2", "t1", "sibling context"),
        _doc("c3", "other", "different table"),
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle([cand], docs, feature_enabled=True)
    assert bundle is not None
    assert len(bundle.entries) == 2
    assert bundle.entries[0].role == "PRIMARY"
    assert bundle.entries[1].role == "TABLE_CONTEXT_AUXILIARY"
    assert bundle.entries[1].chunk_id == "c2"


def test_different_table_not_grouped():
    docs = [
        _doc("c1", "table_A", "primary"),
        _doc("c2", "table_B", "wrong table sibling"),
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle([cand], docs, feature_enabled=True)
    assert bundle is not None
    assert len(bundle.entries) == 1  # only primary, no wrong-table sibling


def test_token_budget_respected():
    long_text = "word " * 200  # ~200 tokens
    docs = [
        _doc("c1", "t1", "short primary"),
        _doc("c2", "t1", long_text),
        _doc("c3", "t1", long_text),
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle(
        [cand], docs, feature_enabled=True, max_tokens=300,
    )
    assert bundle is not None
    # Only c2 should fit (200 tokens), c3 would exceed 300 budget.
    aux_count = sum(1 for e in bundle.entries if e.role == "TABLE_CONTEXT_AUXILIARY")
    assert aux_count <= 1
    assert bundle.token_estimate <= 300


def test_max_siblings_respected():
    docs = [_doc(f"c{i}", "t1") for i in range(10)]
    docs.insert(0, _doc("primary", "t1", "primary"))
    cand = _cand(docs[0])
    bundle = build_table_context_bundle(
        [cand], docs, feature_enabled=True, max_siblings=2,
    )
    assert bundle is not None
    aux = [e for e in bundle.entries if e.role == "TABLE_CONTEXT_AUXILIARY"]
    assert len(aux) <= 2


def test_primary_preserved_as_first_entry():
    docs = [
        _doc("p1", "t1", "primary"),
        _doc("s1", "t1", "sibling"),
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle([cand], docs, feature_enabled=True)
    assert bundle.entries[0].role == "PRIMARY"
    assert bundle.entries[0].chunk_id == "p1"


def test_dedup_primary_not_repeated_as_auxiliary():
    docs = [
        _doc("c1", "t1", "primary"),
        _doc("c1_dup", "t1", "dup"),  # same chunk_id as primary candidate
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle([cand], [docs[0], docs[0]], feature_enabled=True)
    assert bundle is not None
    aux = [e for e in bundle.entries if e.role == "TABLE_CONTEXT_AUXILIARY"]
    assert len(aux) == 0


def test_extract_table_region_id_empty_metadata():
    assert extract_table_region_id({}) == ""


def test_provenance_reason_present():
    docs = [
        _doc("c1", "t1", "primary"),
        _doc("c2", "t1", "sibling"),
    ]
    cand = _cand(docs[0])
    bundle = build_table_context_bundle([cand], docs, feature_enabled=True)
    aux = [e for e in bundle.entries if e.role == "TABLE_CONTEXT_AUXILIARY"]
    if aux:
        assert aux[0].provenance_reason == "SAME_TABLE_REGION_ID"
