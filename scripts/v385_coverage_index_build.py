#!/usr/bin/env python3
"""V3.85 coverage index builder.

Extracts the full chunk text from the frozen runtime index
(vector_db_v369/chroma.sqlite3) and emits a full-text chunk catalog that the
coverage gate can consume, cross-validated against the frozen V377 catalog.

Design constraints
------------------
* Never modifies frozen artifacts. The existing V377 chunk_catalog.json is
  read-only input; the output is a NEW file (chunk_catalog_v385_fulltext.json).
* Every chunk must carry non-empty text; any gap is a FAIL gate.
* chroma_id set must equal the frozen catalog's chroma_id set exactly.
* source distribution must match corpus_manifest.json chunks_per_document.

Usage
-----
    python scripts/v385_coverage_index_build.py

Output
------
    backend/evaluation/benchmark_private/v377_alignment/corpus/chunk_catalog_v385_fulltext.json
    alongside a printed verification summary + output SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "vector_db_v369", "chroma.sqlite3")
CORPUS_DIR = os.path.join(
    ROOT,
    "backend",
    "evaluation",
    "benchmark_private",
    "v377_alignment",
    "corpus",
)
FROZEN_CATALOG = os.path.join(CORPUS_DIR, "chunk_catalog.json")
MANIFEST = os.path.join(CORPUS_DIR, "corpus_manifest.json")
OUTPUT = os.path.join(CORPUS_DIR, "chunk_catalog_v385_fulltext.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    fails: list[str] = []

    # 1. extract from the runtime index
    con = sqlite3.connect(INDEX_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT e.id AS rid, e.embedding_id AS chroma_id, f.string_value AS text
        FROM embeddings e
        JOIN embedding_fulltext_search f ON e.id = f.rowid
        """
    )
    rows = cur.fetchall()
    if len(rows) != 7333:
        fails.append(f"index chunk count {len(rows)} != 7333")

    text_by_rid = {r["rid"]: r["text"] for r in rows}
    chroma_by_rid = {r["rid"]: r["chroma_id"] for r in rows}

    # metadata (flattened key/value per row id)
    cur.execute(
        "SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata"
    )
    meta_rows = cur.fetchall()
    meta: dict[int, dict] = {}
    for r in meta_rows:
        m = meta.setdefault(r["id"], {})
        if r["string_value"] is not None:
            m[r["key"]] = r["string_value"]
        elif r["int_value"] is not None:
            m[r["key"]] = r["int_value"]
        elif r["float_value"] is not None:
            m[r["key"]] = r["float_value"]
        elif r["bool_value"] is not None:
            m[r["key"]] = r["bool_value"]
    con.close()

    # 2. assemble records
    chunks = []
    empty_text = 0
    for rid in sorted(text_by_rid):
        text = text_by_rid[rid]
        if not text or not text.strip():
            empty_text += 1
        m = meta.get(rid, {})
        chunks.append(
            {
                "chroma_id": chroma_by_rid[rid],
                "source": m.get("source", ""),
                "page": m.get("page", None),
                "text": text,
                "metadata": m,
                "text_len": len(text),
            }
        )
    if empty_text:
        fails.append(f"empty-text chunks: {empty_text}")

    # 3. cross-validate against frozen catalog + manifest
    frozen = load_json(FROZEN_CATALOG)
    frozen_ids = {c["chroma_id"] for c in frozen["chunks"]}
    new_ids = {c["chroma_id"] for c in chunks}
    if frozen_ids != new_ids:
        only_frozen = frozen_ids - new_ids
        only_new = new_ids - frozen_ids
        fails.append(
            f"chroma_id mismatch: frozen-only={len(only_frozen)} new-only={len(only_new)}"
        )
    if frozen["n_chunks"] != len(chunks):
        fails.append(f"n_chunks {frozen['n_chunks']} != extracted {len(chunks)}")

    manifest = load_json(MANIFEST)
    expected_per_doc = manifest["chunks_per_document"]
    actual: dict[str, int] = {}
    for c in chunks:
        src = c["source"]
        if not src:
            continue
        actual[src] = actual.get(src, 0) + 1
    if actual != expected_per_doc:
        fails.append(f"source distribution mismatch: {actual} != {expected_per_doc}")

    # 4. persist
    payload = {
        "catalog_version": "V385_FULLTEXT_CATALOG_V1",
        "n_chunks": len(chunks),
        "index": "vector_db_v369",
        "chunks": chunks,
    }
    os.makedirs(CORPUS_DIR, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    out_sha = sha256_file(OUTPUT)
    frozen_sha = sha256_file(FROZEN_CATALOG)

    print("=== V385 coverage index build summary ===")
    print(f"extracted chunks : {len(chunks)}")
    print(f"empty text       : {empty_text}")
    print(f"source dist      : {actual}")
    print(f"frozen catalog   : {frozen['n_chunks']} chunks, sha256={frozen_sha[:16]}")
    print(f"output           : {OUTPUT}")
    print(f"output sha256    : {out_sha}")
    print(f"status           : {'PASS' if not fails else 'FAIL'}")
    for f_ in fails:
        print(f"  FAIL: {f_}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
