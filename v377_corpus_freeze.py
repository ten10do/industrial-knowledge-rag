"""V3.77 Benchmark–Corpus Alignment: frozen-corpus capture.

Read-only capture of the V3.69/V3.76 benchmark corpus identity:

  * SHA256 of every corpus PDF,
  * per-document page counts as loaded by the production ``rag_core.load_pdf``,
  * the complete chunk catalog of the frozen ``vector_db_v369`` Chroma store
    (all texts + metadata, in stored order),
  * resolved document identity metadata (V3.73 resolver),
  * a corpus-level chunk-text digest.

Nothing here mutates the runtime: no re-indexing, no writes to
``vector_db_v369``, no threshold or Evidence change. All detailed output is
private and lands under ``backend/evaluation/benchmark_private/v377_alignment/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402

OUT_DIR = (
    _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment"
)
CORPUS_DIR = OUT_DIR / "corpus"

# Frozen V3.69 corpus — identical to the PDF list used by v369_real_baseline.py.
BASE = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_PDFS = [
    BASE / "v364_generalization" / "documents" / "Siemens_S7_1200_System_Manual_EN.pdf",
    BASE / "v364_generalization" / "documents" / "ABB_ACS580_Firmware_Manual.pdf",
    BASE / "v364_generalization" / "documents" / "Schneider_M221_Hardware_Guide_EN.pdf",
]
PERSIST_DIR = _REPO_ROOT / "vector_db_v369"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. File-level freeze ---------------------------------------------
    file_records = []
    for pdf in CORPUS_PDFS:
        if not pdf.is_file():
            raise SystemExit(f"frozen corpus PDF missing: {pdf}")
        pages = rag_core.load_pdf(str(pdf))
        record = {
            "filename": pdf.name,
            "relative_path": str(pdf.relative_to(_REPO_ROOT)),
            "sha256": sha256_file(pdf),
            "size_bytes": pdf.stat().st_size,
            "loaded_page_count": len(pages),
            "identity_metadata": {},
        }
        try:
            from backend.retrieval.document_identity_v373 import (
                resolve_document_identity,
            )

            record["identity_metadata"] = dict(resolve_document_identity(str(pdf)))
        except Exception as exc:  # noqa: BLE001 — audit must not die on resolver issues
            record["identity_metadata"] = {"resolver_error": repr(exc)}
        page_digest = hashlib.sha256(
            "\n\x00\n".join(
                f"p{doc.metadata.get('page', i)}\x1f{doc.page_content}"
                for i, doc in enumerate(pages)
            ).encode("utf-8")
        ).hexdigest()
        record["page_text_sha256"] = page_digest
        file_records.append(record)
        print(f"  {pdf.name}: {record['loaded_page_count']} pages sha256={record['sha256'][:16]}…")

    # --- 2. Chunk catalog from the frozen Chroma store ---------------------
    from langchain_chroma import Chroma

    embeddings = rag_core.get_embedding_model()
    vector_db = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
    payload = vector_db.get(include=["documents", "metadatas"])
    texts = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])
    ids = payload.get("ids", [])
    if not (len(texts) == len(metadatas) == len(ids)):
        raise SystemExit("Chroma get() returned inconsistent arrays")

    chunks = []
    for cid, text, meta in zip(ids, texts, metadatas):
        chunks.append({
            "chroma_id": cid,
            "source": str((meta or {}).get("source", "")),
            "page": (meta or {}).get("page"),
            "metadata": {k: (meta or {}).get(k) for k in sorted(meta or {})},
            "text_len": len(text),
        })
    chunk_texts_blob = "\n\x00\n".join(texts).encode("utf-8")
    corpus_chunk_digest = hashlib.sha256(chunk_texts_blob).hexdigest()

    source_counts: dict[str, int] = {}
    for chunk in chunks:
        source_counts[chunk["source"]] = source_counts.get(chunk["source"], 0) + 1

    manifest = {
        "corpus_version": "V377_CORPUS_FROZEN_V1",
        "benchmark_runtime_index": "vector_db_v369",
        "documents": file_records,
        "document_count": len(file_records),
        "chunk_count": len(chunks),
        "chunks_per_document": dict(sorted(source_counts.items())),
        "corpus_chunk_text_sha256": corpus_chunk_digest,
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "chunker": {"name": "RecursiveCharacterTextSplitter", "chunk_size": 800, "chunk_overlap": 150},
    }

    json.dump(manifest, open(CORPUS_DIR / "corpus_manifest.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"n_chunks": len(chunks), "chunks": chunks},
              open(CORPUS_DIR / "chunk_catalog.json", "w"), indent=1, ensure_ascii=False)

    print(f"\ndocument_count={manifest['document_count']} chunk_count={len(chunks)}")
    for src, n in sorted(source_counts.items()):
        print(f"  {src}: {n} chunks")
    print(f"corpus_chunk_text_sha256={corpus_chunk_digest}")
    print(f"saved: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
