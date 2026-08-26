"""V3.74 full pipeline trace: document → chunk → Chroma → retrieval → ProductIdentity."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.retrieval.product_identity import (
    identity_from_metadata,
    identities_from_documents,
)

# Test 1: load_pdf metadata.
from rag_core import load_pdf, split_documents

pdf = str(_REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v364_generalization" / "documents" / "ABB_ACS580_Firmware_Manual.pdf")
docs = load_pdf(pdf)
print("== Document Metadata (first doc) ==")
for k in ("source", "page", "manufacturer", "product_series", "equipment_model", "identity_source"):
    print(f"  {k}: {docs[0].metadata.get(k)!r}")

chunks = split_documents(docs[:3])  # small sample for speed
print(f"\n== Chunk Metadata ({len(chunks)} chunks from 3 pages) ==")
if chunks:
    for k in ("source", "page", "manufacturer", "product_series", "equipment_model"):
        print(f"  {k}: {chunks[0].metadata.get(k)!r}")

pi = identity_from_metadata(chunks[0].metadata)
print(f"\n== identity_from_metadata(chunk) ==")
print(f"  manufacturer={pi.manufacturer!r} family={pi.product_family!r} "
      f"series={pi.product_series!r} model={pi.equipment_model!r}")
print(f"  aliases={list(pi.aliases)[:5]}")

# Test 2: Chroma round-trip using existing index.
import rag_core
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")
from rag_core import retrieve_docs

print("\n== Chroma Retrieved Metadata ==")
scored = retrieve_docs("Tell me about the ACS580 drive.", k=4)
for i, (doc, score) in enumerate(scored[:2]):
    meta = doc.metadata or {}
    print(f"\nCandidate {i}: score={score:.4f}")
    for k in ("source", "page", "manufacturer", "product_series", "equipment_model",
              "identity_source"):
        print(f"  {k}: {meta.get(k)!r}")
    pi = identity_from_metadata(meta)
    print(f"  → ProductIdentity: mfr={pi.manufacturer!r} family={pi.product_family!r} "
          f"series={pi.product_series!r} model={pi.equipment_model!r}")

# Test 3: Evidence runtime full path.
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult

cands = [RetrievalCandidate(document=d, retrieval_source="chroma") for d, s in scored]
rr = RetrievalResult(candidates=cands, retrieval_mode="diag")
ev = analyze_retrieval_evidence(
    "Tell me about the ACS580 drive.", rr,
    [d for d, s in scored], retrieval_mode="diag", identity_matching=True,
)
print(f"\n== Evidence Runtime ==")
print(f"  decision={ev.decision} reason={ev.reason}")
print(f"  identity_relation={ev.identity_relation}")
qi = ev.query_identity
ci = ev.candidate_identity
print(f"  query_identity: {dict((k, qi.get(k,'')) for k in ('manufacturer','product_family','product_series','equipment_model'))}")
print(f"  cand_identity:  {dict((k, ci.get(k,'')) for k in ('manufacturer','product_family','product_series','equipment_model'))}")
