"""V3.72 identity FR classification: inspect runtime identity for failing cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")
from rag_core import retrieve_docs
from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.product_identity import (
    IdentityRelation,
    identity_from_query,
    identities_from_documents,
    identity_relation as compute_relation,
)


def main() -> None:
    # Representative failing queries.
    test_cases = [
        ("Tell me about the SINAMICS G120 drive.", "IDENTITY"),
        ("Tell me about the ACS580 drive.", "IDENTITY"),
        ("Tell me about the Altivar ATV320 drive.", "IDENTITY"),
        ("What is a variable frequency drive?", "NON_TABLE"),
        ("How does pulse-width modulation work?", "NON_TABLE"),
    ]

    for query_text, slice_label in test_cases:
        print(f"\n{'='*60}")
        print(f"[{slice_label}] {query_text}")

        scored = retrieve_docs(query_text, k=4)
        docs = [d for d, s in scored]
        cands = [RetrievalCandidate(document=d, retrieval_source="chroma") for d, s in scored]
        rr = RetrievalResult(candidates=cands, retrieval_mode="diag")
        ev = analyze_retrieval_evidence(query_text, rr, docs,
                                        retrieval_mode="diag", identity_matching=True)

        qi = ev.query_identity
        ci = ev.candidate_identity
        print(f"  decision={ev.decision} reason={ev.reason}")
        print(f"  identity_relation={ev.identity_relation}")
        print(f"  query_identity: mfr={qi.get('manufacturer','')!r} "
              f"family={qi.get('product_family','')!r} "
              f"series={qi.get('product_series','')!r} "
              f"model={qi.get('equipment_model','')!r}")
        print(f"  cand_identity:  mfr={ci.get('manufacturer','')!r} "
              f"family={ci.get('product_family','')!r} "
              f"series={ci.get('product_series','')!r} "
              f"model={ci.get('equipment_model','')!r}")

        # Also show what identities_from_documents extracts from top docs.
        doc_identities = identities_from_documents(docs[:2])
        for di, d in zip(doc_identities[:2], docs[:2]):
            print(f"  doc_identity: mfr={d.manufacturer!r} family={d.product_family!r} "
                  f"series={d.product_series!r} model={d.equipment_model!r}")


if __name__ == "__main__":
    main()
