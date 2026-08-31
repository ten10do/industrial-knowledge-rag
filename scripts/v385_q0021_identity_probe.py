#!/usr/bin/env python3
"""V3.85 identity-gate probe for V369-Q0021.

Q0021 ("Tell me about the SINAMICS G120 drive.") is annotated ABSTAIN /
IDENTITY because the frozen corpus (S7-1200, ACS580, M221) contains no
SINAMICS G120 documentation. In the V3.85 control trace it was answered
with STRONG_VECTOR_EVIDENCE at ratio 0.79 (a false answer).

This probe replays the real retrieval chain for Q0021 and reconstructs the
intermediate identity-gate state (has_query_identity, known_identity,
compatible_identity, relation, distance ratio) WITHOUT modifying
backend/retrieval/evidence.py. It establishes, with runtime evidence,
which gate short-circuited and why.

Usage
-----
    python scripts/v385_q0021_identity_probe.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag_core  # noqa: E402
rag_core.PERSIST_DIR = str(ROOT / "vector_db_v369")

from langchain_chroma import Chroma  # noqa: E402
from backend.retrieval.filters import analyze_query  # noqa: E402
from backend.retrieval.product_identity import (  # noqa: E402
    IdentityRelation,
    identity_from_metadata,
    identities_from_documents,
    identities_from_query,
    identity_is_compatible,
    identity_relation,
    normalize_identity_text,
)
from backend.retrieval.evidence import (  # noqa: E402
    _IDENTITY_GENERIC_WORDS,
    ProductIdentity,
    analyze_retrieval_evidence,
    default_policy,
)

QUERY_ID = "V369-Q0021"
QUERY = "Tell me about the SINAMICS G120 drive."


def main() -> int:
    policy = default_policy()
    search_db = Chroma(
        persist_directory=rag_core.PERSIST_DIR,
        embedding_function=rag_core.get_embedding_model(),
    )
    scored = search_db.similarity_search_with_score(QUERY, k=4)
    documents = [d for d, _s in scored]
    top = documents[0]
    candidates = [(d, s) for d, s in scored]

    print("=== V369-Q0021 identity-gate probe ===")
    print(f"query           : {QUERY!r}")
    print(f"top1 source     : {top.metadata.get('source')} p{top.metadata.get('page')}")
    print(f"top1 distance   : {scored[0][1]:.6f}")
    print(f"top1 ratio      : {scored[0][1] / policy.max_vector_distance:.4f} "
          f"(threshold={policy.max_vector_distance:.6f})")

    # --- query-side analysis (filters.analyze_query) ---
    analysis = analyze_query(QUERY, documents)
    print("\n[analysis] analyze_query")
    print(f"  manufacturer   = {analysis.manufacturer!r}")
    print(f"  equipment_model= {analysis.equipment_model!r}")
    print(f"  product_series = {analysis.product_series!r}")
    print(f"  product_family = {analysis.product_family!r}")
    print(f"  identity_conf  = {analysis.identity_confidence!r}")
    print(f"  product_identities = "
          f"{[i.as_dict() for i in analysis.product_identities]}")

    # --- evidence.py :318-371 reconstruction ---
    identifiers = [
        str(getattr(doc, "metadata", {}).get("identifier", "")).strip()
        for doc in documents
    ]
    query_identity = ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    )
    candidate_identities = [identity_from_metadata(doc.metadata) for doc in documents]

    explicit_corpus_identities = {}
    normalized_query = normalize_identity_text(QUERY)
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized_query)
        if len(token) >= 3 and token not in _IDENTITY_GENERIC_WORDS
    }
    query_numbers = set(re.findall(r"\d+", normalized_query))
    for document in documents:
        identity = identity_from_metadata(getattr(document, "metadata", {}) or {})
        terms = tuple(filter(None, (
            identity.equipment_model, identity.product_series,
            identity.product_family, *identity.aliases,
        )))
        normalized_terms = {
            variant
            for term in terms
            for variant in (
                normalize_identity_text(term),
                re.sub(r"(?:-|\s*)series$", "", normalize_identity_text(term)),
            )
            if len(variant) >= 3
            and (any(c.isdigit() for c in variant)
                 or re.search(r"^[a-z][a-z0-9]*[A-Z]", term) is not None)
        }
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized_query)
            or any(
                term.startswith(token)
                and len(token) >= max(3, len(term) // 2)
                and (not (tn := set(re.findall(r"\d+", term))) or query_numbers <= tn)
                for token in query_tokens
            )
            for term in normalized_terms
        ):
            key = tuple(normalize_identity_text(getattr(identity, f)) for f in (
                "manufacturer", "product_family", "product_series", "equipment_model"))
            explicit_corpus_identities[key] = identity
    print("\n[gate state] evidence.py :318-371 reconstruction")
    print(f"  query_tokens            = {sorted(query_tokens)}")
    print(f"  explicit_corpus_matches = {len(explicit_corpus_identities)}")
    for key in explicit_corpus_identities:
        print(f"    matched doc identity  = {key}")

    analyzed_identities = analysis.product_identities or (query_identity,)
    explicit_identities = tuple(explicit_corpus_identities.values())
    analysis_matches_explicit = any(
        identity_is_compatible(analyzed, explicit)
        for analyzed in analyzed_identities
        for explicit in explicit_identities
    )
    query_identities = (
        analyzed_identities
        if analysis_matches_explicit or not explicit_identities
        else explicit_identities
    )
    has_query_identity = any(
        i.product_family or i.product_series or i.equipment_model
        for i in query_identities
    )
    print(f"  has_query_identity      = {has_query_identity}")
    for i in query_identities:
        print(f"    query identity        = {i.as_dict()}")

    # --- relations / compatibility (evidence.py :376-403) ---
    relations = {
        identity_relation(i, cand)
        for i in query_identities
        for cand in candidate_identities
    }
    relation = next(
        (v for v in (
            IdentityRelation.EXACT_MODEL,
            IdentityRelation.SAME_SERIES,
            IdentityRelation.SAME_FAMILY,
            IdentityRelation.MISMATCH,
        ) if v in relations),
        IdentityRelation.UNKNOWN,
    )
    compatible_identity = any(
        identity_is_compatible(i, cand)
        for i in query_identities
        for cand in candidate_identities
    )
    corpus_identities = identities_from_documents(documents)
    known_identity = (
        all(
            any(identity_is_compatible(qi, ci) for ci in corpus_identities)
            for qi in query_identities
        )
        if has_query_identity
        else True
    )
    print(f"  relation                = {relation.value}")
    print(f"  compatible_identity     = {compatible_identity}")
    print(f"  known_identity          = {known_identity}")
    print(f"  corpus_identities       = {[i.as_dict() for i in corpus_identities]}")

    # --- full decision via real chain ---
    rr = None
    from backend.evaluation.score_lineage import build_retrieval_result
    rr = build_retrieval_result(scored)
    evidence = analyze_retrieval_evidence(
        QUERY, rr, documents,
        retrieval_mode="vector_only_v369", identity_matching=True,
    )
    print("\n[decision] real chain")
    print(f"  decision     = {evidence.decision}")
    print(f"  reason       = {evidence.reason}")
    print(f"  relation     = {evidence.identity_relation}")
    print(f"  vector_dist  = {evidence.vector_distance}")
    print(f"  contract     = {json.dumps(evidence.contract, default=str)[:300]}")

    print("\n=== probe conclusion ===")
    if not has_query_identity:
        print("identity gate state : has_query_identity=False -> identity gates")
        print("                      (known_identity / compatible_identity) short-circuited")
        print("route taken        : vector-distance branch -> STRONG_VECTOR_EVIDENCE")
        print("root cause         : query-side identity extraction is corpus-driven")
        print("                      (identities_from_query only matches corpus-known")
        print("                      identities); SINAMICS G120 is outside the frozen")
        print("                      corpus, so no query identity was resolved and the")
        print("                      out-of-corpus signal never reached a gate.")
    else:
        print("identity gate state : has_query_identity=True (probe did not expect this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
