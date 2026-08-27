"""V3.76 Retrieval Score Lineage Fidelity Correction.

Formal E1 run: replays the V3.71 real-evidence benchmark (69 queries) against
the frozen ``vector_db_v369`` index with the real Chroma distance wired through
``RetrievalCandidate.vector_score`` -> Evidence ``vector_distance``.

Outputs:
  * E0 replay (unwired) for baseline reconciliation (must be 48/2/19),
  * E1 (wired) run twice for deterministic replay,
  * ranking diff (E0 vs E1 retrieved chunk-id sequences),
  * per-query score lineage fidelity.

All per-query private details land in the gitignored ``results/v376_score_lineage/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")

from langchain_chroma import Chroma  # noqa: E402
from backend.retrieval.evidence import analyze_retrieval_evidence, default_policy  # noqa: E402
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.coverage_attribution import blocking_mechanism, is_coverage_bound  # noqa: E402
from backend.evaluation.contract_eval_v367 import (  # noqa: E402
    EvidenceEvaluationRecord,
    _accept_as_pass,
    evaluate_contract_native,
)
from backend.evaluation.score_lineage import (  # noqa: E402
    assert_lineage_fidelity,
    build_retrieval_result,
)


def _load_search():
    embeddings = rag_core.get_embedding_model()
    vector_db = Chroma(persist_directory=rag_core.PERSIST_DIR, embedding_function=embeddings)
    return lambda question, k=4: vector_db.similarity_search_with_score(question, k=k)


def make_record(evidence) -> EvidenceEvaluationRecord:
    contract = evidence.contract or {}
    return EvidenceEvaluationRecord(
        decision=evidence.decision,
        reason=evidence.reason,
        query_identity=dict(evidence.query_identity),
        candidate_identity=dict(evidence.candidate_identity),
        identity_relation=evidence.identity_relation,
        contract_requirements_covered=bool(contract.get("sufficient", False)),
        lexical_score=evidence.lexical_score,
        vector_distance=evidence.vector_distance,
        metadata_consistency=evidence.metadata_consistency,
        covered_claim_keys=frozenset(contract.get("covered", [])),
    )


def _chunk_ids(scored_docs):
    return [str((getattr(d, "metadata", {}) or {}).get("chunk_id", "")) for d, _ in scored_docs]


def run(queries, search, *, wire: bool):
    policy = default_policy()
    rows = []
    latencies = []
    for q in queries:
        query_text = q["query"]
        exp = q["expectation"]
        t0 = time.time()
        try:
            scored_docs = search(query_text, k=4)
        except Exception as exc:  # noqa: BLE001
            rows.append({"query_id": exp.query_id, "slice": q["slice"],
                         "decision": "RETRIEVAL_ERROR", "reason": f"RETRIEVAL_ERROR:{exc}"})
            continue

        if wire:
            rr = build_retrieval_result(scored_docs)
        else:
            cands = [RetrievalCandidate(document=d, retrieval_source="chroma") for d, _ in scored_docs]
            for c, (_d, s) in zip(cands, scored_docs):
                c.fusion_score = float(-s)
            rr = RetrievalResult(candidates=cands, retrieval_mode="vector_only_v369")

        candidates = rr.candidates
        ev = analyze_retrieval_evidence(
            query_text, rr, [d for d, _s in scored_docs],
            retrieval_mode="vector_only_v369", identity_matching=True,
        )
        latencies.append(time.time() - t0)

        rec = make_record(ev)
        result = evaluate_contract_native(exp, rec)
        passing = _accept_as_pass(result.verdict)
        lineage = assert_lineage_fidelity(scored_docs, candidates, ev)
        contract = ev.contract or {}
        rows.append({
            "query_id": exp.query_id,
            "slice": q["slice"],
            "difficulty": q["difficulty"],
            "expected": exp.expected_decision.value,
            "decision": ev.decision,
            "reason": ev.reason,
            "identity_relation": ev.identity_relation,
            "has_candidates": ev.has_candidates,
            "vector_distance": ev.vector_distance,
            "lexical_score": ev.lexical_score,
            "contract_has_critical": bool(contract.get("has_critical_requirements", False)),
            "contract_sufficient": bool(contract.get("sufficient", False)),
            "chunk_ids": _chunk_ids(scored_docs),
            "raw_top1_distance": lineage["raw_top1"],
            "fidelity_ok": lineage["fidelity_ok"],
            "blocking_mechanism": blocking_mechanism(ev) if ev.decision == "ABSTAIN" else "",
            "coverage_bound": is_coverage_bound(ev),
            "verdict": result.verdict.value,
            "passing": passing,
            "max_vector_distance": policy.max_vector_distance,
        })
    return rows, latencies


def summarize(rows, label):
    n = len(rows)
    correct = sum(1 for r in rows if r["passing"])
    fa = sum(1 for r in rows if not r["passing"] and r["expected"] == "ABSTAIN" and r["decision"] == "ANSWER")
    fr = sum(1 for r in rows if not r["passing"] and r["expected"] == "ANSWER" and r["decision"] == "ABSTAIN")
    return {"label": label, "n": n, "correct": correct, "fa": fa, "fr": fr, "accuracy": round(correct / max(n, 1), 4)}


def canonical_digest(rows):
    payload = [
        (r["query_id"], tuple(r.get("chunk_ids", [])), r["decision"], r["reason"],
         r["verdict"], r["vector_distance"])
        for r in rows
    ]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    import importlib
    v369 = importlib.import_module("v369_real_baseline")
    queries = v369.generate_queries()
    policy = default_policy()
    print(f"Queries: {len(queries)} | max_vector_distance = {policy.max_vector_distance!r}")

    search = _load_search()

    print("\n== E0 replay (unwired) ==")
    e0_rows, _ = run(queries, search, wire=False)
    print("== E1 run #1 (wired) ==")
    e1_rows, lat = run(queries, search, wire=True)
    print("== E1 run #2 (wired, replay) ==")
    e1_rows2, _ = run(queries, search, wire=True)

    out_dir = _REPO_ROOT / "results" / "v376_score_lineage"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"e0": e0_rows, "e1_run1": e1_rows, "e1_run2": e1_rows2},
              open(out_dir / "full_trace.json", "w"), indent=1, default=str)

    print("\n== AGGREGATE ==")
    for s in (summarize(e0_rows, "E0_unwired"), summarize(e1_rows, "E1_wired")):
        print(f"  {s['label']:12s} correct={s['correct']} fa={s['fa']} fr={s['fr']} acc={s['accuracy']}")

    # Deterministic replay.
    d1 = canonical_digest(e1_rows)
    d2 = canonical_digest(e1_rows2)
    print(f"\nE1 deterministic replay digest match: {d1 == d2}")
    print(f"  digest1={d1}")
    print(f"  digest2={d2}")

    # Ranking diff E0 vs E1.
    rank_diff = 0
    for a, b in zip(e0_rows, e1_rows):
        if a.get("chunk_ids") != b.get("chunk_ids"):
            rank_diff += 1
    print(f"\nRANKING_DIFF_COUNT (E0 vs E1) = {rank_diff}")

    # Score lineage fidelity.
    fid = sum(1 for r in e1_rows if r.get("fidelity_ok"))
    print(f"SCORE_LINEAGE_FIDELITY = {fid}/{len(e1_rows)} ({(100*fid//max(len(e1_rows),1))}%)")
    bad = [r["query_id"] for r in e1_rows if not r.get("fidelity_ok")]
    print(f"  non-fidelity query ids: {bad}")

    # 5 unwired cases (from V3.75) now have real scores.
    unwired_ids = {"V369-Q0021", "V369-Q0022", "V369-Q0027", "V369-Q0028", "V369-Q0055"}
    print("\n== 5 V3.75 unwired FR cases after wiring ==")
    for r in e1_rows:
        if r["query_id"] in unwired_ids:
            print(f"  {r['query_id']} [{r['slice']}] decision={r['decision']} reason={r['reason']} "
                  f"vdist={r['vector_distance']} raw_top1={r['raw_top1_distance']} idrel={r['identity_relation']}")

    # Reason family delta.
    print("\n== FR reason delta (E0 -> E1) ==")
    def fr_reasons(rows):
        return Counter(r["reason"] for r in rows
                       if r["expected"] == "ANSWER" and r["decision"] == "ABSTAIN")
    c0 = fr_reasons(e0_rows); c1 = fr_reasons(e1_rows)
    for k in sorted(set(c0) | set(c1)):
        print(f"  {k:35s} E0={c0.get(k,0)} E1={c1.get(k,0)}")

    if lat:
        lat.sort()
        print(f"\nLatency: median={lat[len(lat)//2]:.4f}s p95={lat[int(len(lat)*0.95)]:.4f}s")

    print(f"\nsaved: {out_dir / 'full_trace.json'}")


if __name__ == "__main__":
    main()
