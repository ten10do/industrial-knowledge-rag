"""V3.75 Evidence Coverage/Sufficiency Calibration audit.

Faithful runtime trace: replays the V3.71 real-evidence baseline (69 queries)
against the frozen ``vector_db_v369`` index, decomposing each decision into
predicates using ONLY the runtime ``RetrievalEvidence`` fields.

Two variants are recorded:

* ``baseline`` -- reproduces the frozen harness exactly (``RetrievalCandidate``
  built WITHOUT wiring ``vector_score``). This is the runtime truth.
* ``wired`` -- a diagnostic that wires ``vector_score = raw Chroma distance``
  (mirroring the production ``backend.rag_core._vector_candidates`` path) to
  characterise what the ``max_vector_distance`` gate WOULD do if the harness
  passed the score. This diagnostic is clearly separated and is NOT the runtime
  truth.

All per-query private details are written to a gitignored artifact under
``results/v375_audit/``. Nothing here changes the Evidence runtime decision.
"""
from __future__ import annotations

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
from backend.retrieval.evidence import (  # noqa: E402
    analyze_retrieval_evidence,
    default_policy,
)
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.coverage_attribution import (  # noqa: E402
    blocking_mechanism,
    is_coverage_bound,
)
from backend.evaluation.contract_eval_v367 import (  # noqa: E402
    EvidenceEvaluationRecord,
    _accept_as_pass,
    evaluate_contract_native,
)


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


def _load_search():
    """Load the embedding model and Chroma DB once, returning a search closure."""
    embeddings = rag_core.get_embedding_model()
    vector_db = Chroma(
        persist_directory=rag_core.PERSIST_DIR,
        embedding_function=embeddings,
    )

    def search(question: str, k: int = 4):
        return vector_db.similarity_search_with_score(question, k=k)

    return search


def run_queries(queries, *, wire_vector: bool, search):
    """Run the 69-query benchmark. Returns (rows, latency_samples)."""
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
            rows.append({
                "query_id": exp.query_id, "slice": q["slice"],
                "difficulty": q["difficulty"], "expected": exp.expected_decision.value,
                "decision": "RETRIEVAL_ERROR", "reason": f"RETRIEVAL_ERROR:{exc}",
            })
            continue

        raw_dists = [float(s) for _, s in scored_docs]
        raw_top1 = raw_dists[0] if raw_dists else None

        candidates = []
        for doc, score in scored_docs:
            cand = RetrievalCandidate(document=doc, retrieval_source="chroma")
            cand.fusion_score = float(-score)
            if wire_vector:
                cand.vector_score = float(score)
            candidates.append(cand)

        rr = RetrievalResult(candidates=candidates, retrieval_mode="vector_only_v369")
        ev = analyze_retrieval_evidence(
            query_text, rr, [d for d, _s in scored_docs],
            retrieval_mode="vector_only_v369", identity_matching=True,
        )
        latencies.append(time.time() - t0)

        rec = make_record(ev)
        result = evaluate_contract_native(exp, rec)
        passing = _accept_as_pass(result.verdict)

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
            "exact_identifier_match": ev.exact_identifier_match,
            "vector_distance": ev.vector_distance,
            "lexical_score": ev.lexical_score,
            "contract_has_critical": bool(contract.get("has_critical_requirements", False)),
            "contract_sufficient": bool(contract.get("sufficient", False)),
            "contract_missing": list(contract.get("missing", [])),
            "raw_top1_distance": round(raw_top1, 4) if raw_top1 is not None else None,
            "raw_distances": [round(d, 4) for d in raw_dists],
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
    acc = round(correct / max(n, 1), 4)
    return {"label": label, "n": n, "correct": correct, "fa": fa, "fr": fr, "accuracy": acc}


def main() -> None:
    import importlib
    v369 = importlib.import_module("v369_real_baseline")
    queries = v369.generate_queries()
    print(f"Queries: {len(queries)}")
    policy = default_policy()
    print(f"max_vector_distance = {policy.max_vector_distance!r}")

    search = _load_search()

    print("\n== Replaying BASELINE (no vector_score wired) ==")
    base_rows, base_lat = run_queries(queries, wire_vector=False, search=search)
    print("== Replaying WIRED diagnostic (vector_score=raw distance) ==")
    wired_rows, _ = run_queries(queries, wire_vector=True, search=search)

    out_dir = _REPO_ROOT / "results" / "v375_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"baseline": base_rows, "wired": wired_rows},
              open(out_dir / "full_trace.json", "w"), indent=1, default=str)

    print("\n== BASELINE SUMMARY ==")
    bs = summarize(base_rows, "baseline")
    for k, v in bs.items():
        print(f"  {k}: {v}")

    fr_rows = [r for r in base_rows if not r["passing"] and r["expected"] == "ANSWER" and r["decision"] == "ABSTAIN"]
    print(f"\n== FR = {len(fr_rows)}: blocking mechanism ==")
    mech = Counter(r["blocking_mechanism"] for r in fr_rows)
    for k, v in sorted(mech.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} n={v}")

    print("\n== FR reason breakdown ==")
    rc = Counter(r["reason"] for r in fr_rows)
    for k, v in sorted(rc.items(), key=lambda x: -x[1]):
        print(f"  {k:35s} n={v}")

    print("\n== FR identity_relation breakdown ==")
    ic = Counter(r["identity_relation"] for r in fr_rows)
    for k, v in sorted(ic.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} n={v}")

    cb = [r for r in fr_rows if r["coverage_bound"]]
    print(f"\nCOVERAGE_BOUND_FR_COUNT = {len(cb)}")

    # Dominant predicate within coverage-bound FR.
    dom = Counter(r["blocking_mechanism"] for r in cb)
    print("Coverage-bound FR mechanisms:", dict(dom))

    # Runtime vector_distance distribution across FR (should be all None).
    vd = Counter("None" if r["vector_distance"] is None else "set" for r in fr_rows)
    print("FR runtime vector_distance:", dict(vd))

    # Raw distance distribution (characterization only -- NOT fed to runtime).
    print("\n== RAW top1 distance distribution (NOT fed to runtime) ==")
    def dist_stats(rows, label):
        ds = [r["raw_top1_distance"] for r in rows if r["raw_top1_distance"] is not None]
        if not ds:
            print(f"  {label}: n=0")
            return
        ds.sort()
        print(f"  {label}: n={len(ds)} min={ds[0]:.2f} p10={ds[len(ds)//10]:.2f} "
              f"median={ds[len(ds)//2]:.2f} p90={ds[9*len(ds)//10]:.2f} max={ds[-1]:.2f}")

    correct_answer = [r for r in base_rows if r["passing"] and r["expected"] == "ANSWER"]
    fr = fr_rows
    hn = [r for r in base_rows if r["slice"] == "HARD_NEGATIVE"]
    ood = [r for r in base_rows if r["slice"] == "OOD"]
    dist_stats(correct_answer, "CORRECT_ANSWER")
    dist_stats(fr, "FR (ANSWER->ABSTAIN)")
    dist_stats(hn, "HARD_NEGATIVE")
    dist_stats(ood, "OOD")

    # WIRED diagnostic: what would the threshold do if wired.
    print("\n== WIRED diagnostic: baseline vs wired ==")
    ws = summarize(wired_rows, "wired")
    for k, v in ws.items():
        print(f"  {k}: {v}")
    wfr = [r for r in wired_rows if not r["passing"] and r["expected"] == "ANSWER" and r["decision"] == "ABSTAIN"]
    wfa = [r for r in wired_rows if not r["passing"] and r["expected"] == "ABSTAIN" and r["decision"] == "ANSWER"]
    print(f"  wired FR={len(wfr)} (baseline FR={bs['fr']})  wired FA={len(wfa)} (baseline FA={bs['fa']})")
    wmech = Counter(r["blocking_mechanism"] for r in wfr)
    print(f"  wired FR mechanisms: {dict(wmech)}")

    # Latency.
    if base_lat:
        base_lat.sort()
        print(f"\nLatency: median={base_lat[len(base_lat)//2]:.4f}s p95={base_lat[int(len(base_lat)*0.95)]:.4f}s")

    print(f"\nsaved full trace: {out_dir / 'full_trace.json'}")


if __name__ == "__main__":
    main()
