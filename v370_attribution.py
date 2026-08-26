"""V3.70 Responsibility Reassessment: replay + full error attribution."""
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

from backend.retrieval.evidence import analyze_retrieval_evidence
from backend.evaluation.contract_eval_v367 import (
    EvidenceEvaluationRecord,
    ExpectedDecision,
    _accept_as_pass,
    evaluate_contract_native,
)
import rag_core
# Point rag_core at existing V3.69 index.
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")
from rag_core import retrieve_docs
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult

# Override persist dir to use existing index.
rag_core_PERSIST = _REPO_ROOT / "vector_db_v369"


def make_record(evidence) -> EvidenceEvaluationRecord:
    return EvidenceEvaluationRecord(
        decision=evidence.decision,
        reason=evidence.reason,
        query_identity=dict(evidence.query_identity),
        candidate_identity=dict(evidence.candidate_identity),
        identity_relation=evidence.identity_relation,
        contract_requirements_covered=evidence.contract.get("requirements_covered", False),
        lexical_score=evidence.lexical_score,
        vector_distance=evidence.vector_distance,
        metadata_consistency=evidence.metadata_consistency,
        covered_claim_keys=frozenset(),
    )


def main() -> None:
    # Import query generation from v369.
    sys.path.insert(0, str(_REPO_ROOT))
    import importlib
    v369 = importlib.import_module("v369_real_baseline")
    queries = v369.generate_queries()
    print(f"Queries: {len(queries)}")

    # Run retrieval + evidence.
    results = []
    for q in queries:
        qt = q["query"]
        exp = q["expectation"]
        try:
            scored_docs = retrieve_docs(qt, k=4)
        except Exception as e:
            print(f"RETRIEVAL_ERROR {exp.query_id}: {e}")
            continue
        cands = [RetrievalCandidate(document=d, retrieval_source="chroma") for d, _s in scored_docs]
        rr = RetrievalResult(candidates=cands, retrieval_mode="vector_only_v369")
        ev = analyze_retrieval_evidence(
            qt, rr, [d for d, _s in scored_docs],
            retrieval_mode="vector_only_v369", identity_matching=True,
        )
        rec = make_record(ev)

        result = evaluate_contract_native(exp, rec)
        passing = _accept_as_pass(result.verdict)

        results.append({
            "query_id": exp.query_id,
            "query_text": qt[:60],
            "slice": q["slice"],
            "difficulty": q["difficulty"],
            "expected_decision": exp.expected_decision.value,
            "runtime_decision": rec.decision,
            "runtime_reason": rec.reason,
            "identity_relation": rec.identity_relation,
            "contract_requirements_covered": rec.contract_requirements_covered,
            "verdict": result.verdict.value,
            "passing": passing,
            "eval_reasons": list(result.reason_codes),
            "candidate_count": len(cands),
            "top_doc_preview": (scored_docs[0][0].page_content[:80] + "...") if scored_docs else "",
        })

    print(f"\nTotal executed: {len(results)}")

    # Aggregate reconciliation.
    total_pass = sum(1 for r in results if r["passing"])
    total_fail = len(results) - total_pass
    fa = sum(1 for r in results if not r["passing"] and r["expected_decision"] == "ANSWER" and r["runtime_decision"] == "ABSTAIN")
    fr = sum(1 for r in results if not r["passing"] and r["expected_decision"] == "ABSTAIN" and r["runtime_decision"] == "ANSWER")
    fa_wrong_claim = sum(1 for r in results if not r["passing"]
                         and r["expected_decision"] == "ANSWER"
                         and r["runtime_decision"] == "ANSWER"
                         and r["verdict"] != "CORRECT")
    fr_wrong_reject = sum(1 for r in results if not r["passing"]
                          and r["expected_decision"] == "ABSTAIN"
                          and r["runtime_decision"] == "ABSTAIN"
                          and False)  # abstain-abstain always passes

    print(f"\n== RECONCILIATION ==")
    print(f"Pass: {total_pass} | Fail: {total_fail}")
    print(f"FA (gold=ANSWER → runtime=ABSTAIN): {fa}")
    print(f"FR (gold=ABSTAIN → runtime=ANSWER): {fr}")
    print(f"FA_WRONG_CLAIM (both ANSWER but not CORRECT): {fa_wrong_claim}")
    print(f"FR_WRONG_REJECT (both ABSTAIN): {fr_wrong_reject}")
    print(f"Total errors: {fa + fr + fa_wrong_claim}")

    # Slice breakdown.
    slice_metrics: dict[str, Counter] = {}
    for r in results:
        sm = slice_metrics.setdefault(r["slice"], Counter())
        sm["n"] += 1
        sm["pass"] += int(r["passing"])
        if not r["passing"]:
            if r["expected_decision"] == "ANSWER":
                sm["fa_or_fr_from_answer"] += 1
                if r["runtime_decision"] == "ABSTAIN":
                    sm["false_refusal"] += 1
                else:
                    sm["wrong_claim_fa"] += 1
            else:
                sm["false_answer_on_abstain"] += 1

    print("\n== SLICE METRICS ==")
    for sl in sorted(slice_metrics):
        c = slice_metrics[sl]
        acc = round(c.get("pass", 0) / max(c["n"], 1), 3)
        print(f"  {sl}: n={c['n']} pass={c.get('pass',0)} acc={acc} "
              f"FR={c.get('false_refusal',0)} wrong_claim_FA={c.get('wrong_claim_FA',0)} "
              f"FA_abstain={c.get('false_answer_on_abstain',0)}")

    # Reason × verdict cross-tab.
    print("\n== REASON × VERDICT CROSS-TAB ==")
    cross_tab: dict[str, Counter] = {}
    for r in results:
        key = r["runtime_reason"].split("_")[0] if r["runtime_reason"] else "?"
        ct = cross_tab.setdefault(key, Counter())
        ct[r["verdict"]] += 1
    for reason in sorted(cross_tab):
        print(f"  {reason}: {dict(cross_tab[reason])}")

    # Contract requirements coverage distribution.
    crc_dist = Counter(r["contract_requirements_covered"] for r in results)
    print(f"\ncontract_requirements_covered distribution: {dict(crc_dist)}")

    # Detailed analysis of failures.
    print("\n== FAILURE DETAIL (first 15) ==")
    for r in [x for x in results if not x["passing"]][:15]:
        print(f"\n{r['query_id']} [{r['slice']}] expected={r['expected_decision']} "
              f"runtime={r['runtime_decision']} ({r['runtime_reason']}) "
              f"→ verdict={r['verdict']}")
        print(f"  reasons: {r['eval_reasons']}")
        print(f"  contract_covered={r['contract_requirements_covered']} "
              f"cands={r['candidate_count']}")
        print(f"  top_doc: {r['top_doc_preview']}")

    # Save full detail.
    out_dir = _REPO_ROOT / "results" / "v370_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / "full_results.json", "w"), indent=1, default=str)
    print(f"\nsaved: {out_dir / 'full_results.json'}")


if __name__ == "__main__":
    main()
