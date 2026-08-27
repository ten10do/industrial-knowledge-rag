"""V3.80-C Shadow replay: upgraded extraction x frozen support-v316.1.

Replays all 69 aligned cases, evaluating the pre-registered merged governance:
new generic extraction families decide UNLESS coarse guards would downgrade an
existing frozen SUPPORTED; everything else keeps the byte-frozen v316.1 path.

Includes:
 - private prediction-blind REQUIREMENT-EXTRACTION gold (typed independently
   from query texts at authoring time; no runtime label was consulted),
 - 3x determinism digests,
 - the ORIGINAL V3.79 gates recomputed unchanged,
 - protected-hash drift abort.
Outputs private: results/v380c_extraction/replay_results.json
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
from backend.evaluation.score_lineage import build_retrieval_result  # noqa: E402
from backend.retrieval import evidence as ev  # noqa: E402
from backend.retrieval.claim_support_shadow import ClaimSupportState, counterfactual_transitions  # noqa: E402
from backend.retrieval.requirement_extraction_shadow import (  # noqa: E402
    EXTRACTION_SHADOW_VERSION,
    ExtractionUpgradedSupportEvaluator,
    classify_query,
)
from backend.retrieval.technical import normalize_technical_text  # noqa: E402

ALIGN_DIR = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment"
OUT_DIR = _REPO_ROOT / "results" / "v380c_extraction"

PROTECTED_FILES = (
    "backend/retrieval/evidence.py",
    "backend/retrieval/evidence_support.py",
    "backend/retrieval/evidence_contract.py",
    "backend/retrieval/product_identity.py",
    "backend/retrieval/semantic_judge_localnli.py",
    "backend/retrieval/requirement_extraction_shadow.py",
    "rag_core.py",
)

CONTRACT_FA_EXPECTED = {"V369-Q0012", "V369-Q0016", "V369-Q0017", "V369-Q0019",
                        "V369-Q0020", "V369-Q0033", "V369-Q0043"}
THRESHOLD = 13.234710693359375
V379_BASELINE = {"fa_blocked_strict": 8, "correct_killed_strict": 7, "precision_case": 0.2778}

# --- PRIVATE prediction-blind requirement-extraction gold ----------------------
# Typed at authoring time from query texts alone (taxonomy section 9); runtime
# outcomes were NOT consulted for these labels.
GOLD_EXTRACTION = {
    # attribute-value forms (VALUE slice ambiguity core)
    "V369-Q0012": {"family": "ATTRIBUTE_VALUE", "subject": "output frequency", "key": "frequency", "legit": False},
    "V369-Q0014": {"family": "ATTRIBUTE_VALUE", "subject": "protection rating", "key": "protection", "legit": False},
    "V369-Q0015": {"family": "ATTRIBUTE_VALUE", "subject": "control method", "key": "control_method", "legit": False},
    "V369-Q0016": {"family": "ATTRIBUTE_VALUE", "subject": "overload capacity", "key": "overload", "legit": False},
    "V369-Q0018": {"family": "ATTRIBUTE_VALUE", "subject": "power rating", "key": "power", "legit": False},
    "V369-Q0019": {"family": "ATTRIBUTE_VALUE", "subject": "efficiency class", "key": "efficiency", "legit": False},
    "V369-Q0020": {"family": "ATTRIBUTE_VALUE", "subject": "noise level", "key": "noise", "legit": False},
    # definitional tell-me-about forms (non-corpus devices)
    "V369-Q0021": {"family": "DEFINITION", "subject": "sinamics g120 drive", "legit": False},
    "V369-Q0027": {"family": "DEFINITION", "subject": "sinamics g120 drive", "legit": False},
    "V369-Q0024": {"family": "DEFINITION", "subject": "powerflex 520 drive", "legit": False},
    "V369-Q0030": {"family": "DEFINITION", "subject": "powerflex 520 drive", "legit": False},
    # generic / world entity -> legitimately unextractable
    "V369-Q0033": {"family": "WORLD_ENTITY", "subject": "the capital of france", "legit": True},
    "V369-Q0034": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    "V369-Q0035": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    "V369-Q0036": {"family": "WORLD_ENTITY", "subject": "quantum entanglement", "legit": True},
    "V369-Q0037": {"family": "WORLD_ENTITY", "subject": "photosynthesis", "legit": True},
    "V369-Q0038": {"family": "WORLD_ENTITY", "subject": "stock price of apple", "legit": True},
    "V369-Q0039": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    "V369-Q0041": {"family": "WORLD_ENTITY", "subject": "the french revolution", "legit": True},
    "V369-Q0042": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    "V369-Q0043": {"family": "WORLD_ENTITY", "subject": "the speed of light", "legit": True},
    "V369-Q0044": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    "V369-Q0045": {"family": "WORLD_ENTITY", "subject": "", "legit": True},
    # concept definitions / explanations / comparisons (industrial domain)
    "V369-Q0048": {"family": "DEFINITION", "subject": "variable frequency drive", "legit": False},
    "V369-Q0049": {"family": "OPEN_EXPLANATION", "subject": "pulse-width modulation", "legit": True},
    "V369-Q0051": {"family": "PURPOSE", "subject": "overload relay", "legit": False},
    "V369-Q0053": {"family": "DEFINITION", "subject": "sourcing and sinking digital inputs", "legit": False},
    "V369-Q0055": {"family": "PURPOSE", "subject": "emergency stop circuit", "legit": False},
    "V369-Q0057": {"family": "COMPARISON", "subject": "ac vs dc motors", "legit": True},
    "V369-Q0058": {"family": "DEFINITION", "subject": "three-phase power", "legit": False},
    "V369-Q0059": {"family": "OPEN_EXPLANATION", "subject": "regenerative braking", "legit": True},
    "V369-Q0069": {"family": "POLAR_COMPARISON", "subject": "", "legit": True},
    # positive controls: previously SUCCESSFUL baseline extractions must stay BASELINE
    "V369-Q0001": {"family": "BASELINE", "subject": "acceleration time setting", "legit": False},
    "V369-Q0007": {"family": "BASELINE", "subject": "", "legit": False},
    "V369-Q0008": {"family": "BASELINE", "subject": "", "legit": False},
    "V369-Q0010": {"family": "BASELINE", "subject": "", "legit": False},
    "V369-Q0011": {"family": "BASELINE", "subject": "", "legit": False},
    "V369-Q0013": {"family": "BASELINE", "subject": "", "legit": False},
}

CRITICAL_13 = ["V369-Q0012", "V369-Q0016", "V369-Q0019", "V369-Q0020", "V369-Q0033", "V369-Q0043",
               "V369-Q0014", "V369-Q0015", "V369-Q0018", "V369-Q0048", "V369-Q0051",
               "V369-Q0053", "V369-Q0055"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_hit(anchors, chunk_texts):
    prepared = [(" ".join(t.casefold().split()), normalize_technical_text(str(t))) for t in chunk_texts]
    for anchor in anchors:
        quote = " ".join(anchor.get("quote", "").casefold().split())
        pattern = anchor.get("regex") or ""
        for folded, tech in prepared:
            if quote and quote in folded:
                return True
            if pattern:
                try:
                    if __import__("re").search(pattern, folded, __import__("re").IGNORECASE):
                        return True
                except __import__("re").error:
                    pass
            needle = normalize_technical_text(anchor.get("quote", ""))
            if needle and needle in tech:
                return True
    return False


def gold_state(case, chunk_texts):
    claims, anchors = case.get("claims") or [], case.get("anchors") or []
    if not claims:
        return ClaimSupportState.UNSUPPORTED.value
    if anchors and anchor_hit(anchors, chunk_texts):
        return ClaimSupportState.SUPPORTED.value
    return ClaimSupportState.AMBIGUOUS.value


def run_once(cases, search_db):
    evaluator = ExtractionUpgradedSupportEvaluator()
    rows = []
    started_all = time.perf_counter()
    for case in cases:
        qid, text = case["query_id"], case["query_text"]
        scored_docs = search_db.similarity_search_with_score(text, k=4)
        rr = build_retrieval_result(scored_docs)
        documents = [d for d, _s in scored_docs]
        chunk_texts = [str(d.page_content) for d in documents]

        real_ev = ev.analyze_retrieval_evidence(
            text, rr, documents, retrieval_mode="vector_only_v369", identity_matching=True)
        expected = case["expected_decision"]
        support_slice = "OOD" if case["query_domain"] == "GENERIC_OUT_OF_DOMAIN" else (
            "HARD_NEGATIVE" if case["slice_labels"][0] == "HARD_NEGATIVE"
            else "CORPUS_UNSUPPORTED" if case["support_state"] == "CORPUS_UNSUPPORTED"
            else "ANSWERABLE")
        answered = real_ev.decision == "ANSWER"
        cohort = ("FA" if (expected == "ABSTAIN" and answered)
                  else "POSITIVE_ANSWERED" if (expected == "ANSWER" and answered)
                  else "SAFE_ABSTAIN" if expected == "ABSTAIN" else "FALSE_REFUSAL")

        req = classify_query(text)
        merged, _ = evaluator.evaluate(text, rr, documents)
        dist_top1 = float(scored_docs[0][1]) if scored_docs else None
        rows.append({
            "query_id": qid,
            "support_slice": support_slice,
            "cohort": cohort,
            "runtime_answered": answered,
            "correct_answer": cohort == "POSITIVE_ANSWERED",
            "gold_support": gold_state(case, chunk_texts),
            "vector_distance_top1": round(dist_top1, 4) if dist_top1 is not None else None,
            "family_predicted": req.as_dict(),
            "merged_state": merged.state,
            "merged_reason": merged.support_reason,
            "merged_source": merged.support_source,
            "admissible_e1": merged.state == ClaimSupportState.SUPPORTED.value,
        })
    wall = time.perf_counter() - started_all
    counters = {
        "shadow_family_invocations": evaluator.shadow_family_invocations,
        "baseline_fallbacks": evaluator.baseline_fallbacks,
    }
    lat = sorted(evaluator.latency_ms)
    latency = {
        "median_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))], 3),
    }
    return rows, wall, counters, latency


def canonical_digest(rows):
    blob = json.dumps(
        [[r["query_id"], r["merged_state"], r["merged_reason"], r["merged_source"],
          r["family_predicted"]] for r in rows],
        sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def extraction_metrics(rows):
    stats = {"critical_newly_structured": [], "critical_correct": [],
             "critical_false_extraction": [], "legit_honored": 0, "legit_violated": []}
    for r in rows:
        qid = r["query_id"]
        gold = GOLD_EXTRACTION.get(qid)
        if not gold:
            continue
        fam = r["family_predicted"]["family"]
        if gold["legit"]:
            routed_ok = (
                r["family_predicted"].get("legitimately_unextractable")
                or r["merged_source"].startswith("struct:")
            )
            if fam in {"WORLD_ENTITY"} or r["family_predicted"].get("legitimately_unextractable"):
                stats["legit_honored"] += 1
            elif routed_ok and r["merged_state"] != "SUPPORTED":
                stats["legit_honored"] += 1  # safe unextractable via baseline fallback
            else:
                stats["legit_violated"].append(qid)
        if qid in CRITICAL_13:
            if fam in {"DEFINITION", "PURPOSE", "ATTRIBUTE_VALUE"}:
                stats["critical_newly_structured"].append(qid)
                subject_pred = r["family_predicted"].get("subject", "")
                subject_ok = (not gold["subject"] or gold["subject"] in subject_pred
                              or subject_pred in gold["subject"])
                if fam == gold["family"] and subject_ok:
                    stats["critical_correct"].append(qid)
            elif fam != "BASELINE" and not gold["legit"]:
                stats["critical_false_extraction"].append(qid)
            elif gold["family"] == "BASELINE" and fam != "BASELINE":
                stats["critical_false_extraction"].append(qid)
    return stats


def main():
    hashes_start = {p: sha256(_REPO_ROOT / p) for p in PROTECTED_FILES}
    payload = json.load(open(ALIGN_DIR / "aligned_benchmark_v2.json", encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) == 69

    search_db = Chroma(persist_directory=rag_core.PERSIST_DIR,
                       embedding_function=rag_core.get_embedding_model())
    print(f"candidate={EXTRACTION_SHADOW_VERSION}")

    runs = []
    for index in range(3):
        rows, wall, counters, latency = run_once(cases, search_db)
        runs.append({"rows": rows, "wall_s": round(wall, 1), "digest": canonical_digest(rows)})
        print(f"run{index + 1}: wall={wall:.1f}s digest={runs[-1]['digest']}")
    if len({r['digest'] for r in runs}) != 1:
        print("DETERMINISM_FAILURE")
        raise SystemExit(1)

    rows = runs[0]["rows"]
    cohort_counts = Counter(r["cohort"] for r in rows)
    if dict(cohort_counts) != {"FA": 9, "POSITIVE_ANSWERED": 23, "SAFE_ABSTAIN": 34, "FALSE_REFUSAL": 3}:
        print("RUNTIME_COHORT_MISMATCH:", cohort_counts)
        raise SystemExit(1)

    contract_fas = [r for r in rows if r["cohort"] == "FA" and r["query_id"] in CONTRACT_FA_EXPECTED]
    other_fas = [r for r in rows if r["cohort"] == "FA" and r["query_id"] not in CONTRACT_FA_EXPECTED]
    positives = [r for r in rows if r["cohort"] == "POSITIVE_ANSWERED"]
    positives_le_thr = [r for r in positives if (r["vector_distance_top1"] or 99) <= THRESHOLD]
    fr_rows = [r for r in rows if r["cohort"] == "FALSE_REFUSAL"]
    hn = [r for r in rows if r["support_slice"] == "HARD_NEGATIVE"]

    print("\n== CRITICAL 13 CASES ==")
    for qid in CRITICAL_13:
        r = next(x for x in rows if x["query_id"] == qid)
        print(f"  {qid} [{r['cohort'][:3]}|{r['support_slice'][:4]}] "
              f"fam={r['family_predicted']['family']:15s} merged={r['merged_state']:12s} "
              f"{r['merged_reason'][:28]}")
    m = extraction_metrics(rows)
    print(f"  newly_structured={len(m['critical_newly_structured'])}/13 "
          f"correct={len(m['critical_correct'])}/13 "
          f"false_extraction={len(m['critical_false_extraction'])}")
    print(f"  legit-honored={m['legit_honored']} violated={m['legit_violated']}")

    print("\n== V3.79 GATES RECOMPUTED (unchanged thresholds) ==")
    a_block = sum(1 for r in contract_fas if not r["admissible_e1"])
    b_kill_all = sum(1 for r in positives if not r["admissible_e1"])
    b_kill_strict = sum(1 for r in positives_le_thr if not r["admissible_e1"])
    c_block_total = sum(1 for r in rows if r["cohort"] == "FA" and not r["admissible_e1"])
    fr_new = sum(1 for r in fr_rows if not r["admissible_e1"]) - 3  # were already rejected in V3.79
    hn_supported = sum(1 for r in hn if r["admissible_e1"])
    tp = sum(1 for r in rows if r["admissible_e1"] and r["gold_support"] == "SUPPORTED")
    fp = sum(1 for r in rows if r["admissible_e1"] and r["gold_support"] != "SUPPORTED")
    precision_case = round(tp / (tp + fp), 4) if (tp + fp) else None
    print(f"  A contract-FA blocked      : {a_block}/7   (need >=5)")
    print(f"  B correct killed           : all={b_kill_all}/23 strict10={b_kill_strict}/10 (need <=2)")
    print(f"  C overall FA blocked       : {c_block_total}/9   (need >=5)")
    print(f"  D new FR potential         : {max(fr_new, 0)}     (need <=2)")
    print(f"  E hard-negative supported  : {hn_supported}/10  (need 0)")
    print(f"  I case-level precision     : {precision_case}  (V3.79 was {V379_BASELINE['precision_case']})")

    print("\n== PER-COHORT merged STATE DISTRIBUTION ==")
    for name, pop in (("FA", [r for r in rows if r["cohort"] == "FA"]),
                      ("POS", positives), ("SAFE", [r for r in rows if r["cohort"] == "SAFE_ABSTAIN"]),
                      ("FR", fr_rows), ("HN", hn)):
        states = Counter(r["merged_state"] for r in pop)
        print(f"  {name:5s} n={len(pop):2d} {dict(states)}")

    cf_rows = [{**r, "admissible": r["admissible_e1"]} for r in rows]
    transitions = counterfactual_transitions(cf_rows, policy_name="E1_merged_strict")
    print(f"\n  transitions: SAFE_BLOCK={len(transitions['SAFE_BLOCK'])} "
          f"FALSE_REFUSAL_REGRESSION={transitions['FALSE_REFUSAL_REGRESSION']} "
          f"UNSAFE_CHANGE={transitions['UNSAFE_CHANGE']}")

    print("\n== KILLED POSITIVES DETAIL (if any) ==")
    for r in positives:
        if not r["admissible_e1"]:
            print(f"  {r['query_id']} reason={r['merged_reason']}")

    hashes_end = {p: sha256(_REPO_ROOT / p) for p in PROTECTED_FILES}
    drift = [p for p in PROTECTED_FILES if hashes_start[p] != hashes_end[p]]
    if drift:
        print("PROTECTED_HASH_DRIFT:", drift)
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({
        "candidate": EXTRACTION_SHADOW_VERSION,
        "digest": runs[0]["digest"],
        "rows": rows,
        "extraction_metrics": m,
        "gates": {"A_contract_fa_block": a_block, "B_kill_all": b_kill_all,
                  "B_kill_strict10": b_kill_strict, "C_overall_block": c_block_total,
                  "D_new_fr": max(fr_new, 0), "E_hn_supported": hn_supported,
                  "I_precision_case": precision_case},
        "counters": counters,
        "latency_evaluator": latency,
        "protected_hashes": hashes_end,
    }, open(OUT_DIR / "replay_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nsaved:", OUT_DIR / "replay_results.json")
    print("PROTECTED_HASHES_OK")


if __name__ == "__main__":
    main()
