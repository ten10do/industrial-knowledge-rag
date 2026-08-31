"""V3.85 Production Readiness Analysis — offline policy simulation.

Scope and honesty boundary
--------------------------
This script performs OFFLINE analysis over two frozen artifacts:

  results/v377_aligned_baseline/run1_full_trace.json
  backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json

It does NOT execute the runtime, does NOT touch the vector index, and does
NOT modify any decision code. Every number it prints is a policy simulation
over recorded decisions. A simulated decision flip is an APPROXIMATION of what
the real evaluator would score: the real contract evaluator also inspects
claim-level support (ExpectedClaim / _accept_as_pass). Any proposed gate must
be confirmed by a live ``v377_aligned_baseline.py`` re-run producing a
REAL_EVIDENCE_RUNTIME trace with digest comparison before it can be reported
as a PASS.

Scoring convention (matches the published V3.77 convention):
  CORRECT, ABSTAIN_CORRECT  -> correct
  INCORRECT + ANSWER        -> false answer  (FA)  [safety hazard]
  INCORRECT + ABSTAIN       -> false refusal (FR)  [usability cost]
  PARTIAL_INCOMPLETE        -> not-correct; counted as 1.0 FR-equivalent in
                               the risk metric and reported separately

Risk metric is auditable and explicitly weighted:
  risk = W_FALSE_ANSWER * FA + W_FALSE_REFUSAL * FR + W_PARTIAL * PARTIAL
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "results" / "v377_aligned_baseline" / "run1_full_trace.json"
BENCH = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "aligned_benchmark_v2.json"

W_FALSE_ANSWER = 10
W_FALSE_REFUSAL = 1
W_PARTIAL = 1

# Q0021..Q0026 are byte-identical to Q0027..Q0032 (verified over query_text).
DUPLICATE_IDS = {
    "V369-Q0027", "V369-Q0028", "V369-Q0029",
    "V369-Q0030", "V369-Q0031", "V369-Q0032",
}

TAU_GRID = [i / 100 for i in range(80, 300, 5)]


def load():
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    texts = {c["query_id"]: c["query_text"] for c in bench["cases"]}
    return trace, texts


def ratio_of(record):
    return record["raw_top1_distances"] / record["threshold"]


def apply_distance_ceiling(records, tau):
    """Abstain when the contract branch answers above a relative distance ceiling.

    Target is the contract ANSWER branch at backend/retrieval/evidence.py:456,
    which answers whenever the contract reports all critical requirements
    covered, with no retrieval-distance check.
    """
    out = []
    for r in records:
        r = dict(r)
        r["flipped"] = False
        if r["contract_has_critical"] and r["contract_sufficient"] and ratio_of(r) > tau:
            r["decision"] = "ABSTAIN"
            r["verdict"] = "ABSTAIN_CORRECT" if r["expected_decision"] == "ABSTAIN" else "INCORRECT"
            r["flipped"] = True
        out.append(r)
    return out


def metrics(records):
    correct = fa = fr = partial = 0
    for r in records:
        v = r["verdict"]
        if v in ("CORRECT", "ABSTAIN_CORRECT"):
            correct += 1
        elif v == "PARTIAL_INCOMPLETE":
            partial += 1
        elif r["decision"] == "ANSWER":
            fa += 1
        else:
            fr += 1
    n = len(records)
    risk = W_FALSE_ANSWER * fa + W_FALSE_REFUSAL * fr + W_PARTIAL * partial
    return {
        "n": n, "correct": correct, "fa": fa, "fr": fr, "partial": partial,
        "accuracy": correct / n, "risk": risk,
    }


def main():
    trace, texts = load()

    print("=" * 78)
    print("V3.85 PRODUCTION READINESS ANALYSIS (offline simulation)")
    print("=" * 78)

    # ---- 1. duplicate audit -------------------------------------------------
    by_text = {}
    for qid, text in texts.items():
        by_text.setdefault(text.strip().lower(), []).append(qid)
    dup_groups = {t: ids for t, ids in by_text.items() if len(ids) > 1}
    print("\n[1] DUPLICATE AUDIT")
    print(f"    benchmark cases      : {len(texts)}")
    print(f"    duplicate groups     : {len(dup_groups)}")
    for t, ids in sorted(dup_groups.items()):
        print(f"      x{len(ids)}  {ids}  {t!r}")
    print(f"    redundant cases      : {sum(len(v) - 1 for v in dup_groups.values())}"
          f"  ({sum(len(v) - 1 for v in dup_groups.values()) / len(texts) * 100:.1f}% of benchmark)")

    dedup = [r for r in trace if r["query_id"] not in DUPLICATE_IDS]

    # ---- 2. published vs dedup baseline ------------------------------------
    print("\n[2] BASELINE: PUBLISHED vs DEDUPLICATED")
    for label, recs in (("published 69", trace), ("deduplicated 63", dedup)):
        m = metrics(recs)
        print(f"    {label:<18} correct={m['correct']:>3} FA={m['fa']} FR={m['fr']} "
              f"PARTIAL={m['partial']}  acc={m['accuracy']:.4f}  risk={m['risk']}")

    # ---- 3. slice decomposition -------------------------------------------
    print("\n[3] SLICE DECOMPOSITION (deduplicated)")
    print(f"    {'slice':<20}{'n':>4}{'correct':>9}{'FA':>5}{'FR':>5}{'PART':>6}{'acc':>9}{'FA rate':>9}")
    for sl in ("ANSWERABLE", "CORPUS_UNSUPPORTED", "HARD_NEGATIVE", "OOD"):
        sub = [r for r in dedup if r["support_slice"] == sl]
        if not sub:
            continue
        m = metrics(sub)
        print(f"    {sl:<20}{m['n']:>4}{m['correct']:>9}{m['fa']:>5}{m['fr']:>5}{m['partial']:>6}"
              f"{m['accuracy']:>9.4f}{m['fa'] / m['n']:>9.4f}")

    # ---- 4. error attribution ---------------------------------------------
    print("\n[4] ATTRIBUTION OF THE 12 PUBLISHED ERRORS")
    reasons = Counter()
    for r in trace:
        if r["verdict"] in ("INCORRECT", "PARTIAL_INCOMPLETE"):
            reasons[r["reason"]] += 1
    for reason, n in reasons.most_common():
        print(f"    {reason:<34} {n}")

    # ---- 5. distance discriminative power ---------------------------------
    print("\n[5] DISCRIMINATIVE POWER OF THE GLOBAL DISTANCE THRESHOLD")
    thr = sorted({round(r["threshold"], 4) for r in trace})
    print(f"    distinct thresholds in benchmark : {len(thr)}  value={thr}")
    ratios = {v: [ratio_of(r) for r in dedup if r["verdict"] == v] for v in
              ("CORRECT", "ABSTAIN_CORRECT", "PARTIAL_INCOMPLETE", "INCORRECT")}
    for v, xs in ratios.items():
        if xs:
            print(f"    {v:<20} n={len(xs):>3}  ratio min={min(xs):.2f} max={max(xs):.2f}")

    # ---- 6. rejected intervention -----------------------------------------
    print("\n[6] REJECTED INTERVENTION: blanket abstention on unanchored value queries")
    import re
    unanchored = re.compile(r"^what is the .+\??$", re.I)
    hits = [r for r in dedup if unanchored.match(texts[r["query_id"]].strip())]
    gain = sum(1 for r in hits if r["verdict"] == "INCORRECT" and r["decision"] == "ANSWER")
    loss = sum(1 for r in hits if r["verdict"] in ("CORRECT", "ABSTAIN_CORRECT") and r["decision"] == "ANSWER")
    m0 = metrics(dedup)
    print(f"    matched unanchored value queries : {len(hits)}")
    print(f"    false answers removed            : {gain}")
    print(f"    correct answers destroyed        : {loss}")
    print(f"    net accuracy                     : {m0['accuracy']:.4f} -> "
          f"{(m0['correct'] + gain - loss) / m0['n']:.4f}   VERDICT: REJECTED")

    # ---- 7. distance ceiling sweep ----------------------------------------
    print("\n[7] DISTANCE CEILING SWEEP ON CONTRACT BRANCH (deduplicated)")
    print(f"    {'tau':>6}{'correct':>9}{'FA':>5}{'FR':>5}{'PART':>6}{'acc':>10}{'risk':>7}")
    for tau in TAU_GRID:
        m = metrics(apply_distance_ceiling(dedup, tau))
        if 1.10 <= tau <= 2.00 or tau in (TAU_GRID[0], TAU_GRID[-1]):
            print(f"    {tau:>6.2f}{m['correct']:>9}{m['fa']:>5}{m['fr']:>5}{m['partial']:>6}"
                  f"{m['accuracy']:>10.4f}{m['risk']:>7}")

    # ---- 8. split-half validation -----------------------------------------
    print("\n[8] SPLIT-HALF VALIDATION (tau tuned on calibration half only)")
    random.seed(7)
    trials = 300
    tuned_acc, tuned_risk, chosen = [], [], []
    base_acc, base_risk = [], []
    for _ in range(trials):
        idx = list(range(len(dedup)))
        random.shuffle(idx)
        cal = [dedup[i] for i in idx[:31]]
        test = [dedup[i] for i in idx[31:]]

        def key(tau):
            m = metrics(apply_distance_ceiling(cal, tau))
            return (m["accuracy"], -m["risk"])

        best = max(TAU_GRID, key=key)
        chosen.append(best)
        mt = metrics(apply_distance_ceiling(test, best))
        tuned_acc.append(mt["accuracy"])
        tuned_risk.append(mt["risk"])
        mb = metrics(test)
        base_acc.append(mb["accuracy"])
        base_risk.append(mb["risk"])

    def stat(xs):
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        return mu, sd

    ta_mu, ta_sd = stat(tuned_acc)
    ba_mu, ba_sd = stat(base_acc)
    tr_mu, tr_sd = stat(tuned_risk)
    br_mu, br_sd = stat(base_risk)
    print(f"    held-out accuracy  baseline={ba_mu:.4f} (sd {ba_sd:.4f})")
    print(f"    held-out accuracy  gated   ={ta_mu:.4f} (sd {ta_sd:.4f})   delta={ta_mu - ba_mu:+.4f}")
    print(f"    held-out risk      baseline={br_mu:.1f} (sd {br_sd:.1f})")
    print(f"    held-out risk      gated   ={tr_mu:.1f} (sd {tr_sd:.1f})   "
          f"delta={(tr_mu - br_mu) / br_mu * 100:+.1f}%")
    print(f"    selected tau distribution : {Counter(chosen).most_common(5)}")

    # ---- 9. residual errors at the recommended point ----------------------
    print("\n[9] RESIDUAL ERRORS AT tau=1.35")
    recs = apply_distance_ceiling(dedup, 1.35)
    for r in recs:
        if r["verdict"] in ("INCORRECT", "PARTIAL_INCOMPLETE"):
            kind = ("FALSE_ANSWER" if r["decision"] == "ANSWER" and r["verdict"] == "INCORRECT"
                    else "FALSE_REFUSAL" if r["verdict"] == "INCORRECT" else "PARTIAL")
            print(f"    {r['query_id']:<12}{r['support_slice']:<20}{kind:<15}"
                  f"ratio={ratio_of(r):.2f}  {texts[r['query_id']]}")

    print("\n[10] STATUS")
    print("    OFFLINE_SIMULATION_ONLY — live replay required before any PASS claim.")


if __name__ == "__main__":
    main()
