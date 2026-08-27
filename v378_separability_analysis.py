"""V3.78 secondary separability analysis over saved traces.

Questions answered (from frozen-trace data only):
 1. Which branches produced SAFE_ABSTAIN / FR cohorts (control-cohort view)?
 2. Do any MONOTONE rules over AVAILABLE runtime signals separate the
    6 above-threshold branch#11 FAs from the 13 above-threshold branch#11
    correct answers? (top1 distance, mean/min distance, n-within-threshold,
    margins, identity flags...) Each rule reports blocked-FA vs killed-positive.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = json.load(open("results/v378_audit/full_traces.json", encoding="utf-8"))
THRESHOLD = 13.234710693359375

print("== COHORT x BRANCH crosstab ==")
from collections import Counter
ct = Counter((r["cohort"], f"#{r['branch']} {r['branch_name']}") for r in rows)
for (cohort, branch), n in sorted(ct.items()):
    print(f"  {cohort:18s} {branch:38s} {n}")

print("\n== BRANCH#11 OVER-THRESHOLD SUBGROUP (the decisive comparison) ==")
sub = [r for r in rows if r["branch"] == 11 and r["vector_distance_top1"] > THRESHOLD]
for r in sorted(sub, key=lambda x: x["vector_distance_top1"]):
    tag = "FA" if r["cohort"] == "FA" else "POS"
    print(f"  {tag} {r['query_id']} [{r['support_slice']}] vdist={r['vector_distance_top1']:.4f} "
          f"slice={r['benchmark_slice']}")

fa_sub = [r for r in sub if r["cohort"] == "FA"]
pos_sub = [r for r in sub if r["cohort"] == "POSITIVE_ANSWERED"]

def dists(rs):
    return sorted(r["vector_distance_top1"] for r in rs)

fd, pd_ = dists(fa_sub), dists(pos_sub)
print(f"\nFA distances:       {[round(x,3) for x in fd]}")
print(f"POSITIVE distances: {[round(x,3) for x in pd_]}")

rules = {}
# Rule family A: global top1 cutoff t (would deviate from frozen threshold -> shown for completeness)
for t in (14, 15, 16, 17, 18, 20, 24, 25):
    blocked_fa = sum(1 for d in fd if d > t)
    killed_pos = sum(1 for d in pd_ if d > t)
    rules[f"top1_distance > {t}"] = (blocked_fa, killed_pos)
# Rule family B: all-of-topk beyond threshold fraction
def frac_beyond(r, mult):
    vals = r["vector_distances_all"]
    return sum(1 for v in vals if v > THRESHOLD * mult) / len(vals)
for mult in (1.0, 1.1, 1.2, 1.3):
    thr_key = f"min(top4) > threshold*{mult}"
    b = sum(1 for r in fa_sub if min(r["vector_distances_all"]) > THRESHOLD * mult)
    k = sum(1 for r in pos_sub if min(r["vector_distances_all"]) > THRESHOLD * mult)
    rules[thr_key] = (b, k)
for n in (1, 2, 3, 4):
    key = f"zero-of-top4 within threshold (n_required={n})"
    b = sum(1 for r in fa_sub if sum(1 for v in r["vector_distances_all"] if v <= THRESHOLD) < n)
    k = sum(1 for r in pos_sub if sum(1 for v in r["vector_distances_all"] if v <= THRESHOLD) < n)
    rules[key] = (b, k)
# margins
# margins (top1-top2 gap over the retrieved distance vector)
def gap(r):
    vals = sorted(v for v in r["vector_distances_all"] if v is not None)
    return round(vals[0] - vals[1], 4) if len(vals) >= 2 else None
fm = sorted(g for g in (gap(r) for r in fa_sub) if g is not None)
pm = sorted(g for g in (gap(r) for r in pos_sub) if g is not None)
print(f"\nFA top1-top2 gaps: {[round(x,3) for x in fm]}")
print(f"POS gaps:          {[round(x,3) for x in pm]}")
# margin-based veto candidates
for cap in (0.5, 0.75, 1.0, 1.5, 2.0):
    b = sum(1 for r in fa_sub if gap(r) is None or gap(r) > cap)
    k = sum(1 for r in pos_sub if gap(r) is None or gap(r) > cap)
    rules[f"top1-top2 gap > {cap}"] = (b, k)

print("\n== RULE SWEEP: (FA blocked, correct positives killed) ==")
for name, (b, k) in rules.items():
    print(f"  {name:44s} blocked_FA={b}/6  killed_POS={k}/13")
clean = [name for name, (b, k) in rules.items() if k == 0 and b >= 1]
print(f"\nrules with zero collateral and >=1 FA blocked: {clean or 'NONE'}")
perfect = [name for name, (b, k) in rules.items() if k == 0 and b == 6]
print(f"rules achieving 6/6 with zero collateral:      {perfect or 'NONE'}")

print("\n== G120 in-threshold FA pair context ==")
for r in rows:
    if r["query_id"] in {"V369-Q0021", "V369-Q0027"}:
        print(f"  {r['query_id']} branch#{r['branch']} vdist={r['vector_distance_top1']:.4f} "
              f"corpus_ids={r['corpus_identities']} foreign={r['foreign_equipment_signal']} "
              f"unknown_param={r['unknown_parameter']}")
