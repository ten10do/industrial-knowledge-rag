"""V3.79 post-hoc policy-variant + anatomy analysis over shadow_results.json."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = json.load(open("results/v379_claim_support/shadow_results.json", encoding="utf-8"))["rows"]
THRESHOLD = 13.234710693359375

fa = [r for r in rows if r["cohort"] == "FA"]
pos = [r for r in rows if r["cohort"] == "POSITIVE_ANSWERED"]
safe = [r for r in rows if r["cohort"] == "SAFE_ABSTAIN"]
fr = [r for r in rows if r["cohort"] == "FALSE_REFUSAL"]
hn = [r for r in rows if r["support_slice"] == "HARD_NEGATIVE"]
cu = [r for r in rows if r["support_slice"] == "CORPUS_UNSUPPORTED"]
ood = [r for r in rows if r["support_slice"] == "OOD"]

print("== PER-SIGNAL STATE DISTRIBUTION BY COHORT ==")
for name in ("s0", "s2", "s3"):
    print(f"-- {name}")
    for label, pop in (("FA", fa), ("POS", pos), ("SAFE", safe), ("FR", fr), ("HN", hn)):
        states = {}
        for r in pop:
            key = (r[name] or {}).get("state")
            states[key] = states.get(key, 0) + 1
        print(f"   {label:5s} n={len(pop):2d} {states}")

print("\n== S0 STRICT vs DEFINITE-ONLY POLICY VARIANTS ==")
for variant, keep_states in (("strict(SUPPORTED only)", {"SUPPORTED"}),
                             ("definite-block(UNSUPPORTED vetoes)", {"SUPPORTED", "AMBIGUOUS"})):
    blocked_fa = [r["query_id"] for r in fa if r["s0"]["state"] not in keep_states]
    killed_pos = [r["query_id"] for r in pos if r["s0"]["state"] not in keep_states]
    fr_new_block = [r["query_id"] for r in fr if r["s0"]["state"] not in keep_states]
    print(f"  {variant}:")
    print(f"    FA blocked {len(blocked_fa)}/9: {blocked_fa}")
    print(f"    POS killed {len(killed_pos)}/23: {killed_pos}")
    print(f"    FR additional refusals: {fr_new_block}")

print("\n== ANATOMY: S0-rejected correct positives ==")
for r in pos:
    if r["s0"]["state"] != "SUPPORTED":
        print(f"  {r['query_id']} [{r['support_slice']}] state={r['s0']['state']:12s} "
              f"reason={r['s0']['support_reason']:28s} miss={r['s0']['provenance']['missing_requirements'][:3]} "
              f"vdist={r['vector_distance_top1']}")

print("\n== ANATOMY: FA Q0017 false support ==")
q17 = next(r for r in fa if r["query_id"] == "V369-Q0017")
print(json.dumps(q17["s0"], ensure_ascii=False, indent=1)[:800])

print("\n== CORPUS_UNSUPPORTED slice validator view (S0) ==")
tp = sum(1 for r in cu if r["gold"] == "UNSUPPORTED" and r["s0"]["state"] != "SUPPORTED")
print(f"  correctly non-supported among gold-UNSUPPORTED CU rows: {tp}/{len(cu)}")
print("\n== HARD NEGATIVE anatomy (S0) ==")
for r in hn:
    if r["s0"]["state"] == "SUPPORTED":
        print(f"  FALSE-SUPPORT: {r['query_id']} reason={r['s0']['support_reason']}")
print("\n== OOD FAs ==")
for r in ood:
    if r["cohort"] == "FA":
        print(f"  {r['query_id']} s0={r['s0']['state']} s2={(r['s2'] or {}).get('state')}")

print("\n== FR rows behavior (record only) ==")
for r in fr:
    print(f"  {r['query_id']} s0={r['s0']['state']:12s} reason={r['s0']['support_reason'][:36]}")

print("\n== latency summary ==")
meta = json.load(open("results/v379_claim_support/shadow_results.json", encoding="utf-8"))
print(json.dumps(meta["latency"], indent=1))
