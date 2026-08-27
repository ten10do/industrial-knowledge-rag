"""V3.80-C full NO_EXTRACTABLE_REQUIREMENT audit over saved V3.79 traces."""
import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

meta = json.load(open("results/v379_claim_support/shadow_results.json", encoding="utf-8"))
rows = {r["query_id"]: r for r in meta["rows"]}
cases = json.load(open("backend/evaluation/benchmark_private/v377_alignment/aligned_benchmark_v2.json",
                       encoding="utf-8"))["cases"]
by_qid = {c["query_id"]: c for c in cases}

blind = []
for qid, r in sorted(rows.items()):
    if r["s0"]["support_reason"] != "NO_EXTRACTABLE_REQUIREMENT":
        continue
    c = by_qid[qid]
    blind.append({
        "query_id": qid,
        "query_text": c["query_text"],
        "cohort": r["cohort"],
        "gold": r["gold"],
        "slice": c["slice_labels"][0],
        "support_state": c["support_state"],
        "support_reason_gold": c["support_reason"][:80],
    })

print(f"NO_EXTRACTABLE_REQUIREMENT total = {len(blind)} / 69\n")
for b in blind:
    print(f"{b['query_id']} [{b['slice']:12s}|{b['cohort']:16s}|gold={b['gold']}] {b['query_text']}")

print("\nby cohort:", dict(Counter(b['cohort'] for b in blind)))
print("by slice :", dict(Counter(b['slice'] for b in blind)))

# quick form probe: leading interrogative + head noun phrase
def form_of(q):
    ql = q.casefold()
    if re.match(r"^what\s+is\s+(?:the\s+)?(?:a\s+)?[a-z]", ql):
        return "WHAT_IS_<np>"
    if ql.startswith("what"):
        return "WHAT_OTHER"
    if ql.startswith(("which", "where", "when", "who")):
        return ql.split()[0].upper()
    if ql.startswith("how"):
        return "HOW"
    if ql.startswith(("does", "do ", "is ", "are ", "can ")):
        return "POLAR"
    return "IMPERATIVE/OTHER"

print("\nform census:")
for form, n in Counter(form_of(b["query_text"]) for b in blind).most_common():
    print(f"  {form:14s} {n}")
