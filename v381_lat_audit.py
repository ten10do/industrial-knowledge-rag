"""V3.81-LAT audit analysis: apply hand-audit labels to the seeded sample.

Labels below were assigned by manual review of each sampled quote against its
claimed family/kind (authoritative review, recorded verbatim here). Gate:
strict precision >= 0.95 => GO for a full spec-lattice feasibility phase;
otherwise NO-GO -> system freezes at the audited boundary (owner decision path).
"""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# index -> label  (V=VALID, X=INVALID, A=AMBIGUOUS)
LABELS = {
    0:"V",1:"X",2:"V",3:"X",4:"V",5:"V",6:"V",7:"X",8:"V",9:"X",
    10:"V",11:"X",12:"V",13:"V",14:"X",15:"X",16:"V",17:"V",18:"V",19:"V",
    20:"V",21:"V",22:"X",23:"V",24:"V",25:"V",26:"V",27:"X",28:"X",29:"V",
    30:"X",31:"X",32:"V",33:"A",34:"V",35:"V",36:"V",37:"X",38:"X",39:"V",
    40:"V",41:"V",42:"X",43:"V",44:"V",45:"V",46:"V",47:"V",48:"X",49:"X",
    50:"V",51:"V",52:"V",53:"V",54:"V",55:"X",56:"V",57:"A",58:"X",59:"X",
    60:"X",61:"V",62:"V",63:"V",64:"V",65:"V",66:"V",67:"X",68:"V",69:"X",
    70:"X",71:"X",72:"X",73:"V",74:"V",75:"X",76:"V",77:"X",78:"X",79:"X",
    80:"V",81:"V",82:"X",83:"V",84:"V",85:"X",86:"V",87:"V",88:"V",89:"V",
    90:"A",91:"V",92:"V",93:"V",94:"V",95:"V",96:"X",97:"A",98:"X",99:"X",
}

GATE_PRECISION = 0.95

def main():
    payload = json.load(open("results/v381_lat/lattice_probe.json", encoding="utf-8"))
    sample = payload["sample"]
    assert len(sample) == len(LABELS)

    rows = []
    for i, e in enumerate(sample):
        rows.append({**e, "audit_index": i, "label": LABELS[i],
                     "attribute_display": e.get("attribute_name") or e["family"]})

    counts = Counter(r["label"] for r in rows)
    n_valid = counts["V"]
    n_amb = counts["A"]
    n_inv = counts["X"]
    strict_precision = round(n_valid / len(rows), 4)
    loose_precision = round((n_valid + n_amb) / len(rows), 4)

    fail_modes = Counter()
    for r in rows:
        if r["label"] != "X":
            continue
        q = r["quote"].casefold()
        val, fam = r["value"].casefold(), r["family"]
        if r["kind"] == "FIXED_RATING" and re.search(r"\b(?:min|max|=\s*1)\b", q):
            fail_modes["scale_factor_or_menu_fragment_capture"] += 1
        elif (fam == "voltage" and ("ma" in val)) or (fam == "current" and any(u in val for u in (" v", "vd"))):
            fail_modes["unit_family_mismatch"] += 1
        elif (fam == "time" and (" v" in val or "hz" in val)) or (fam != "power" and " w" in val):
            fail_modes["unit_family_mismatch"] += 1
        elif r["kind"] == "FIXED_RATING":
            fail_modes["rating_adjective_distant_from_value"] += 1
        elif r["kind"] == "RANGE":
            fail_modes["range_crosses_adjacent_column"] += 1
        else:
            fail_modes["other_misattribution"] += 1

    verdict = "GO" if strict_precision >= GATE_PRECISION else "NO_GO"
    print(f"n={len(rows)} VALID={n_valid} AMBIGUOUS={n_amb} INVALID={n_inv}")
    print(f"strict precision={strict_precision}  loose(with AMBIGUOUS as pass)={loose_precision}")
    print(f"gate >= {GATE_PRECISION}: {verdict}")
    print("failure modes:", dict(fail_modes))

    by_kind_label = Counter((r["kind"], r["label"]) for r in rows)
    print("kind x label:", {f"{k}={v}" for k, v in sorted(by_kind_label.items())})

    json.dump({
        "gate_precision_required": GATE_PRECISION,
        "strict_precision": strict_precision,
        "loose_precision": loose_precision,
        "counts": dict(counts),
        "failure_modes": dict(fail_modes),
        "verdict": verdict,
        "labeled_rows": [{k: r[k] for k in ("audit_index", "kind", "subject", "family",
                                            "attribute_display", "value", "source_document",
                                            "page", "label")} for r in rows],
    }, open("results/v381_lat/audit_verdict.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved: results/v381_lat/audit_verdict.json")


import re  # noqa: E402  (used inside loop)

if __name__ == "__main__":
    main()
