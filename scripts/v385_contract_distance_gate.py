"""V3.85 experiment: contract-branch relative distance ceiling.

Runs the frozen V377 aligned benchmark through the real chain four times:

  A. control  - gate disabled, must reproduce the frozen baseline digest
  B. control  - second pass, determinism check
  C. gated    - gate enabled at EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO
  D. gated    - second pass, determinism check

The control run is an equivalence check: if the new policy field changed any
decision while disabled, the control digest will differ from the frozen digest
and the whole experiment is invalid.

Usage:
    python scripts/v385_contract_distance_gate.py [ratio]      # default 1.20
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RATIO = sys.argv[1] if len(sys.argv) > 1 else "1.20"

FROZEN_DIGEST = "dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d"
FROZEN_SUMMARY = ROOT / "results" / "v377_aligned_baseline" / "aligned_baseline_summary.json"
OUT_DIR = ROOT / "results" / "v385_contract_distance_gate"

# Q0021..Q0026 are byte-identical to Q0027..Q0032.
DUPLICATE_IDS = {
    "V369-Q0027", "V369-Q0028", "V369-Q0029",
    "V369-Q0030", "V369-Q0031", "V369-Q0032",
}

W_FALSE_ANSWER, W_FALSE_REFUSAL, W_PARTIAL = 10, 1, 1


def set_ratio(value: str | None) -> None:
    if value is None:
        os.environ.pop("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", None)
    else:
        os.environ["EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO"] = value


def summarise(rows: list[dict]) -> dict:
    formal = [r for r in rows if r.get("eligible_formal", True)]
    dedup = [r for r in formal if r["query_id"] not in DUPLICATE_IDS]

    def block(subset):
        correct = sum(1 for r in subset if r["passing"])
        fa = sum(1 for r in subset if not r["passing"] and r["decision"] == "ANSWER"
                 and r["verdict"] == "INCORRECT")
        fr = sum(1 for r in subset if not r["passing"] and r["decision"] == "ABSTAIN"
                 and r["verdict"] == "INCORRECT")
        partial = sum(1 for r in subset if r["verdict"] == "PARTIAL_INCOMPLETE")
        risk = W_FALSE_ANSWER * fa + W_FALSE_REFUSAL * fr + W_PARTIAL * partial
        return {
            "n": len(subset), "correct": correct, "false_answer": fa,
            "false_refusal": fr, "partial": partial,
            "accuracy": round(correct / len(subset), 4), "risk": risk,
        }

    return {
        "published_69": block(formal),
        "dedup_63": block(dedup),
        "slices": {
            sl: block([r for r in dedup if r["support_slice"] == sl])
            for sl in ("ANSWERABLE", "CORPUS_UNSUPPORTED", "HARD_NEGATIVE", "OOD")
        },
    }


def run_condition(base, cases, label, ratio):
    set_ratio(ratio)
    from backend.retrieval.evidence import default_policy
    pol = default_policy()
    print(f"\n--- {label}: contract_max_distance_ratio={pol.contract_max_distance_ratio} ---", flush=True)
    r1 = base.run_once(cases, f"{label}_run1")
    r2 = base.run_once(cases, f"{label}_run2")
    digest_match = r1["digest"] == r2["digest"]
    decisions_match = all(
        (a.get("decision"), a.get("reason"), a.get("verdict"))
        == (b.get("decision"), b.get("reason"), b.get("verdict"))
        for a, b in zip(r1["rows"], r2["rows"], strict=True)
    )
    print(f"    digest run1 = {r1['digest']}")
    print(f"    digest run2 = {r2['digest']}")
    print(f"    replay digest_match={digest_match} decisions_match={decisions_match}")
    return {
        "label": label,
        "ratio": pol.contract_max_distance_ratio,
        "threshold": pol.max_vector_distance,
        "digest": r1["digest"],
        "digest_run2": r2["digest"],
        "replay_ok": bool(digest_match and decisions_match),
        "summary": summarise(r1["rows"]),
        "metrics_69": r1["metrics"],
        "capability_coverage": base.capability_coverage(
            [r for r in r1["rows"] if r.get("eligible_formal")]
        ),
        "score_lineage_fidelity": {
            "ok": sum(1 for r in r1["rows"] if r.get("fidelity_ok")),
            "total": len(r1["rows"]),
        },
        "rows": r1["rows"],
    }


def main() -> int:
    set_ratio(None)
    import v377_aligned_baseline as base

    cases = base.load_cases()
    print(f"Loaded {len(cases)} frozen aligned cases ({base.BENCHMARK_VERSION_EXPECTED})")

    control = run_condition(base, cases, "control_gate_off", None)
    gated = run_condition(base, cases, f"gated_tau_{RATIO}", RATIO)

    print("\n" + "=" * 78)
    print("EQUIVALENCE CHECK")
    print("=" * 78)
    frozen = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    frozen_digest = frozen["run1"]["digest"]
    print(f"    frozen baseline digest : {frozen_digest}")
    print(f"    control digest         : {control['digest']}")
    equivalence_ok = control["digest"] == frozen_digest == FROZEN_DIGEST
    print(f"    EQUIVALENCE_OK         : {equivalence_ok}")
    if not equivalence_ok:
        print("    STATUS: EQUIVALENCE_CHECK_FAILED - new policy field changed "
              "decisions while disabled. Experiment invalid.")

    print("\n" + "=" * 78)
    print("BEFORE / AFTER (deduplicated 63)")
    print("=" * 78)
    c, g = control["summary"]["dedup_63"], gated["summary"]["dedup_63"]
    hdr = f"{'':<22}{'control':>12}{'gated':>12}{'delta':>12}"
    print(hdr)
    for key in ("correct", "false_answer", "false_refusal", "partial", "accuracy", "risk"):
        cv, gv = c[key], g[key]
        d = f"{gv - cv:+}" if isinstance(cv, int) else f"{gv - cv:+.4f}"
        print(f"    {key:<18}{cv:>12}{gv:>12}{d:>12}")

    print("\n" + "=" * 78)
    print("BEFORE / AFTER (published 69)")
    print("=" * 78)
    c69, g69 = control["summary"]["published_69"], gated["summary"]["published_69"]
    for key in ("correct", "false_answer", "false_refusal", "partial", "accuracy", "risk"):
        cv, gv = c69[key], g69[key]
        d = f"{gv - cv:+}" if isinstance(cv, int) else f"{gv - cv:+.4f}"
        print(f"    {key:<18}{cv:>12}{gv:>12}{d:>12}")

    print("\n" + "=" * 78)
    print("SLICE DETAIL (deduplicated 63)")
    print("=" * 78)
    print(f"    {'slice':<22}{'n':>4}{'acc_before':>12}{'acc_after':>11}{'FA_before':>11}{'FA_after':>10}")
    for sl, sb in gated["summary"]["slices"].items():
        cb = control["summary"]["slices"][sl]
        print(f"    {sl:<22}{sb['n']:>4}{cb['accuracy']:>12.4f}{sb['accuracy']:>11.4f}"
              f"{cb['false_answer']:>11}{sb['false_answer']:>10}")

    # ---- decision flips ----------------------------------------------------
    ctrl_by_id = {r["query_id"]: r for r in control["rows"]}
    texts = {c["query_id"]: c["query_text"] for c in cases}
    flips, gains, losses = [], [], []
    for r in gated["rows"]:
        o = ctrl_by_id[r["query_id"]]
        if (o["decision"], o["verdict"]) != (r["decision"], r["verdict"]):
            flips.append((r, o))
            if not o["passing"] and r["passing"]:
                gains.append(r)
            elif o["passing"] and not r["passing"]:
                losses.append(r)

    print("\n" + "=" * 78)
    print(f"DECISION FLIPS ({len(flips)})")
    print("=" * 78)
    print("\n  FIXED (was error, now correct):")
    for r in gains:
        o = ctrl_by_id[r["query_id"]]
        print(f"    {r['query_id']:<12}{r['support_slice']:<20}"
              f"{o['decision']}/{o['verdict']} -> {r['decision']}/{r['verdict']}  {texts[r['query_id']]}")
    print("\n  BROKEN (was correct, now error):")
    for r in losses:
        o = ctrl_by_id[r["query_id"]]
        print(f"    {r['query_id']:<12}{r['support_slice']:<20}"
              f"{o['decision']}/{o['verdict']} -> {r['decision']}/{r['verdict']}  {texts[r['query_id']]}")

    print("\n  RESIDUAL ERRORS AFTER GATE (deduplicated):")
    for r in gated["rows"]:
        if r["query_id"] in DUPLICATE_IDS or r["passing"]:
            continue
        kind = ("FALSE_ANSWER" if r["decision"] == "ANSWER" and r["verdict"] == "INCORRECT"
                else "FALSE_REFUSAL" if r["verdict"] == "INCORRECT" else "PARTIAL")
        ratio = r["raw_top1_distances"] / r["threshold"]
        print(f"    {r['query_id']:<12}{r['support_slice']:<20}{kind:<15}ratio={ratio:.2f}  {texts[r['query_id']]}")

    # ---- persist -----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "V385_CONTRACT_DISTANCE_GATE",
        "ratio": RATIO,
        "equivalence_ok": equivalence_ok,
        "frozen_digest": frozen_digest,
        "control": {k: v for k, v in control.items() if k != "rows"},
        "gated": {k: v for k, v in gated.items() if k != "rows"},
    }
    (OUT_DIR / f"experiment_tau_{RATIO}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_DIR / f"gated_tau_{RATIO}_trace.json").write_text(
        json.dumps(gated["rows"], indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_DIR / "control_trace.json").write_text(
        json.dumps(control["rows"], indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    gate_ok = bool(control["replay_ok"] and gated["replay_ok"] and equivalence_ok)
    print("\n" + "=" * 78)
    print(f"    control replay OK : {control['replay_ok']}")
    print(f"    gated   replay OK : {gated['replay_ok']}")
    print(f"    equivalence OK    : {equivalence_ok}")
    print(f"    STATUS: {'EXPERIMENT_VALID' if gate_ok else 'EXPERIMENT_INVALID'}")
    print(f"    saved: {OUT_DIR}")
    print("=" * 78)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
