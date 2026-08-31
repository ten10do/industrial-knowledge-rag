#!/usr/bin/env python3
"""V3.85 Phase C holdout validation: coverage gate on independent sets (T5).

The coverage gate must be validated on data that was NOT part of the V377
aligned benchmark, otherwise a "PASS" could just be overfitting to the 69
V377 queries (in particular V369-Q0021 "Tell me about the SINAMICS G120
drive."). This script replays the gate on two real, annotated external
sets plus independent anchor queries.

Holdout sources
---------------
* hms_m40_queries.json (15 rows): real annotated queries about the Anybus
  CompactCom M40 (HMS Networks), an industrial-communication module that is
  OUT-OF-CORPUS for the frozen V377 corpus (S7-1200 / ACS580 / M221).
* v327_extract_benchmark/train.json (39 rows): EXTRACTION queries over
  Corpus F (Rockwell Micro800, Beckhoff EL6652, Omron R88D-1SN-ECT), all
  three devices OUT-OF-CORPUS for V377. Queries carry candidate_chunk_ids.

Neither source is part of the V377 benchmark and neither was used to build
the coverage index, so both are independent for gate validation.

Protocol
--------
G1  no-false-positive on real holdout sets: all 54 queries must return
    gate_ok=True. Every real query references an out-of-corpus device or
    no device at all; a gate rejection here would be a spurious fire on a
    query the corpus cannot legitimately answer, so any fire is a FAIL.
G2  out-of-corpus pattern anchors (positive): synthesized queries phrased
    independently of the V377 benchmark, each mentioning a pattern-covered
    model that is NOT in the frozen corpus (s7-1500, acs880, m241,
    powerflex 520, fr-e800, fx3u, cj2m, vlt fc101, sinamics g120
    paraphrase). Expected: OUT_OF_CORPUS_MODEL. This proves the true
    positive path generalizes beyond V369-Q0021's exact wording.
G3  in-corpus anchors (negative): synthesized queries mentioning only
    corpus-known models (s7-1200, acs580, modicon m221) in varied
    phrasings. Expected: gate silent (no over-rejection).
G4  window independence: with the build-time index supplied, the verdict
    must not depend on the retrieval window (documents=[] vs a synthetic
    window) and both key forms of each known model ("s7 1200" / "s7-1200")
    must resolve.
G5  coverage boundary (informational): for each real holdout query, report
    whether its referenced device is pattern-covered. These two sets
    exercise false-positive safety; they cannot exercise the true-positive
    path without extending the pattern tables, which is a documented blind
    spot of the gate, not a gate failure.

Every condition is evaluated twice (determinism check). The script exits
non-zero if any gate FAILs.

Usage
-----
    python scripts/v385_coverage_gate_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "v385_coverage_gate"
INDEX_PATH = (
    ROOT / "backend" / "evaluation" / "benchmark_private"
    / "v377_alignment" / "corpus" / "coverage_index_v385.json"
)
HMS_PATH = ROOT / "backend" / "evaluation" / "benchmark_private" / "hms_m40_queries.json"
V327_PATH = ROOT / "backend" / "evaluation" / "benchmark_private" / "v327_extract_benchmark" / "train.json"

from backend.retrieval.coverage_gate import (  # noqa: E402
    corpus_known_models,
    coverage_gate_verdict,
    detect_model_mentions,
)

_INDEX = json.load(open(INDEX_PATH, encoding="utf-8"))
GLOBAL_KNOWN_MODELS: dict[str, dict] = dict(_INDEX["known_models"])

# --- real holdout sets ----------------------------------------------------

HMS_DEVICE = "Anybus CompactCom M40"


def load_hms() -> list[dict]:
    rows = json.load(open(HMS_PATH, encoding="utf-8"))
    return [{
        "source": "hms_m40",
        "query_id": f"HMS-{i:02d}",
        "query": r["query"],
        "answerable": r.get("answerable"),
        "expected_device": HMS_DEVICE,
        "target": r.get("target", ""),
    } for i, r in enumerate(rows)]


def load_v327() -> list[dict]:
    data = json.load(open(V327_PATH, encoding="utf-8"))
    by_id = {d["document_id"]: d for d in data["documents"]}
    out = []
    for i, q in enumerate(data["queries"]):
        doc = by_id.get(q.get("document_id"), {})
        device = doc.get("equipment_model", "")
        aliases = [a for a in doc.get("model_aliases", []) if a != device]
        out.append({
            "source": "v327",
            "query_id": f"V327-{i:02d}",
            "query": q["query"],
            "answerable": q.get("query_type", ""),
            "expected_device": device,
            "target": ", ".join([device, *aliases, doc.get("product_series", "")]),
        })
    return out


# --- anchor queries (independent of the V377 benchmark) --------------------

OUT_ANCHORS = [
    ("OUT-1", "What is the rated current of the Siemens S7-1500 controller?", "s7-1500"),
    ("OUT-2", "How do I configure the ABB ACS880 drive for fieldbus control?", "acs880"),
    ("OUT-3", "What is the M241 PLC built-in Ethernet port used for?", "m241"),
    ("OUT-4", "Does the PowerFlex 520 drive support EtherNet/IP communication?", "powerflex 520"),
    ("OUT-5", "Set the FR-E800 inverter acceleration time to 3 seconds.", "fr-e800"),
    ("OUT-6", "Which FX3U CPU model should I order for this line?", "fx3u"),
    ("OUT-7", "How do I program the CJ2M CPU unit with CX-Programmer?", "cj2m"),
    ("OUT-8", "What is the control word layout of the VLT FC101 drive?", "vlt fc101"),
    ("OUT-9", "Tell me about the SINAMICS G120 frequency converter options.", "sinamics g120"),
]

IN_ANCHORS = [
    ("IN-1", "What is the CPU variant of the S7-1200 controller?", "s7-1200"),
    ("IN-2", "Does the s7 1200 support PROFINET communication?", "s7-1200"),
    ("IN-3", "What is the current rating of the ACS580 drive?", "acs580"),
    ("IN-4", "How do I wire the Modicon M221 programmable controller?", "modicon m221"),
    ("IN-5", "What is the m221 built-in Ethernet speed?", "modicon m221"),
]

# --- synthetic retrieval window for the window-independence check ----------


class _FakeDoc:
    def __init__(self, metadata: dict):
        self.metadata = metadata


_SYNTH_WINDOW = [
    _FakeDoc({"source": "Siemens_S7_1200_System_Manual_EN.pdf", "page": 10,
              "equipment_model": "s7-1200", "product_series": "s7-1200"}),
    _FakeDoc({"source": "ABB_ACS580_Firmware_Manual.pdf", "page": 42,
              "equipment_model": "acs580", "product_series": "acs580"}),
]


def evaluate_verdict(query: str, window: list | None, known_models: dict | None):
    ok, reason = coverage_gate_verdict(query, window or [], known_models=known_models)
    mentions = detect_model_mentions(query)
    return {"ok": ok, "reason": reason, "mentions": mentions}


def run_all(known_models: dict | None) -> dict:
    rows: list[dict] = []
    fails: list[dict] = []

    # G1 real holdout sets
    for q in [*load_hms(), *load_v327()]:
        v = evaluate_verdict(q["query"], [], known_models)
        pattern_covered = any(
            m in corpus_known_models([], known_models=known_models) or True
            for m in v["mentions"]
        ) if False else bool(v["mentions"])
        rows.append({
            **q,
            "gate_ok": v["ok"],
            "gate_reason": v["reason"],
            "mentions": v["mentions"],
            "pattern_covered_mention": pattern_covered,
            "check": "G1",
        })
        if not v["ok"]:
            fails.append({"query_id": q["query_id"], "check": "G1", "detail": v["reason"]})

    # G2 out-of-corpus pattern anchors
    for qid, text, expected in OUT_ANCHORS:
        v = evaluate_verdict(text, [], known_models)
        rows.append({
            "source": "anchor-out", "query_id": qid, "query": text,
            "answerable": None, "expected_device": expected,
            "target": expected, "gate_ok": v["ok"],
            "gate_reason": v["reason"], "mentions": v["mentions"],
            "pattern_covered_mention": True, "check": "G2",
        })
        if v["ok"] or expected not in v["reason"]:
            fails.append({"query_id": qid, "check": "G2",
                          "detail": f"expected OUT_OF_CORPUS_MODEL:{expected}, got ok={v['ok']} reason={v['reason']!r}"})

    # G3 in-corpus anchors
    for qid, text, expected in IN_ANCHORS:
        v = evaluate_verdict(text, [], known_models)
        rows.append({
            "source": "anchor-in", "query_id": qid, "query": text,
            "answerable": None, "expected_device": expected,
            "target": expected, "gate_ok": v["ok"],
            "gate_reason": v["reason"], "mentions": v["mentions"],
            "pattern_covered_mention": True, "check": "G3",
        })
        if not v["ok"]:
            fails.append({"query_id": qid, "check": "G3", "detail": v["reason"]})

    return {"rows": rows, "fails": fails}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== V3.85 Phase C T5 holdout validation ===\n")
    print(f"coverage index: {INDEX_PATH}")
    print(f"known models  : {sorted(GLOBAL_KNOWN_MODELS)}")

    # G4 window independence + dual key forms
    g4 = {"known_model_keys": sorted(corpus_known_models([], known_models=GLOBAL_KNOWN_MODELS))}
    window_rows = []
    for qid, text, _expected in [*IN_ANCHORS, *OUT_ANCHORS]:
        v_empty = evaluate_verdict(text, [], GLOBAL_KNOWN_MODELS)
        v_window = evaluate_verdict(text, _SYNTH_WINDOW, GLOBAL_KNOWN_MODELS)
        same = v_empty == v_window
        window_rows.append({"query_id": qid, "empty_window": v_empty, "synth_window": v_window, "identical": same})
        if not same:
            g4.setdefault("fails", []).append({"query_id": qid, "detail": "window-dependent verdict"})
    g4["window_rows"] = window_rows
    g4["identical_all"] = all(r["identical"] for r in window_rows)
    g4["has_dual_key_forms"] = (
        "s7 1200" in g4["known_model_keys"] and "s7-1200" in g4["known_model_keys"]
        and "acs580" in g4["known_model_keys"] and "modicon m221" in g4["known_model_keys"]
    )
    print(f"\n[G4] window independence: identical_all={g4['identical_all']} "
          f"dual_key_forms={g4['has_dual_key_forms']}")
    print(f"     known model keys ({len(g4['known_model_keys'])}): {g4['known_model_keys']}")

    # determinism: run all checks twice
    pass1 = run_all(GLOBAL_KNOWN_MODELS)
    pass2 = run_all(GLOBAL_KNOWN_MODELS)
    deterministic = pass1["rows"] == pass2["rows"]
    rows, fails = pass1["rows"], pass1["fails"]
    print(f"determinism (two passes identical): {deterministic}")

    # G5 coverage boundary over the real sets
    g5 = {"covered": [], "uncovered": []}
    for r in rows:
        if r["source"] not in ("hms_m40", "v327"):
            continue
        entry = {"query_id": r["query_id"], "expected_device": r["expected_device"],
                 "mentions": r["mentions"]}
        (g5["covered"] if r["mentions"] else g5["uncovered"]).append(entry)
    print(f"\n[G5] coverage boundary (real holdout sets, n={len(g5['covered']) + len(g5['uncovered'])}):")
    print(f"     pattern-covered device mentions : {len(g5['covered'])}")
    print(f"     not pattern-covered (blind spot): {len(g5['uncovered'])}")

    # summary: a row passes when it is not listed in fails
    fail_ids = {(f["query_id"], f["check"]) for f in fails}
    by_check: dict[str, dict] = {}
    for r in rows:
        c = by_check.setdefault(r["check"], {"n": 0, "ok": 0})
        c["n"] += 1
        if (r["query_id"], r["check"]) not in fail_ids:
            c["ok"] += 1

    print("\n=== summary ===")
    for chk in ("G1", "G2", "G3"):
        c = by_check[chk]
        print(f"  {chk}: {c['ok']}/{c['n']} pass")
    print(f"  G4: {'PASS' if g4['identical_all'] and g4['has_dual_key_forms'] else 'FAIL'}")
    print(f"  determinism: {'PASS' if deterministic else 'FAIL'}")
    if fails:
        print("\n=== FAILS ===")
        for f in fails:
            print(f"  {f}")

    payload = {
        "experiment": "V385_COVERAGE_GATE_HOLDOUT_T5",
        "index_sha256": _INDEX.get("index_sha256", _INDEX.get("catalog_sha256", "")),
        "holdout_counts": {
            "hms_m40": sum(1 for r in rows if r["source"] == "hms_m40"),
            "v327_train": sum(1 for r in rows if r["source"] == "v327"),
            "anchor_out": sum(1 for r in rows if r["source"] == "anchor-out"),
            "anchor_in": sum(1 for r in rows if r["source"] == "anchor-in"),
        },
        "g4": {k: v for k, v in g4.items() if k != "window_rows"},
        "g5": g5,
        "determinism_ok": deterministic,
        "by_check": {k: v for k, v in by_check.items()},
        "fails": fails,
        "passed": deterministic and not fails and g4["identical_all"] and g4["has_dual_key_forms"],
        "rows": rows,
    }
    out = OUT_DIR / "holdout.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\noutput: {out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
