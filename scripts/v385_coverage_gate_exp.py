#!/usr/bin/env python3
"""V3.85 Phase C experiment: coverage gate (out-of-corpus equipment).

Replays the frozen V377 aligned benchmark through the real chain with the
Phase C coverage gate enabled/disabled, mirroring the Phase B protocol:

  A. control - gate disabled, must reproduce the frozen baseline digest
  B. control - second pass, determinism check
  C. gated   - gate enabled (OUT_OF_CORPUS_MODEL -> ABSTAIN)
  D. gated   - second pass, determinism check

The control run is the equivalence check: if the new module changed any
decision while disabled, the digest differs from the frozen digest and the
experiment is invalid. The gated run reports decision flips (FIXED/BROKEN)
and slice-level changes.

Usage
-----
    python scripts/v385_coverage_gate_exp.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN_DIGEST = "dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d"
OUT_DIR = ROOT / "results" / "v385_coverage_gate"

DUPLICATE_IDS = {
    "V369-Q0027", "V369-Q0028", "V369-Q0029",
    "V369-Q0030", "V369-Q0031", "V369-Q0032",
}

# Build-time global corpus identity (coverage_index_v385.json). Corpus identity
# is a corpus-level fact and must not depend on the retrieval window.
INDEX_PATH = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "corpus" / "coverage_index_v385.json"

W_FALSE_ANSWER, W_FALSE_REFUSAL, W_PARTIAL = 10, 1, 1

import rag_core  # noqa: E402
rag_core.PERSIST_DIR = str(ROOT / "vector_db_v369")

import v377_aligned_baseline as base  # noqa: E402
from backend.evaluation.score_lineage import build_retrieval_result  # noqa: E402
from backend.retrieval.coverage_gate import coverage_gate_verdict, out_of_corpus_models  # noqa: E402
from backend.retrieval.evidence import analyze_retrieval_evidence, default_policy  # noqa: E402

_INDEX = json.load(open(INDEX_PATH, encoding="utf-8"))
GLOBAL_KNOWN_MODELS = dict(_INDEX["known_models"])  # canonical model -> identity dict


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


def run_condition(cases, label, gate_enabled):
    from langchain_chroma import Chroma
    search_db = Chroma(persist_directory=rag_core.PERSIST_DIR,
                       embedding_function=rag_core.get_embedding_model())
    policy = default_policy()
    print(f"\n--- {label}: gate_enabled={gate_enabled} ---", flush=True)

    runs = []
    for pass_no in (1, 2):
        rows = []
        for case in cases:
            query_text = case["query_text"]
            exp = base.to_expectation(case)
            scored = search_db.similarity_search_with_score(query_text, k=4)
            documents = [d for d, _s in scored]
            rr = build_retrieval_result(scored)
            evidence = analyze_retrieval_evidence(
                query_text, rr, documents,
                retrieval_mode="vector_only_v369", identity_matching=True,
            )
            lineage = base.assert_lineage_fidelity(scored, rr.candidates, evidence)

            unknown_models = out_of_corpus_models(query_text, documents, known_models=GLOBAL_KNOWN_MODELS) if gate_enabled else []
            gate_ok, gate_reason = (True, "") if not gate_enabled else coverage_gate_verdict(
                query_text, documents, known_models=GLOBAL_KNOWN_MODELS)
            gated = False
            if gate_enabled and not gate_ok:
                evidence = replace(evidence, decision="ABSTAIN", reason="OUT_OF_CORPUS_MODEL")
                gated = True

            record = base.make_record(evidence)
            result = base.evaluate_contract_native(exp, record)
            eligible = bool(lineage["fidelity_ok"])
            chunk_ids = [str((getattr(d, "metadata", {}) or {}).get("source", "")) + "#p" +
                         str((getattr(d, "metadata", {}) or {}).get("page", "")) for d, _s in scored]
            contract = evidence.contract or {}
            rows.append({
                "query_id": case["query_id"],
                "benchmark_slice": case["slice_labels"][0],
                "support_slice": base.support_slice(case),
                "support_state": case["support_state"],
                "expected_decision": exp.expected_decision.value,
                "decision": record.decision,
                "reason": record.reason,
                "identity_relation": record.identity_relation,
                "has_candidates": bool(rr.candidates),
                "vector_distance": record.vector_distance,
                "lexical_score": record.lexical_score,
                "contract_has_critical": bool(contract.get("has_critical_requirements", False)),
                "contract_sufficient": bool(contract.get("sufficient", False)),
                "raw_top1_distances": lineage["raw_top1"],
                "fidelity_ok": lineage["fidelity_ok"],
                "chunk_ids": chunk_ids,
                "verdict": result.verdict.value,
                "passing": base._accept_as_pass(result.verdict),
                "eval_reasons": list(result.reason_codes),
                "blocking_mechanism": base.blocking_mechanism(evidence) if record.decision == "ABSTAIN" else "",
                "threshold": policy.max_vector_distance,
                "record_origin": "REAL_EVIDENCE_RUNTIME",
                "retrieval_score_origin": "REAL_CHROMA_RUNTIME",
                "eligible_formal": eligible,
                "gate_applied": gated,
                "gate_reason": gate_reason,
                "unknown_models": unknown_models,
            })
            rows[-1]["primary_attribution"] = (
                base.attribute_error(rows[-1], case) if not rows[-1]["passing"] else ""
            )
        formal = [r for r in rows if r.get("eligible_formal", True)]
        digest = base.canonical_digest(formal)
        runs.append({"rows": rows, "digest": digest})
        print(f"    pass{pass_no} digest={digest}", flush=True)

    ok = runs[0]["digest"] == runs[1]["digest"]
    return {
        "label": label,
        "gate_enabled": gate_enabled,
        "digest": runs[0]["digest"],
        "digest_run2": runs[1]["digest"],
        "replay_ok": ok,
        "summary": summarise(runs[0]["rows"]),
        "rows": runs[0]["rows"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = base.load_cases()

    control = run_condition(cases, "control", gate_enabled=False)
    gated = run_condition(cases, "gated", gate_enabled=True)

    print("\n=== equivalence ===")
    print(f"control digest == frozen digest: {control['digest'] == FROZEN_DIGEST}")
    print(f"  control  = {control['digest']}")
    print(f"  frozen   = {FROZEN_DIGEST}")
    print(f"gated determinism: {gated['replay_ok']}")

    print("\n=== decision flips (gated vs control) ===")
    ctrl = {r["query_id"]: r for r in control["rows"]}
    flips = []
    for r in gated["rows"]:
        c = ctrl[r["query_id"]]
        if (c["decision"], c["reason"]) != (r["decision"], r["reason"]):
            fixed = (not c["passing"] and r["passing"])
            broken = (c["passing"] and not r["passing"])
            flips.append({
                "query_id": r["query_id"],
                "slice": r["support_slice"],
                "before": (c["decision"], c["reason"], c["verdict"]),
                "after": (r["decision"], r["reason"], r["verdict"]),
                "kind": "FIXED" if fixed else "BROKEN" if broken else "NEUTRAL",
                "gate_reason": r["gate_reason"],
            })
    for f in flips:
        print(f"  {f['query_id']} [{f['kind']}] {f['slice']}: {f['before']} -> {f['after']} gate={f['gate_reason']}")

    payload = {
        "experiment": "V385_COVERAGE_GATE",
        "frozen_digest": FROZEN_DIGEST,
        "equivalence_ok": control["digest"] == FROZEN_DIGEST,
        "control": {k: v for k, v in control.items() if k != "rows"},
        "gated": {k: v for k, v in gated.items() if k != "rows"},
        "flips": flips,
        "control_trace": control["rows"],
        "gated_trace": gated["rows"],
    }
    out = OUT_DIR / "experiment.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\noutput: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
