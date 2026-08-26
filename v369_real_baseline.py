"""V3.69 Real Mixed Evidence Runtime Baseline.

Builds a real Chroma index from selected PDFs, runs real retrieval +
real Evidence runtime, and evaluates via contract-native v367.
All predictions have record_origin=REAL_EVIDENCE_RUNTIME.
"""
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

import rag_core
from rag_core import build_knowledge_base, retrieve_docs, get_representative_docs
from backend.retrieval.evidence import analyze_retrieval_evidence, RetrievalEvidence
from backend.evaluation.contract_eval_v367 import (
    EvidenceEvaluationRecord,
    EvidenceExpectation,
    ExpectedDecision,
    ExpectedClaim,
    ForbiddenClaim,
    _accept_as_pass,
    evaluate_contract_native,
)

# --- Corpus selection (from available benchmark_private PDFs) -------------------

BASE = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS = [
    BASE / "v364_generalization" / "documents" / "Siemens_S7_1200_System_Manual_EN.pdf",
    BASE / "v364_generalization" / "documents" / "ABB_ACS580_Firmware_Manual.pdf",
    BASE / "v364_generalization" / "documents" / "Schneider_M221_Hardware_Guide_EN.pdf",
]

PERSIST_DIR = _REPO_ROOT / "vector_db_v369"


def build_index() -> tuple[int, int]:
    if PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
    rag_core.PERSIST_DIR = str(PERSIST_DIR)
    n_docs, n_chunks = build_knowledge_base([str(p) for p in CORPUS])
    return n_docs, n_chunks


def make_evidence_record(evidence: RetrievalEvidence) -> EvidenceEvaluationRecord:
    """Adapter: serialize real Evidence output."""
    return EvidenceEvaluationRecord(
        decision=evidence.decision,
        reason=evidence.reason,
        query_identity=dict(evidence.query_identity),
        candidate_identity=dict(evidence.candidate_identity),
        identity_relation=evidence.identity_relation,
        contract_requirements_covered=evidence.contract.get(
            "requirements_covered", False,
        ),
        lexical_score=evidence.lexical_score,
        vector_distance=evidence.vector_distance,
        metadata_consistency=evidence.metadata_consistency,
    )


def generate_queries() -> list[dict]:
    """Generate structured queries with gold expectations."""
    queries = []
    qid_counter = [0]

    def next_id():
        qid_counter[0] += 1
        return f"V369-Q{qid_counter[0]:04d}"

    # --- DIRECT_FACT (10): parameter defaults ---
    for i, (param, value, mfr_hint) in enumerate([
        ("acceleration time", "5.0 seconds", "powerflex"),
        ("deceleration time", "5.0 seconds", "powerflex"),
        ("motor overload protection", "class 10", "siemens"),
        ("braking resistor", "integrated", "abb"),
        ("communication protocol", "Modbus RTU", "schneider"),
        ("rated current", "varies by frame size", "abb"),
        ("analog input range", "0-10V or 4-20mA", "siemens"),
        ("digital input count", "6 or 8", "schneider"),
        ("PID control mode", "built-in", "mitsubishi"),
        ("safety function", "STO", "siemens"),
    ]):
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ANSWER,
            required_claims=(
                ExpectedClaim(subject=param, relation="has default/specification",
                              obj_value=value),
            ),
            slice_labels=("DIRECT_FACT",),
            difficulty="L2" if i < 5 else "L3",
        )
        queries.append({
            "query": f"What is the {param} setting?",
            "expectation": exp, "slice": "DIRECT_FACT", "difficulty": exp.difficulty,
        })

    # --- VALUE (10) ---
    for i in range(10):
        params = [
            ("operating voltage range", "380-480V"),
            ("output frequency", "0-590Hz"),
            ("ambient temperature", "-10 to 50°C"),
            ("protection rating", "IP20"),
            ("control method", "vector control"),
            ("overload capacity", "150% for 60s"),
            ("response time", "< 2ms"),
            ("power rating", "0.37 to 132 kW"),
            ("efficiency class", "IE2"),
            ("noise level", "< 70dB"),
        ]
        param, val = params[i]
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ANSWER,
            required_claims=(
                ExpectedClaim(subject=param, relation="is specified as", obj_value=val),
            ),
            slice_labels=("VALUE",),
            difficulty="L3",
        )
        queries.append({
            "query": f"What is the {param}?",
            "expectation": exp, "slice": "VALUE", "difficulty": exp.difficulty,
        })

    # --- IDENTITY (12): model identification ---
    models = [
        "SINAMICS G120", "ACS580", "Altivar ATV320",
        "PowerFlex 520", "FR-E800", "FC51",
    ]
    for i in range(12):
        model = models[i % len(models)]
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ANSWER,
            slice_labels=("IDENTITY",),
            difficulty="L3",
        )
        queries.append({
            "query": f"Tell me about the {model} drive.",
            "expectation": exp, "slice": "IDENTITY", "difficulty": exp.difficulty,
        })

    # --- OOD (15): out-of-domain queries → expect ABSTAIN ---
    ood = [
        "What is the capital of France?",
        "How do I bake bread?",
        "Who wrote Hamlet?",
        "What is quantum entanglement?",
        "Explain photosynthesis.",
        "What is the stock price of Apple?",
        "How do I fix a leaky faucet?",
        "What's the best recipe for pasta?",
        "Explain the French Revolution.",
        "How many moons does Jupiter have?",
        "What is the speed of light?",
        "Who won World War II?",
        "How does a refrigerator work?",
        "What causes rain?",
        "Why is the sky blue?",
    ]
    for i, q in enumerate(ood):
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ABSTAIN,
            slice_labels=("OOD",),
            difficulty="L2" if i < 8 else "L3",
        )
        queries.append({"query": q, "expectation": exp, "slice": "OOD",
                        "difficulty": exp.difficulty})

    # --- NON_TABLE (12): general industrial knowledge ---
    non_table = [
        "What is a variable frequency drive?",
        "How does pulse-width modulation work?",
        "What safety precautions should be taken before electrical work?",
        "What is the purpose of an overload relay?",
        "Explain Profinet communication basics.",
        "What are sourcing and sinking digital inputs?",
        "What is Modbus RTU protocol?",
        "Describe the function of an emergency stop circuit.",
        "How do you select wire gauge for motor connection?",
        "What is the difference between AC and DC motors?",
        "What is three-phase power?",
        "How does regenerative braking work?",
    ]
    for i, q in enumerate(non_table):
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ANSWER,
            slice_labels=("NON_TABLE",),
            difficulty="L2" if i < 6 else "L3",
        )
        queries.append({"query": q, "expectation": exp, "slice": "NON_TABLE",
                        "difficulty": exp.difficulty})

    # --- HARD_NEGATIVE (10): cross-model confusion → expect ABSTAIN ---
    confusions = [
        "What is the acceleration time for ACS580 when using PowerFlex 520 parameters?",
        "Can SINAMICS G120 use ABB drive firmware settings directly?",
        "Is the FR-E800 parameter P040 compatible with Schneider ATV320?",
        "Does Altivar ATV320 support Danfoss FC51 communication profiles?",
        "Can I use PowerFlex 520 wiring diagrams for Siemens G120 installation?",
        "Are ABB ACS580 I/O terminals identical to Mitsubishi FR-E800?",
        "Does the Danfoss FC51 support the same safety functions as PowerFlex 755?",
        "Can Schneider ATV320 replace a Siemens S7-1200 PLC directly?",
        "Is the PowerFlex 525 mounting bracket the same as ABB ACS355?",
        "Do all drives listed here share identical fault codes?",
    ]
    for i, q in enumerate(confusions):
        exp = EvidenceExpectation(
            query_id=next_id(),
            expected_decision=ExpectedDecision.ABSTAIN,
            slice_labels=("HARD_NEGATIVE",),
            difficulty="L4" if i < 5 else "L5",
        )
        queries.append({"query": q, "expectation": exp, "slice": "HARD_NEGATIVE",
                        "difficulty": exp.difficulty})

    return queries


def main() -> None:
    print("== Phase 1: Building Chroma index ==")
    t0 = time.time()
    n_docs, n_chunks = build_index()
    print(f"  {n_docs} documents → {n_chunks} chunks in {time.time()-t0:.1f}s")

    print("\n== Phase 2: Generating benchmark queries ==")
    queries = generate_queries()
    print(f"Total queries: {len(queries)}")
    dist = Counter(q["slice"] for q in queries)
    print("Slices:", dict(sorted(dist.items())))

    print("\n== Phase 3: Running real Evidence pipeline ==")
    results = []
    latencies = []

    from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult

    for qi, q in enumerate(queries):
        query_text = q["query"]
        exp = q["expectation"]

        t_start = time.time()
        try:
            scored_docs = retrieve_docs(query_text, k=4)
        except Exception as e:
            print(f"  RETRIEVAL ERROR {exp.query_id}: {e}")
            continue

        # Build candidates list for Evidence.
        candidates = []
        for doc, score in scored_docs:
            cand = RetrievalCandidate(document=doc, retrieval_source="chroma")
            cand.fusion_score = float(-score)   # higher = better (negate distance)
            candidates.append(cand)

        rr = RetrievalResult(
            candidates=candidates,
            retrieval_mode="vector_only_v369",
        )

        evidence = analyze_retrieval_evidence(
            query_text, rr, [d for d, _s in scored_docs],
            retrieval_mode="vector_only_v369",
            identity_matching=True,
        )
        elapsed = time.time() - t_start
        latencies.append(elapsed)

        record = make_evidence_record(evidence)
        result = evaluate_contract_native(exp, record)
        passing = _accept_as_pass(result.verdict)

        results.append({
            "query_id": exp.query_id,
            "slice": q["slice"],
            "difficulty": q["difficulty"],
            "runtime_decision": record.decision,
            "runtime_reason": record.reason,
            "identity_relation": record.identity_relation,
            "verdict": result.verdict.value,
            "passing": passing,
            "reason_codes": result.reason_codes,
            "record_origin": "REAL_EVIDENCE_RUNTIME",
        })

    print(f"\nExecuted: {len(results)} queries")

    # Metrics.
    total_pass = sum(1 for r in results if r["passing"])
    total_fail = len(results) - total_pass
    accuracy = round(total_pass / max(len(results), 1), 4)

    answer_queries = [r for r in results if r["slice"] not in ("OOD", "HARD_NEGATIVE")]
    abstain_queries = [r for r in results if r["slice"] in ("OOD", "HARD_NEGATIVE")]

    ar_correct = sum(1 for r in answer_queries if r["passing"])
    abr_correct = sum(1 for r in abstain_queries if r["passing"])

    fa_count = sum(1 for r in results if not r["passing"] and r["runtime_decision"] == "ANSWER"
                   and r["expectation_expected"] == "ABSTAIN") if False else 0

    # Simpler: classify by runtime vs expectation.
    fa_on_abstain = sum(1 for r in results if r["runtime_decision"] == "ANSWER"
                        and r["slice"] in ("OOD", "HARD_NEGATIVE"))
    fr_on_answer = sum(1 for r in results if r["runtime_decision"] == "ABSTAIN"
                       and r["slice"] not in ("OOD", "HARD_NEGATIVE"))

    print(f"\n== REAL EVIDENCE BASELINE METRICS ==")
    print(f"Total: {len(results)} | Pass: {total_pass} | Fail: {total_fail} | Acc: {accuracy}")

    slice_metrics: dict[str, Counter] = {}
    for r in results:
        sm = slice_metrics.setdefault(r["slice"], Counter())
        sm["n"] += 1
        sm["pass"] += int(r["passing"])
        sm["fa"] += int(not r["passing"] and r["runtime_decision"] == "ANSWER")
        sm["fr"] += int(not r["passing"] and r["runtime_decision"] == "ABSTAIN")

    print("\nSlice metrics:")
    for sl in sorted(slice_metrics):
        c = slice_metrics[sl]
        acc = round(c.get("pass", 0) / max(c["n"], 1), 3)
        print(f"  {sl}: n={c['n']} pass={c.get('pass',0)} acc={acc} "
              f"FA={c.get('fa',0)} FR={c.get('fr',0)}")

    print(f"\nFA on ABSTAIN-expected: {fa_on_abstain}")
    print(f"FR on ANSWER-expected: {fr_on_answer}")

    # Latency.
    sorted_lat = sorted(latencies)
    med_lat = sorted_lat[len(sorted_lat)//2] if sorted_lat else 0
    p95_lat = sorted_lat[int(len(sorted_lat)*0.95)] if sorted_lat else 0
    print(f"\nLatency: median={med_lat:.4f}s p95={p95_lat:.4f}s")

    # Decision distribution.
    dec_dist = Counter(r["runtime_decision"] for r in results)
    print(f"Runtime decisions: {dict(dec_dist)}")
    reason_dist = Counter(r["runtime_reason"].split("_")[0] if r["runtime_reason"] else "?"
                          for r in results)
    print(f"Reason families: {dict(reason_dist)}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_queries": len(results),
        "accuracy": accuracy,
        "fa_on_abstain": fa_on_abstain,
        "fr_on_answer": fr_on_answer,
        "slice_metrics": {k: dict(v) for k, v in sorted(slice_metrics.items())},
        "latency_median": round(med_lat, 4),
        "latency_p95": round(p95_lat, 4),
        "decision_distribution": dict(dec_dist),
    }, open(out_dir / "v369_real_baseline.json", "w"), indent=1)
    print(f"saved: {out_dir / 'v369_real_baseline.json'}")


if __name__ == "__main__":
    main()
