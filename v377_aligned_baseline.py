"""V3.77 Aligned Real Evidence Baseline (formal).

Runs the completely frozen V3.76 runtime over the frozen
``V377_ALIGNED_BENCHMARK_V2`` through the complete real chain:

    query -> real Chroma retrieval -> raw distance -> V3.76 score lineage
          -> RetrievalCandidate -> identity -> Evidence -> V3.71-style adapter
          -> contract-native evaluator

No mock, no score recomputation, no threshold change, no runtime modification.

Formal provenance per case:

    record_origin         = REAL_EVIDENCE_RUNTIME
    retrieval_score_origin = REAL_CHROMA_RUNTIME

A case may only enter the formal denominator when both origins are real AND
per-case score-lineage fidelity holds.

The run executes the full benchmark TWICE and requires byte-identical
retrieval ids/ranks/scores, Evidence decisions, EvaluationRecords, contract
verdicts, metrics, and canonical digests across runs.

All per-case details are private under ``results/v377_aligned_baseline/``.
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
from backend.evaluation.benchmark_v2_schema import QueryDomain, canonical_json, sha256_text  # noqa: E402
from backend.evaluation.contract_eval_v367 import (  # noqa: E402
    EvidenceEvaluationRecord,
    ExpectedClaim,
    EvidenceExpectation,
    ExpectedDecision,
    _accept_as_pass,
    evaluate_contract_native,
)
from backend.evaluation.score_lineage import assert_lineage_fidelity, build_retrieval_result  # noqa: E402
from backend.retrieval.coverage_attribution import blocking_mechanism  # noqa: E402
from backend.retrieval.evidence import analyze_retrieval_evidence, default_policy  # noqa: E402

ALIGN_DIR = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment"
RESULTS_DIR = _REPO_ROOT / "results" / "v377_aligned_baseline"

BENCHMARK_VERSION_EXPECTED = "V377_ALIGNED_BENCHMARK_V2"
THRESHOLD = 13.234710693359375

# Primary-error attribution map for gold=ANSWER refusals (unique attribution).
ABSTAIN_REASON_ATTRIBUTION = {
    "NO_CANDIDATE": "RETRIEVAL_MISS",
    "MODEL_MISMATCH": "IDENTITY",
    "UNKNOWN_IDENTIFIER": "IDENTIFIER",
    "UNKNOWN_PARAMETER": "IDENTIFIER",
    "CROSS_EQUIPMENT": "OTHER",
    "UNSUPPORTED_PROCEDURE": "OTHER",
    # Contract-coverage gate family:
    "IDENTIFIER_NOT_IN_EVIDENCE": "CONTRACT_COVERAGE",
    "PROTOCOL_MISMATCH": "CONTRACT_COVERAGE",
    "MISSING_ATTRIBUTE_EVIDENCE": "CONTRACT_COVERAGE",
    "MISSING_VALUE_EVIDENCE": "CONTRACT_COVERAGE",
    "MISSING_REQUIREMENT_EVIDENCE": "CONTRACT_COVERAGE",
    "MISSING_ACTION_EVIDENCE": "CONTRACT_COVERAGE",
    "PARTIAL_EVIDENCE_ONLY": "CONTRACT_COVERAGE",
    # Coverage/sufficiency gate:
    "INSUFFICIENT_EVIDENCE": "COVERAGE_SCORE",
    "WEAK_RETRIEVAL_EVIDENCE": "RETRIEVAL_MISS",
}


def load_cases() -> list[dict]:
    payload = json.load(open(ALIGN_DIR / "aligned_benchmark_v2.json", encoding="utf-8"))
    if payload["benchmark_version"] != BENCHMARK_VERSION_EXPECTED:
        raise SystemExit("unexpected benchmark version")
    freeze = payload["freeze"]
    verify_frozen_state(payload, freeze)
    return payload["cases"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_state(payload: dict, freeze: dict) -> None:
    """Verify benchmark artifact + runtime integrity against the freeze ledger."""
    corpus_manifest = json.load(open(ALIGN_DIR / "corpus" / "corpus_manifest.json", encoding="utf-8"))
    doc_dir = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v364_generalization" / "documents"
    for doc in corpus_manifest["documents"]:
        if file_sha256(doc_dir / doc["filename"]) != doc["sha256"]:
            raise SystemExit(f"corpus drift detected for {doc['filename']}")
    corpus_hash = sha256_text(canonical_json({
        "documents": [{"filename": d["filename"], "sha256": d["sha256"]} for d in corpus_manifest["documents"]],
        "chunk_count": corpus_manifest["chunk_count"],
        "corpus_chunk_text_sha256": corpus_manifest["corpus_chunk_text_sha256"],
    }))
    if corpus_hash != freeze["CORPUS_HASH"]:
        raise SystemExit("CORPUS_HASH mismatch vs frozen ledger")

    cases = payload["cases"]
    checks = {
        "QUERY_TEXT_HASH": sha256_text(canonical_json([[c["query_id"], c["query_text"]] for c in cases])),
        "EXPECTATION_HASH": sha256_text(canonical_json([
            [c["query_id"], c["expected_decision"],
             [[cl.get("subject"), cl.get("relation"), cl.get("obj_value")] for cl in c["claims"]]]
            for c in cases
        ])),
        "SLICE_MANIFEST_HASH": sha256_text(canonical_json(
            [[c["query_id"], tuple(c["slice_labels"]), c["difficulty"]] for c in cases]
        )),
    }
    for key, value in checks.items():
        if value != freeze[key]:
            raise SystemExit(f"{key} mismatch vs frozen ledger")
    if freeze["EVALUATOR_HASH"] != file_sha256(_REPO_ROOT / "backend" / "evaluation" / "contract_eval_v367.py"):
        raise SystemExit("EVALUATOR_HASH drift - contract evaluator changed since freeze")
    if freeze["SCORE_LINEAGE_HASH"] != file_sha256(_REPO_ROOT / "backend" / "evaluation" / "score_lineage.py"):
        raise SystemExit("SCORE_LINEAGE_HASH drift - score lineage module changed since freeze")
    if default_policy().max_vector_distance != THRESHOLD:
        raise SystemExit("Evidence threshold drifted from frozen value")


def to_expectation(case: dict) -> EvidenceExpectation:
    claims = tuple(
        ExpectedClaim(
            subject=claim["subject"],
            relation=claim["relation"],
            obj_value=claim.get("obj_value", ""),
            scope=claim.get("scope", ""),
        )
        for claim in case["claims"]
    )
    return EvidenceExpectation(
        query_id=case["query_id"],
        expected_decision=ExpectedDecision(case["expected_decision"]),
        required_claims=claims,
        slice_labels=tuple(case["slice_labels"]),
        difficulty=case["difficulty"],
    )


def make_record(evidence) -> EvidenceEvaluationRecord:
    contract = evidence.contract or {}
    return EvidenceEvaluationRecord(
        decision=evidence.decision,
        reason=evidence.reason,
        query_identity=dict(evidence.query_identity),
        candidate_identity=dict(evidence.candidate_identity),
        identity_relation=evidence.identity_relation,
        contract_requirements_covered=bool(contract.get("sufficient", False)),
        lexical_score=evidence.lexical_score,
        vector_distance=evidence.vector_distance,
        metadata_consistency=evidence.metadata_consistency,
        covered_claim_keys=frozenset(contract.get("covered", [])),
    )


def support_slice(case: dict) -> str:
    """Closed-corpus safety slices (§37)."""
    if case["query_domain"] == QueryDomain.GENERIC_OUT_OF_DOMAIN.value:
        return "OOD"
    if case["slice_labels"][0] == "HARD_NEGATIVE":
        return "HARD_NEGATIVE"
    if case["support_state"] == "CORPUS_UNSUPPORTED":
        return "CORPUS_UNSUPPORTED"
    return "ANSWERABLE"


def attribute_error(row: dict, case: dict) -> str:
    """Unique primary attribution on the aligned benchmark."""
    expected = row["expected_decision"]
    decision = row["decision"]
    if expected == "ABSTAIN" and decision == "ANSWER":
        return "OOD_FALSE_ANSWER" if case["query_domain"] == "GENERIC_OUT_OF_DOMAIN" \
            else "CORPUS_UNSUPPORTED_FALSE_ANSWER"
    if expected == "ANSWER" and decision == "ABSTAIN":
        reason = row["reason"]
        attribution = ABSTAIN_REASON_ATTRIBUTION.get(reason, "OTHER")
        if attribution == "COVERAGE_SCORE" and not row["has_candidates"]:
            attribution = "RETRIEVAL_MISS"
        return attribution
    if expected == "ANSWER" and decision == "ANSWER" and row["verdict"] != "CORRECT":
        return "WRONG_CLAIM"
    return "OTHER"


def canonical_digest(rows: list[dict]) -> str:
    payload = [
        (
            r["query_id"],
            r["chunk_ids"],
            r["raw_top1_distances"],
            r["decision"],
            r["reason"],
            r["identity_relation"],
            r["vector_distance"],
            r["lexical_score"],
            r["contract_has_critical"],
            r["contract_sufficient"],
            r["verdict"],
        )
        for r in rows
    ]
    return sha256_text(canonical_json(payload))


def run_once(cases: list[dict], label: str):
    search_db = Chroma(persist_directory=rag_core.PERSIST_DIR, embedding_function=rag_core.get_embedding_model())
    policy = default_policy()
    rows = []
    latencies = []
    for case in cases:
        query_text = case["query_text"]
        exp = to_expectation(case)
        t0 = time.time()
        try:
            scored_docs = search_db.similarity_search_with_score(query_text, k=4)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "query_id": case["query_id"], "error": f"RETRIEVAL_ERROR:{exc}",
                "record_origin": "REAL_EVIDENCE_RUNTIME",
                "retrieval_score_origin": "REAL_CHROMA_RUNTIME", "eligible_formal": False,
            })
            continue
        latencies.append(time.time() - t0)

        rr = build_retrieval_result(scored_docs)
        evidence = analyze_retrieval_evidence(
            query_text, rr, [d for d, _s in scored_docs],
            retrieval_mode="vector_only_v369", identity_matching=True,
        )
        lineage = assert_lineage_fidelity(scored_docs, rr.candidates, evidence)
        record = make_record(evidence)
        result = evaluate_contract_native(exp, record)

        eligible = bool(lineage["fidelity_ok"])
        chunk_ids = [str((getattr(d, "metadata", {}) or {}).get("source", "")) + "#p" +
                     str((getattr(d, "metadata", {}) or {}).get("page", "")) for d, _s in scored_docs]
        contract = evidence.contract or {}
        rows.append({
            "query_id": case["query_id"],
            "benchmark_slice": case["slice_labels"][0],
            "support_slice": support_slice(case),
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
            "passing": _accept_as_pass(result.verdict),
            "eval_reasons": list(result.reason_codes),
            "blocking_mechanism": blocking_mechanism(evidence) if record.decision == "ABSTAIN" else "",
            "threshold": policy.max_vector_distance,
            "latency_seconds": latencies[-1],
            "record_origin": "REAL_EVIDENCE_RUNTIME",
            "retrieval_score_origin": "REAL_CHROMA_RUNTIME",
            "eligible_formal": eligible,
        })
        row = rows[-1]
        row["primary_attribution"] = (
            attribute_error(row, case) if not row["passing"] else ""
        )

    ineligible = [r for r in rows if not r.get("eligible_formal", True)]
    formal_rows = [r for r in rows if r.get("eligible_formal", True)]
    metrics = compute_metrics(formal_rows)
    print(f"\n[{label}] n={len(rows)} formal={len(formal_rows)} ineligible={len(ineligible)}")
    print_metrics(metrics)
    digest = canonical_digest(formal_rows)
    print(f"[{label}] digest={digest}")
    return {"label": label, "rows": rows, "metrics": metrics, "digest": digest,
            "ineligible_ids": [r["query_id"] for r in ineligible],
            "latency_median_ms": round(sorted(latencies)[len(latencies)//2] * 1000, 1) if latencies else 0.0}


def compute_metrics(rows: list[dict]) -> dict:
    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    answer_rows = [r for r in rows if r["expected_decision"] == "ANSWER"]
    abstain_rows = [r for r in rows if r["expected_decision"] == "ABSTAIN"]

    correct_answer = sum(1 for r in answer_rows if r["verdict"] == "CORRECT")
    correct_abstain = sum(1 for r in abstain_rows if r["verdict"] == "ABSTAIN_CORRECT")
    false_refusal = sum(
        1 for r in answer_rows if r["support_state"] == "SUPPORTED_ANSWER" and r["decision"] == "ABSTAIN"
    )
    false_answer_total = sum(
        1 for r in abstain_rows if r["decision"] == "ANSWER"
    )
    false_answer_corpus_unsupported = sum(
        1 for r in abstain_rows
        if r["decision"] == "ANSWER" and r["support_slice"] == "CORPUS_UNSUPPORTED"
    )
    false_answer_hard_negative = sum(
        1 for r in abstain_rows if r["decision"] == "ANSWER" and r["support_slice"] == "HARD_NEGATIVE"
    )
    false_answer_ood = sum(
        1 for r in abstain_rows if r["decision"] == "ANSWER" and r["support_slice"] == "OOD"
    )
    verdict_counts = Counter(r["verdict"] for r in rows)
    total_correct = sum(1 for r in rows if r["passing"])

    attribution_counts = Counter(r["primary_attribution"] for r in rows if not r["passing"])
    slice_metrics = {}
    for slice_name in ("ANSWERABLE", "CORPUS_UNSUPPORTED", "HARD_NEGATIVE", "OOD"):
        subset = [r for r in rows if r["support_slice"] == slice_name]
        if not subset:
            continue
        slice_metrics[slice_name] = {
            "n": len(subset),
            "correct": sum(1 for r in subset if r["passing"]),
            "false_answer": sum(1 for r in subset if r["decision"] == "ANSWER" and not r["passing"]),
            "false_refusal": sum(1 for r in subset if r["decision"] == "ABSTAIN" and r["expected_decision"] == "ANSWER"),
            "accuracy": rate(sum(1 for r in subset if r["passing"]), len(subset)),
        }

    return {
        "n_formal": len(rows),
        "correct": total_correct,
        "accuracy": rate(total_correct, len(rows)),
        "false_answer": false_answer_total,
        "false_answer_rate": rate(false_answer_total, len(abstain_rows)) if abstain_rows else None,
        "false_refusal": false_refusal,
        "false_refusal_rate": rate(false_refusal, len(answer_rows)) if answer_rows else None,
        "answerable_recall": rate(correct_answer, len(answer_rows)),
        "abstention_recall": rate(correct_abstain, len(abstain_rows)),
        "answerable_n": len(answer_rows),
        "abstainable_n": len(abstain_rows),
        "correct_answer_count": correct_answer,
        "correct_abstain_count": correct_abstain,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "partial_incomplete_count": verdict_counts.get("PARTIAL_INCOMPLETE", 0),
        "contract_unsupported_verdict_count": verdict_counts.get("CONTRACT_UNSUPPORTED", 0),
        "false_answer_breakdown": {
            "corpus_unsupported": false_answer_corpus_unsupported,
            "hard_negative": false_answer_hard_negative,
            "ood": false_answer_ood,
        },
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "support_slice_metrics": slice_metrics,
    }


def print_metrics(metrics: dict) -> None:
    print(json.dumps(metrics, indent=2))


def capability_coverage(rows: list[dict]) -> dict:
    """Actual invocation counts per capability path (§42)."""
    return {
        "identity_path_invocations": sum(1 for r in rows if r["identity_relation"] not in ("UNKNOWN", "", None)),
        "score_gate_threshold_evaluations": sum(1 for r in rows if r["vector_distance"] is not None),
        "contract_coverage_gate_invocations": sum(1 for r in rows if r["contract_has_critical"]),
        "identifier_gate_evaluations": sum(1 for r in rows if r["reason"] == "UNKNOWN_IDENTIFIER"),
        "ood_abstain_executions": sum(1 for r in rows if r["support_slice"] == "OOD"),
        "evidence_answers": sum(1 for r in rows if r["decision"] == "ANSWER"),
        "evidence_abstains": sum(1 for r in rows if r["decision"] == "ABSTAIN"),
    }


def main() -> None:
    cases = load_cases()
    print(f"Loaded {len(cases)} frozen aligned cases ({BENCHMARK_VERSION_EXPECTED})")

    run1 = run_once(cases, "run1")
    run2 = run_once(cases, "run2")

    replay_equal = run1["digest"] == run2["digest"]
    rank_equal = all(
        r1["chunk_ids"] == r2["chunk_ids"]
        for r1, r2 in zip(run1["rows"], run2["rows"], strict=True)
        if r1.get("chunk_ids") is not None
    )
    decisions_equal = all(
        (r1.get("decision"), r1.get("reason"), r1.get("verdict"))
        == (r2.get("decision"), r2.get("reason"), r2.get("verdict"))
        for r1, r2 in zip(run1["rows"], run2["rows"], strict=True)
    )
    summary = {
        "benchmark_version": BENCHMARK_VERSION_EXPECTED,
        "run1": {k: v for k, v in run1.items() if k != "rows"},
        "run2": {k: v for k, v in run2.items() if k != "rows"},
        "replay": {
            "digest_match": replay_equal,
            "rank_sequence_match": rank_equal,
            "decisions_and_verdicts_match": decisions_equal,
            "digest_run1": run1["digest"],
            "digest_run2": run2["digest"],
        },
        "capability_coverage": capability_coverage(
            [r for r in run1["rows"] if r.get("eligible_formal")]
        ),
        "score_lineage_fidelity": {
            "ok": sum(1 for r in run1["rows"] if r.get("fidelity_ok")),
            "total": len(run1["rows"]),
        },
        "protected_runtime_note": "runtime untouched; see report protected hashes section",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "run1_full_trace.json", "w", encoding="utf-8") as handle:
        json.dump(run1["rows"], handle, indent=1, ensure_ascii=False, default=str)
    with open(RESULTS_DIR / "run2_full_trace.json", "w", encoding="utf-8") as handle:
        json.dump(run2["rows"], handle, indent=1, ensure_ascii=False, default=str)
    with open(RESULTS_DIR / "aligned_baseline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False, default=str)

    cap = summary["capability_coverage"]
    print("\n== CAPABILITY COVERAGE ==")
    print(json.dumps(cap, indent=2))
    fid = summary["score_lineage_fidelity"]
    print(f"\nSCORE_LINEAGE_FIDELITY (real chain): {fid['ok']}/{fid['total']}")
    print("\nREPLAY:", json.dumps(summary["replay"], indent=2))
    if not replay_equal or not rank_equal or not decisions_equal:
        raise SystemExit("deterministic replay FAILED")
    print(f"\nsaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
