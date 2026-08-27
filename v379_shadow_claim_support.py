"""V3.79 Shadow Claim-Support Validation Feasibility runner.

SHADOW ONLY - zero runtime modifications. Replays the frozen V3.77 aligned
baseline chain (vector_db_v369, k=4), recomputes the real Evidence decision,
then evaluates candidate CLAIM-SUPPORT quality signals against the benchmark's
own prediction-blind claim layer (claims/anchors were built at freeze time from
corpus evidence, never from runtime predictions):

    gold(SUPPORTED)   = case.claims nonempty AND an anchor matches retrieved text
    gold(UNSUPPORTED) = case.claims empty  (no documented fact anywhere)
    gold(AMBIGUOUS)   = claims nonempty but no anchor hit among retrieved chunks

Signals (all shadow):
    S0 structured replay  : support-v316.1 validate_evidence_support
    S2 local NLI          : cross-encoder/nli-deberta-v3-xsmall, FROZEN thresholds
    S3 hybrid             : S0 verdict, NLI contradiction can only downgrade

Validity gates enforced before any conclusion:
    RUNTIME_COHORT_MISMATCH if cohorts != FA9 / POSITIVE23 / SAFE34 / FR3
    PROTECTED_HASH_DRIFT    if any protected file hash changes start->end
Determinism: full pipeline executed 3x, canonical digests must match.
Outputs are private under results/v379_claim_support/.
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
from backend.evaluation.score_lineage import build_retrieval_result  # noqa: E402
from backend.retrieval import evidence as ev  # noqa: E402
from backend.retrieval.claim_support_shadow import (  # noqa: E402
    CLAIM_SUPPORT_SIGNAL_VERSION,
    ClaimSupportState,
    NliClaimSupportValidator,
    case_admissible,
    counterfactual_transitions,
    hybrid_claim_support,
    structured_claim_support,
)
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION  # noqa: E402
from backend.retrieval.technical import normalize_technical_text  # noqa: E402

ALIGN_DIR = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment"
OUT_DIR = _REPO_ROOT / "results" / "v379_claim_support"

PROTECTED_FILES = (
    "backend/retrieval/evidence.py",
    "backend/retrieval/evidence_support.py",
    "backend/retrieval/evidence_contract.py",
    "backend/retrieval/claim_support_shadow.py",
    "backend/retrieval/semantic_judge_localnli.py",
    "rag_core.py",
    "backend/evaluation/contract_eval_v367.py",
)

CONTRACT_FA_EXPECTED = {"V369-Q0012", "V369-Q0016", "V369-Q0017", "V369-Q0019",
                        "V369-Q0020", "V369-Q0033", "V369-Q0043"}
G120_FA_EXPECTED = {"V369-Q0021", "V369-Q0027"}
THRESHOLD = 13.234710693359375


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def anchor_hit(anchors: list[dict], chunk_texts: list[str]) -> bool:
    """Mechanical locality check: does any retrieved chunk contain an anchored span?"""
    prepared = []
    for text in chunk_texts:
        prepared.append((_norm(text), normalize_technical_text(str(text))))
    for anchor in anchors:
        quote = _norm(anchor.get("quote", ""))
        pattern = anchor.get("regex") or ""
        for folded, tech in prepared:
            if quote and quote in folded:
                return True
            if pattern:
                try:
                    if __import__("re").search(pattern, folded, __import__("re").IGNORECASE):
                        return True
                except __import__("re").error:
                    pass
            needle_tech = normalize_technical_text(anchor.get("quote", ""))
            if needle_tech and needle_tech in tech:
                return True
    return False


def gold_state(case: dict, chunk_texts: list[str]) -> str:
    claims, anchors = case.get("claims") or [], case.get("anchors") or []
    if not claims:
        return ClaimSupportState.UNSUPPORTED.value
    if anchors and anchor_hit(anchors, chunk_texts):
        return ClaimSupportState.SUPPORTED.value
    return ClaimSupportState.AMBIGUOUS.value


def run_once(cases, search_db, nli_validator=None, collect_latency=False) -> tuple[list[dict], dict]:
    rows, timing = [], {"structured_ms": [], "nli_ms": []}
    for case in cases:
        qid, text = case["query_id"], case["query_text"]
        scored_docs = search_db.similarity_search_with_score(text, k=4)
        rr = build_retrieval_result(scored_docs)
        documents = [d for d, _s in scored_docs]
        chunk_texts = [str(d.page_content) for d in documents]

        real_ev = ev.analyze_retrieval_evidence(
            text, rr, documents, retrieval_mode="vector_only_v369", identity_matching=True,
        )
        expected = case["expected_decision"]
        support_slice = "OOD" if case["query_domain"] == "GENERIC_OUT_OF_DOMAIN" else (
            "HARD_NEGATIVE" if case["slice_labels"][0] == "HARD_NEGATIVE"
            else "CORPUS_UNSUPPORTED" if case["support_state"] == "CORPUS_UNSUPPORTED"
            else "ANSWERABLE")
        answered = real_ev.decision == "ANSWER"
        cohort = ("FA" if (expected == "ABSTAIN" and answered)
                  else "POSITIVE_ANSWERED" if (expected == "ANSWER" and answered)
                  else "SAFE_ABSTAIN" if expected == "ABSTAIN" else "FALSE_REFUSAL")

        started = time.perf_counter()
        s0 = structured_claim_support(text, rr, documents)
        s0_ms = (time.perf_counter() - started) * 1000.0

        if nli_validator is not None:
            t0 = time.perf_counter()
            s2 = nli_validator.judge_case(text, chunk_texts)
            s2_ms = (time.perf_counter() - t0) * 1000.0
            timing["nli_ms"].append(s2_ms)
        else:
            s2 = None
        timing["structured_ms"].append(s0_ms)
        s3 = hybrid_claim_support(s0, s2) if s2 is not None else None

        gs = gold_state(case, chunk_texts)
        dist_top1 = float(scored_docs[0][1]) if scored_docs else None
        rows.append({
            "query_id": qid,
            "support_slice": support_slice,
            "cohort": cohort,
            "runtime_answered": answered,
            "correct_answer": cohort == "POSITIVE_ANSWERED",
            "gold": gs,
            "has_claims": bool(case.get("claims")),
            "vector_distance_top1": round(dist_top1, 4) if dist_top1 is not None else None,
            "s0": s0.as_dict(),
            "s2": s2.as_dict() if s2 is not None else None,
            "s3": s3.as_dict() if s3 is not None else None,
            "admissible_s0": case_admissible(s0),
            "admissible_s2": case_admissible(s2) if s2 is not None else None,
            "admissible_s3": case_admissible(s3) if s3 is not None else None,
        })
    return rows, timing


def signal_metrics(rows: list[dict], key: str) -> dict:
    tp = sum(1 for r in rows if r[key] and r["gold"] == "SUPPORTED")
    fp = sum(1 for r in rows if r[key] and r["gold"] != "SUPPORTED")
    fn = sum(1 for r in rows if not r[key] and r["gold"] == "SUPPORTED")
    tn_unsupported = sum(1 for r in rows if r["gold"] == "UNSUPPORTED")
    detected_unsupported = sum(1 for r in rows if r["gold"] == "UNSUPPORTED" and not r[key])
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    udr = round(detected_unsupported / tn_unsupported, 4) if tn_unsupported else None
    fsr = round(fp / (fp + tp + fn + tn_unsupported - tn_unsupported), 6) if fp else 0.0
    fur = round(fn / (tp + fn), 4) if (tp + fn) else None
    return {
        "signal_supported": tp + fp,
        "true_positive": tp, "false_support": fp, "false_unsupported": fn,
        "precision": precision, "recall": recall,
        "unsupported_detection_rate": udr, "false_support_rate_of_all": fsr,
        "false_unsupported_rate": fur,
    }


def canonical_digest(rows: list[dict], keys=("s0", "s2", "s3")) -> dict:
    out = {}
    for key in keys:
        blob = json.dumps([[r["query_id"], r[key]] for r in rows],
                          sort_keys=True, ensure_ascii=False).encode("utf-8")
        out[key] = hashlib.sha256(blob).hexdigest()
    return out


def main() -> None:
    hashes_start = {p: sha256(_REPO_ROOT / p) for p in PROTECTED_FILES}

    payload = json.load(open(ALIGN_DIR / "aligned_benchmark_v2.json", encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) == 69

    search_db = Chroma(persist_directory=rag_core.PERSIST_DIR,
                       embedding_function=rag_core.get_embedding_model())

    print(f"signal version={CLAIM_SUPPORT_SIGNAL_VERSION} support_rule={SUPPORT_RULE_VERSION}")
    print("loading local NLI (frozen thresholds)...")
    load_started = time.perf_counter()
    from backend.retrieval.semantic_judge_localnli import load_cross_encoder
    nli_model = load_cross_encoder("cross-encoder/nli-deberta-v3-xsmall")
    nli_load_s = time.perf_counter() - load_started
    print(f"NLI loaded in {nli_load_s:.1f}s")

    runs = []
    validator = NliClaimSupportValidator(model=nli_model)
    for run_index in range(3):
        started = time.perf_counter()
        rows, timing = run_once(cases, search_db, validator, collect_latency=(run_index == 2))
        wall = time.perf_counter() - started
        runs.append({"rows": rows, "timing": timing, "wall_s": round(wall, 1),
                     "digest": canonical_digest(rows)})
        print(f"run{run_index + 1}: wall={wall:.1f}s digest={runs[-1]['digest']}")

    digests = {tuple(r["digest"][k] for k in ("s0", "s2", "s3")) for r in runs}
    deterministic = len(digests) == 1
    if not deterministic:
        print("DETERMINISM_FAILURE:", digests)
        raise SystemExit(1)

    rows = runs[0]["rows"]

    cohort_counts = Counter(r["cohort"] for r in rows)
    expected_counts = {"FA": 9, "POSITIVE_ANSWERED": 23, "SAFE_ABSTAIN": 34, "FALSE_REFUSAL": 3}
    if dict(cohort_counts) != expected_counts:
        print("RUNTIME_COHORT_MISMATCH:", cohort_counts)
        raise SystemExit(1)

    contract_fas = [r for r in rows if r["cohort"] == "FA"
                    and r["query_id"] in CONTRACT_FA_EXPECTED]
    g120_fas = [r for r in rows if r["cohort"] == "FA" and r["query_id"] in G120_FA_EXPECTED]
    positives_contract = [r for r in rows if r["cohort"] == "POSITIVE_ANSWERED"
                          and r["vector_distance_top1"] is not None
                          and r["vector_distance_top1"] <= THRESHOLD]
    positives_all = [r for r in rows if r["cohort"] == "POSITIVE_ANSWERED"]
    hard_negatives = [r for r in rows if r["support_slice"] == "HARD_NEGATIVE"]

    print("\n== PRIMARY 20-CASE COHORT (contract path) ==")
    print("cohort A - 7 contract-path FAs:")
    for r in contract_fas:
        print(f"  {r['query_id']} gold={r['gold']:12s} "
              f"s0={r['s0']['state']:12s} s2={(r['s2'] or {}).get('state')} "
              f"s3={(r['s3'] or {}).get('state')}")
    print(f"  detected unsupported: "
          f"S0={sum(1 for r in contract_fas if not r['admissible_s0'])}/7 "
          f"S2={sum(1 for r in contract_fas if not r['admissible_s2'])}/7 "
          f"S3={sum(1 for r in contract_fas if not r['admissible_s3'])}/7")
    print("cohort B - contract-path correct answers preservation (strictest set =<=thr):")
    for name, key, pop in (("S0", "admissible_s0", positives_contract),
                           ("S2", "admissible_s2", positives_contract),
                           ("S3", "admissible_s3", positives_contract)):
        killed = sum(1 for r in pop if not r[key])
        print(f"  {name}: preserved={len(pop) - killed}/{len(pop)} killed={killed}")

    print("\n== G120 PAIR (#13, in-threshold) ==")
    for r in g120_fas:
        print(f"  {r['query_id']} gold={r['gold']} s0={r['s0']['state']} "
              f"s2={(r['s2'] or {}).get('state')} s3={(r['s3'] or {}).get('state')}")

    print("\n== CASE METRICS BY SIGNAL ==")
    metrics = {}
    for name, key in (("S0_structured_v3161", "admissible_s0"),
                      ("S2_local_nli", "admissible_s2"),
                      ("S3_hybrid", "admissible_s3")):
        m = signal_metrics(rows, key)
        metrics[name] = m
        print(f"  {name}: {m}")

    print("\n== HAR NEGATIVES false-support check ==")
    hn_false_support = {
        "S0": sum(1 for r in hard_negatives if r["admissible_s0"]),
        "S2": sum(1 for r in hard_negatives if r["admissible_s2"]),
        "S3": sum(1 for r in hard_negatives if r["admissible_s3"]),
    }
    print(f"  false SUPPORTED among 10 hard negatives: {hn_false_support}")

    print("\n== COUNTERFACTUAL TRANSITIONS (shadow only) ==")
    transitions = {}
    for name, key in (("S0", "admissible_s0"), ("S2", "admissible_s2"), ("S3", "admissible_s3")):
        cf_rows = [{**r, "admissible": r[key]} for r in rows]
        t = counterfactual_transitions(cf_rows, policy_name=name)
        # note: transition table treats admissible=True on answered rows as NO_EFFECT;
        # recompute explicitly per class for reporting clarity
        fa_blocked = sum(1 for r in rows if r["cohort"] == "FA" and not r[key])
        correct_killed = sum(1 for r in rows if r["cohort"] == "POSITIVE_ANSWERED" and not r[key])
        fr_rows = [r for r in rows if r["cohort"] == "FALSE_REFUSAL"]
        safe_abstains = [r for r in rows if r["cohort"] == "SAFE_ABSTAIN"]
        transitions[name] = {
            "FA_blocked_total": fa_blocked,
            "correct_answered_killed": correct_killed,
            "FR_unchanged_note": [r["query_id"] for r in fr_rows],
            "safe_abstain_unchanged": len(safe_abstains),
        }
        print(f"  {name}: FA_blocked={fa_blocked}/9 correct_killed={correct_killed}")

    stats = runs[0]["timing"]
    flat_nli = sorted(stats["nli_ms"])
    flat_s0 = sorted(stats["structured_ms"])

    def pct(vals, p):
        idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
        return round(vals[idx], 3)

    latency = {
        "structured_validate_ms_median": pct(flat_s0, 0.5),
        "structured_validate_ms_p95": pct(flat_s0, 0.95),
        "nli_pair_count_invocations": validator.invocations,
        "nli_per_case_ms_median": pct(flat_nli, 0.5),
        "nli_per_case_ms_p95": pct(flat_nli, 0.95),
        "nli_model_load_s": round(nli_load_s, 1),
    }

    hashes_end = {p: sha256(_REPO_ROOT / p) for p in PROTECTED_FILES}
    drift = [p for p in PROTECTED_FILES if hashes_start[p] != hashes_end[p]]
    if drift:
        print("PROTECTED_HASH_DRIFT:", drift)
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({
        "signal_version": CLAIM_SUPPORT_SIGNAL_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "deterministic": deterministic,
        "digests": runs[0]["digest"],
        "rows": rows,
        "metrics": metrics,
        "transitions": transitions,
        "hard_negative_false_support": hn_false_support,
        "latency": latency,
        "protected_hashes": hashes_end,
    }, open(OUT_DIR / "shadow_results.json", "w", encoding="utf-8"),
        indent=1, ensure_ascii=False)

    print("\nsaved:", OUT_DIR / "shadow_results.json")
    print("PROTECTED_HASHES_OK")


if __name__ == "__main__":
    main()
