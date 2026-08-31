"""V3.85 runtime smoke: verify the frozen chain reproduces the frozen distances.

Checks that real Chroma retrieval + V3.76 score lineage reproduces the
top-1 distances recorded in results/v377_aligned_baseline/run1_full_trace.json
before any decision-logic experiment is attempted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag_core  # noqa: E402

rag_core.PERSIST_DIR = str(ROOT / "vector_db_v369")

from langchain_chroma import Chroma  # noqa: E402
from backend.evaluation.score_lineage import build_retrieval_result  # noqa: E402
from backend.retrieval.evidence import analyze_retrieval_evidence, default_policy  # noqa: E402

TRACE = ROOT / "results" / "v377_aligned_baseline" / "run1_full_trace.json"
BENCH = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "aligned_benchmark_v2.json"

# Spot-check ids spanning the distance range, including every slice.
SPOT_IDS = [
    "V369-Q0001", "V369-Q0012", "V369-Q0019", "V369-Q0021",
    "V369-Q0033", "V369-Q0043", "V369-Q0060", "V369-Q0069",
]


def main():
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    by_id = {r["query_id"]: r for r in trace}
    cases = {c["query_id"]: c for c in bench["cases"]}

    print("threshold from default_policy():", default_policy().max_vector_distance)
    print("loading Chroma ...", flush=True)
    db = Chroma(persist_directory=rag_core.PERSIST_DIR,
                embedding_function=rag_core.get_embedding_model())
    print("Chroma loaded.", flush=True)

    print(f"\n{'id':<12}{'expected_dist':>15}{'actual_dist':>14}{'delta':>10}{'match':>8}")
    ok = 0
    for qid in SPOT_IDS:
        case = cases[qid]
        scored = db.similarity_search_with_score(case["query_text"], k=4)
        rr = build_retrieval_result(scored)
        ev = analyze_retrieval_evidence(
            case["query_text"], rr, [d for d, _s in scored],
            retrieval_mode="vector_only_v369", identity_matching=True,
        )
        expected = by_id[qid]["raw_top1_distances"]
        actual = ev.vector_distance
        delta = abs(actual - expected)
        match = delta < 1e-6
        ok += match
        print(f"{qid:<12}{expected:>15.6f}{actual:>14.6f}{delta:>10.2e}{str(match):>8}")

    print(f"\nreproduced {ok}/{len(SPOT_IDS)}")
    if ok != len(SPOT_IDS):
        print("STATUS: RUNTIME_REPRODUCTION_FAILED - do not proceed with experiments")
        return 1
    print("STATUS: RUNTIME_REPRODUCTION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
