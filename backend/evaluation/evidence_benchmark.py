"""Independent calibration and evaluation for the V2.7 evidence gate."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

from langchain_core.documents import Document

from backend import rag_core
from backend.evaluation.benchmark_runner import CHALLENGE_PATH, _benchmark_documents
from backend.evaluation.benchmark_schema import rank_of
from backend.evaluation.full_vector_benchmark import (
    FULL_BENCHMARK_KNOWLEDGE_BASE_ID,
    full_vector_knowledge_base,
)
from backend.evaluation.retrieval_benchmark import benchmark_knowledge_base
from backend.retrieval.evidence import EvidencePolicy


CALIBRATION_PATH = Path(__file__).resolve().parent / "fixtures" / "evidence_calibration.json"
EVALUATION_PATH = Path(__file__).resolve().parent / "fixtures" / "evidence_evaluation.json"
MODES = ("bm25", "vector", "hybrid")


def load_query_set(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", [])
    if not queries or any("answerable" not in item or "ood_type" not in item for item in queries):
        raise ValueError("Evidence fixture requires labelled answerable and OOD queries.")
    return payload


def calibrate_vector_policy(rows: list[dict]) -> EvidencePolicy:
    """Choose a single distance cutoff from independent labelled calibration queries."""
    samples = [
        row for row in rows
        if row["vector_distance"] is not None
        and not row["exact_identifier_match"]
        and row["reason"] not in {"UNKNOWN_IDENTIFIER", "MODEL_MISMATCH"}
    ]
    if not samples:
        raise ValueError("Calibration did not produce usable vector distances.")
    thresholds = sorted({row["vector_distance"] for row in samples})

    def score(threshold: float) -> tuple[float, float, float]:
        decisions = [row["vector_distance"] <= threshold for row in samples]
        correct = sum(decision == row["answerable"] for decision, row in zip(decisions, samples))
        answerable = [row for row in samples if row["answerable"]]
        answerable_recall = sum(
            row["vector_distance"] <= threshold for row in answerable
        ) / len(answerable)
        false_answer_rate = sum(
            row["vector_distance"] <= threshold for row in samples if not row["answerable"]
        ) / len([row for row in samples if not row["answerable"]])
        return correct / len(samples), answerable_recall, -false_answer_rate

    best = max(thresholds, key=lambda threshold: (score(threshold), -threshold))
    return EvidencePolicy(max_vector_distance=best)


def _decision_metrics(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    ood = [row for row in rows if not row["answerable"]]
    abstained = [row for row in rows if row["decision"] == "ABSTAIN"]
    false_answers = [row for row in ood if row["decision"] == "ANSWER"]
    false_refusals = [row for row in answerable if row["decision"] == "ABSTAIN"]
    return {
        "decision_accuracy": sum(
            (row["decision"] == "ANSWER") == row["answerable"] for row in rows
        ) / len(rows),
        "ood_recall": sum(row["decision"] == "ABSTAIN" for row in ood) / len(ood),
        "answerable_recall": sum(row["decision"] == "ANSWER" for row in answerable) / len(answerable),
        "false_answer_rate": len(false_answers) / len(ood),
        "false_refusal_rate": len(false_refusals) / len(answerable),
        "abstention_precision": sum(not row["answerable"] for row in abstained) / len(abstained) if abstained else 0.0,
        "abstention_recall": sum(row["decision"] == "ABSTAIN" for row in ood) / len(ood),
    }


def _retrieval_metrics(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    ranks = [row["retrieval_rank"] for row in answerable]
    return {
        "hit_rate_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum((rank or 99) <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks if rank) / len(ranks),
    }


def _failure_type(row: dict) -> str | None:
    if not row["answerable"] and row["decision"] == "ANSWER":
        if row["ood_type"] == "unknown_identifier":
            return "UNKNOWN_IDENTIFIER_MISSED"
        if row["ood_type"] in {"unknown_model", "cross_equipment"}:
            return "MODEL_MISMATCH_MISSED"
        return "WEAK_EVIDENCE_ACCEPTED"
    if row["answerable"] and row["decision"] == "ABSTAIN":
        return "STRONG_EVIDENCE_REJECTED"
    return None


def _run_queries(query_set: dict, light_rag, light_id, policy: EvidencePolicy) -> dict:
    reports = {}
    for mode in MODES:
        rows = []
        for query in query_set["queries"]:
            if mode == "bm25":
                result = light_rag.retrieve_docs(query["query"], k=5, knowledge_base_id=light_id, retrieval_mode="lexical")
                evidence = light_rag.analyze_evidence(query["query"], result, "lexical", policy=policy)
            else:
                result = rag_core.retrieve_docs(query["query"], k=5, knowledge_base_id=FULL_BENCHMARK_KNOWLEDGE_BASE_ID, retrieval_mode=mode)
                evidence = rag_core.analyze_evidence(query["query"], result, mode, policy=policy)
            row = {
                "query_id": query["query_id"],
                "query": query["query"],
                "answerable": query["answerable"],
                "ood_type": query["ood_type"],
                "retrieval_rank": rank_of(
                    [{"chunk_id": candidate.chunk_id} for candidate in result.candidates],
                    query["relevant_chunk_ids"],
                ),
                **evidence.as_dict(),
            }
            row["correct"] = (row["decision"] == "ANSWER") == row["answerable"]
            row["failure_type"] = _failure_type(row)
            rows.append(row)
        reports[mode] = {
            "decision_metrics": _decision_metrics(rows),
            "retrieval_metrics": _retrieval_metrics(rows),
            "ood_category_metrics": {
                category: {
                    "count": len(items),
                    "recall": sum(item["decision"] == "ABSTAIN" for item in items) / len(items),
                }
                for category in sorted({row["ood_type"] for row in rows if row["ood_type"]})
                for items in [[row for row in rows if row["ood_type"] == category]]
            },
            "decision_table": rows,
            "failures": [row for row in rows if row["failure_type"]],
        }
    return reports


def run_evidence_benchmark() -> dict:
    challenge = json.loads(CHALLENGE_PATH.read_text(encoding="utf-8"))
    benchmark = _benchmark_documents(CHALLENGE_PATH, challenge)
    calibration = load_query_set(CALIBRATION_PATH)
    evaluation = load_query_set(EVALUATION_PATH)
    with ExitStack() as stack:
        light_rag, light_id = stack.enter_context(benchmark_knowledge_base(benchmark))
        documents = [Document(page_content=item["content"], metadata=dict(item["metadata"])) for item in benchmark["documents"]]
        stack.enter_context(full_vector_knowledge_base(documents))
        provisional = EvidencePolicy(max_vector_distance=float("inf"))
        calibration_reports = _run_queries(calibration, light_rag, light_id, provisional)
        policy = calibrate_vector_policy(calibration_reports["vector"]["decision_table"])
        return {
            "calibration": {
                "query_count": len(calibration["queries"]),
                "answerable_count": sum(item["answerable"] for item in calibration["queries"]),
                "ood_count": sum(not item["answerable"] for item in calibration["queries"]),
                "policy": {"max_vector_distance": policy.max_vector_distance},
            },
            "evaluation": {
                "query_count": len(evaluation["queries"]),
                "answerable_count": sum(item["answerable"] for item in evaluation["queries"]),
                "ood_count": sum(not item["answerable"] for item in evaluation["queries"]),
                "reports": _run_queries(evaluation, light_rag, light_id, policy),
            },
        }


if __name__ == "__main__":
    print(json.dumps(run_evidence_benchmark(), ensure_ascii=False, indent=2))
