"""One-shot offline A-E replay and V3.20 safety/utility report assembly."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v312_replay_runner import ensure_private_path, replay_artifact
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION
from backend.retrieval.technical import EVIDENCE_SUPPORT_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "backend" / "evaluation" / "benchmark_private"
RUNTIME_ROOT = PRIVATE_ROOT / "v312_runtime"
RESULT_ROOT = PRIVATE_ROOT / "v320_results"
ARTIFACTS = {
    "A": PRIVATE_ROOT / "v312_artifacts" / "v312-frozen-a-20260814-r4.json",
    "B": PRIVATE_ROOT / "v312_artifacts" / "v312-frozen-b-20260814-r4.json",
    "C": PRIVATE_ROOT / "v314_artifacts" / "v314-frozen-c-20260814.json",
    "D": PRIVATE_ROOT / "v317_artifacts" / "v317-frozen-d-v1.json",
    "E": PRIVATE_ROOT / "v319_artifacts" / "v319-frozen-e-v1.json",
}
BASELINES = {
    "A": RUNTIME_ROOT / "v318-final-a" / "summary.json",
    "B": RUNTIME_ROOT / "v318-final-b" / "summary.json",
    "C": RUNTIME_ROOT / "v318-final-c" / "summary.json",
    "D": RUNTIME_ROOT / "v318-final-d" / "summary.json",
    "E": RUNTIME_ROOT / "v319-replay-e" / "summary.json",
}
RUN_IDS = {corpus: f"v320-final-{corpus.casefold()}" for corpus in ARTIFACTS}


def _evidence_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]["evidence"]
    return {
        "accuracy": metrics["decision_accuracy"],
        "answerable_recall": metrics["answerable_recall"],
        "abstain_recall": metrics["ood_recall"],
        "false_answer_rate": metrics["false_answer_rate"],
        "false_refusal_rate": metrics["false_refusal_rate"],
        "false_answer_ids": metrics["false_answer_ids"],
        "false_refusal_ids": metrics["false_refusal_ids"],
    }


def _final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    false_answers = [row["query_id"] for row in unsupported if row["final_decision"] == "ANSWER"]
    false_refusals = [row["query_id"] for row in supported if row["final_decision"] == "ABSTAIN"]
    return {
        "accuracy": sum((row["final_decision"] == "ANSWER") == row["expected_supported"] for row in rows) / len(rows),
        "answerable_recall": 1 - len(false_refusals) / len(supported) if supported else None,
        "abstain_recall": 1 - len(false_answers) / len(unsupported) if unsupported else None,
        "false_final_answer_rate": len(false_answers) / len(unsupported) if unsupported else None,
        "false_final_refusal_rate": len(false_refusals) / len(supported) if supported else None,
        "false_final_answer_ids": false_answers,
        "false_final_refusal_ids": false_refusals,
    }


def _correct(row: dict[str, Any], *, final: bool) -> bool:
    if final:
        return (row["final_decision"] == "ANSWER") == bool(row["expected_supported"])
    return (row["base_decision"] == "ANSWER") == bool(row["answerable"])


def _introduced(before: dict[str, Any], after: dict[str, Any], *, final: bool) -> list[str]:
    before_rows = {row["query_id"]: row for row in before["rows"]}
    after_rows = {row["query_id"]: row for row in after["rows"]}
    return sorted(
        query_id for query_id in before_rows.keys() & after_rows.keys()
        if _correct(before_rows[query_id], final=final) and not _correct(after_rows[query_id], final=final)
    )


def _range(matrix: dict[str, dict[str, Any]], name: str) -> dict[str, float | None]:
    values = [metrics[name] for metrics in matrix.values() if metrics[name] is not None]
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "spread": max(values) - min(values) if values else None,
    }


def _class_matrix(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for corpus, result in results.items():
        for row in result["rows"]:
            truth = row.get("ground_truth", {})
            failure_class = truth.get("failure_class") or truth.get("requirement_category") or "UNCLASSIFIED"
            grouped[str(failure_class)].append({**row, "corpus": corpus})
    matrix = {}
    for failure_class, rows in sorted(grouped.items()):
        answerable = [row for row in rows if row["answerable"]]
        abstain = [row for row in rows if not row["answerable"]]
        false_answers = [f"{row['corpus']}:{row['query_id']}" for row in abstain if row["base_decision"] == "ANSWER"]
        false_refusals = [f"{row['corpus']}:{row['query_id']}" for row in answerable if row["base_decision"] == "ABSTAIN"]
        matrix[failure_class] = {
            "queries": len(rows),
            "accuracy": sum(_correct(row, final=False) for row in rows) / len(rows),
            "answerable_recall": 1 - len(false_refusals) / len(answerable) if answerable else None,
            "abstain_recall": 1 - len(false_answers) / len(abstain) if abstain else None,
            "false_answer_ids": false_answers,
            "false_refusal_ids": false_refusals,
        }
    return matrix


def assemble_report(
    baselines: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before_matrix = {corpus: _evidence_metrics(result) for corpus, result in baselines.items()}
    after_matrix = {corpus: _evidence_metrics(result) for corpus, result in after.items()}
    return {
        "validity": "VALID",
        "evidence_rule_version": EVIDENCE_SUPPORT_RULE_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "matrix": {
            corpus: {
                "before": before_matrix[corpus],
                "after": after_matrix[corpus],
                "final_before": _final_metrics(baselines[corpus]["rows"]),
                "final_after": _final_metrics(after[corpus]["rows"]),
                "evidence_regressions": _introduced(baselines[corpus], after[corpus], final=False),
                "final_regressions": _introduced(baselines[corpus], after[corpus], final=True),
                "false_answer_plus_false_refusal_before": len(before_matrix[corpus]["false_answer_ids"]) + len(before_matrix[corpus]["false_refusal_ids"]),
                "false_answer_plus_false_refusal_after": len(after_matrix[corpus]["false_answer_ids"]) + len(after_matrix[corpus]["false_refusal_ids"]),
                "artifact_id": after[corpus]["artifact_id"],
                "artifact_hash": after[corpus]["artifact_hash"],
                "runtime_seconds": after[corpus]["replay_elapsed_seconds"],
            }
            for corpus in ARTIFACTS
        },
        "generalization_range": {
            "before": {name: _range(before_matrix, name) for name in (
                "answerable_recall", "abstain_recall", "false_answer_rate", "false_refusal_rate",
            )},
            "after": {name: _range(after_matrix, name) for name in (
                "answerable_recall", "abstain_recall", "false_answer_rate", "false_refusal_rate",
            )},
        },
        "failure_class_matrix": {
            "before": _class_matrix(baselines),
            "after": _class_matrix(after),
        },
        "live_retrieval_isolation": {
            corpus: {"bm25": False, "chroma": False, "cross_encoder": False, "pdf_parser": False, "live_retrieval": False}
            for corpus in ARTIFACTS
        },
    }


def run_once(output: Path) -> dict[str, Any]:
    existing = [run_id for run_id in RUN_IDS.values() if (RUNTIME_ROOT / run_id).exists()]
    if existing:
        raise RuntimeError(f"V320_ONE_SHOT_RUN_ALREADY_EXISTS:{existing}")
    baselines = {corpus: read_json(path) for corpus, path in BASELINES.items()}
    after = {
        corpus: replay_artifact(path, RUN_IDS[corpus])
        for corpus, path in ARTIFACTS.items()
    }
    report = assemble_report(baselines, after)
    atomic_write_json(ensure_private_path(output), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "v320_frozen_a_e_report.json")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_once(args.output)
    else:
        baselines = {corpus: read_json(path) for corpus, path in BASELINES.items()}
        after = {corpus: read_json(RUNTIME_ROOT / RUN_IDS[corpus] / "summary.json") for corpus in ARTIFACTS}
        report = assemble_report(baselines, after)
        atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
