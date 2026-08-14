"""V3.15 Support precision calibration and frozen-replay reporting helpers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.evaluation.v313_support_calibration import (
    run_calibration,
    validate_calibration,
)
from backend.evaluation.v314_holdout_validation import generalization_range, support_matrix
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION


V315_FAILURE_TAXONOMY = frozenset({
    "RETRIEVAL_CONTEXT_ERROR",
    "COMPATIBILITY_COVERAGE_FAILURE",
    "VALUE_COVERAGE_FAILURE",
    "ATTRIBUTE_COVERAGE_FAILURE",
    "GENERIC_CONCEPT_OVERMATCH",
    "PARTIAL_SUPPORT_ACCEPTED",
})
REQUIRED_MANUFACTURERS = frozenset({
    "Rockwell Automation", "ABB", "Omron", "Beckhoff",
})


def calibration_distribution(calibration: dict[str, Any]) -> dict[str, Any]:
    queries = calibration["queries"]
    return {
        "queries": len(queries),
        "support": dict(sorted(Counter(row["expected_support"] for row in queries).items())),
        "manufacturer": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "category": dict(sorted(Counter(row["category"] for row in queries).items())),
    }


def validate_precision_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    validate_calibration(calibration)
    distribution = calibration_distribution(calibration)
    if not 24 <= distribution["queries"] <= 36:
        raise ValueError("V315_CALIBRATION_SIZE")
    if distribution["support"].get("SUPPORTED", 0) < 12:
        raise ValueError("V315_SUPPORTED_MINIMUM")
    if distribution["support"].get("INSUFFICIENT", 0) < 12:
        raise ValueError("V315_UNSUPPORTED_MINIMUM")
    if set(distribution["manufacturer"]) != REQUIRED_MANUFACTURERS:
        raise ValueError("V315_MANUFACTURER_COVERAGE")
    return distribution


def failure_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    def failures(result: dict[str, Any], expected: bool, predicted: bool) -> set[str]:
        return {
            row["query_id"] for row in result["rows"]
            if bool(row["expected_supported"]) is expected
            and bool(row["predicted_supported"]) is predicted
        }

    before_fs, after_fs = failures(before, False, True), failures(after, False, True)
    before_fi, after_fi = failures(before, True, False), failures(after, True, False)
    return {
        "false_support_fixed": sorted(before_fs - after_fs),
        "false_support_introduced": sorted(after_fs - before_fs),
        "false_insufficient_fixed": sorted(before_fi - after_fi),
        "false_insufficient_introduced": sorted(after_fi - before_fi),
    }


def abc_report(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before_matrix = support_matrix(before)
    after_matrix = support_matrix(after)
    return {
        "support_rule_version": SUPPORT_RULE_VERSION,
        "before": before_matrix,
        "after": after_matrix,
        "generalization_range": {
            "before": generalization_range(before_matrix),
            "after": generalization_range(after_matrix),
        },
        "failure_delta": {
            corpus: failure_delta(before[corpus], after[corpus])
            for corpus in sorted(after)
        },
        "replay_runtime_seconds": {
            corpus: after[corpus]["replay_elapsed_seconds"]
            for corpus in sorted(after)
        },
        "live_retrieval": {
            "BM25": "NO", "Chroma": "NO", "CrossEncoder": "NO", "PDF parser": "NO",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-calibration")
    validate.add_argument("--calibration", type=Path, required=True)
    run = subparsers.add_parser("run-calibration")
    run.add_argument("--calibration", type=Path, required=True)
    for corpus in "abc":
        run.add_argument(f"--artifact-{corpus}", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration_path = ensure_private_path(args.calibration)
    calibration = read_json(calibration_path)
    distribution = validate_precision_calibration(calibration)
    if args.command == "validate-calibration":
        report = {"validity": "VALID", "distribution": distribution, "freeze": calibration["freeze"]}
    else:
        report = run_calibration(
            calibration_path,
            {
                "A": ensure_private_path(args.artifact_a),
                "B": ensure_private_path(args.artifact_b),
                "C": ensure_private_path(args.artifact_c),
            },
            rule_version=SUPPORT_RULE_VERSION,
        )
        atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
