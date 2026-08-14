"""V3.16 Support recall calibration and frozen-replay reporting helpers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evaluation.resumable import atomic_write_json, read_json
from backend.evaluation.v312_replay_runner import ensure_private_path
from backend.evaluation.v313_support_calibration import run_calibration, validate_calibration
from backend.evaluation.v315_support_precision import abc_report
from backend.retrieval.evidence_support import SUPPORT_RULE_VERSION


REQUIRED_MANUFACTURERS = frozenset({
    "Rockwell Automation", "ABB", "Omron", "Beckhoff",
})
V316_FAILURE_CLASSES = frozenset({
    "OVER_CONSTRAINED_ATTRIBUTE",
    "OVER_CONSTRAINED_VALUE",
    "OVER_CONSTRAINED_REQUIREMENT",
    "OVER_CONSTRAINED_QUALIFIER",
    "SEMANTIC_EQUIVALENCE_MISSED",
    "LOCAL_ASSOCIATION_TOO_STRICT",
    "MULTI_CHUNK_AGGREGATION_MISSED",
    "PARTIAL_SUPPORT_ACCEPTED",
})


def calibration_distribution(calibration: dict[str, Any]) -> dict[str, Any]:
    queries = calibration["queries"]
    return {
        "queries": len(queries),
        "support": dict(sorted(Counter(row["expected_support"] for row in queries).items())),
        "manufacturer": dict(sorted(Counter(row["manufacturer"] for row in queries).items())),
        "category": dict(sorted(Counter(row["category"] for row in queries).items())),
        "failure_class": dict(sorted(Counter(row["failure_class"] for row in queries).items())),
        "semantic_positives": sum(bool(row.get("semantic_positive")) for row in queries),
    }


def validate_recall_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    validate_calibration(calibration)
    distribution = calibration_distribution(calibration)
    if not 24 <= distribution["queries"] <= 36:
        raise ValueError("V316_CALIBRATION_SIZE")
    if distribution["support"].get("SUPPORTED", 0) < 16:
        raise ValueError("V316_SUPPORTED_MINIMUM")
    if distribution["support"].get("INSUFFICIENT", 0) < 10:
        raise ValueError("V316_UNSUPPORTED_MINIMUM")
    if set(distribution["manufacturer"]) != REQUIRED_MANUFACTURERS:
        raise ValueError("V316_MANUFACTURER_COVERAGE")
    if distribution["semantic_positives"] < 4:
        raise ValueError("V316_SEMANTIC_POSITIVE_MINIMUM")
    unknown = set(distribution["failure_class"]) - V316_FAILURE_CLASSES
    if unknown:
        raise ValueError(f"V316_UNKNOWN_FAILURE_CLASS:{sorted(unknown)}")
    return distribution


def calibration_failure_matrix(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    return {
        failure_class: {
            "count": len(items),
            "correct": sum(row["expected_support"] == row["predicted_support"] for row in items),
            "false_support": [
                row["calibration_id"] for row in items
                if row["expected_support"] == "INSUFFICIENT" and row["predicted_support"] == "SUPPORTED"
            ],
            "false_insufficient": [
                row["calibration_id"] for row in items
                if row["expected_support"] == "SUPPORTED" and row["predicted_support"] != "SUPPORTED"
            ],
        }
        for failure_class in sorted({row["failure_class"] for row in rows})
        for items in [[row for row in rows if row["failure_class"] == failure_class]]
    }


def final_report(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = abc_report(before, after)
    report["combined_replay_runtime_seconds"] = sum(report["replay_runtime_seconds"].values())
    return report


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
    distribution = validate_recall_calibration(calibration)
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
        report["failure_class_matrix"] = calibration_failure_matrix(report)
        atomic_write_json(ensure_private_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
