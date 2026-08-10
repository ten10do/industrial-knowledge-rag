"""Run the ignored V3.2 calibration set without evaluating frozen V3.1 queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.evaluation.benchmark_runner import PRIVATE_PATH
from backend.evaluation.private_benchmark import run_private_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_private_calibration(PRIVATE_PATH)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
