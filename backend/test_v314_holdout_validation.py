"""Unit coverage for V3.14 holdout-only evaluation helpers."""

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.evaluation.v314_holdout_validation import (
    LOCATION_GROUND_TRUTH_POLICY_V1, generalization_range, query_hash,
    support_analysis, support_matrix, validate_holdout_manifest,
    validate_location_annotation,
)
from backend.evaluation.private_benchmark import annotation_hash


def _replay_row(query_id, expected, predicted):
    return {"query_id": query_id, "expected_supported": expected, "predicted_supported": predicted}


def _replay(rows):
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    return {"rows": rows, "metrics": {"support": {
        "unsupported_recall": sum(not row["predicted_supported"] for row in unsupported) / len(unsupported) if unsupported else None,
        "false_support_rate": sum(row["predicted_supported"] for row in unsupported) / len(unsupported) if unsupported else None,
        "supported_recall": sum(row["predicted_supported"] for row in supported) / len(supported) if supported else None,
        "false_insufficient_rate": sum(not row["predicted_supported"] for row in supported) / len(supported) if supported else None,
    }}}


class HoldoutValidationTests(unittest.TestCase):
    def test_location_policy_validation(self):
        self.assertIn("general_where", LOCATION_GROUND_TRUTH_POLICY_V1)
        validate_location_annotation({"query_id": "x", "location_expectation": "specific_page", "expected_page": "5"})
        with self.assertRaisesRegex(ValueError, "LOCATION_SUBSECTION_REQUIRED"):
            validate_location_annotation({"query_id": "x", "location_expectation": "exact_subsection"})

    def test_holdout_freeze_hash(self):
        documents = [{"document_id": f"d{index}", "manufacturer": "Omron" if index < 2 else "Beckhoff"} for index in range(3)]
        queries = [{
            "query_id": f"q{index:02d}", "query": f"holdout query {index}",
            "support_gate_truth": "SUPPORTED" if index < 19 else "INSUFFICIENT",
            "requirement_category": "semantic", "failure_class": "OTHER",
            "manufacturer": "Omron", "location_expectation": "none",
        } for index in range(31)]
        manifest = {"documents": documents, "queries": queries, "freeze": {}}
        manifest["freeze"] = {"query_sha256": query_hash(manifest), "annotation_sha256": annotation_hash(manifest)}
        validate_holdout_manifest(manifest)
        manifest["freeze"]["query_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "QUERY_HASH"):
            validate_holdout_manifest(manifest)

    def test_category_manufacturer_and_failure_taxonomy(self):
        manifest = {"queries": [
            {"query_id": "a", "requirement_category": "semantic", "manufacturer": "Omron", "support_gate_truth": "SUPPORTED", "failure_class": "OVER_CONSTRAINED_REQUIREMENT", "location_expectation": "none", "relevant_chunk_ids": ["ca"]},
            {"query_id": "b", "requirement_category": "protocol", "manufacturer": "Beckhoff", "support_gate_truth": "INSUFFICIENT", "failure_class": "PARTIAL_SUPPORT_ACCEPTED", "location_expectation": "none", "relevant_chunk_ids": ["cb"]},
        ]}
        replay = _replay([_replay_row("a", True, False), _replay_row("b", False, True)])
        p2 = [{"query_id": "a", "candidate_ids": ["ca"]}, {"query_id": "b", "candidate_ids": ["cb"]}]
        report = support_analysis(replay, manifest, p2)
        self.assertEqual(report["by_requirement_category"]["semantic"]["false_insufficient"], ["a"])
        self.assertEqual(report["by_manufacturer"]["Beckhoff"]["false_support"], ["b"])
        self.assertEqual(report["over_constrained_failures"][0]["query_id"], "a")
        self.assertEqual(report["partial_support_failures"][0]["query_id"], "b")

    def test_abc_matrix_and_generalization_range(self):
        results = {corpus: {"metrics": {"support": values}} for corpus, values in {
            "A": {"support_accuracy": .9333, "supported_recall": .9167, "unsupported_recall": 1, "false_support_rate": 0, "false_insufficient_rate": .0833},
            "B": {"support_accuracy": .9, "supported_recall": .8889, "unsupported_recall": .9167, "false_support_rate": .0833, "false_insufficient_rate": .1111},
            "C": {"support_accuracy": .8, "supported_recall": .7, "unsupported_recall": .9, "false_support_rate": .1, "false_insufficient_rate": .3},
        }.items()}
        matrix = support_matrix(results)
        self.assertEqual(set(matrix), {"A", "B", "C"})
        self.assertAlmostEqual(generalization_range(matrix)["false_support_rate"]["range"], .1)

    def test_retrieval_missing_evidence_is_separated_from_support_failure(self):
        manifest = {"queries": [{"query_id": "x", "requirement_category": "semantic", "manufacturer": "Omron", "support_gate_truth": "SUPPORTED", "failure_class": "OVER_CONSTRAINED_REQUIREMENT", "location_expectation": "none", "relevant_chunk_ids": ["needed"]}]}
        report = support_analysis(_replay([_replay_row("x", True, False)]), manifest, [{"query_id": "x", "candidate_ids": ["other"]}])
        self.assertEqual(report["failures"][0]["failure_class"], "RETRIEVAL_MISSING_EVIDENCE")

    def test_artifact_replay_c_delegates_to_offline_replay(self):
        manifest = {"documents": [], "queries": []}
        path = Path("backend/evaluation/benchmark_private/v314_artifacts/test-c.json")
        with patch("backend.evaluation.v314_holdout_validation.load_holdout_manifest", return_value=manifest), patch(
            "backend.evaluation.v314_holdout_validation.replay_artifact", return_value={"validity": "VALID"}
        ) as replay:
            from backend.evaluation.v314_holdout_validation import replay_holdout_artifact
            self.assertEqual(replay_holdout_artifact(path, "test-c")["validity"], "VALID")
        replay.assert_called_once_with(path.resolve(), "test-c", expected_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
