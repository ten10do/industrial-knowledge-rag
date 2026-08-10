import copy
import unittest
from unittest import mock

from backend.evaluation import run


class OfflineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = run.load_dataset()
        cls.report = run.run_evaluation()

    def test_dataset_has_required_coverage(self):
        categories = [case["category"] for case in self.dataset["cases"]]
        self.assertEqual(len(self.dataset["documents"]), 5)
        self.assertEqual(len(self.dataset["cases"]), 13)
        self.assertEqual(len(self.dataset["multi_turn_cases"]), 2)
        self.assertEqual(len(self.dataset["fallback_cases"]), 20)
        self.assertEqual(categories.count("single-document"), 8)
        self.assertEqual(categories.count("cross-document"), 2)
        self.assertEqual(categories.count("out-of-scope"), 2)
        self.assertEqual(categories.count("hard-paraphrase"), 1)
        fallback_categories = [
            case["category"]
            for case in self.dataset["fallback_cases"]
        ]
        self.assertEqual(fallback_categories.count("lexical-boundary"), 9)
        self.assertEqual(
            fallback_categories.count("deterministic-fallback"),
            11,
        )
        explicit_subtopic = next(
            case
            for case in self.dataset["fallback_cases"]
            if case["id"]
            == "fallback-explicit-subtopic-no-assistant-echo"
        )
        prior_assistant_text = " ".join(
            turn["content"]
            for turn in explicit_subtopic["history"][:2]
            if turn["role"] == "assistant"
        )
        self.assertNotIn("反馈环节", prior_assistant_text)

    def test_dataset_rejects_an_invalid_question(self):
        invalid_dataset = copy.deepcopy(self.dataset)
        invalid_dataset["cases"][0]["question"] = None
        with self.assertRaises(ValueError):
            run.validate_dataset(invalid_dataset)

    def test_real_light_retriever_meets_quality_gates(self):
        metrics = self.report["metrics"]
        self.assertEqual(self.report["page_count"], 10)
        self.assertEqual(self.report["chunk_count"], 10)
        self.assertGreaterEqual(metrics["hit_rate_at_3"], 0.80)
        self.assertGreaterEqual(metrics["mrr"], 0.70)
        self.assertEqual(metrics["metadata_completeness"], 1.00)
        self.assertGreaterEqual(metrics["refusal_accuracy"], 0.80)
        self.assertEqual(metrics["decision_accuracy"], 1.00)
        self.assertEqual(metrics["multi_turn_accuracy"], 1.00)
        self.assertEqual(metrics["deterministic_fallback_accuracy"], 1.00)
        self.assertEqual(metrics["lexical_boundary_accuracy"], 1.00)
        self.assertTrue(self.report["gates_passed"])

    def test_multiturn_followups_use_standalone_query_and_real_light_retrieval(self):
        results = {
            result["id"]: result
            for result in self.report["multi_turn_results"]
        }
        for case in self.dataset["multi_turn_cases"]:
            result = results[case["id"]]
            self.assertEqual(
                result["standalone_query"],
                case["standalone_query"],
            )
            self.assertTrue(result["query_terms_present"])
            self.assertTrue(result["source_hit"])
            self.assertTrue(result["keyword_match"])
            self.assertTrue(result["metadata_complete"])
            self.assertFalse(result["actual_refuse"])
            self.assertEqual(result["query_rewrite_status"], "fallback")
            self.assertTrue(result["fallback_used"])

    def test_fallback_matrix_uses_production_deterministic_rewriter(self):
        results = {
            result["id"]: result
            for result in self.report["fallback_results"]
        }
        for case in self.dataset["fallback_cases"]:
            result = results[case["id"]]
            self.assertEqual(
                result["standalone_query"],
                case["expected_standalone_query"],
            )
            self.assertEqual(
                result["query_rewrite_status"],
                case["expected_status"],
            )
            self.assertEqual(
                result["fallback_used"],
                case["expected_fallback_used"],
            )
            self.assertTrue(result["passed"])

    def test_fallback_quality_gates_cannot_be_bypassed(self):
        for metric in (
            "deterministic_fallback_accuracy",
            "lexical_boundary_accuracy",
        ):
            failed_report = copy.deepcopy(self.report)
            failed_report["metrics"][metric] = 0.99
            self.assertFalse(run.quality_gates_pass(failed_report))

    def test_expected_keywords_are_present_in_retrieved_chunks(self):
        results_by_id = {
            result["id"]: result for result in self.report["case_results"]
        }
        for case in self.dataset["cases"]:
            if not case["should_refuse"]:
                self.assertTrue(
                    results_by_id[case["id"]]["keyword_match"],
                    msg=f"Missing expected keyword for {case['id']}",
                )

    def test_cross_document_cases_keep_sources_distinct(self):
        results_by_id = {
            result["id"]: result for result in self.report["case_results"]
        }
        cross_document_cases = [
            case
            for case in self.dataset["cases"]
            if case["category"] == "cross-document"
        ]
        for case in cross_document_cases:
            actual_sources = {
                item["source"]
                for item in results_by_id[case["id"]]["results"][:3]
            }
            self.assertTrue(set(case["expected_sources"]).issubset(actual_sources))

    def test_out_of_scope_questions_are_refused(self):
        results_by_id = {
            result["id"]: result for result in self.report["case_results"]
        }
        for case in self.dataset["cases"]:
            if case["should_refuse"]:
                self.assertTrue(results_by_id[case["id"]]["actual_refuse"])

    def test_repeated_runs_are_stable(self):
        self.assertTrue(self.report["stable"])

    def test_threshold_calibration_reports_a_reproducible_candidate(self):
        calibration = self.report["threshold_calibration"]
        self.assertIsInstance(calibration["recommended_threshold"], float)
        self.assertGreaterEqual(calibration["accuracy"], 0.80)

    def test_metric_math(self):
        cases = [
            {
                "id": "first",
                "expected_sources": ["a.pdf"],
                "should_refuse": False,
            },
            {
                "id": "second",
                "expected_sources": ["b.pdf"],
                "should_refuse": False,
            },
            {
                "id": "refuse",
                "expected_sources": [],
                "should_refuse": True,
            },
        ]
        complete = lambda source: {
            "source": source,
            "page": 0,
            "content": "content",
            "score": 0.2,
        }
        results = [
            {
                "id": "first",
                "actual_refuse": False,
                "results": [complete("a.pdf")],
            },
            {
                "id": "second",
                "actual_refuse": False,
                "results": [complete("x.pdf"), complete("b.pdf")],
            },
            {
                "id": "refuse",
                "actual_refuse": True,
                "results": [complete("x.pdf")],
            },
        ]
        metrics = run.calculate_metrics(cases, results)
        self.assertEqual(metrics["hit_rate_at_1"], 0.5)
        self.assertEqual(metrics["hit_rate_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)
        self.assertEqual(metrics["metadata_completeness"], 1.0)
        self.assertEqual(metrics["refusal_accuracy"], 1.0)
        self.assertEqual(metrics["decision_accuracy"], 1.0)

    def test_metadata_completeness_detects_missing_fields(self):
        cases = [
            {
                "id": "case",
                "expected_sources": ["expected.pdf"],
                "should_refuse": False,
            }
        ]
        results = [
            {
                "id": "case",
                "actual_refuse": False,
                "results": [
                    {
                        "source": "expected.pdf",
                        "page": 0,
                        "content": "valid",
                        "score": 0.1,
                    },
                    {
                        "source": "",
                        "page": None,
                        "content": "",
                        "score": float("nan"),
                    },
                ],
            }
        ]
        metrics = run.calculate_metrics(cases, results)
        self.assertEqual(metrics["metadata_completeness"], 0.5)

    def test_empty_knowledge_base_and_empty_question_fail_cleanly(self):
        light_rag_core = run._load_light_rag_core()
        previous_state = dict(light_rag_core._knowledge_bases)
        try:
            light_rag_core._knowledge_bases.clear()
            with self.assertRaises(ValueError):
                light_rag_core.retrieve_docs("feedback")
        finally:
            light_rag_core._knowledge_bases.clear()
            light_rag_core._knowledge_bases.update(previous_state)

        with run.offline_knowledge_base(self.dataset) as (light_rag_core, _, _):
            with self.assertRaises(ValueError):
                light_rag_core.retrieve_docs("   ")

    def test_cli_returns_nonzero_when_a_gate_fails(self):
        failed_report = {
            "document_count": 4,
            "case_count": 12,
            "chunk_count": 8,
            "stable": True,
            "gates_passed": False,
            "threshold_calibration": {
                "recommended_threshold": 0.5,
                "accuracy": 1.0,
            },
            "metrics": {
                "hit_rate_at_1": 0.0,
                "hit_rate_at_3": 0.0,
                "mrr": 0.0,
                "metadata_completeness": 1.0,
                "refusal_accuracy": 1.0,
                "decision_accuracy": 1.0,
                "multi_turn_accuracy": 1.0,
                "deterministic_fallback_accuracy": 0.0,
                "lexical_boundary_accuracy": 0.0,
            },
        }
        with mock.patch.object(run, "run_evaluation", return_value=failed_report):
            self.assertEqual(run.main(), 1)


if __name__ == "__main__":
    unittest.main()
