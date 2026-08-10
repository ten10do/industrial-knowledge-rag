"""Run a deterministic, offline regression evaluation of the light RAG pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from backend.conversation.context_manager import ConversationContextManager
from backend.conversation.models import ContextOptions, ConversationTurn


DATASET_PATH = Path(__file__).resolve().parent / "fixtures" / "dataset.json"
QUALITY_GATES = {
    "hit_rate_at_3": 0.80,
    "mrr": 0.70,
    "metadata_completeness": 1.00,
    "refusal_accuracy": 0.80,
    "decision_accuracy": 0.85,
    "multi_turn_accuracy": 1.00,
    "deterministic_fallback_accuracy": 1.00,
    "lexical_boundary_accuracy": 1.00,
}


class _FixturePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FixturePdfReader:
    def __init__(self, path: str, documents_by_source: dict[str, dict]):
        source = Path(path).name
        if source not in documents_by_source:
            raise FileNotFoundError(f"No offline fixture for {source}")
        self.pages = [
            _FixturePage(page_text)
            for page_text in documents_by_source[source]["pages"]
        ]


def load_dataset(path: Path = DATASET_PATH) -> dict:
    with path.open("r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict) -> None:
    documents = dataset.get("documents")
    cases = dataset.get("cases")
    multi_turn_cases = dataset.get("multi_turn_cases")
    fallback_cases = dataset.get("fallback_cases")
    if not isinstance(documents, list) or len(documents) < 4:
        raise ValueError("Evaluation dataset requires at least four documents.")
    if not isinstance(cases, list) or len(cases) < 12:
        raise ValueError("Evaluation dataset requires at least twelve cases.")
    if not isinstance(multi_turn_cases, list) or len(multi_turn_cases) < 2:
        raise ValueError(
            "Evaluation dataset requires at least two multi-turn cases."
        )
    if not isinstance(fallback_cases, list) or len(fallback_cases) < 20:
        raise ValueError(
            "Evaluation dataset requires at least twenty fallback cases."
        )

    sources = []
    for document in documents:
        source = document.get("source")
        pages = document.get("pages")
        keywords = document.get("keywords")
        if not isinstance(source, str) or not source.endswith(".pdf"):
            raise ValueError("Every fixture document requires a PDF source name.")
        if not isinstance(pages, list) or not pages or not all(
            isinstance(page, str) and page.strip() for page in pages
        ):
            raise ValueError(f"Fixture {source} requires non-empty text pages.")
        if not isinstance(keywords, list) or not keywords or not all(
            isinstance(keyword, str) and keyword.strip() for keyword in keywords
        ):
            raise ValueError(f"Fixture {source} requires topic keywords.")
        sources.append(source)
    if len(sources) != len(set(sources)):
        raise ValueError("Fixture source names must be unique.")

    source_set = set(sources)
    case_ids = set()
    required_case_fields = {
        "question",
        "expected_sources",
        "expected_keywords",
        "should_refuse",
    }
    for case in cases:
        if not required_case_fields.issubset(case):
            raise ValueError("Each evaluation case is missing a required field.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("Evaluation case IDs must be non-empty and unique.")
        case_ids.add(case_id)
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Case {case_id} requires a non-empty question.")
        if not isinstance(case["should_refuse"], bool):
            raise ValueError(f"Case {case_id} has an invalid should_refuse value.")
        if not isinstance(case["expected_sources"], list) or not set(
            case["expected_sources"]
        ).issubset(source_set):
            raise ValueError(f"Case {case_id} references an unknown source.")
        if not isinstance(case["expected_keywords"], list) or not all(
            isinstance(keyword, str) and keyword
            for keyword in case["expected_keywords"]
        ):
            raise ValueError(f"Case {case_id} has invalid expected keywords.")
        if case["should_refuse"] and case["expected_sources"]:
            raise ValueError(f"Refusal case {case_id} must not expect a source.")
        if not case["should_refuse"] and not case["expected_sources"]:
            raise ValueError(f"Answerable case {case_id} must expect a source.")

    multi_turn_ids = set()
    required_multi_turn_fields = {
        "first_question",
        "first_answer",
        "follow_up",
        "standalone_query",
        "required_query_terms",
        "expected_sources",
        "expected_keywords",
    }
    for case in multi_turn_cases:
        case_id = case.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in multi_turn_ids
        ):
            raise ValueError(
                "Multi-turn case IDs must be non-empty and unique."
            )
        multi_turn_ids.add(case_id)
        if not required_multi_turn_fields.issubset(case):
            raise ValueError(
                f"Multi-turn case {case_id} is missing a required field."
            )
        for field in (
            "first_question",
            "first_answer",
            "follow_up",
            "standalone_query",
        ):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(
                    f"Multi-turn case {case_id} has invalid {field}."
                )
        for field in ("required_query_terms", "expected_keywords"):
            if not isinstance(case[field], list) or not all(
                isinstance(value, str) and value.strip()
                for value in case[field]
            ):
                raise ValueError(
                    f"Multi-turn case {case_id} has invalid {field}."
                )
        if not isinstance(case["expected_sources"], list) or not set(
            case["expected_sources"]
        ).issubset(source_set):
            raise ValueError(
                f"Multi-turn case {case_id} references an unknown source."
            )

    fallback_ids = set()
    required_fallback_fields = {
        "category",
        "history",
        "question",
        "expected_standalone_query",
        "expected_status",
        "expected_fallback_used",
    }
    for case in fallback_cases:
        case_id = case.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in fallback_ids
        ):
            raise ValueError(
                "Fallback case IDs must be non-empty and unique."
            )
        fallback_ids.add(case_id)
        if not required_fallback_fields.issubset(case):
            raise ValueError(
                f"Fallback case {case_id} is missing a required field."
            )
        if case["category"] not in {
            "lexical-boundary",
            "deterministic-fallback",
        }:
            raise ValueError(
                f"Fallback case {case_id} has an invalid category."
            )
        if not isinstance(case["history"], list):
            raise ValueError(
                f"Fallback case {case_id} has invalid history."
            )
        for history_turn in case["history"]:
            ConversationTurn.model_validate(history_turn)
        for field in ("question", "expected_standalone_query"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(
                    f"Fallback case {case_id} has invalid {field}."
                )
        if case["expected_status"] not in {
            "not_needed",
            "fallback",
            "unresolved",
        }:
            raise ValueError(
                f"Fallback case {case_id} has invalid expected_status."
            )
        if not isinstance(case["expected_fallback_used"], bool):
            raise ValueError(
                f"Fallback case {case_id} has invalid fallback flag."
            )


def _load_light_rag_core():
    """Import the production retriever without loading environment files."""
    dotenv = importlib.import_module("dotenv")
    original_load_dotenv = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        return importlib.import_module("backend.light_rag_core")
    finally:
        dotenv.load_dotenv = original_load_dotenv


@contextmanager
def offline_knowledge_base(dataset: dict):
    """Build the real in-memory light index from deterministic fixture pages."""
    light_rag_core = _load_light_rag_core()
    documents_by_source = {
        document["source"]: document for document in dataset["documents"]
    }
    previous_state = dict(light_rag_core._knowledge_bases)
    reader_factory = lambda path: _FixturePdfReader(path, documents_by_source)

    try:
        with TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                light_rag_core,
                "DATA_DIR",
                Path(temp_dir) / "data",
            ):
                with mock.patch.object(
                    light_rag_core, "PdfReader", side_effect=reader_factory
                ):
                    page_count, chunk_count = light_rag_core.build_knowledge_base(
                        [Path(source) for source in documents_by_source]
                    )
                yield light_rag_core, page_count, chunk_count
    finally:
        light_rag_core._knowledge_bases.clear()
        light_rag_core._knowledge_bases.update(previous_state)


def _normalize_result(document, score: float) -> dict:
    return {
        "source": document.metadata.get("source"),
        "page": document.metadata.get("page"),
        "content": document.page_content,
        "score": score,
    }


def _has_complete_metadata(item: dict) -> bool:
    return bool(
        isinstance(item.get("source"), str)
        and item["source"]
        and isinstance(item.get("page"), int)
        and not isinstance(item["page"], bool)
        and item["page"] >= 0
        and isinstance(item.get("content"), str)
        and item["content"].strip()
        and isinstance(item.get("score"), (int, float))
        and not isinstance(item["score"], bool)
        and math.isfinite(item["score"])
    )


class _FixtureSummarizer:
    def summarize(
        self,
        turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        return ""


class _FailingQueryRewriter:
    def rewrite(
        self,
        current_question: str,
        summary: str,
        recent_turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        raise RuntimeError("offline deterministic fallback")


def evaluate_multi_turn_cases(
    light_rag_core,
    cases: list[dict],
    top_k: int,
) -> list[dict]:
    results = []
    for case in cases:
        manager = ConversationContextManager(
            summarizer=_FixtureSummarizer(),
            query_rewriter=_FailingQueryRewriter(),
        )
        context = manager.process(
            current_question=case["follow_up"],
            history=[
                ConversationTurn(
                    role="user",
                    content=case["first_question"],
                ),
                ConversationTurn(
                    role="assistant",
                    content=case["first_answer"],
                ),
            ],
            conversation_id=f"evaluation-{case['id']}",
            options=ContextOptions(),
        )
        scored_documents = light_rag_core.retrieve_docs(
            context.standalone_query,
            k=top_k,
        )
        normalized_results = [
            _normalize_result(document, score)
            for document, score in scored_documents
        ]
        combined_content = "\n".join(
            item["content"] for item in normalized_results
        )
        ranked_sources = [
            item["source"] for item in normalized_results[:3]
        ]
        result = {
            "id": case["id"],
            "standalone_query": context.standalone_query,
            "query_rewrite_status": (
                context.metadata.query_rewrite_status
            ),
            "fallback_used": context.metadata.fallback_used,
            "query_terms_present": all(
                term.lower() in context.standalone_query.lower()
                for term in case["required_query_terms"]
            ),
            "source_hit": bool(
                set(case["expected_sources"]).intersection(ranked_sources)
            ),
            "keyword_match": all(
                keyword in combined_content
                for keyword in case["expected_keywords"]
            ),
            "metadata_complete": bool(normalized_results)
            and all(
                _has_complete_metadata(item)
                for item in normalized_results
            ),
            "actual_refuse": not light_rag_core.has_relevant_docs(
                scored_documents
            ),
            "results": normalized_results,
        }
        result["passed"] = bool(
            result["standalone_query"] == case["standalone_query"]
            and result["query_terms_present"]
            and result["source_hit"]
            and result["keyword_match"]
            and result["metadata_complete"]
            and not result["actual_refuse"]
            and result["query_rewrite_status"] == "fallback"
            and result["fallback_used"]
        )
        results.append(result)
    return results


def evaluate_fallback_cases(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        manager = ConversationContextManager(
            summarizer=_FixtureSummarizer(),
            query_rewriter=_FailingQueryRewriter(),
        )
        context = manager.process(
            current_question=case["question"],
            history=[
                ConversationTurn.model_validate(turn)
                for turn in case["history"]
            ],
            conversation_id=f"evaluation-{case['id']}",
            options=ContextOptions(),
        )
        result = {
            "id": case["id"],
            "category": case["category"],
            "standalone_query": context.standalone_query,
            "query_rewrite_status": (
                context.metadata.query_rewrite_status
            ),
            "fallback_used": context.metadata.fallback_used,
        }
        result["passed"] = bool(
            result["standalone_query"]
            == case["expected_standalone_query"]
            and result["query_rewrite_status"]
            == case["expected_status"]
            and result["fallback_used"]
            == case["expected_fallback_used"]
        )
        results.append(result)
    return results


def calculate_metrics(cases: list[dict], case_results: list[dict]) -> dict:
    results_by_id = {result["id"]: result for result in case_results}
    answerable_cases = [case for case in cases if not case["should_refuse"]]

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_total = 0.0
    correct_refusal_decisions = 0
    refused_out_of_scope = 0
    refusal_cases = [case for case in cases if case["should_refuse"]]
    complete_metadata = 0
    total_results = 0

    for case in cases:
        result = results_by_id[case["id"]]
        actual_refuse = result["actual_refuse"]
        if actual_refuse == case["should_refuse"]:
            correct_refusal_decisions += 1
        if case["should_refuse"] and actual_refuse:
            refused_out_of_scope += 1

        for item in result["results"]:
            total_results += 1
            if _has_complete_metadata(item):
                complete_metadata += 1

        if case["should_refuse"]:
            continue
        expected_sources = set(case["expected_sources"])
        ranked_sources = [item["source"] for item in result["results"]]
        if ranked_sources and ranked_sources[0] in expected_sources:
            hit_at_1 += 1
        if expected_sources.intersection(ranked_sources[:3]):
            hit_at_3 += 1
        for rank, source in enumerate(ranked_sources, start=1):
            if source in expected_sources:
                reciprocal_rank_total += 1.0 / rank
                break

    answerable_count = len(answerable_cases)
    refusal_count = len(refusal_cases)
    return {
        "hit_rate_at_1": hit_at_1 / answerable_count if answerable_count else 0.0,
        "hit_rate_at_3": hit_at_3 / answerable_count if answerable_count else 0.0,
        "mrr": (
            reciprocal_rank_total / answerable_count if answerable_count else 0.0
        ),
        "metadata_completeness": (
            complete_metadata / total_results if total_results else 0.0
        ),
        "refusal_accuracy": (
            refused_out_of_scope / refusal_count if refusal_count else 0.0
        ),
        "decision_accuracy": (
            correct_refusal_decisions / len(cases) if cases else 0.0
        ),
    }


def calibrate_relevance_threshold(
    cases: list[dict],
    case_results: list[dict],
) -> dict:
    labels_and_scores = []
    for case, result in zip(cases, case_results):
        if not result["results"]:
            continue
        labels_and_scores.append(
            (
                not case["should_refuse"],
                float(result["results"][0]["score"]),
            )
        )
    if not labels_and_scores:
        return {"recommended_threshold": None, "accuracy": 0.0}

    scores = sorted({score for _, score in labels_and_scores})
    candidates = [0.0, *scores, 1.0]
    ranked = []
    for threshold in candidates:
        correct = sum(
            (score <= threshold) == should_accept
            for should_accept, score in labels_and_scores
        )
        ranked.append((correct / len(labels_and_scores), threshold))
    accuracy, threshold = max(ranked, key=lambda item: (item[0], -item[1]))
    return {
        "recommended_threshold": threshold,
        "accuracy": accuracy,
    }


def evaluate_once(dataset: dict, top_k: int = 3) -> dict:
    case_results = []
    with offline_knowledge_base(dataset) as (
        light_rag_core,
        page_count,
        chunk_count,
    ):
        for case in dataset["cases"]:
            scored_documents = light_rag_core.retrieve_docs(
                case["question"], k=top_k
            )
            results = [
                _normalize_result(document, score)
                for document, score in scored_documents
            ]
            combined_content = "\n".join(item["content"] for item in results)
            case_results.append(
                {
                    "id": case["id"],
                    "actual_refuse": not light_rag_core.has_relevant_docs(
                        scored_documents
                    ),
                    "keyword_match": all(
                        keyword in combined_content
                        for keyword in case["expected_keywords"]
                    ),
                    "results": results,
                }
            )
        multi_turn_results = evaluate_multi_turn_cases(
            light_rag_core,
            dataset["multi_turn_cases"],
            top_k,
        )
    fallback_results = evaluate_fallback_cases(
        dataset["fallback_cases"]
    )

    metrics = calculate_metrics(dataset["cases"], case_results)
    metrics["multi_turn_accuracy"] = (
        sum(result["passed"] for result in multi_turn_results)
        / len(multi_turn_results)
        if multi_turn_results
        else 0.0
    )
    deterministic_results = multi_turn_results + [
        result
        for result, case in zip(
            fallback_results,
            dataset["fallback_cases"],
        )
        if case["expected_fallback_used"]
    ]
    metrics["deterministic_fallback_accuracy"] = (
        sum(result["passed"] for result in deterministic_results)
        / len(deterministic_results)
        if deterministic_results
        else 0.0
    )
    lexical_results = [
        result
        for result in fallback_results
        if result["category"] == "lexical-boundary"
    ]
    metrics["lexical_boundary_accuracy"] = (
        sum(result["passed"] for result in lexical_results)
        / len(lexical_results)
        if lexical_results
        else 0.0
    )
    return {
        "document_count": len(dataset["documents"]),
        "case_count": len(dataset["cases"]),
        "multi_turn_case_count": len(dataset["multi_turn_cases"]),
        "fallback_case_count": len(dataset["fallback_cases"]),
        "page_count": page_count,
        "chunk_count": chunk_count,
        "case_results": case_results,
        "multi_turn_results": multi_turn_results,
        "fallback_results": fallback_results,
        "metrics": metrics,
        "threshold_calibration": calibrate_relevance_threshold(
            dataset["cases"],
            case_results,
        ),
    }


def _stability_signature(report: dict) -> tuple:
    single_turn_signature = tuple(
        (
            case_result["id"],
            case_result["actual_refuse"],
            tuple(
                (
                    item["source"],
                    item["page"],
                    round(item["score"], 10),
                )
                for item in case_result["results"]
            ),
        )
        for case_result in report["case_results"]
    )
    multi_turn_signature = tuple(
        (
            case_result["id"],
            case_result["standalone_query"],
            case_result["actual_refuse"],
            tuple(
                (
                    item["source"],
                    item["page"],
                    round(item["score"], 10),
                )
                for item in case_result["results"]
            ),
        )
        for case_result in report["multi_turn_results"]
    )
    fallback_signature = tuple(
        (
            case_result["id"],
            case_result["standalone_query"],
            case_result["query_rewrite_status"],
            case_result["fallback_used"],
        )
        for case_result in report["fallback_results"]
    )
    return (
        single_turn_signature,
        multi_turn_signature,
        fallback_signature,
    )


def run_evaluation(
    dataset_path: Path = DATASET_PATH, repeat_runs: int = 2
) -> dict:
    if repeat_runs < 2:
        raise ValueError("repeat_runs must be at least 2 to verify stability.")
    dataset = load_dataset(dataset_path)
    reports = [evaluate_once(dataset) for _ in range(repeat_runs)]
    report = reports[0]
    first_signature = _stability_signature(report)
    report["stable"] = all(
        _stability_signature(candidate) == first_signature
        for candidate in reports[1:]
    )
    report["gates_passed"] = quality_gates_pass(report)
    return report


def quality_gates_pass(report: dict) -> bool:
    metrics = report["metrics"]
    return bool(report.get("stable")) and all(
        metrics[metric] >= threshold
        for metric, threshold in QUALITY_GATES.items()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic light-RAG retrieval evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to a production-like evaluation dataset JSON file.",
    )
    args = parser.parse_args(list(argv or []))
    report = run_evaluation(args.dataset)
    metrics = report["metrics"]
    print(
        "Offline RAG evaluation: "
        f"{report['document_count']} documents, "
        f"{report['case_count']} cases, "
        f"{report.get('multi_turn_case_count', 0)} multi-turn cases, "
        f"{report.get('fallback_case_count', 0)} fallback cases, "
        f"{report['chunk_count']} chunks"
    )
    print(f"Hit Rate@1: {metrics['hit_rate_at_1']:.3f}")
    print(f"Hit Rate@3: {metrics['hit_rate_at_3']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(
        "Source metadata completeness: "
        f"{metrics['metadata_completeness']:.3f}"
    )
    print(f"Refusal accuracy: {metrics['refusal_accuracy']:.3f}")
    print(f"Relevance decision accuracy: {metrics['decision_accuracy']:.3f}")
    calibration = report["threshold_calibration"]
    print(
        "Recommended light distance threshold: "
        f"{calibration['recommended_threshold']:.4f} "
        f"(calibration accuracy {calibration['accuracy']:.3f})"
    )
    print(
        "Multi-turn follow-up accuracy: "
        f"{metrics.get('multi_turn_accuracy', 0.0):.3f}"
    )
    print(
        "Deterministic fallback accuracy: "
        f"{metrics.get('deterministic_fallback_accuracy', 0.0):.3f}"
    )
    print(
        "Lexical boundary accuracy: "
        f"{metrics.get('lexical_boundary_accuracy', 0.0):.3f}"
    )
    print(f"Stable across repeated runs: {'yes' if report['stable'] else 'no'}")
    print(f"Quality gates: {'PASS' if report['gates_passed'] else 'FAIL'}")
    return 0 if report["gates_passed"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
