"""Benchmark-only overlays and reports for retrieval lifecycle traces."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from backend.retrieval.section import normalize_section


SURVIVAL_STAGES = ("RETRIEVAL", "SCOPE", "RRF", "SECTION_MERGE", "BUDGET", "RERANK", "FINAL")
FAILURE_STAGE = {
    "RETRIEVAL": "RETRIEVAL",
    "SCOPE": "SCOPE",
    "RRF": "RETRIEVAL",
    "SECTION_MERGE": "SECTION",
    "BUDGET": "BUDGET",
    "RERANK": "RERANK",
    "FINAL": "RERANK",
}


def _matches_model(candidate_model: str, expected_model: str) -> bool:
    candidate = candidate_model.strip().casefold()
    expected = expected_model.strip().casefold()
    return bool(candidate and expected and (candidate == expected or expected in candidate))


def overlay_relevance(trace: dict, query: dict) -> dict:
    """Add private labels to a copied trace; labels never enter ranking code."""
    result = deepcopy(trace or {})
    relevant = set(query.get("relevant_chunk_ids", []))
    expected_section = normalize_section(query.get("expected_section", ""))
    expected_model = str(query.get("expected_model", ""))
    by_id = {}
    for candidate in result.get("candidates", []):
        candidate["is_relevant"] = candidate.get("chunk_id") in relevant
        candidate["is_expected_section"] = (
            normalize_section(candidate.get("section", "")) == expected_section
            if expected_section else None
        )
        candidate["is_expected_model"] = (
            _matches_model(candidate.get("equipment_model", ""), expected_model)
            if expected_model else None
        )
        by_id[candidate.get("chunk_id", "")] = candidate
    for displacement in result.get("displacements", []):
        displaced = by_id.get(displacement.get("displaced_chunk", ""))
        replacement = by_id.get(displacement.get("replacement_chunk", ""))
        displaced_relevant = displaced.get("is_relevant") if displaced else None
        replacement_relevant = replacement.get("is_relevant") if replacement else None
        displacement["displaced_relevant"] = displaced_relevant
        displacement["replacement_relevant"] = replacement_relevant
        if displaced_relevant is True and replacement_relevant is False:
            classification = "HARMFUL_DISPLACEMENT"
        elif displaced_relevant is False and replacement_relevant is True:
            classification = "BENEFICIAL_DISPLACEMENT"
        elif displaced_relevant is not None and replacement_relevant is not None:
            classification = "NEUTRAL_DISPLACEMENT"
        else:
            classification = "UNKNOWN_DISPLACEMENT"
        displacement["classification"] = classification
    return result


def _rank_at_stage(trace: dict, relevant: set[str], stage: str) -> int | None:
    for rank, chunk_id in enumerate(trace.get("stage_chunk_ids", {}).get(stage, []), start=1):
        if chunk_id in relevant:
            return rank
    return None


def query_trace_summary(query: dict, trace: dict) -> dict:
    relevant = set(query.get("relevant_chunk_ids", []))
    ranks = {stage: _rank_at_stage(trace, relevant, stage) for stage in SURVIVAL_STAGES}
    first_failure = "SUCCESS"
    if query.get("answerable") and ranks["FINAL"] is None:
        for index, stage in enumerate(SURVIVAL_STAGES):
            if ranks[stage] is None and all(ranks[later] is None for later in SURVIVAL_STAGES[index:]):
                first_failure = FAILURE_STAGE[stage]
                break
    candidates = {item.get("chunk_id", ""): item for item in trace.get("candidates", [])}
    relevant_candidates = [candidates[item] for item in relevant if item in candidates]
    drop_reasons = sorted({
        item.get("drop_reason") or "UNKNOWN_DROP_REASON"
        for item in relevant_candidates
        if not item.get("final_selected")
    })
    if query.get("answerable") and not relevant_candidates:
        drop_reasons = ["NOT_RETRIEVED"]
    replacements = [
        item for item in trace.get("displacements", [])
        if item.get("displaced_chunk") in relevant
    ]
    return {
        "query_id": query.get("query_id", ""),
        "query": query.get("query", ""),
        "query_intent": trace.get("query_intent", ""),
        "query_identity": trace.get("query_identity", {}),
        "identifiers": trace.get("identifiers", []),
        "section_hint": trace.get("section_hint", {}),
        "requested_scope": trace.get("scope", {}).get("requested_scope", ""),
        "effective_scope": trace.get("scope", {}).get("effective_scope", ""),
        "candidate_counts_by_stage": trace.get("candidate_counts_by_stage", {}),
        "relevant_rank_by_stage": ranks,
        "first_failure_stage": first_failure,
        "drop_reasons": drop_reasons,
        "replacements": replacements,
        "identifier_protection": trace.get("identifier_protection", {}),
    }


def _survival(queries: list[dict], traces: dict[str, dict]) -> dict:
    answerable = [item for item in queries if item.get("answerable")]
    return {
        stage: (
            sum(
                _rank_at_stage(traces[item["query_id"]], set(item.get("relevant_chunk_ids", [])), stage)
                is not None
                for item in answerable
            ) / len(answerable)
            if answerable else 0.0
        )
        for stage in SURVIVAL_STAGES
    }


def _section_candidate_metrics(traces: dict[str, dict]) -> dict:
    selected = []
    rejected = []
    for trace in traces.values():
        for candidate in trace.get("candidates", []):
            if not candidate.get("section_expanded"):
                continue
            if candidate.get("budget_selected"):
                selected.append(candidate)
            elif candidate.get("budget_selected") is False:
                rejected.append(candidate)
    relevant_selected = sum(item.get("is_relevant") is True for item in selected)
    expected_selected = sum(item.get("is_expected_section") is True for item in selected)
    return {
        "selected_expanded_candidates": len(selected),
        "rejected_expanded_candidates": len(rejected),
        "relevant_precision": relevant_selected / len(selected) if selected else 0.0,
        "expected_section_precision": expected_selected / len(selected) if selected else 0.0,
        "relevant_candidates_rejected": sum(item.get("is_relevant") is True for item in rejected),
        "irrelevant_candidates_selected": sum(item.get("is_relevant") is False for item in selected),
    }


def analyze_observability(queries: list[dict], baseline_report: dict, section_report: dict) -> dict:
    query_by_id = {item["query_id"]: item for item in queries}
    baseline_rows = {item["query_id"]: item for item in baseline_report.get("rows", [])}
    section_rows = {item["query_id"]: item for item in section_report.get("rows", [])}
    baseline_traces = {
        query_id: overlay_relevance(row.get("trace", {}), query_by_id[query_id])
        for query_id, row in baseline_rows.items()
    }
    section_traces = {
        query_id: overlay_relevance(row.get("trace", {}), query_by_id[query_id])
        for query_id, row in section_rows.items()
    }
    query_summaries = [
        query_trace_summary(query, section_traces[query["query_id"]])
        for query in queries
    ]
    displacements = [
        {"query_id": query_id, **item}
        for query_id, trace in section_traces.items()
        for item in trace.get("displacements", [])
    ]
    displacement_counts = Counter(item["classification"] for item in displacements)
    counterfactual = Counter()
    counterfactual_rows = []
    regression_rows = []
    exact_model_regressions = []
    for query in queries:
        if not query.get("answerable"):
            continue
        query_id = query["query_id"]
        relevant = set(query.get("relevant_chunk_ids", []))
        before_rank = _rank_at_stage(baseline_traces[query_id], relevant, "FINAL")
        after_rank = _rank_at_stage(section_traces[query_id], relevant, "FINAL")
        if after_rank is not None and (before_rank is None or after_rank < before_rank):
            outcome = "SECTION_CAUSED_IMPROVEMENT"
        elif before_rank is not None and (after_rank is None or after_rank > before_rank):
            outcome = "SECTION_CAUSED_REGRESSION"
        else:
            outcome = "SECTION_NO_EFFECT"
        counterfactual[outcome] += 1
        counterfactual_rows.append({
            "query_id": query_id,
            "baseline_relevant_rank": before_rank,
            "section_relevant_rank": after_rank,
            "outcome": outcome,
        })
        if before_rank is not None and after_rank is None:
            summary = next(item for item in query_summaries if item["query_id"] == query_id)
            trace_candidates = {
                item.get("chunk_id", ""): item for item in section_traces[query_id].get("candidates", [])
            }
            relevant_candidate = next((trace_candidates[item] for item in relevant if item in trace_candidates), None)
            replacement = summary["replacements"][0] if summary["replacements"] else None
            row = {
                "query_id": query_id,
                "query": query.get("query", ""),
                "expected_model": query.get("expected_model", ""),
                "baseline_relevant_rank": before_rank,
                "section_relevant_rank": after_rank,
                "first_failure_stage": summary["first_failure_stage"],
                "drop_reason": (
                    relevant_candidate.get("drop_reason") if relevant_candidate else "NOT_RETRIEVED"
                ),
                "replacement_candidate": replacement,
                "scope": section_traces[query_id].get("scope", {}),
                "relevant_candidate": relevant_candidate,
            }
            regression_rows.append(row)
            if query.get("expected_model"):
                exact_model_regressions.append(row)
    section_candidates = _section_candidate_metrics(section_traces)
    relevant_rejected = sum(
        candidate.get("is_relevant") is True and candidate.get("budget_selected") is False
        for trace in section_traces.values()
        for candidate in trace.get("candidates", [])
    )
    reranker_drops = sum(
        candidate.get("is_relevant") is True and candidate.get("drop_reason") == "RERANK_TRUNCATED"
        for trace in section_traces.values()
        for candidate in trace.get("candidates", [])
    )
    scope_drops = sum(
        candidate.get("is_relevant") is True and candidate.get("drop_reason") == "SCOPE_FILTERED"
        for trace in section_traces.values()
        for candidate in trace.get("candidates", [])
    )
    return {
        "relevant_candidate_survival": {
            "baseline": _survival(queries, baseline_traces),
            "section": _survival(queries, section_traces),
        },
        "first_failure_stage": dict(Counter(
            item["first_failure_stage"] for item in query_summaries
            if query_by_id[item["query_id"]].get("answerable")
        )),
        "displacement": {
            "beneficial": displacement_counts["BENEFICIAL_DISPLACEMENT"],
            "neutral": displacement_counts["NEUTRAL_DISPLACEMENT"],
            "harmful": displacement_counts["HARMFUL_DISPLACEMENT"],
            "unknown": displacement_counts["UNKNOWN_DISPLACEMENT"],
            "rows": displacements,
        },
        "section_candidate_precision": section_candidates,
        "budget_reject_relevant_count": relevant_rejected,
        "reranker_drop_relevant_count": reranker_drops,
        "scope_drop_relevant_count": scope_drops,
        "counterfactual": {**dict(counterfactual), "rows": counterfactual_rows},
        "regression_queries": regression_rows,
        "exact_model_regression_queries": exact_model_regressions,
        "query_traces": query_summaries,
        "baseline_candidate_traces": baseline_traces,
        "section_candidate_traces": section_traces,
    }


def write_trace_artifacts(directory: Path, reports: dict[str, dict]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    candidate_payload = {
        name: {
            "baseline": report.get("baseline_candidate_traces", {}),
            "section": report.get("section_candidate_traces", {}),
        }
        for name, report in reports.items()
    }
    query_payload = {name: report.get("query_traces", []) for name, report in reports.items()}
    displacement_payload = {
        name: {
            "displacement": report.get("displacement", {}),
            "counterfactual": report.get("counterfactual", {}),
            "section_candidate_precision": report.get("section_candidate_precision", {}),
        }
        for name, report in reports.items()
    }
    payloads = {
        "candidate_trace": candidate_payload,
        "query_trace": query_payload,
        "displacement_report": displacement_payload,
    }
    paths = {}
    for name, payload in payloads.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = str(path)
    return paths
