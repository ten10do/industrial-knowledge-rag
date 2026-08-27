"""V3.77 Aligned Benchmark Schema (``V377_ALIGNED_BENCHMARK_V2``).

Defines the corpus-grounded support states used by the benchmark–corpus
alignment rebuild, the derivation rule from support state to gold decision,
and structural invariant validators.

Machine properties (contract):

    BENCHMARK_GOLD_INDEPENDENT_OF_PREDICTION = TRUE
    QUERY_TEXT_V2 MUST EQUAL QUERY_TEXT_V1
    CORPUS_UNSUPPORTED => EXPECTED_DECISION = ABSTAIN
    SUPPORTED_ANSWER   => EXPECTED_DECISION = ANSWER (+ >=1 anchored claim)

The schema reuses ``EvidenceExpectation`` / ``ExpectedDecision`` /
``ExpectedClaim`` from the frozen contract evaluator
(``backend.evaluation.contract_eval_v367``); it only adds the support-state
layer. Out-of-domain queries keep their own classification
(``QueryDomain.GENERIC_OUT_OF_DOMAIN``) and are strictly separated from
``CorpusSupportState.CORPUS_UNSUPPORTED`` (in-domain but not covered by the
frozen corpus).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


BENCHMARK_VERSION = "V377_ALIGNED_BENCHMARK_V2"


class CorpusSupportState(str, Enum):
    """Corpus-grounded answerability of one benchmark case."""

    SUPPORTED_ANSWER = "SUPPORTED_ANSWER"
    CORPUS_UNSUPPORTED = "CORPUS_UNSUPPORTED"
    AMBIGUOUS_CORPUS_SUPPORT = "AMBIGUOUS_CORPUS_SUPPORT"
    INVALID_GOLD_ANNOTATION = "INVALID_GOLD_ANNOTATION"


class QueryDomain(str, Enum):
    """Domain classification; orthogonal to corpus support."""

    INDUSTRIAL_IN_DOMAIN = "INDUSTRIAL_IN_DOMAIN"
    GENERIC_OUT_OF_DOMAIN = "GENERIC_OUT_OF_DOMAIN"


# Derivation rule (single source of truth) applied WITHIN a query domain.
DECISION_FOR_SUPPORT_STATE = {
    CorpusSupportState.SUPPORTED_ANSWER: "ANSWER",
    CorpusSupportState.CORPUS_UNSUPPORTED: "ABSTAIN",
    CorpusSupportState.AMBIGUOUS_CORPUS_SUPPORT: None,
    CorpusSupportState.INVALID_GOLD_ANNOTATION: None,
}

# Out-of-domain queries abstain BY DOMAIN, independent of corpus support.
# This keeps OOD strictly separated from CORPUS_UNSUPPORTED at every level.
DECISION_FOR_QUERY_DOMAIN = {
    QueryDomain.GENERIC_OUT_OF_DOMAIN: "ABSTAIN",
    QueryDomain.INDUSTRIAL_IN_DOMAIN: None,
}


def derive_expected_decision(
    state: CorpusSupportState | str,
    domain: QueryDomain | str = QueryDomain.INDUSTRIAL_IN_DOMAIN,
) -> str | None:
    """Gold decision derived ONLY from domain + corpus-support state."""
    domain_decision = DECISION_FOR_QUERY_DOMAIN[QueryDomain(domain)]
    if domain_decision is not None:
        return domain_decision
    return DECISION_FOR_SUPPORT_STATE[CorpusSupportState(state)]


@dataclass(frozen=True)
class AlignedBenchmarkCase:
    """One benchmark-V2 case: V1 identity + V2 corpus-grounded gold."""

    query_id: str
    query_text: str                       # byte-equal to V1
    slice_labels: tuple[str, ...]         # V1 slice continuity (private manifest)
    difficulty: str
    support_state: CorpusSupportState
    query_domain: QueryDomain
    expected_decision: str                # ANSWER | ABSTAIN (derived)
    support_reason: str                   # corpus-driven rationale (no predictions)
    anchors: tuple[dict, ...] = field(default_factory=tuple)
    # Anchors are private provenance: {"document", "page", "quote"}. They are
    # never committed to the public repository.
    claims: tuple[dict, ...] = field(default_factory=tuple)  # corpus-grounded V2 claims
    changed_from_v1: bool = False


def canonical_json(payload) -> str:
    """Deterministic serialization used for every hash freeze field."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_text_hash(query_texts: list[tuple[str, str]]) -> str:
    """sha256 over ``[(qid, text)]`` in listed order."""
    return sha256_text(canonical_json([[q, t] for q, t in query_texts]))


def validate_case(case: AlignedBenchmarkCase) -> list[str]:
    """Structural invariants for one case. Returns a list of violations."""
    problems: list[str] = []
    state = case.support_state
    expected = derive_expected_decision(state, case.query_domain)
    if expected is None:
        problems.append(
            f"{case.query_id}: support state {state} / domain {case.query_domain} "
            "has no gold decision"
        )
        return problems
    if case.expected_decision != expected:
        problems.append(
            f"{case.query_id}: expected_decision {case.expected_decision} != "
            f"derived {expected} for state {state} domain {case.query_domain}"
        )
    # OOD must be strictly separated from corpus-unsupported reporting.
    if (
        case.query_domain == QueryDomain.GENERIC_OUT_OF_DOMAIN
        and state != CorpusSupportState.CORPUS_UNSUPPORTED
    ):
        problems.append(f"{case.query_id}: OOD-domain cases must carry CORPUS_UNSUPPORTED state")
    if (
        case.query_domain == QueryDomain.GENERIC_OUT_OF_DOMAIN
        and case.anchors
    ):
        problems.append(f"{case.query_id}: OOD-domain cases must not claim support anchors")
    if state == CorpusSupportState.SUPPORTED_ANSWER:
        if case.query_domain != QueryDomain.INDUSTRIAL_IN_DOMAIN:
            problems.append(f"{case.query_id}: SUPPORTED_ANSWER requires in-domain query")
        if not case.anchors:
            problems.append(f"{case.query_id}: SUPPORTED_ANSWER requires >=1 anchor")
        if not case.claims:
            problems.append(f"{case.query_id}: SUPPORTED_ANSWER requires corpus-grounded claims")
    # Unchanged claims required when gold was not edited (V1 continuity guard is
    # performed by the builder using the V1 expectation claims).
    if (
        state == CorpusSupportState.CORPUS_UNSUPPORTED
        and case.query_domain == QueryDomain.INDUSTRIAL_IN_DOMAIN
        and not case.support_reason
    ):
        problems.append(f"{case.query_id}: in-domain CORPUS_UNSUPPORTED requires a support reason")
    if not case.support_reason:
        problems.append(f"{case.query_id}: every audited case needs a support reason")
    if not case.query_text or not case.query_text.strip():
        problems.append(f"{case.query_id}: empty query text")
    return problems


def validate_benchmark(cases: list[AlignedBenchmarkCase]) -> list[str]:
    """Benchmark-level invariants (deterministic ordering enforced)."""
    problems: list[str] = []
    ids = [c.query_id for c in cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate query ids")
    if cases != sorted(cases, key=lambda c: c.query_id):
        problems.append("cases are not sorted by query id")
    ambiguous = [c.query_id for c in cases if c.support_state == CorpusSupportState.AMBIGUOUS_CORPUS_SUPPORT]
    invalid = [c.query_id for c in cases if c.support_state == CorpusSupportState.INVALID_GOLD_ANNOTATION]
    if ambiguous:
        problems.append(f"AMBIGUOUS_CORPUS_SUPPORT present: {ambiguous}")
    if invalid:
        problems.append(f"INVALID_GOLD_ANNOTATION present: {invalid}")
    for case in cases:
        problems.extend(validate_case(case))
    return problems


def aggregate_counts(cases: list[AlignedBenchmarkCase]) -> dict:
    """Public-safe aggregates only (no query text, no anchors)."""
    support_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    slice_support: dict[str, dict[str, int]] = {}
    for case in cases:
        support_counts[case.support_state.value] = support_counts.get(case.support_state.value, 0) + 1
        domain_counts[case.query_domain.value] = domain_counts.get(case.query_domain.value, 0) + 1
        decision_counts[case.expected_decision] = decision_counts.get(case.expected_decision, 0) + 1
        bucket = slice_support.setdefault(case.slice_labels[0] if case.slice_labels else "", {})
        bucket[case.support_state.value] = bucket.get(case.support_state.value, 0) + 1
    return {
        "n_cases": len(cases),
        "support_states": dict(sorted(support_counts.items())),
        "query_domains": dict(sorted(domain_counts.items())),
        "expected_decisions": dict(sorted(decision_counts.items())),
        "slice_by_support": {k: dict(sorted(v.items())) for k, v in sorted(slice_support.items())},
    }
