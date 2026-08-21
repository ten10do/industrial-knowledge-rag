"""Conservative evidence-sufficiency candidate for V3.39.

The candidate is additive: it delegates to the unchanged V3.36 identity
boundary and V3.32 mixed Evidence path first.  It may reconsider only a soft
Evidence abstention when one locally bounded candidate claim explicitly covers
the verification proposition.  Identity conflicts, procedures, negation,
cross-document joins, and conflicting values are never relaxed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .identity_reasoning import analyze_identity_aware_evidence
from .product_identity import normalize_identity_text


EVIDENCE_SUFFICIENCY_CANDIDATE_VERSION = "evidence-v339-sufficiency-candidate"
EVIDENCE_SUFFICIENCY_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

_SOFT_REASONS = frozenset({
    "INSUFFICIENT_EVIDENCE",
    "PARTIAL_EVIDENCE_ONLY",
    "MISSING_ATTRIBUTE_EVIDENCE",
    "MISSING_VALUE_EVIDENCE",
    "MISSING_REQUIREMENT_EVIDENCE",
    "MISSING_ACTION_EVIDENCE",
    "MODEL_MISMATCH",
})
_PROCEDURE_MARKERS = re.compile(
    r"\b(?:before|after|first|next|then|procedure|step|remove|install|wire|wiring|load|download|exchange)\b",
    re.IGNORECASE,
)
_NEGATION_MARKERS = re.compile(
    r"\b(?:not|without|skip|bypass|regardless|instead|never|only)\b",
    re.IGNORECASE,
)
_EVIDENCE_CONTRADICTION = re.compile(
    r"\b(?:cannot|must\s+not|may\s+not|should\s+not|not\s+permitted|prohibited)\b",
    re.IGNORECASE,
)
_ADVERSE_SCOPE = re.compile(
    r"\b(?:unstable|insufficient\s+supplied\s+power|error\s+will\s+occur|may\s+occur|will\s+not\s+operate)\b",
    re.IGNORECASE,
)
_MODEL_TOKEN = re.compile(r"(?<![a-z0-9])(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+(?![a-z0-9])", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?(?![a-z0-9])", re.IGNORECASE)
_UNIT = re.compile(r"(?<![a-z0-9])(?:vdc|vac|mv|kv|v|ma|a|ms|s|mm|cm|m|hz|khz|mhz|mbit/s|bytes?|slots?|ports?|racks?)(?![a-z0-9])", re.IGNORECASE)
_WORD = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "between", "both", "can", "do", "does",
    "either", "for", "from", "has", "have", "in", "is", "it", "its", "may", "more",
    "most", "must", "no", "of", "on", "or", "per", "should", "than", "the", "their", "to",
    "use", "used", "using", "with", "within",
})
_GENERIC_IDENTITY = frozenset({"series", "system", "product", "unit", "manual", "controller"})


@dataclass(frozen=True)
class CompatibleEvidenceRelation:
    supported: bool
    reason_code: str
    chunk_id: str = ""
    document_id: str = ""
    relation: str = "NONE"
    lexical_coverage: float = 0.0
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_terms"] = list(self.matched_terms)
        payload["missing_terms"] = list(self.missing_terms)
        return payload


@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    query: str
    decision: str
    reason: str
    final_decision_source: str
    query_path: str
    baseline_decision: str
    baseline_reason: str
    identity_result: str
    relaxed: bool
    relation: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized(value: object) -> str:
    return " ".join(normalize_identity_text(value).split())


def _variants(token: str) -> tuple[str, ...]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
        variants.add(token[:-1])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return tuple(variants)


def _contains_term(text: str, token: str) -> bool:
    if token == "direction":
        return "off to on" in text and "on to off" in text
    return any(re.search(rf"(?<![a-z0-9]){re.escape(item)}(?![a-z0-9])", text) for item in _variants(token))


def _contains_value(text: str, value: str) -> bool:
    return bool(re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", text))


def _contains_unit(text: str, unit: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(unit)}(?![a-z])", text))


def _windows(text: str, size: int = 16) -> list[str]:
    lines = [_normalized(line) for line in str(text or "").splitlines() if _normalized(line)]
    if not lines:
        return [_normalized(text)]
    if len(lines) <= size:
        return [" ".join(lines)]
    return [" ".join(lines[index:index + size]) for index in range(len(lines) - size + 1)]


def _query_terms(query: str, model_tokens: tuple[str, ...] = ()) -> tuple[str, ...]:
    model_parts = {part for item in model_tokens for part in item.split()}
    terms = []
    for item in _WORD.findall(_normalized(query)):
        if item in _STOPWORDS or item in _GENERIC_IDENTITY or item in model_parts or len(item) < 3:
            continue
        terms.append(item)
    return tuple(dict.fromkeys(terms))


def _explicit_values(query: str) -> tuple[str, ...]:
    without_models = _normalized(_MODEL_TOKEN.sub(" ", query))
    return tuple(dict.fromkeys(_NUMBER.findall(without_models)))


def evaluate_compatible_evidence_relation(query: str, candidates: list[Any]) -> CompatibleEvidenceRelation:
    """Find one bounded candidate claim that explicitly covers a safe proposition."""
    if _NEGATION_MARKERS.search(query):
        return CompatibleEvidenceRelation(False, "NEGATED_OR_EXCLUSIVE_PROPOSITION")
    if _PROCEDURE_MARKERS.search(query):
        return CompatibleEvidenceRelation(False, "PROCEDURE_RELAXATION_FORBIDDEN")

    query_norm = _normalized(query)
    model_tokens = tuple(dict.fromkeys(_normalized(item) for item in _MODEL_TOKEN.findall(query.casefold())))
    values = _explicit_values(query)
    units = tuple(dict.fromkeys(item.casefold() for item in _UNIT.findall(query_norm)))
    terms = _query_terms(query_norm, model_tokens)
    best: CompatibleEvidenceRelation | None = None

    for candidate in candidates:
        document = getattr(candidate, "document", candidate)
        metadata = getattr(document, "metadata", {}) or {}
        document_text = str(getattr(document, "page_content", "") or "")
        if _ADVERSE_SCOPE.search(document_text):
            continue
        metadata_identity = _normalized(" ".join(str(metadata.get(key, "")) for key in (
            "manufacturer", "product_family", "product_series", "equipment_model",
        )))
        for window in _windows(document_text):
            if _EVIDENCE_CONTRADICTION.search(window):
                continue
            identity_text = f"{window} {metadata_identity}"
            if model_tokens and not all(_contains_term(identity_text, item) for item in model_tokens):
                continue
            if values and not all(_contains_value(window, item) for item in values):
                continue
            if units and not all(_contains_unit(window, item) for item in units):
                continue
            matched = tuple(item for item in terms if _contains_term(window, item) or _contains_term(metadata_identity, item))
            missing = tuple(item for item in terms if item not in matched)
            coverage = len(matched) / len(terms) if terms else 1.0
            relation = CompatibleEvidenceRelation(
                supported=coverage >= 0.72,
                reason_code="LOCAL_PROPOSITION_COVERED" if coverage >= 0.72 else "LEXICAL_COVERAGE_INSUFFICIENT",
                chunk_id=str(metadata.get("chunk_id", "")),
                document_id=str(metadata.get("document_id", "")),
                relation="SAME_PARAMETER_BLOCK" if values or units else "SAME_PRODUCT_CLAIM",
                lexical_coverage=coverage,
                matched_terms=matched,
                missing_terms=missing,
            )
            if best is None or relation.lexical_coverage > best.lexical_coverage:
                best = relation
            if relation.supported:
                return relation
    return best or CompatibleEvidenceRelation(False, "NO_LOCAL_CANDIDATE_RELATION")


def analyze_evidence_sufficiency(
    query: str,
    result,
    documents: list,
    retrieval_mode: str,
    *,
    judge: Any = None,
    policy: Any = None,
    identity_matching: bool = True,
    requirement: Any = None,
    apply_open_sufficiency: bool = True,
) -> EvidenceSufficiencyDecision:
    """Run the formal identity/mixed path, then apply the bounded DEV candidate."""
    baseline = analyze_identity_aware_evidence(
        query, result, documents, retrieval_mode,
        judge=judge,
        policy=policy,
        identity_matching=identity_matching,
        requirement=requirement,
        apply_open_sufficiency=apply_open_sufficiency,
    )
    baseline_dict = baseline.as_dict()
    boundary = baseline.identity_boundary or {}
    identity_status = str(boundary.get("status", "UNKNOWN"))
    existing = baseline.existing_evidence or {}
    base_reason = str(existing.get("base_rule_reason", baseline.reason))

    if baseline.decision == "ANSWER":
        return EvidenceSufficiencyDecision(
            query, "ANSWER", baseline.reason, baseline.final_decision_source, baseline.query_path,
            baseline.decision, base_reason, identity_status, False, baseline=baseline_dict,
        )
    if identity_status == "INCOMPATIBLE" or not baseline.delegated_to_existing_evidence:
        return EvidenceSufficiencyDecision(
            query, "ABSTAIN", "IDENTITY_BOUNDARY_PRESERVED", baseline.final_decision_source, baseline.query_path,
            baseline.decision, base_reason, identity_status, False, baseline=baseline_dict,
        )
    if baseline.query_path != "VERIFICATION" or base_reason not in _SOFT_REASONS:
        return EvidenceSufficiencyDecision(
            query, "ABSTAIN", "NON_RELAXABLE_EVIDENCE_REASON", baseline.final_decision_source, baseline.query_path,
            baseline.decision, base_reason, identity_status, False, baseline=baseline_dict,
        )

    relation = evaluate_compatible_evidence_relation(query, list(getattr(result, "candidates", [])))
    if relation.supported:
        return EvidenceSufficiencyDecision(
            query, "ANSWER", "COMPATIBLE_EVIDENCE_RELATION", "V339_SUFFICIENCY", baseline.query_path,
            baseline.decision, base_reason, identity_status, True, relation.as_dict(), baseline_dict,
        )
    return EvidenceSufficiencyDecision(
        query, "ABSTAIN", relation.reason_code, baseline.final_decision_source, baseline.query_path,
        baseline.decision, base_reason, identity_status, False, relation.as_dict(), baseline_dict,
    )
