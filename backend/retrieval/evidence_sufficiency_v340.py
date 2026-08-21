"""Coverage-relation evidence sufficiency candidate for V3.40.

V3.39 relaxed a soft Evidence abstention whenever one locally bounded window
lexically covered the proposition (coverage >= 0.72).  That recovered false
refusals but also relaxed high-similarity scope conflicts, because lexical
overlap alone cannot distinguish a same-parameter block from a sibling-model
block.  V3.40 models the Evidence COVERAGE RELATION explicitly and relaxes only
a safe relation type that carries an explicit textual anchor:

* DIRECT      - same parameter definition block (model + parameter terms + value/range)
* REFERENCED  - explicit same-section anchor or cross-section/manual reference link
* INHERITED   - explicit family-wide statement ("all models", "the entire family")
* DEPENDENT   - explicit module-parent compatibility or configuration dependency
* UNSUPPORTED - scope or value conflicts; never relaxed

Only this Evidence sufficiency layer changes.  Identity reasoning, retrieval,
parser, NLI judge, support, and open sufficiency are untouched.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence_sufficiency_v339 import EvidenceSufficiencyDecision
from .identity_reasoning import analyze_identity_aware_evidence
from .product_identity import normalize_identity_text


EVIDENCE_COVERAGE_CANDIDATE_VERSION = "evidence-v340-coverage-candidate"
EVIDENCE_COVERAGE_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

# Relax only when the modeled relation carries an explicit anchor and the
# confidence reaches this floor.
RELAX_CONFIDENCE_FLOOR = 0.75

_SOFT_REASONS = frozenset({
    "INSUFFICIENT_EVIDENCE",
    "PARTIAL_EVIDENCE_ONLY",
    "MISSING_ATTRIBUTE_EVIDENCE",
    "MISSING_VALUE_EVIDENCE",
    "MISSING_REQUIREMENT_EVIDENCE",
    "MISSING_ACTION_EVIDENCE",
    "MODEL_MISMATCH",
})

# ---------------------------------------------------------------------------
# Inherited guard patterns (identical semantics to the V3.39 guards)
# ---------------------------------------------------------------------------

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
_UNIT = re.compile(r"(?<![a-z0-9])(?:vdc|vac|mv|kv|v|ma|a|ms|s|mm|cm|m|hz|khz|mhz|mbit/s|bytes?|slots?|ports?|racks?|kw|w|nm|°c|°f)(?![a-z0-9])", re.IGNORECASE)
_WORD = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "between", "both", "can", "do", "does",
    "either", "for", "from", "has", "have", "in", "is", "it", "its", "may", "more",
    "most", "must", "no", "of", "on", "or", "per", "should", "than", "the", "their", "to",
    "use", "used", "using", "with", "within",
})
_GENERIC_IDENTITY = frozenset({"series", "system", "product", "unit", "manual", "controller"})

# ---------------------------------------------------------------------------
# Coverage-relation anchor patterns (new in V3.40)
# ---------------------------------------------------------------------------

# Family-wide quantifiers that license an INHERITED relation.
_FAMILY_ANCHOR = re.compile(
    r"\b(?:all\s+(?:other\s+)?(?:models?|variants?|versions?|units?|adapters?|drives?|modules?|types?|sizes?)"
    r"|(?:all|every|each|any|both)\s+(?:\w+\s+){0,3}(?:models?|variants?|versions?|units?|adapters?|drives?|modules?|types?|sizes?"
    r"|parameters?|sources?|functions?)"
    r"|the\s+(?:entire|whole|complete)\s+(?:family|series|range|line)"
    r"|(?:all|any)\s+(?:models?|units?)\s+of\s+the\s+(?:\w+\s+)?(?:family|series|range)"
    r"|may\s+be\s+added\s+to\s+all\s+models?|common\s+to\s+all)\b",
    re.IGNORECASE,
)
# Explicit reference markers that license a REFERENCED relation.
_REFERENCE_ANCHOR = re.compile(
    r"\b(?:refer(?:s|red|ring)?\s+(?:to|the\s+(?:\w+\s+)*document)|see\s+(?:the\s+)?(?:section|chapter|appendix|manual|document)"
    r"|described\s+in|detailed\s+in|documented\s+in|given\s+in|listed\s+in|specified\s+in"
    r"|in\s+the\s+(?:\w+\s+){0,3}(?:section|chapter|appendix|manual)"
    r"|user['’]?s?\s+manual|operating\s+manual|instruction\s+manual|cat\.?\s*no\.?|part\s+number)\b",
    re.IGNORECASE,
)
# Same-section anchors: the chunk itself is the referenced section content.
_SAME_SECTION_ANCHOR = re.compile(
    r"\b(?:this\s+section|this\s+chapter|in\s+this\s+section|as\s+described\s+(?:above|earlier)"
    r"|as\s+described\s+in\s+detail\s+below|described\s+in\s+detail\s+below|see\s+below"
    r"|as\s+shown\s+in|presented\s+in"
    r"|the\s+following\s+section|section\s+\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
# Module-parent compatibility / containment anchors (DEPENDENT relation).
_COMPATIBILITY_ANCHOR = re.compile(
    r"\b(?:supports?|supported|can\s+be\s+connected|can\s+be\s+mounted|can\s+be\s+used|can\s+be\s+added"
    r"|can\s+be\s+extended|compatible|accommodates?|accepts?|allows?|connect(?:ed)?\s+to"
    r"|mount(?:ed)?\s+(?:on|to|immediately)|install(?:ed)?\s+(?:on|to|immediately)|attach(?:ed)?)\b",
    re.IGNORECASE,
)
# Configuration-dependency anchors (DEPENDENT relation).
_DEPENDENCY_ANCHOR = re.compile(
    r"\b(?:requires?|required|must\s+be(?:\s+\w+){0,3}|only\s+(?:when|if|after|by)|must\s+first"
    r"|needs?\s+to\s+be|shall\s+be|is\s+needed|is\s+necessary\s+to|prerequisite|depends?\s+on|defined\s+in|configured\s+in)\b",
    re.IGNORECASE,
)
# Range/value assertion markers in the query (DIRECT_PARAMETER intent).
_VALUE_ASSERTION = re.compile(
    r"\b(?:range|rated|rating|default|maximum|minimum|max\.?|min\.?|up\s+to|at\s+least|at\s+most"
    r"|no\s+more\s+than|output\s+current|supply\s+voltage|operating\s+voltage|capacity|limit(?:ed)?\s+to)\b",
    re.IGNORECASE,
)


# Intents whose safety rests on an explicit anchor (lower coverage floor).
_TYPED_INTENTS = frozenset({
    "PRODUCT_FAMILY_INHERITANCE",
    "CROSS_SECTION_REFERENCE",
    "MODULE_PARENT_RELATION",
    "CONFIGURATION_DEPENDENCY",
})


@dataclass(frozen=True)
class EvidenceCoverageRelation:
    """Typed coverage relation between one query and one candidate claim."""

    relation: str                 # DIRECT | INHERITED | REFERENCED | DEPENDENT | UNSUPPORTED
    coverage_type: str            # benchmark relation label this window best matches
    confidence: float
    reason_code: str
    chunk_id: str = ""
    document_id: str = ""
    lexical_coverage: float = 0.0
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_terms"] = list(self.matched_terms)
        payload["missing_terms"] = list(self.missing_terms)
        payload["anchors"] = list(self.anchors)
        return payload


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
    if not token.endswith("s"):
        variants.add(token + "s")
        if token.endswith(("x", "sh", "ch", "o")):
            variants.add(token + "es")
    if token.endswith("y") and len(token) > 2:
        variants.add(token[:-1] + "ies")
    return tuple(variants)


def _contains_term(text: str, token: str) -> bool:
    if token == "direction":
        return "off to on" in text and "on to off" in text
    return any(re.search(rf"(?<![a-z0-9]){re.escape(item)}(?![a-z0-9])", text) for item in _variants(token))


def _contains_value(text: str, value: str) -> bool:
    return bool(re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", text))


def _contains_unit(text: str, unit: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(unit)}(?![a-z])", text))


def _windows(text: str, size: int = 16) -> list[tuple[str, list[str]]]:
    """Sliding windows over a document, keeping raw lines for scope checks.

    Each entry is ``(normalized_window, raw_lines)``; the normalized window is
    used for term/value matching, the raw lines (hyphens intact) for model-token
    detection inside multi-model tables.
    """
    raw_lines = [line for line in str(text or "").splitlines() if line.strip()]
    norm_lines = [_normalized(line) for line in raw_lines]
    if not norm_lines:
        return [(_normalized(text), [str(text or "")])]
    if len(norm_lines) <= size:
        return [("\n".join(norm_lines), raw_lines)]
    return [
        ("\n".join(norm_lines[index:index + size]), raw_lines[index:index + size])
        for index in range(len(norm_lines) - size + 1)
    ]


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


def _model_tokens(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normalized(item) for item in _MODEL_TOKEN.findall(text.casefold())))


def _anchors(window: str) -> tuple[str, ...]:
    found: list[str] = []
    if _FAMILY_ANCHOR.search(window):
        found.append("FAMILY_QUANTIFIER")
    if _REFERENCE_ANCHOR.search(window):
        found.append("REFERENCE_MARKER")
    if _SAME_SECTION_ANCHOR.search(window):
        found.append("SAME_SECTION_MARKER")
    if _COMPATIBILITY_ANCHOR.search(window):
        found.append("COMPATIBILITY_VERB")
    if _DEPENDENCY_ANCHOR.search(window):
        found.append("DEPENDENCY_VERB")
    return tuple(found)


def _classify_query(query: str) -> str:
    """Coarse query intent used to select the relation anchor to demand."""
    if _FAMILY_ANCHOR.search(_normalized(query)):
        return "PRODUCT_FAMILY_INHERITANCE"
    if _VALUE_ASSERTION.search(query) and _explicit_values(query):
        return "DIRECT_PARAMETER"
    if _REFERENCE_ANCHOR.search(query) or re.search(
            r"\b(?:section|chapter|appendix|manual|document|catalogue|catalog)\b", query, re.IGNORECASE):
        return "CROSS_SECTION_REFERENCE"
    if _DEPENDENCY_ANCHOR.search(query):
        return "CONFIGURATION_DEPENDENCY"
    if _COMPATIBILITY_ANCHOR.search(query):
        return "MODULE_PARENT_RELATION"
    return "SAME_SECTION_REFERENCE"


def _scope_conflict(
    query: str,
    window: str,
    raw_lines: list[str],
    values: tuple[str, ...],
    query_models: tuple[str, ...],
) -> str | None:
    """Detect sibling-model or foreign-range coverage (never relaxable).

    A value assertion is only covered when the queried model appears in the
    window text itself; counting metadata identity here is exactly how V3.39
    relaxed sibling-model blocks (its identity hard-negative FA regression).
    When the window is a multi-model table (several distinct model tokens),
    the asserted values must share a LINE with the queried model, so a value
    belonging to a sibling row can never license an answer.
    """
    if not values:
        return None
    lines = [line for line in window.split("\n") if line.strip()]
    if query_models and not any(
        _contains_term(line, item) for line in lines for item in query_models
    ):
        return "NEGATIVE_SCOPE_CONFLICT"
    window_models: set[str] = set()
    for line in raw_lines:
        window_models.update(_model_tokens(line))
    if len(window_models) >= 2 and query_models:
        co_present = any(
            all(_contains_term(line, item) for item in query_models)
            and all(_contains_value(_normalized(line), value) for value in values)
            for line in raw_lines
        )
        if not co_present:
            return "VALUE_SCOPE_CONFLICT"
        return None
    missing = [item for item in values if not _contains_value(window, item)]
    if missing:
        return "VALUE_SCOPE_CONFLICT"
    return None


def _relation_for_window(
    query: str,
    window: str,
    raw_lines: list[str],
    metadata_identity: str,
    terms: tuple[str, ...],
    values: tuple[str, ...],
    units: tuple[str, ...],
    intent: str,
    query_models: tuple[str, ...],
) -> EvidenceCoverageRelation:
    matched = tuple(item for item in terms if _contains_term(window, item) or _contains_term(metadata_identity, item))
    missing = tuple(item for item in terms if item not in matched)
    coverage = len(matched) / len(terms) if terms else 1.0
    anchors = _anchors(window)
    conflict = _scope_conflict(query, window, raw_lines, values, query_models)
    if conflict:
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.95, conflict, lexical_coverage=coverage,
            matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    if units and not all(_contains_unit(window, item) for item in units):
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.6, "VALUE_SCOPE_CONFLICT", lexical_coverage=coverage,
            matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    # Value assertions must co-occur with a proposition term on ONE physical
    # line: a correct number elsewhere in the window (another row, another
    # parameter) never licenses a direct answer.
    if intent == "DIRECT_PARAMETER" and values:
        line_ok = any(
            all(_contains_value(_normalized(line), value) for value in values)
            and any(_contains_term(_normalized(line), term) for term in matched)
            for line in raw_lines
        )
        if not line_ok:
            return EvidenceCoverageRelation(
                "UNSUPPORTED", intent, 0.9, "VALUE_SCOPE_CONFLICT", lexical_coverage=coverage,
                matched_terms=matched, missing_terms=missing, anchors=anchors,
            )
    # When every asserted value is present in the window, the values themselves
    # carry the proposition; the lexical floor drops accordingly (V3.39 kept
    # 0.72 even for fully-valued propositions and lost those rescues).
    floor = 0.60 if (intent == "DIRECT_PARAMETER" and values) else (0.60 if intent in _TYPED_INTENTS else 0.72)
    if coverage < floor:
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, round(coverage, 3), "LEXICAL_COVERAGE_INSUFFICIENT",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )

    # Relation-defining anchors decide the typed relation only after the
    # proposition itself is lexically covered by this window.  Typed relations
    # carry an explicit anchor, so their coverage floor is lower.
    if intent == "PRODUCT_FAMILY_INHERITANCE":
        if "FAMILY_QUANTIFIER" in anchors:
            return EvidenceCoverageRelation(
                "INHERITED", intent, 0.9, "FAMILY_INHERITANCE_SUPPORTED",
                lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
            )
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.85, "FAMILY_INHERITANCE_UNSUPPORTED",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    if intent == "CROSS_SECTION_REFERENCE":
        if "REFERENCE_MARKER" in anchors or "SAME_SECTION_MARKER" in anchors:
            return EvidenceCoverageRelation(
                "REFERENCED", intent, 0.88, "CROSS_SECTION_LINK_EXPLICIT",
                lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
            )
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.8, "CROSS_SECTION_LINK_MISSING",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    if intent == "MODULE_PARENT_RELATION":
        if "COMPATIBILITY_VERB" in anchors:
            return EvidenceCoverageRelation(
                "DEPENDENT", intent, 0.88, "MODULE_PARENT_SUPPORTED",
                lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
            )
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.8, "MODULE_PARENT_UNSUPPORTED",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    if intent == "CONFIGURATION_DEPENDENCY":
        if "DEPENDENCY_VERB" in anchors or "COMPATIBILITY_VERB" in anchors:
            return EvidenceCoverageRelation(
                "DEPENDENT", intent, 0.88, "CONFIGURATION_DEPENDENCY_SUPPORTED",
                lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
            )
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.8, "CONFIGURATION_DEPENDENCY_MISSING",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    # DIRECT_PARAMETER and SAME_SECTION_REFERENCE: a fully covered window with
    # matching values/units inside the same product scope is a direct block.
    if intent == "SAME_SECTION_REFERENCE" and not anchors:
        return EvidenceCoverageRelation(
            "UNSUPPORTED", intent, 0.7, "SECTION_ANCHOR_MISSING",
            lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
        )
    reason = "DIRECT_PARAMETER_SUPPORTED" if intent == "DIRECT_PARAMETER" else "SAME_SECTION_SUPPORTED"
    confidence = 0.8 + (0.1 if anchors else 0.0)
    return EvidenceCoverageRelation(
        "DIRECT", intent, round(confidence, 3), reason,
        lexical_coverage=coverage, matched_terms=matched, missing_terms=missing, anchors=anchors,
    )


def classify_coverage_relation(query: str, candidates: list[Any]) -> EvidenceCoverageRelation:
    """Find the best typed coverage relation across the retrieved candidate claims."""
    if _NEGATION_MARKERS.search(query):
        return EvidenceCoverageRelation("UNSUPPORTED", "NONE", 0.95, "NEGATED_OR_EXCLUSIVE_PROPOSITION")
    if _PROCEDURE_MARKERS.search(query):
        return EvidenceCoverageRelation("UNSUPPORTED", "NONE", 0.95, "PROCEDURE_RELAXATION_FORBIDDEN")

    query_norm = _normalized(query)
    model_tokens = _model_tokens(query)
    values = _explicit_values(query)
    units = tuple(dict.fromkeys(item.casefold() for item in _UNIT.findall(query_norm)))
    terms = _query_terms(query_norm, model_tokens)
    intent = _classify_query(query)
    best: EvidenceCoverageRelation | None = None

    for candidate in candidates:
        document = getattr(candidate, "document", candidate)
        metadata = getattr(document, "metadata", {}) or {}
        document_text = str(getattr(document, "page_content", "") or "")
        if _ADVERSE_SCOPE.search(document_text):
            continue
        metadata_identity = _normalized(" ".join(str(metadata.get(key, "")) for key in (
            "manufacturer", "product_family", "product_series", "equipment_model",
        )))
        for window, raw_lines in _windows(document_text):
            if _EVIDENCE_CONTRADICTION.search(window):
                continue
            identity_text = f"{window} {metadata_identity}"
            if model_tokens and not all(_contains_term(identity_text, item) for item in model_tokens):
                continue
            relation = _relation_for_window(
                query, window, raw_lines, metadata_identity, terms, values, units, intent, model_tokens,
            )
            relation = EvidenceCoverageRelation(
                relation.relation, relation.coverage_type, relation.confidence, relation.reason_code,
                chunk_id=str(metadata.get("chunk_id", "")),
                document_id=str(metadata.get("document_id", "")),
                lexical_coverage=relation.lexical_coverage,
                matched_terms=relation.matched_terms,
                missing_terms=relation.missing_terms,
                anchors=relation.anchors,
            )
            if best is None or relation.confidence > best.confidence:
                best = relation
            if relation.relation != "UNSUPPORTED" and relation.confidence >= RELAX_CONFIDENCE_FLOOR:
                return relation
    if best is None:
        return EvidenceCoverageRelation("UNSUPPORTED", intent, 0.0, "NO_LOCAL_CANDIDATE_RELATION")
    return best


def analyze_coverage_evidence_sufficiency(
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
    """Run the formal identity/mixed path, then apply the typed coverage candidate."""
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

    relation = classify_coverage_relation(query, list(getattr(result, "candidates", [])))
    safe = relation.relation in ("DIRECT", "INHERITED", "REFERENCED", "DEPENDENT")
    if safe and relation.confidence >= RELAX_CONFIDENCE_FLOOR:
        return EvidenceSufficiencyDecision(
            query, "ANSWER", relation.reason_code, "V340_COVERAGE", baseline.query_path,
            baseline.decision, base_reason, identity_status, True, relation.as_dict(), baseline_dict,
        )
    return EvidenceSufficiencyDecision(
        query, "ABSTAIN", relation.reason_code, baseline.final_decision_source, baseline.query_path,
        baseline.decision, base_reason, identity_status, False, relation.as_dict(), baseline_dict,
    )