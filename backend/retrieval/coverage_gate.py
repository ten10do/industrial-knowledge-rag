"""V3.85 coverage gate: out-of-corpus equipment detection (Phase C).

Q0021 ("Tell me about the SINAMICS G120 drive.") is annotated ABSTAIN / IDENTITY
because the frozen corpus (S7-1200, ACS580, M221) contains no SINAMICS G120
documentation. The identity-gate probe (scripts/v385_q0021_identity_probe.py)
established the root cause with runtime evidence:

* query-side identity extraction (product_identity.identities_from_query) is
  corpus-driven: it only resolves identities that already exist in the corpus,
  so "SINAMICS G120" resolves to an empty identity;
* has_query_identity becomes False and the known_identity / compatible_identity
  gates short-circuit to True;
* foreign_equipment_signal (technical.py) is vendor-grained: "siemens" IS in
  the corpus (S7-1200 manual), so the whole siemens group is skipped and the
  out-of-corpus model SINAMICS G120 never reaches a gate;
* the query falls through to the vector-distance branch and is answered.

This module adds the missing model-grained gate. It is an ADDITIVE pre-gate:
it does not modify backend/retrieval/evidence.py and therefore preserves the
frozen digest when disabled.

Design
------
* model mentions are detected with the same deterministic series patterns the
  document resolver uses (document_identity_v373.SERIES_PATTERNS) plus the
  concrete model tokens in VENDOR_EQUIPMENT_TERMS (brand words excluded);
* a mention is out-of-corpus when its normalized form is not among the
  corpus-known models (metadata equipment_model / product_series, aliases
  included);
* the gate reports (model, reason) only for detected out-of-corpus mentions.
  No mention detected -> no signal (conservative: miss > false positive).
"""

from __future__ import annotations

import re

from .document_identity_v373 import SERIES_PATTERNS
from .product_identity import identity_from_metadata, normalize_identity_text
from .technical import VENDOR_EQUIPMENT_TERMS

# Brand / series-level tokens inside VENDOR_EQUIPMENT_TERMS that are NOT
# concrete equipment models. A query mentioning "siemens", "sinamics" or
# "tia portal" is not necessarily referencing an out-of-corpus *model*;
# "sinamics g120" / "powerflex 520" are still detected via the series
# patterns below.
_BRAND_TOKENS = frozenset({
    "siemens", "simatic", "step 7", "tia portal",
    "schneider", "schneider electric", "modicon", "telemecanique",
    "mitsubishi", "mitsubishi electric", "melsec",
    "rockwell", "rockwell automation", "allen-bradley", "allen bradley",
    "abb", "omron", "beckhoff", "twincat", "yaskawa", "danfoss", "vlt",
    "sinamics", "powerflex", "cx-programmer",
})

# Concrete model tokens drawn from VENDOR_EQUIPMENT_TERMS (brand words removed).
_CONCRETE_TERMS: tuple[str, ...] = tuple(sorted({
    term.casefold()
    for terms in VENDOR_EQUIPMENT_TERMS.values()
    for term in terms
    if term.casefold() not in _BRAND_TOKENS
}))

# Series patterns produce normalized model strings like "s7-1200",
# "sinamics g120", "acs580", "modicon m221", "powerflex 520", "fr-e800".
_SERIES_OUTPUTS: tuple[str, ...] = tuple({
    tpl.replace("\\1", "x").replace("\\2", "x")
    for _pattern, _mfr, tpl in SERIES_PATTERNS
})


def _series_mentions(query: str) -> list[str]:
    """Model mentions via document_identity_v373 series patterns.

    The frozen patterns use ``[-_]?`` / ``[\\s_-]*`` between brand and model
    tokens. normalize_identity_text collapses hyphens to spaces, which would
    break ``s7[-_]?\\d{3,4}`` for "s7-1200". We therefore run the patterns on
    both the casefolded text and a variant where hyphen/underscore/space runs
    are folded to a single hyphen. Detection is a union; dedup happens in
    detect_model_mentions.
    """
    raw = (query or "").casefold()
    variants = [raw, re.sub(r"[-_\s]+", "-", raw)]
    mentions: list[str] = []
    for text in variants:
        for pattern, expected_mfr, template in SERIES_PATTERNS:
            for match in re.finditer(pattern, text):
                group = match.group(1) if match.groups() else ""
                if group:
                    mentions.append(template.replace("\\1", group))
                else:
                    mentions.append(template)
    return mentions


def _term_mentions(query: str) -> list[str]:
    """Concrete model tokens (from VENDOR_EQUIPMENT_TERMS) as whole tokens."""
    text = normalize_identity_text(query)
    mentions: list[str] = []
    for term in _CONCRETE_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
            mentions.append(term)
    return mentions


def detect_model_mentions(query: str) -> list[str]:
    """Return normalized model mentions found in a query (deduplicated)."""
    seen: list[str] = []
    for mention in (*_series_mentions(query), *_term_mentions(query)):
        if mention and mention not in seen:
            seen.append(mention)
    return seen


def corpus_known_models(
    documents: list | None = None,
    *,
    known_models: dict[str, str] | None = None,
) -> dict[str, str]:
    """Normalized model -> canonical form for every corpus-known model.

    ``known_models`` is the build-time global index (coverage_index_v385.json)
    and is the preferred source: corpus identity is a corpus-level FACT and
    must not depend on which documents happened to be retrieved. When omitted,
    the function falls back to the document metadata of the current window
    (top-k retrieval), which is a weaker approximation.
    """
    if known_models is not None:
        # Store BOTH the normalized key ("s7 1200") and the raw key
        # ("s7-1200"): series-pattern mentions keep their hyphen ("s7-1200")
        # while normalize_identity_text collapses hyphens to spaces.
        result: dict[str, str] = {}
        for key, value in known_models.items():
            result[normalize_identity_text(key)] = value
            result[key.casefold()] = value
        return result

    known: dict[str, str] = {}
    for document in documents or []:
        metadata = getattr(document, "metadata", {}) or {}
        identity = identity_from_metadata(metadata)
        for value in (
            identity.equipment_model,
            identity.product_series,
            identity.product_family,
            *identity.aliases,
        ):
            if not value:
                continue
            normalized = normalize_identity_text(value)
            if normalized:
                known.setdefault(normalized, value)
                known.setdefault(value.casefold(), value)
    return known


def out_of_corpus_models(
    query: str,
    documents: list,
    *,
    known_models: dict[str, str] | None = None,
) -> list[str]:
    """Detected model mentions that are NOT covered by the corpus.

    Returns the normalized mention strings. Empty list means no signal.
    """
    known = corpus_known_models(documents, known_models=known_models)
    return [mention for mention in detect_model_mentions(query) if mention not in known]


def coverage_gate_verdict(
    query: str,
    documents: list,
    *,
    known_models: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Phase C coverage gate verdict.

    Returns (ok, reason). ``ok=True`` when the query is inside the corpus
    whitelist scope (no out-of-corpus model detected). ``ok=False`` with a
    reason when an out-of-corpus model is referenced.
    """
    unknown = out_of_corpus_models(query, documents, known_models=known_models)
    if unknown:
        return False, f"OUT_OF_CORPUS_MODEL:{','.join(sorted(unknown))}"
    return True, ""
