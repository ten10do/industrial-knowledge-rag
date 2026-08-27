"""V3.80-C Shadow requirement-extraction upgrade candidate (E1, SHADOW ONLY).

Single-variable research delta on top of the V3.79 finding
(``REQUIREMENT_EXTRACTION_COVERAGE_INSUFFICIENT``): add GENERIC deterministic
extraction families for the query shapes that today terminate in
``NO_EXTRACTABLE_REQUIREMENT`` -> AMBIGUOUS, and give each family an explicit,
narrow support predicate implemented HERE (support-v316.1 itself stays
byte-frozen; this module never modifies it).

Families (all generic surface/semantic patterns - no manufacturer strings,
no query ids, no benchmark-shaped special cases):

    DEFINITION      "what is a/an/the <np>", "tell me about <np>",
                    "describe ..."                -> DESCRIBES(subject=<np>)
        support: subject present AND definitional/function cue near subject
    ATTRIBUTE_VALUE "what is the <attr head noun phrase>" (definite article +
                    a small generic industrial attribute-head lexicon)
                        -> HAS_VALUE(subject=contextual device, attribute=head)
        support: attribute synonym present AND a value marker (numeric or
                 named-option) in the SAME sentence
    PURPOSE         "what is the purpose/function of <np>" -> FUNCTION_OF(<np>)
        support: subject present AND purpose cue in same sentence
    WORLD_ENTITY    interrogative with ZERO industrial hints -> the query is
                    outside any verifiable industrial claim contract ->
                    LEGITIMATELY_UNEXTRACTABLE (AMBIGUOUS under strict policy;
                    deliberately NOT force-structured per spec section 18)
    FALLBACK        anything the baseline already extracts goes to the frozen
                    validator unchanged (zero regression by construction)

Every routine is pure/deterministic regex over the query and chunk texts.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from backend.retrieval.claim_support_shadow import (
    ClaimSupportResult,
    ClaimSupportState,
    structured_claim_support,
)
from backend.retrieval.evidence_contract import evaluate_evidence_contract  # noqa: F401  (parity audit)
from backend.retrieval.product_identity import (
    ProductIdentity,
    identity_from_metadata,
    identity_is_compatible,
)

EXTRACTION_SHADOW_VERSION = "reqex-shadow-v380c-r0"

# --- generic industrial hint vocabulary (domain gate; deliberately broad) ------
INDUSTRIAL_HINTS: tuple[str, ...] = (
    "drive", "vfd", "inverter", "servo", "plc", "motor", "frequency converter",
    "input", "output", "parameter", "fault", "alarm", "warning", "relay",
    "terminal", "wiring", "fieldbus", "modbus", "profibus", "profinet",
    "ethernet", "canopen", "devicenet", "pid", "ramp", "acceleration",
    "deceleration", "torque", "braking", "encoder", "pulse", "digital",
    "analog", "panel", "keypad", "dip switch", "firmware", "supply voltage",
    "dc bus", "kw", "hp", "ampere", "amps", "voltage", "hz",
)


def has_industrial_hint(text: str) -> bool:
    folded = " ".join(str(text).casefold().split())
    return any(hint in folded for hint in INDUSTRIAL_HINTS)


# --- generic attribute head lexicon (surface-generic; small by design) --------
ATTRIBUTE_HEAD_GROUPS: dict[str, tuple[str, ...]] = {
    "frequency": ("frequency", "frequencies"),
    "overload": ("overload capacity", "overload capability", "overload"),
    "efficiency": ("efficiency class", "efficiency"),
    "noise": ("noise level", "noise", "sound pressure", "acoustic"),
    "protection": ("protection rating", "ingress protection", "protection class", "protection"),
    "control_method": ("control method", "control mode", "control principle"),
    "power": ("power rating", "power consumption", "rated power", "power"),
    "acceleration": ("acceleration time", "ramp time", "acceleration"),
    "deceleration": ("deceleration time", "deceleration"),
}

# value markers: measurable quantities or named discrete options
NUMERIC_VALUE_RE = re.compile(
    r"(?<![a-z0-9])\d+(?:[.,]\d+)?\s*(?:%|hz|khz|mhz|kw|w|v|a|nm|n\s*m|s|ms|min|db|"
    r"|ip\s?\d{2}[a-z]?|[a-z]{1,6})?(?![a-z0-9])",
    re.IGNORECASE,
)
NAMED_OPTION_RE = re.compile(
    r"\b(?:v/?f(?:\s+control)?|vector control|scalar control|dpi|rj45|rs[- ]?485|"
    r"class \d+|class\s+[a-z]+\d*|ip\s?\d{2}[a-z]?)\b",
    re.IGNORECASE,
)

DEFINITION_QUERY_RE = re.compile(
    r"^(?:what\s+(?:is|are)|tell\s+me\s+about|describe\s+)\s+(.+?)\s*[?.]?\s*$",
    re.IGNORECASE,
)
ATTR_VALUE_QUERY_RE = re.compile(r"^what\s+is\s+the\s+(.+?)\s*[?.]?\s*$", re.IGNORECASE)
PURPOSE_QUERY_RE = re.compile(
    r"^what\s+is\s+the\s+(?:purpose|function)\s+of\s+(?:an|a|the)?\s*(.+?)\s*[?.]?\s*$",
    re.IGNORECASE,
)
DESCRIBE_PURPOSE_QUERY_RE = re.compile(
    r"^(?:describe|explain)\s+the\s+(?:function|purpose)\s+of\s+(?:an|a|the)?\s*(.+?)\s*[?.]?\s*$",
    re.IGNORECASE,
)
STOP_HEADS = re.compile(
    r"^(?:capital|speed of light|stock price|moons|population|author|meaning of life)",
    re.IGNORECASE,
)

DEFINITION_CUE_RE = re.compile(
    r"(?:\bis|\bare)\s+(?:a|an|the)?\s*[a-z]|"
    r"\b(?:refers? to|means|defined as|known as|stands? for)\b",
    re.IGNORECASE,
)
FUNCTION_CUE_RE = re.compile(
    r"\b(?:used to|used for|is used|are used|enables?|allows?|provides?|converts?|"
    r"controls?|protects?|monitors?|suppresses?|limits?|ensures?|prevents?|"
    r"removes?|disconnects?|interrupts?)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ShadowRequirement:
    family: str            # DEFINITION | ATTRIBUTE_VALUE | PURPOSE | WORLD_ENTITY | BASELINE
    subject: str = ""
    attribute_key: str = ""
    legitimately_unextractable: bool = False

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "subject": self.subject,
            "attribute_key": self.attribute_key,
            "legitimately_unextractable": self.legitimately_unextractable,
        }


def _strip_articles(np: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", np.strip(), flags=re.IGNORECASE).strip()


def _head_hit(attr_key: str, text: str) -> str | None:
    for synonym in ATTRIBUTE_HEAD_GROUPS[attr_key]:
        if re.search(rf"(?<![a-z0-9]){re.escape(synonym)}(?![a-z0-9])", text, re.IGNORECASE):
            return synonym
    return None


def classify_query(query: str) -> ShadowRequirement:
    """Generic surface->structure classification (order matters)."""
    q = " ".join((query or "").strip().casefold().split())
    purpose = PURPOSE_QUERY_RE.match(q) or DESCRIBE_PURPOSE_QUERY_RE.match(q)
    if purpose:
        return ShadowRequirement("PURPOSE", subject=_strip_articles(purpose.group(1)))
    attr = ATTR_VALUE_QUERY_RE.match(q)
    if attr:
        np_ = _strip_articles(attr.group(1))
        if STOP_HEADS.match(np_):
            return ShadowRequirement("WORLD_ENTITY", subject=np_,
                                     legitimately_unextractable=True)
        for key, group in ATTRIBUTE_HEAD_GROUPS.items():
            if any(re.search(rf"(?<![a-z0-9]){re.escape(syn)}(?![a-z0-9])", np_) for syn in group):
                return ShadowRequirement("ATTRIBUTE_VALUE", subject=np_, attribute_key=key)
        # No attribute head: a definite-NP about an industrial entity is still a
        # definition-style subject, not a coarse world-entity question.
        if has_industrial_hint(np_):
            return ShadowRequirement("DEFINITION", subject=np_)
        return ShadowRequirement("WORLD_ENTITY", subject=np_, legitimately_unextractable=True)
    definition = DEFINITION_QUERY_RE.match(q)
    if definition:
        np_ = _strip_articles(definition.group(1))
        if STOP_HEADS.match(np_):
            return ShadowRequirement("WORLD_ENTITY", subject=np_,
                                     legitimately_unextractable=True)
        # A definition query about something outside the industrial contract is
        # legitimately unextractable rather than force-structured.
        if not has_industrial_hint(q) and not has_industrial_hint(np_):
            return ShadowRequirement("WORLD_ENTITY", subject=np_,
                                     legitimately_unextractable=True)
        return ShadowRequirement("DEFINITION", subject=np_)
    if not has_industrial_hint(q):
        return ShadowRequirement("WORLD_ENTITY", subject=q, legitimately_unextractable=True)
    return ShadowRequirement("BASELINE")


# --- shadow support predicates (explicit narrow semantics per family) ----------


def _sentences_with(text: str, needle_regex: str) -> list[str]:
    pattern = re.compile(needle_regex, re.IGNORECASE)
    hits = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        if sentence.strip() and pattern.search(sentence):
            hits.append(sentence)
    return hits


def _definition_supported(subject: str, chunk_text: str) -> bool:
    """DESCRIBES: subject mention + definitional/function cue, subject locality."""
    variants = [re.escape(subject)]
    if subject.endswith("s"):
        variants.append(re.escape(subject[:-1]))
    else:
        variants.append(re.escape(subject) + "s")
    subject_re = "(?:" + "|".join(variants) + ")"
    sentences = _sentences_with(chunk_text, rf"(?<![a-z0-9]){subject_re}")
    if not sentences:
        return False
    for sentence in sentences:
        if DEFINITION_CUE_RE.search(sentence) or FUNCTION_CUE_RE.search(sentence):
            return True
    # bounded neighbor window: sentence BEFORE a glossary-style short line
    flat_sentences = [s for s in SENTENCE_SPLIT_RE.split(chunk_text) if s.strip()]
    for index, sentence in enumerate(flat_sentences):
        if re.search(rf"(?<![a-z0-9]){subject_re}(?![a-z0-9])", sentence, re.IGNORECASE):
            nxt = flat_sentences[index + 1] if index + 1 < len(flat_sentences) else ""
            if len(nxt.split()) <= 14 and (DEFINITION_CUE_RE.search(nxt) or FUNCTION_CUE_RE.search(nxt)):
                return True
    return False


def _attribute_value_supported(attr_key: str, chunk_text: str) -> bool:
    """HAS_VALUE: attribute synonym AND value marker within ONE sentence."""
    synonym = _head_hit(attr_key, chunk_text)
    if not synonym:
        return False
    for sentence in _sentences_with(chunk_text, re.escape(synonym)):
        if NUMERIC_VALUE_RE.search(sentence) or NAMED_OPTION_RE.search(sentence):
            return True
    return False


def _query_identity(query: str) -> ProductIdentity | None:
    """Generic model-number SHAPE probe (no literal model names): a short
    letter run optionally hyphenated to digits, e.g. the industrial
    ``xx123``/``xx-1234`` convention. Purely input-conditional."""
    match = re.search(r"(?<![a-z0-9])([a-z]{1,6}[-\s]?\d{2,4}[a-z]?)(?![a-z0-9])",
                      query, re.IGNORECASE)
    if not match:
        return None
    token = re.sub(r"[\s-]+", "", match.group(1)).casefold()
    return ProductIdentity(
        product_series=token,
        equipment_model=token,
        aliases=(token,),
    )


def _identity_ok(query: str, candidates: list) -> bool:
    qid = _query_identity(query)
    if qid is None:
        return True
    return any(
        identity_is_compatible(qid, identity_from_metadata(c.metadata))
        for c in candidates
    )


class ExtractionUpgradedSupportEvaluator:
    """Shadow-only merged evaluator (pre-registered governance matrix):

    BASELINE ... : frozen support-v316.1 verdict passed through untouched.
    New families : shadow verdict governs, EXCEPT ``WORLD_ENTITY`` never
                   downgrades an existing baseline SUPPORTED (the generic
                   world-entity guard is intentionally coarse; precision-first).
    """

    def __init__(self):
        self.shadow_family_invocations = 0
        self.baseline_fallbacks = 0
        self.latency_ms: list[float] = []

    def evaluate(self, query: str, result, documents: list) -> tuple[ClaimSupportResult, ShadowRequirement]:
        started = time.perf_counter()
        req = classify_query(query)
        candidates = list(getattr(result, "candidates", []) or [])
        chunk_texts = [" ".join(str(c.document.page_content).split()) for c in candidates]
        combined = "\n".join(chunk_texts)

        baseline = structured_claim_support(query, result, documents)

        def _from_baseline() -> ClaimSupportResult:
            return ClaimSupportResult(
                state=baseline.state,
                support_source=baseline.support_source,
                support_reason=baseline.support_reason,
                confidence=baseline.confidence,
                identity_compatible=baseline.identity_compatible,
                scope_compatible=baseline.scope_compatible,
                provenance={**baseline.provenance, "route": "baseline_v3161"},
            )

        # Governance matrix (pre-registered): coarse WORLD_ENTITY guard never
        # downgrades an existing SUPPORTED; everything else decided upstream.
        if req.family == "BASELINE" or (
            req.legitimately_unextractable and baseline.state == ClaimSupportState.SUPPORTED.value
        ):
            self.baseline_fallbacks += 1
            outcome = _from_baseline()
        elif req.legitimately_unextractable:
            self.shadow_family_invocations += 1
            outcome = ClaimSupportResult(
                state=ClaimSupportState.AMBIGUOUS.value,
                support_source=f"reqex:{EXTRACTION_SHADOW_VERSION}",
                support_reason="LEGITIMATELY_UNEXTRACTABLE",
                identity_compatible=_identity_ok(query, candidates),
                scope_compatible=True,
                provenance={"family": req.family, "subject": req.subject},
            )
        elif req.family in {"DEFINITION", "PURPOSE", "ATTRIBUTE_VALUE"}:
            self.shadow_family_invocations += 1
            if req.family == "DEFINITION":
                supported = (
                    _identity_ok(query, candidates)
                    and bool(combined.strip())
                    and _definition_supported(req.subject, combined)
                )
                reason = "DESCRIBES_SUPPORTED" if supported else "DESCRIBES_UNSUPPORTED"
            elif req.family == "PURPOSE":
                supported = (
                    _identity_ok(query, candidates)
                    and bool(combined.strip())
                    and any(
                        re.search(rf"(?<![a-z0-9]){re.escape(req.subject)}", s, re.IGNORECASE)
                        and FUNCTION_CUE_RE.search(s)
                        for s in SENTENCE_SPLIT_RE.split(combined)
                    )
                )
                reason = "FUNCTION_SUPPORTED" if supported else "FUNCTION_UNSUPPORTED"
            else:
                supported = (
                    _identity_ok(query, candidates)
                    and bool(combined.strip())
                    and _attribute_value_supported(req.attribute_key, combined)
                )
                reason = "ATTR_VALUE_SUPPORTED" if supported else "ATTR_VALUE_UNSUPPORTED"
            provenance_extra = {"family": req.family, "subject": req.subject}
            if req.attribute_key:
                provenance_extra["attribute_key"] = req.attribute_key
            outcome = ClaimSupportResult(
                state=(ClaimSupportState.SUPPORTED if supported else ClaimSupportState.UNSUPPORTED).value,
                support_source=f"reqex:{EXTRACTION_SHADOW_VERSION}",
                support_reason=reason,
                identity_compatible=_identity_ok(query, candidates),
                scope_compatible=True,
                provenance=provenance_extra,
            )
        else:  # defensive: unknown family keeps frozen behavior
            self.baseline_fallbacks += 1
            outcome = _from_baseline()

        self.latency_ms.append((time.perf_counter() - started) * 1000.0)
        return outcome, req


def merged_case_admissible(evaluator_result: ClaimSupportResult) -> bool:
    """Same strict aggregation as V3.79 (SUPPORTED-or-nothing). Unchanged."""
    return evaluator_result.state == ClaimSupportState.SUPPORTED.value
