"""Identity claim expansion candidate for V3.41.

V3.40 proved the coverage relation layer is safe (0 unsafe relax, FA flat) and
that its residual false refusals are all frozen identity-boundary blocks:
``NO_COMPATIBLE_IDENTITY_CLAIM``.  The frozen identity extractor only binds
explicit hyphenated identity tokens, while industrial manuals constantly refer
to their own product implicitly — document title ("Optidrive E3"), section
context, pronouns ("this drive", "the unit"), and family references.

This candidate adds an explicit **identity claim layer** on top of the frozen
identity path (which is never modified).  When the frozen boundary refuses with
``NO_COMPATIBLE_IDENTITY_CLAIM``, the layer builds a document-local
``IdentityClaim`` and re-evaluates compatibility under three hard safety
constraints: a different manufacturer is rejected, a different product line is
rejected, and a parameter owner outside the document is rejected.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence_mixed import analyze_mixed_evidence
from .identity_reasoning import analyze_identity_aware_evidence
from .product_identity import normalize_identity_text


IDENTITY_CLAIM_CANDIDATE_VERSION = "identity-v341-claim-expansion-candidate"
IDENTITY_CLAIM_CANDIDATE_STATUS = "EXPERIMENTAL_CANDIDATE"

# Pronoun / generic product nouns that may bind to a single-product document.
_GENERIC_PRODUCT_NOUN = re.compile(
    r"\b(?:this|the|its|your)\s+(?:drive|inverter|controller|unit|device|plc|amplifier|servo|system|product|equipment|amplifiers?)\b"
    r"|\b(?:the|this)\s+(?:user's\s+)?manual\b",
    re.IGNORECASE,
)
_SECTION_CONTEXT = re.compile(r"\b(?:this\s+section|section\s+\d|chapter\s+\d|in\s+this\s+chapter|the\s+manual)\b", re.IGNORECASE)
_PARAMETER_TOKEN = re.compile(r"\b[A-Za-z]{1,2}\d{0,2}[-]\d{1,4}\b|\b[A-Za-z]{2}\d{1,2}[-]\d{1,4}\b|\bP\d{3,4}\b")
_HYPHENATED_TOKEN = re.compile(r"(?<![a-z0-9])[a-z0-9]+(?:-[a-z0-9]+)+(?![a-z0-9])", re.IGNORECASE)
# "refer(s) X to Y" assertions: the claimed target must be concrete (a section,
# page, or document code owned by the corpus) before an identity expansion may
# stand behind it.
_REFERENCE_ASSERTION = re.compile(
    r"\b(?:refer(?:s|red|ring)?|pointed?|directs?|directed)\b[^?]*?\b(?:to|on)\b",
    re.IGNORECASE,
)
_CONCRETE_TARGET = re.compile(r"(?<![a-z0-9])[a-z]*\d+[a-z0-9]*(?:-[a-z0-9]+)*", re.IGNORECASE)

# Deterministic deny-list of major industrial manufacturers: a query naming one
# of these (and not the claim manufacturer) can never bind to this document.
_KNOWN_MANUFACTURERS = (
    "siemens", "mitsubishi", "omron", "rockwell", "allen-bradley", "schneider",
    "abb", "delta", "lenze", "hitachi", "panasonic", "weg", "invertek",
    "danfoss", "yaskawa", "fuji", "toshiba", "keyence", "beckhoff", "eaton",
    "phoenix contact", "turck", "moxa", "wago", "pilz", "sick", "festo",
    "advantech", "hilscher", "nord", "unitronics", "weidmueller",
    "automationdirect", "emerson", "b&r", "beckhoff", "sew-eurodrive",
)


@dataclass(frozen=True)
class IdentityClaim:
    """A document-local identity claim established from corpus metadata."""

    product: str
    family: str
    model: str
    module: str
    scope: str          # DOCUMENT | SECTION
    source: str         # corpus metadata provenance
    aliases: tuple[str, ...] = ()
    manufacturer: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        return payload


@dataclass(frozen=True)
class IdentityClaimDecision:
    query: str
    decision: str
    reason: str
    final_decision_source: str
    query_path: str
    baseline_decision: str
    baseline_reason: str
    identity_result: str
    expanded: bool
    claim_relation: str          # IdentityEvidenceRelation value
    claim_reason_code: str
    claim: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized(value: object) -> str:
    return " ".join(normalize_identity_text(value).split())


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Word-boundary containment on normalized (hyphen-folded) text."""
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", haystack) is not None


def build_document_claims(documents: list) -> dict[str, IdentityClaim]:
    """Build one document-local identity claim per distinct corpus document.

    A claim is established per document (not per corpus): every chunk of a
    manual inherits that manual's declared identity.  Documents whose own
    metadata is ambiguous (mixed manufacturer/family inside one document_id)
    are skipped.
    """
    groups: dict[str, list[dict]] = {}
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        groups.setdefault(str(metadata.get("document_id", "")), []).append(metadata)
    claims: dict[str, IdentityClaim] = {}
    for document_id, metas in groups.items():
        if not document_id:
            continue
        manufacturers = {_normalized(m.get("manufacturer", "")) for m in metas}
        families = {_normalized(m.get("product_family", "")) for m in metas}
        manufacturers.discard("")
        families.discard("")
        if len(manufacturers) != 1 or len(families) != 1:
            continue
        models = {_normalized(m.get("equipment_model", "")) for m in metas}
        models.discard("")
        aliases: set[str] = set()
        for m in metas:
            raw = m.get("model_aliases", "")
            items = raw if isinstance(raw, list) else str(raw or "").split("|")
            aliases.update(_normalized(a) for a in items if _normalized(a))
        family = sorted(families)[0]
        model = sorted(models)[0] if len(models) == 1 else ""
        claims[document_id] = IdentityClaim(
            product=family,
            family=family,
            model=model,
            module="",
            scope="DOCUMENT",
            source="document_metadata",
            aliases=tuple(sorted((aliases | models | families) - {""})),
            manufacturer=sorted(manufacturers)[0],
        )
    return claims


def _dominant_document(result) -> str | None:
    """The single document all retrieved candidates belong to, else None."""
    ids = {
        str((getattr(candidate, "metadata", {}) or {}).get("document_id", ""))
        for candidate in list(getattr(result, "candidates", []) or [])
    }
    ids.discard("")
    if len(ids) != 1:
        return None
    return next(iter(ids))


def _owned_in_corpus(token_norm: str, documents: list) -> bool:
    """A hyphenated token that appears inside the corpus is owned by it (e.g.
    genuine parameter/register codes); anything unknown is foreign."""
    return any(
        _contains_phrase(_normalized(getattr(document, "page_content", "")), token_norm)
        for document in documents
    )


def _query_conflict(query: str, query_norm: str, claim: IdentityClaim, documents: list) -> str | None:
    """Return a rejection reason when the query cannot bind to the claim."""
    for manufacturer in _KNOWN_MANUFACTURERS:
        if _contains_phrase(query_norm, manufacturer) and manufacturer not in claim.manufacturer:
            return "MANUFACTURER_MISMATCH_REJECTED"
    for token in _HYPHENATED_TOKEN.findall(query):
        token_norm = _normalized(token)
        if any(
            _contains_phrase(a, token_norm) or _contains_phrase(token_norm, a)
            for a in claim.aliases if len(a) > 2
        ):
            continue
        if _owned_in_corpus(token_norm, documents):
            continue  # document-owned register/parameter code
        return "PRODUCT_LINE_MISMATCH_REJECTED"
    for match in re.finditer(r"\bP\d{3,4}\b", query):
        if not _owned_in_corpus(_normalized(match.group(0)), documents):
            return "PARAMETER_OWNER_MISMATCH_REJECTED"
    return None


def bind_query_to_claim(query: str, claim: IdentityClaim, documents: list) -> tuple[str, str] | None:
    """Return ``(IdentityEvidenceRelation, reason_code)`` or None when rejected."""
    query_norm = _normalized(query)
    conflict = _query_conflict(query, query_norm, claim, documents)
    if conflict:
        return None
    if _contains_phrase(query_norm, claim.family):
        return "FAMILY_INHERITED", "FAMILY_REFERENCE_BOUND"
    explicit = [
        a for a in claim.aliases
        if len(a) > 2 and a != claim.family and _contains_phrase(query_norm, a)
    ]
    if explicit:
        return "EXPLICIT", "EXPLICIT_MODEL_ALIAS_MATCH"
    if _GENERIC_PRODUCT_NOUN.search(query) or _SECTION_CONTEXT.search(query):
        relation = "SECTION_INHERITED" if _SECTION_CONTEXT.search(query) else "DOCUMENT_INHERITED"
        code = {
            "SECTION_INHERITED": "SECTION_CONTEXT_INHERITED",
            "DOCUMENT_INHERITED": "PRONOUN_BOUND_TO_DOCUMENT_PRODUCT",
        }[relation]
        return relation, code
    return None


def analyze_identity_claim_evidence(
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
) -> IdentityClaimDecision:
    """Run the frozen identity path, then expand identity via document claims."""
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
    identity_reason = str(boundary.get("reason", ""))

    if baseline.decision == "ANSWER":
        return IdentityClaimDecision(
            query, "ANSWER", baseline.reason, baseline.final_decision_source, baseline.query_path,
            baseline.decision, baseline.reason, identity_status, False,
            "EXPLICIT", "ALREADY_ANSWERED", baseline=baseline_dict,
        )
    identity_blocked = identity_status in ("INCOMPATIBLE", "UNKNOWN") and (
        "NO_COMPATIBLE_IDENTITY_CLAIM" in identity_reason
        or "IDENTITY_EVIDENCE_UNKNOWN" in identity_reason
        or "CLAIM_IDENTITY_UNKNOWN" in identity_reason
        or "QUERY_IDENTITY_UNKNOWN" in identity_reason
    )
    if not identity_blocked or baseline.delegated_to_existing_evidence:
        # Non-identity blocks (evidence, scope, parser, other identity reasons)
        # are preserved untouched.
        return IdentityClaimDecision(
            query, "ABSTAIN", "NON_IDENTITY_BLOCK_PRESERVED", baseline.final_decision_source,
            baseline.query_path, baseline.decision, baseline.reason, identity_status, False,
            "UNSUPPORTED", "NON_IDENTITY_BLOCK_PRESERVED", baseline=baseline_dict,
        )

    claims = build_document_claims(documents)
    document_id = _dominant_document(result)
    claim = claims.get(document_id) if document_id else None
    if claim is None:
        return IdentityClaimDecision(
            query, "ABSTAIN", "MULTI_PRODUCT_CORPUS_UNRESOLVED", baseline.final_decision_source,
            baseline.query_path, baseline.decision, baseline.reason, identity_status, False,
            "UNSUPPORTED", "MULTI_PRODUCT_CORPUS_UNRESOLVED", baseline=baseline_dict,
        )
    target_documents = [
        document for document in documents
        if str((getattr(document, "metadata", {}) or {}).get("document_id", "")) == document_id
    ]
    bound = bind_query_to_claim(query, claim, target_documents)
    if bound is None:
        # Which safety constraint fired?
        query_norm = _normalized(query)
        reason_code = "NO_IDENTITY_ANCHOR"
        for manufacturer in _KNOWN_MANUFACTURERS:
            if _contains_phrase(query_norm, manufacturer) and manufacturer not in claim.manufacturer:
                reason_code = "MANUFACTURER_MISMATCH_REJECTED"
                break
        else:
            for token in _HYPHENATED_TOKEN.findall(query):
                token_norm = _normalized(token)
                if any(
                    _contains_phrase(a, token_norm) or _contains_phrase(token_norm, a)
                    for a in claim.aliases if len(a) > 2
                ):
                    continue
                if _owned_in_corpus(token_norm, documents):
                    continue
                reason_code = "PRODUCT_LINE_MISMATCH_REJECTED"
                break
            else:
                for match in re.finditer(r"\bP\d{3,4}\b", query):
                    if not _owned_in_corpus(_normalized(match.group(0)), documents):
                        reason_code = "PARAMETER_OWNER_MISMATCH_REJECTED"
                        break
        return IdentityClaimDecision(
            query, "ABSTAIN", reason_code, baseline.final_decision_source,
            baseline.query_path, baseline.decision, baseline.reason, identity_status, False,
            "UNSUPPORTED", reason_code, claim.as_dict(), baseline_dict,
        )

    relation, reason_code = bound
    if (
        _REFERENCE_ASSERTION.search(query)
        and relation != "EXPLICIT"
        and not _CONCRETE_TARGET.findall(query)
    ):
        # A cross-reference assertion without a concrete, corpus-ownable target
        # cannot be verified; expanding identity behind it would let the
        # evidence layer guess the target.
        return IdentityClaimDecision(
            query, "ABSTAIN", "REFERENCE_TARGET_UNVERIFIED", baseline.final_decision_source,
            baseline.query_path, baseline.decision, baseline.reason, identity_status, False,
            "UNSUPPORTED", "REFERENCE_TARGET_UNVERIFIED", claim.as_dict(), baseline_dict,
        )
    mixed = analyze_mixed_evidence(
        query, result, documents, retrieval_mode,
        judge=judge, policy=policy, identity_matching=identity_matching,
        requirement=requirement, apply_open_sufficiency=apply_open_sufficiency,
    )
    return IdentityClaimDecision(
        query, mixed.decision, reason_code, "V341_IDENTITY_CLAIM", mixed.query_path,
        baseline.decision, baseline.reason, "COMPATIBLE", True,
        relation, reason_code, claim.as_dict(), baseline_dict,
    )