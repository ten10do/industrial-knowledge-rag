"""Explainable product- and identifier-aware retrieval scopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .filters import IDENTIFIER_PATTERN, QueryAnalysis
from .product_identity import (
    IdentityRelation,
    ProductIdentity,
    identity_from_metadata,
    identity_relation,
)


class RetrievalScope(str, Enum):
    EXACT_MODEL_SCOPE = "EXACT_MODEL_SCOPE"
    SERIES_SCOPE = "SERIES_SCOPE"
    FAMILY_SCOPE = "FAMILY_SCOPE"
    GLOBAL_SCOPE = "GLOBAL_SCOPE"
    UNKNOWN_SCOPE = "UNKNOWN_SCOPE"
    MULTI_IDENTITY_SCOPE = "MULTI_IDENTITY_SCOPE"


@dataclass(frozen=True)
class ScopeTier:
    level: str
    documents: tuple[object, ...]
    reason: str = ""


@dataclass
class RetrievalScopeDecision:
    requested_scope: str
    effective_scope: str
    tiers: tuple[ScopeTier, ...]
    candidate_counts: dict[str, int]
    identifiers: tuple[str, ...] = ()
    identifier_found: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    events: list[str] = field(default_factory=list)
    query_identities: tuple[ProductIdentity, ...] = ()

    def as_dict(self) -> dict:
        return {
            "requested_scope": self.requested_scope,
            "effective_scope": self.effective_scope,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "candidate_counts": dict(self.candidate_counts),
            "identifiers": list(self.identifiers),
            "identifier_found": self.identifier_found,
            "events": list(self.events),
        }


def _document_key(document: object) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("chunk_id", "")) or str(id(document))


def _deduplicate(documents: list[object]) -> list[object]:
    unique = {}
    for document in documents:
        unique.setdefault(_document_key(document), document)
    return list(unique.values())


def _query_identity(analysis: QueryAnalysis) -> ProductIdentity:
    return ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    )


def _relations(documents: list, identities: tuple[ProductIdentity, ...]) -> dict[str, set[IdentityRelation]]:
    result = {}
    for document in documents:
        candidate = identity_from_metadata(getattr(document, "metadata", {}) or {})
        result[_document_key(document)] = {identity_relation(query, candidate) for query in identities}
    return result


def _identifier_documents(documents: list, identifiers: tuple[str, ...]) -> list:
    if not identifiers:
        return []
    wanted = {value.casefold() for value in identifiers}
    matches = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        found = {
            match.group(0).casefold()
            for match in IDENTIFIER_PATTERN.finditer(str(getattr(document, "page_content", "") or ""))
        }
        metadata_identifier = str(metadata.get("error_code", "")).strip().casefold()
        if metadata_identifier:
            found.add(metadata_identifier)
        if wanted.issubset(found):
            matches.append(document)
    return matches


def build_retrieval_scope(question: str, documents: list, analysis: QueryAnalysis) -> RetrievalScopeDecision:
    """Build ordered candidate tiers without mutating source metadata."""
    del question  # The parsed analysis is the single source of scope decisions.
    all_documents = list(documents)
    resolved = tuple(analysis.product_identities)
    query_identity = _query_identity(analysis)
    identities = resolved or ((query_identity,) if any((
        query_identity.product_family,
        query_identity.product_series,
        query_identity.equipment_model,
    )) else ())
    relation_map = _relations(all_documents, identities) if identities else {}

    exact = [doc for doc in all_documents if IdentityRelation.EXACT_MODEL in relation_map.get(_document_key(doc), set())]
    series = [doc for doc in all_documents if relation_map.get(_document_key(doc), set()) & {
        IdentityRelation.EXACT_MODEL, IdentityRelation.SAME_SERIES,
    }]
    family = [doc for doc in all_documents if relation_map.get(_document_key(doc), set()) & {
        IdentityRelation.EXACT_MODEL, IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY,
    }]
    counts = {
        "exact_candidates": len(exact),
        "series_candidates": len(series),
        "family_candidates": len(family),
        "global_candidates": len(all_documents),
    }

    if len(resolved) > 1:
        requested = RetrievalScope.MULTI_IDENTITY_SCOPE
        base_tiers = [ScopeTier(requested.value, tuple(exact))]
        broader = [doc for doc in family if _document_key(doc) not in {_document_key(item) for item in exact}]
        if broader:
            base_tiers.append(ScopeTier(RetrievalScope.FAMILY_SCOPE.value, tuple(broader), "insufficient_multi_identity_candidates"))
    elif analysis.identity_confidence == "EXACT_MODEL":
        if exact:
            requested = RetrievalScope.EXACT_MODEL_SCOPE
            base_tiers = [ScopeTier(requested.value, tuple(exact))]
            siblings = [doc for doc in family if _document_key(doc) not in {_document_key(item) for item in exact}]
            if siblings:
                base_tiers.append(ScopeTier(RetrievalScope.FAMILY_SCOPE.value, tuple(siblings), "insufficient_exact_candidates"))
        else:
            requested = RetrievalScope.UNKNOWN_SCOPE
            base_tiers = [ScopeTier(RetrievalScope.GLOBAL_SCOPE.value, tuple(all_documents), "unknown_model")]
    elif analysis.identity_confidence == "SERIES":
        requested = RetrievalScope.SERIES_SCOPE
        base_tiers = [ScopeTier(requested.value, tuple(series or family))]
    elif analysis.identity_confidence == "FAMILY":
        requested = RetrievalScope.FAMILY_SCOPE
        base_tiers = [ScopeTier(requested.value, tuple(family))]
    else:
        requested = RetrievalScope.GLOBAL_SCOPE
        base_tiers = [ScopeTier(requested.value, tuple(all_documents))]

    identifier_docs = _identifier_documents(all_documents, analysis.identifiers)
    if analysis.identifiers and identifier_docs:
        base_keys = {_document_key(doc) for tier in base_tiers for doc in tier.documents}
        scoped_identifiers = [doc for doc in identifier_docs if _document_key(doc) in base_keys]
        protected = scoped_identifiers or identifier_docs
        protected_keys = {_document_key(doc) for doc in protected}
        tiers = [ScopeTier("IDENTIFIER_SCOPE", tuple(protected))]
        for tier in base_tiers:
            remaining = tuple(doc for doc in tier.documents if _document_key(doc) not in protected_keys)
            if remaining:
                tiers.append(ScopeTier(tier.level, remaining, "insufficient_identifier_candidates"))
    else:
        tiers = base_tiers

    tiers = [ScopeTier(tier.level, tuple(_deduplicate(list(tier.documents))), tier.reason) for tier in tiers if tier.documents]
    if not tiers:
        tiers = [ScopeTier(RetrievalScope.GLOBAL_SCOPE.value, tuple(all_documents), "scope_resolution_failed")]
    decision = RetrievalScopeDecision(
        requested_scope=requested.value,
        effective_scope=tiers[0].level,
        tiers=tuple(tiers),
        candidate_counts=counts,
        identifiers=analysis.identifiers,
        identifier_found=bool(identifier_docs),
        query_identities=identities,
    )
    if requested == RetrievalScope.UNKNOWN_SCOPE:
        decision.fallback_used = True
        decision.fallback_reason = "unknown_model"
        decision.events.append("SCOPE_FALLBACK")
    return decision


def collect_scoped_candidates(decision: RetrievalScopeDecision, top_k: int, retrieve) -> list:
    """Fill a channel's candidate pool tier-by-tier, preserving primary eligibility."""
    selected, seen = [], set()
    for tier_index, tier in enumerate(decision.tiers):
        remaining = top_k - len(selected)
        if remaining <= 0:
            break
        candidates = retrieve(list(tier.documents), remaining)
        for candidate in candidates:
            key = candidate.chunk_id or str(id(candidate.document))
            if key in seen:
                continue
            candidate.scope_match = "primary" if tier_index == 0 else "fallback"
            candidate.scope_level = tier.level
            relations = {
                identity_relation(identity, identity_from_metadata(candidate.metadata))
                for identity in decision.query_identities
            }
            candidate.identity_relation = next(
                (
                    relation.value for relation in (
                        IdentityRelation.EXACT_MODEL,
                        IdentityRelation.SAME_SERIES,
                        IdentityRelation.SAME_FAMILY,
                        IdentityRelation.MISMATCH,
                    )
                    if relation in relations
                ),
                IdentityRelation.UNKNOWN.value,
            )
            selected.append(candidate)
            seen.add(key)
            if len(selected) == top_k:
                break
        if tier_index == 0 and len(tier.documents) >= top_k:
            break
        if tier_index > 0 and candidates:
            decision.fallback_used = True
            decision.fallback_reason = tier.reason or "insufficient_primary_candidates"
            decision.effective_scope = tier.level
    return selected
