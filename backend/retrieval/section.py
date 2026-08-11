"""Lightweight, scope-safe section candidate retrieval."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from .candidates import RetrievalCandidate
from .product_identity import IdentityRelation, identity_from_metadata, identity_relation
from .tokenizer import tokenize


DEFAULT_SECTION_NEIGHBOR_WINDOW = 1
DEFAULT_SECTION_CANDIDATE_K = 2
DEFAULT_SECTION_MAX_EXPANDED = 3


@dataclass(frozen=True)
class SectionIdentity:
    document_id: str
    section: str
    subsection: str
    normalized_section: str
    section_path: str
    page_start: int | None
    page_end: int | None
    parent_section: str = ""


@dataclass(frozen=True)
class SectionHint:
    preferred_concepts: tuple[str, ...] = ()
    vocabulary_matches: tuple[str, ...] = ()
    matched_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectionConfig:
    enabled: bool = False
    neighbor_window: int = DEFAULT_SECTION_NEIGHBOR_WINDOW
    candidate_k: int = DEFAULT_SECTION_CANDIDATE_K
    max_expanded: int = DEFAULT_SECTION_MAX_EXPANDED

    def __post_init__(self):
        if self.neighbor_window < 0:
            raise ValueError("SECTION_NEIGHBOR_WINDOW must be zero or positive.")
        if self.candidate_k <= 0 or self.max_expanded <= 0:
            raise ValueError("Section candidate limits must be positive.")


@dataclass
class SectionExpansionReport:
    section_requested: bool
    section_effective: bool
    section_fallback_reason: str = ""
    section_candidates_added: int = 0
    section_expansion_used: bool = False
    candidate_budget_overflow: int = 0
    selected_sections: list[dict] = field(default_factory=list)
    hint: SectionHint = field(default_factory=SectionHint)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["hint"] = asdict(self.hint)
        return payload


_SEPARATOR_PATTERN = re.compile(r"\s*(?:[|>\u203a\u00bb\u2192]+|\s[-\u2013\u2014]\s)\s*")
_CHAPTER_PATTERN = re.compile(r"^(?:chapter\s*)?(\d+)\s*[.:/\-\u2013\u2014]*\s*", re.IGNORECASE)
_APPENDIX_PATTERN = re.compile(r"^appendix\s+([a-z0-9]+)\s*[.:\-\u2013\u2014]*\s*", re.IGNORECASE)


def normalize_section(value: str) -> str:
    """Normalize retrieval identity while leaving citation metadata untouched."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"/uni00a0", " ", text, flags=re.IGNORECASE)
    text = "".join(" " if unicodedata.category(char) == "Zs" else char for char in text)
    text = _SEPARATOR_PATTERN.sub(" / ", text)
    text = re.sub(r"\s+", " ", text).strip()
    chapter = _CHAPTER_PATTERN.match(text)
    if chapter:
        number = str(int(chapter.group(1)))
        text = f"{number} {text[chapter.end():]}".strip()
    else:
        appendix = _APPENDIX_PATTERN.match(text)
        if appendix:
            text = f"appendix {appendix.group(1)} {text[appendix.end():]}".strip()
    return re.sub(r"\s+", " ", text).strip().casefold()


def section_identity(document: object) -> SectionIdentity:
    metadata = getattr(document, "metadata", {}) or {}
    section = str(metadata.get("section", ""))
    subsection = str(metadata.get("subsection", ""))
    normalized = normalize_section(section)
    parent = str(metadata.get("parent_section", ""))
    path = " / ".join(item for item in (normalized, normalize_section(subsection)) if item)

    def page(name: str, fallback: str) -> int | None:
        value = metadata.get(name, metadata.get(fallback))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return SectionIdentity(
        document_id=str(metadata.get("document_id", "")),
        section=section,
        subsection=subsection,
        normalized_section=normalized,
        section_path=path,
        page_start=page("page_start", "page"),
        page_end=page("page_end", "page"),
        parent_section=parent,
    )


# Small, generic bilingual aliases. Corpus section titles remain the vocabulary source.
_HINT_RULES = (
    (re.compile(r"\u6545\u969c|\u51b2\u7a81|\u6062\u590d|\bfault\b|\bconflict\b|\brecover", re.I),
     ("fault", "conflict", "troubleshooting", "diagnostic", "resolution", "recover", "mode"), "fault"),
    (re.compile(r"\u5b89\u5168|\u65ad\u7535|\u653e\u7535|\u7535\u5bb9|\u76f4\u6d41\u6bcd\u7ebf|\bsafety\b|\bpower\b|discharg|capacitor|dc bus", re.I),
     ("safety", "warning", "attention", "power", "mains", "discharge", "capacitor", "dc", "bus", "voltage"), "safety"),
    (re.compile(r"\u6a21\u62df|\u7cbe\u5ea6|\u89c4\u683c|\u53c2\u6570|\banalog\b|accuracy|specification|parameter", re.I),
     ("parameter", "specification", "control", "accuracy", "resolution", "analog", "output"), "parameter"),
    (re.compile(r"\u7f51\u7edc|\u5730\u5740|\u914d\u7f6e|\u6295\u5165\u8fd0\u884c|\bip\b|network|address|configur|commission", re.I),
     ("ip", "network", "ethernet", "connect", "configuration", "address", "commissioning", "startup", "device"), "network"),
    (re.compile(r"\u72b6\u6001|\u6307\u793a|status|indicator", re.I),
     ("status", "indicator", "diagnostic", "display"), "status"),
    (re.compile(r"\u7ef4\u62a4|\u7ef4\u4fee|\u4fdd\u517b|maintenance|service", re.I),
     ("maintenance", "servicing", "repair", "replace", "procedure", "safety"), "maintenance"),
    (re.compile(r"\u6b65\u9aa4|\u5982\u4f55|\u600e\u6837|procedure|\bhow\b", re.I),
     ("procedure", "step", "configure", "set", "verify"), "procedure"),
)
_SECTION_STOPWORDS = {"powerflex", "compactlogix", "controller", "controllers", "drive", "drives"}


def _enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("SECTION_EXPANSION_ENABLED must be true or false.")


def load_section_config() -> tuple[SectionConfig, str]:
    try:
        return SectionConfig(
            enabled=_enabled(os.getenv("SECTION_EXPANSION_ENABLED", "false")),
            neighbor_window=int(os.getenv("SECTION_NEIGHBOR_WINDOW", str(DEFAULT_SECTION_NEIGHBOR_WINDOW))),
            candidate_k=int(os.getenv("SECTION_CANDIDATE_K", str(DEFAULT_SECTION_CANDIDATE_K))),
            max_expanded=int(os.getenv("SECTION_MAX_EXPANDED", str(DEFAULT_SECTION_MAX_EXPANDED))),
        ), ""
    except (TypeError, ValueError) as exc:
        return SectionConfig(), f"Invalid section retrieval configuration: {exc}"


def _terms(text: str) -> set[str]:
    return {
        token.casefold() for token in tokenize(text)
        if (len(token) > 1 or token in {"ip", "dc"})
        and not token.isdigit()
        and token.casefold() not in _SECTION_STOPWORDS
    }


def _counter(text: str) -> Counter:
    return Counter(
        token.casefold() for token in tokenize(text)
        if (len(token) > 1 or token in {"ip", "dc"})
        and not token.isdigit()
        and token.casefold() not in _SECTION_STOPWORDS
    )


def infer_section_hint(query: str, vocabulary: set[str]) -> SectionHint:
    concepts = set(_terms(query))
    aliases = []
    for pattern, additions, name in _HINT_RULES:
        if pattern.search(query or ""):
            concepts.update(additions)
            aliases.append(name)
    vocabulary_matches = sorted(concepts & vocabulary)
    return SectionHint(
        preferred_concepts=tuple(sorted(concepts)),
        vocabulary_matches=tuple(vocabulary_matches),
        matched_aliases=tuple(aliases),
    )


def _overlap(counter: Counter, concepts: set[str]) -> float:
    return sum(1.0 + math.log1p(counter[token]) for token in concepts if counter[token])


@dataclass
class _SectionRecord:
    identity: SectionIdentity
    documents: list[object]
    title_tokens: Counter
    chunk_tokens: list[Counter]


class SectionIndex:
    def __init__(self, documents: list[object]):
        groups = defaultdict(list)
        for document in documents:
            identity = section_identity(document)
            if identity.document_id and identity.normalized_section:
                groups[(identity.document_id, identity.normalized_section)].append(document)
        self.records = []
        self.vocabulary = set()
        for group in groups.values():
            def order(item):
                metadata = getattr(item, "metadata", {}) or {}
                try:
                    chunk_index = int(metadata.get("chunk_index", 10**9))
                except (TypeError, ValueError):
                    chunk_index = 10**9
                return chunk_index, str(metadata.get("chunk_id", ""))

            group.sort(key=order)
            identity = section_identity(group[0])
            subtitles = " ".join(str((getattr(item, "metadata", {}) or {}).get("subsection", "")) for item in group)
            title_tokens = _counter(identity.normalized_section)
            chunk_tokens = [_counter(str(getattr(item, "page_content", ""))) for item in group]
            self.vocabulary.update(title_tokens)
            self.vocabulary.update(_terms(subtitles))
            self.records.append(_SectionRecord(identity, group, title_tokens, chunk_tokens))

    def select(self, query: str, allowed_chunk_ids: set[str], candidate_k: int):
        allowed_records = [
            record for record in self.records
            if any(str((getattr(doc, "metadata", {}) or {}).get("chunk_id", "")) in allowed_chunk_ids for doc in record.documents)
        ]
        hint = infer_section_hint(query, self.vocabulary)
        concepts = set(hint.preferred_concepts)
        ranked = []
        for record in allowed_records:
            chunk_scores = sorted((_overlap(tokens, concepts) for tokens in record.chunk_tokens), reverse=True)
            padded = chunk_scores[:3] + [0.0, 0.0, 0.0]
            score = 1.5 * _overlap(record.title_tokens, concepts) + padded[0] + .3 * padded[1] + .1 * padded[2]
            if score > 0:
                ranked.append((score, record, chunk_scores))
        ranked.sort(key=lambda item: (-item[0], item[1].identity.document_id, item[1].identity.normalized_section))
        return ranked[:candidate_k], hint


_INDEX_CACHE: dict[str, tuple[tuple, SectionIndex]] = {}


def _signature(documents: list[object]) -> tuple:
    chunk_ids = tuple(str((getattr(item, "metadata", {}) or {}).get("chunk_id", "")) for item in documents)
    return len(chunk_ids), hash(chunk_ids)


def _section_index(cache_key: str, documents: list[object]) -> SectionIndex:
    signature = _signature(documents)
    cached = _INDEX_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]
    index = SectionIndex(documents)
    _INDEX_CACHE[cache_key] = (signature, index)
    if len(_INDEX_CACHE) > 4:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    return index


def _chunk_id(document: object) -> str:
    return str((getattr(document, "metadata", {}) or {}).get("chunk_id", ""))


def expand_section_candidates(
    query: str,
    base_candidates: list[RetrievalCandidate],
    corpus_documents: list[object],
    scope_decision,
    *,
    budget: int,
    cache_key: str,
    config: SectionConfig,
) -> tuple[list[RetrievalCandidate], SectionExpansionReport]:
    report = SectionExpansionReport(section_requested=config.enabled, section_effective=False)
    if not config.enabled:
        report.section_fallback_reason = "disabled"
        return base_candidates, report
    primary_documents = list(scope_decision.tiers[0].documents) if getattr(scope_decision, "tiers", ()) else []
    allowed_ids = {_chunk_id(document) for document in primary_documents if _chunk_id(document)}
    if not allowed_ids:
        report.section_fallback_reason = "no_allowed_scope_documents"
        return base_candidates, report
    index = _section_index(cache_key, corpus_documents)
    ranked_sections, hint = index.select(query, allowed_ids, config.candidate_k)
    report.hint = hint
    if not ranked_sections:
        report.section_fallback_reason = "section_metadata_unavailable"
        return base_candidates, report

    base_by_id = {candidate.chunk_id: candidate for candidate in base_candidates}
    base_sections = defaultdict(list)
    for candidate in base_candidates:
        identity = section_identity(candidate.document)
        base_sections[(identity.document_id, identity.normalized_section)].append(candidate)
        candidate.pre_section_rank = candidate.final_rank

    anchors = []
    new_candidates = []
    selected_records = []
    for section_rank, (score, record, _) in enumerate(ranked_sections, start=1):
        key = (record.identity.document_id, record.identity.normalized_section)
        existing = base_sections.get(key, [])
        if existing:
            anchor = min(existing, key=lambda item: item.final_rank or 10**9)
        else:
            concepts = set(hint.preferred_concepts)
            scored = [
                (_overlap(tokens, concepts), document)
                for document, tokens in zip(record.documents, record.chunk_tokens)
                if _chunk_id(document) in allowed_ids
            ]
            scored.sort(key=lambda item: (-item[0], _chunk_id(item[1])))
            if not scored:
                continue
            anchor = RetrievalCandidate(document=scored[0][1], retrieval_source="section")
            anchor.section_expanded = True
            anchor.scope_match = "primary"
            anchor.scope_level = scope_decision.tiers[0].level
            relations = {
                identity_relation(identity, identity_from_metadata(anchor.metadata))
                for identity in getattr(scope_decision, "query_identities", ())
            }
            anchor.identity_relation = next((relation.value for relation in (
                IdentityRelation.EXACT_MODEL, IdentityRelation.SAME_SERIES,
                IdentityRelation.SAME_FAMILY, IdentityRelation.MISMATCH,
            ) if relation in relations), IdentityRelation.UNKNOWN.value)
            new_candidates.append(anchor)
        anchor.section_rank = section_rank
        anchor.section_candidate_source = "section_retrieval"
        anchors.append((anchor, record))
        selected_records.append({
            "rank": section_rank,
            "document_id": record.identity.document_id,
            "normalized_section": record.identity.normalized_section,
            "score": score,
        })

    # New section anchors take precedence; neighbors are round-robin by distance.
    additions = []
    seen = set(base_by_id)
    for candidate in new_candidates:
        if candidate.chunk_id not in seen:
            additions.append(candidate)
            seen.add(candidate.chunk_id)
    for distance in range(1, config.neighbor_window + 1):
        for direction in (-1, 1):
            for anchor, record in anchors:
                anchor_index = next((index for index, document in enumerate(record.documents) if _chunk_id(document) == anchor.chunk_id), None)
                if anchor_index is None:
                    continue
                neighbor_index = anchor_index + direction * distance
                if not 0 <= neighbor_index < len(record.documents):
                    continue
                document = record.documents[neighbor_index]
                chunk_id = _chunk_id(document)
                if chunk_id in seen or chunk_id not in allowed_ids:
                    continue
                neighbor = RetrievalCandidate(
                    document=document,
                    retrieval_source="section",
                    section_expanded=True,
                    section_rank=anchor.section_rank,
                    neighbor_distance=distance,
                    section_candidate_source="same_section_neighbor",
                    identity_relation=anchor.identity_relation,
                    scope_match="primary",
                    scope_level=scope_decision.tiers[0].level,
                )
                additions.append(neighbor)
                seen.add(chunk_id)

    protected = []
    anchor_ids = {anchor.chunk_id for anchor, _ in anchors}
    for candidate in base_candidates:
        if candidate.chunk_id in anchor_ids or (getattr(scope_decision, "identifiers", ()) and candidate.exact_metadata_match):
            protected.append(candidate)
    max_new = max(0, min(config.max_expanded, budget - len(protected)))
    selected_additions = additions[:max_new]
    keep_count = max(0, budget - len(selected_additions))
    kept = []
    for candidate in protected + base_candidates:
        if candidate not in kept and len(kept) < keep_count:
            kept.append(candidate)
    merged = kept + selected_additions
    for rank, candidate in enumerate(merged, start=1):
        candidate.final_rank = rank
    report.section_effective = True
    report.section_expansion_used = bool(selected_additions)
    report.section_candidates_added = len(selected_additions)
    report.candidate_budget_overflow = max(0, len(additions) - len(selected_additions))
    report.selected_sections = selected_records
    return merged, report
