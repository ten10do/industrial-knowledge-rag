"""Generic, data-driven product identity normalization and comparison."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum


HYPHENS = "‐‑‒–—−﹣－"
SERIES_PATTERN = re.compile(
    r"^(?P<family>.+?)\s+(?P<number>\d{2,5})(?:\s+|-)series(?:\s+\((?P<members>[^)]+)\))?$",
    re.IGNORECASE,
)
MODEL_SUFFIX_PATTERN = re.compile(r"^(?P<family>.+?)\s+(?P<model>[a-z]*\d[a-z0-9]*)$", re.IGNORECASE)


class IdentityRelation(str, Enum):
    EXACT_MODEL = "EXACT_MODEL"
    SAME_SERIES = "SAME_SERIES"
    SAME_FAMILY = "SAME_FAMILY"
    UNKNOWN = "UNKNOWN"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True)
class ProductIdentity:
    manufacturer: str = ""
    product_family: str = ""
    product_series: str = ""
    equipment_model: str = ""
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {**asdict(self), "aliases": list(self.aliases)}


def _display_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("/uni00A0", " ")
    for hyphen in HYPHENS:
        text = text.replace(hyphen, "-")
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip(" ,;:")


def normalize_identity_text(value: object) -> str:
    """Normalize formatting without removing model-significant letters or digits."""
    return re.sub(r"\s+", " ", _display_text(value).replace("-", " ")).casefold()


def _metadata_aliases(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = re.split(r"[|;]", str(value or ""))
    return tuple(item for item in (_display_text(part) for part in raw) if item)


def identity_from_metadata(metadata: dict) -> ProductIdentity:
    manufacturer = _display_text(metadata.get("manufacturer", ""))
    model = _display_text(metadata.get("equipment_model", ""))
    family = _display_text(metadata.get("product_family", ""))
    series = _display_text(metadata.get("product_series", ""))
    aliases = list(_metadata_aliases(metadata.get("model_aliases", metadata.get("aliases", ""))))

    series_match = SERIES_PATTERN.match(model)
    if series_match:
        family = family or series_match.group("family")
        series = series or f"{series_match.group('family')} {series_match.group('number')}"
        aliases.extend((model, series, f"{series} series"))
        members = re.findall(r"[a-z]*\d[a-z0-9]*", series_match.group("members") or "", re.IGNORECASE)
        aliases.extend(f"{family} {member}" for member in members)
    elif model:
        suffix = MODEL_SUFFIX_PATTERN.match(model)
        if suffix:
            family = family or suffix.group("family")
        elif match := re.match(r"^([a-z]+\d*)-", model, re.IGNORECASE):
            family = family or match.group(1)
        elif match := re.match(r"^([a-z]+)\d", model, re.IGNORECASE):
            family = family or match.group(1)
        aliases.append(model)

    aliases.extend(value for value in (series, f"{series} series" if series else "") if value)
    unique = {}
    for alias in aliases:
        if normalized := normalize_identity_text(alias):
            unique.setdefault(normalized, _display_text(alias))
    return ProductIdentity(manufacturer, family, series, model, tuple(unique.values()))


def identities_from_documents(documents: list) -> list[ProductIdentity]:
    identities = {}
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        identity = identity_from_metadata(metadata)
        key = (
            normalize_identity_text(identity.manufacturer),
            normalize_identity_text(identity.product_family),
            normalize_identity_text(identity.product_series),
            normalize_identity_text(identity.equipment_model),
        )
        if any(key):
            identities.setdefault(key, identity)
    return list(identities.values())


def _bare_model_token(identity: ProductIdentity) -> str:
    """Return a safe standalone model token for corpus-unique resolution."""
    if identity.product_series:
        return ""
    match = MODEL_SUFFIX_PATTERN.match(identity.equipment_model)
    return match.group("model") if match else ""


def identities_from_query(
    query: str,
    documents: list,
    *,
    corpus_identities: list[ProductIdentity] | None = None,
) -> tuple[ProductIdentity, ...]:
    """Resolve every explicitly mentioned known identity, including comparisons."""
    identities = corpus_identities if corpus_identities is not None else identities_from_documents(documents)
    matches: list[tuple[int, int, ProductIdentity]] = []
    for identity in identities:
        aliases = {identity.equipment_model, *identity.aliases} - {""}
        # A family-level manual may use the family name as its metadata model.
        # It establishes family scope, but must not turn an unknown sibling
        # model mentioned in the query into an exact-model match.
        if (
            not identity.product_series
            and normalize_identity_text(identity.equipment_model)
            == normalize_identity_text(identity.product_family)
        ):
            aliases = set()
        if identity.product_series:
            series_aliases = {
                normalize_identity_text(identity.product_series),
                normalize_identity_text(f"{identity.product_series} series"),
                normalize_identity_text(identity.equipment_model),
            }
            aliases = {
                alias for alias in aliases
                if normalize_identity_text(alias) not in series_aliases
            }
        aliases = sorted(
            aliases,
            key=lambda value: -len(normalize_identity_text(value)),
        )
        for alias in aliases:
            position = _mention_position(query, alias)
            if position is not None:
                matches.append((position, -len(normalize_identity_text(alias)), identity))
                break

    bare_tokens: dict[str, list[ProductIdentity]] = {}
    for identity in identities:
        if token := normalize_identity_text(_bare_model_token(identity)):
            bare_tokens.setdefault(token, []).append(identity)
    for token, token_identities in bare_tokens.items():
        if len(token_identities) != 1 or any(item[2] == token_identities[0] for item in matches):
            continue
        position = _mention_position(query, token)
        if position is not None:
            matches.append((position, -len(token), token_identities[0]))

    ordered: list[ProductIdentity] = []
    for _, _, identity in sorted(matches, key=lambda item: (item[0], item[1])):
        if identity not in ordered:
            ordered.append(identity)
    return tuple(ordered)


def _mentioned(query: str, value: str) -> bool:
    normalized_query = f" {normalize_identity_text(query)} "
    normalized_value = normalize_identity_text(value)
    return bool(normalized_value and f" {normalized_value} " in normalized_query)


def _mention_position(query: str, value: str) -> int | None:
    normalized_query = normalize_identity_text(query)
    normalized_value = normalize_identity_text(value)
    if not normalized_value:
        return None
    match = re.search(rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])", normalized_query)
    return match.start() if match else None


def identity_from_query(
    query: str,
    documents: list,
    *,
    corpus_identities: list[ProductIdentity] | None = None,
    explicit_identities: tuple[ProductIdentity, ...] | None = None,
) -> tuple[ProductIdentity, str]:
    identities = corpus_identities if corpus_identities is not None else identities_from_documents(documents)
    normalized_query = normalize_identity_text(query)
    manufacturers = [item.manufacturer for item in identities if item.manufacturer]
    manufacturer = next((value for value in manufacturers if _mentioned(query, value)), "")

    explicit_identities = (
        explicit_identities
        if explicit_identities is not None
        else identities_from_query(query, documents, corpus_identities=identities)
    )
    if explicit_identities:
        identity = explicit_identities[0]
        return ProductIdentity(
            manufacturer or identity.manufacturer,
            identity.product_family,
            identity.product_series,
            identity.equipment_model,
            identity.aliases,
        ), "EXACT_MODEL"

    matches = []
    for identity in identities:
        if identity.product_series and (position := _mention_position(query, identity.product_series)) is not None:
            matches.append((position, 1, -len(normalize_identity_text(identity.product_series)), "", identity))
    for identity in identities:
        if identity.product_series:
            series_aliases = {
                normalize_identity_text(identity.product_series),
                normalize_identity_text(f"{identity.product_series} series"),
                normalize_identity_text(identity.equipment_model),
            }
            exact_aliases = [alias for alias in identity.aliases if normalize_identity_text(alias) not in series_aliases]
        else:
            exact_aliases = list(identity.aliases)
            if normalize_identity_text(identity.equipment_model) == normalize_identity_text(identity.product_family):
                exact_aliases = []
        for alias in exact_aliases:
            if (position := _mention_position(query, alias)) is not None:
                matches.append((position, 0, -len(normalize_identity_text(alias)), alias, identity))
    if matches:
        _, specificity, _, alias, identity = min(matches)
        if specificity == 1:
            return ProductIdentity(
                manufacturer or identity.manufacturer,
                identity.product_family,
                identity.product_series,
                "",
                identity.aliases,
            ), "SERIES"
        return ProductIdentity(
            manufacturer or identity.manufacturer,
            identity.product_family,
            identity.product_series,
            alias,
            identity.aliases,
        ), "EXACT_MODEL"

    for identity in identities:
        family = identity.product_family
        if not family:
            continue
        unknown_model = re.search(
            rf"\b{re.escape(normalize_identity_text(family))}\s+[a-z]*\d[a-z0-9]*\b",
            normalized_query,
            re.IGNORECASE,
        )
        if unknown_model:
            model = _display_text(unknown_model.group(0))
            return ProductIdentity(manufacturer or identity.manufacturer, family, "", model, (model,)), "EXACT_MODEL"
        if _mentioned(query, family):
            return ProductIdentity(manufacturer or identity.manufacturer, family, "", "", (family,)), "FAMILY"
    return ProductIdentity(manufacturer=manufacturer), "UNKNOWN"


def identity_relation(query: ProductIdentity, candidate: ProductIdentity) -> IdentityRelation:
    if not (query.product_family or query.product_series or query.equipment_model):
        return IdentityRelation.UNKNOWN
    if not (candidate.product_family or candidate.product_series or candidate.equipment_model):
        return IdentityRelation.UNKNOWN
    if (
        query.manufacturer
        and candidate.manufacturer
        and normalize_identity_text(query.manufacturer) != normalize_identity_text(candidate.manufacturer)
    ):
        return IdentityRelation.MISMATCH

    query_model = normalize_identity_text(query.equipment_model)
    candidate_aliases = {normalize_identity_text(value) for value in candidate.aliases}
    if query_model and query_model in candidate_aliases:
        return IdentityRelation.EXACT_MODEL
    if query.product_series and normalize_identity_text(candidate.equipment_model) in {
        normalize_identity_text(value) for value in query.aliases
    }:
        return IdentityRelation.SAME_SERIES
    if query.product_series and candidate.product_series and (
        normalize_identity_text(query.product_series) == normalize_identity_text(candidate.product_series)
    ):
        return IdentityRelation.SAME_SERIES
    if query.product_family and candidate.product_family and (
        normalize_identity_text(query.product_family) == normalize_identity_text(candidate.product_family)
    ):
        return IdentityRelation.SAME_FAMILY
    return IdentityRelation.MISMATCH


def identity_is_compatible(query: ProductIdentity, candidate: ProductIdentity) -> bool:
    relation = identity_relation(query, candidate)
    if relation == IdentityRelation.EXACT_MODEL:
        return True
    if relation == IdentityRelation.SAME_SERIES:
        return not query.equipment_model
    if relation == IdentityRelation.SAME_FAMILY:
        return not query.equipment_model and not query.product_series
    return relation == IdentityRelation.UNKNOWN and not (
        query.product_family or query.product_series or query.equipment_model
    )
