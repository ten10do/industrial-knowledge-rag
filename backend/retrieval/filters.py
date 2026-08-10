from __future__ import annotations

import re
from dataclasses import dataclass

from .product_identity import identity_from_query


IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:0x[0-9a-f]+|[fae]\d{2,5})(?![a-z0-9])", re.IGNORECASE
)
MODEL_PATTERN = re.compile(
    r"\b(?:[a-z]+\d*(?:-[a-z0-9]+)+|[a-z]+\s+g\d+)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class QueryAnalysis:
    error_code: str = ""
    equipment_model: str = ""
    manufacturer: str = ""
    equipment_type: str = ""
    document_type: str = ""
    knowledge_type: str = ""
    product_family: str = ""
    product_series: str = ""
    identity_confidence: str = "UNKNOWN"


def _values(documents: list, key: str) -> dict[str, str]:
    values = {}
    for document in documents:
        value = str((getattr(document, "metadata", {}) or {}).get(key, "")).strip()
        if value:
            values[value.lower()] = value
    return values


def analyze_query(query: str, documents: list) -> QueryAnalysis:
    normalized = (query or "").lower()
    codes = IDENTIFIER_PATTERN.findall(query or "")
    models = _values(documents, "equipment_model")
    manufacturers = _values(documents, "manufacturer")
    equipment_types = _values(documents, "equipment_type")
    document_types = _values(documents, "document_type")
    knowledge_types = _values(documents, "knowledge_type")

    def mentioned(values: dict[str, str]) -> str:
        return next((original for value, original in values.items() if value in normalized), "")

    identity, identity_confidence = identity_from_query(query, documents)
    model = identity.equipment_model or mentioned(models)
    if not model:
        model_match = MODEL_PATTERN.search(query or "")
        model = model_match.group(0) if model_match else ""
    return QueryAnalysis(
        error_code=(codes[0].upper() if codes else ""),
        equipment_model=model,
        manufacturer=identity.manufacturer or mentioned(manufacturers),
        equipment_type=mentioned(equipment_types),
        document_type=mentioned(document_types),
        knowledge_type=mentioned(knowledge_types),
        product_family=identity.product_family,
        product_series=identity.product_series,
        identity_confidence=identity_confidence,
    )


def filter_documents(documents: list, analysis: QueryAnalysis) -> tuple[list, bool]:
    """Apply strict exact metadata filters only when they produce candidates."""
    filtered = list(documents)
    applied = False
    for field in ("error_code", "equipment_model"):
        value = getattr(analysis, field)
        if not value:
            continue
        matches = [
            document
            for document in filtered
            if str((getattr(document, "metadata", {}) or {}).get(field, "")).lower()
            == value.lower()
        ]
        if matches:
            filtered, applied = matches, True
    return filtered, applied


def has_exact_metadata_match(document, analysis: QueryAnalysis) -> bool:
    metadata = getattr(document, "metadata", {}) or {}
    return bool(
        analysis.error_code
        and str(metadata.get("error_code", "")).lower() == analysis.error_code.lower()
    ) or bool(
        analysis.equipment_model
        and str(metadata.get("equipment_model", "")).lower()
        == analysis.equipment_model.lower()
    )
