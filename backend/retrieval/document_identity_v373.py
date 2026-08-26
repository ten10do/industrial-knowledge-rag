"""V3.73 DocumentIdentityResolver: query-agnostic document-level identity.

Extracts manufacturer/product_family/product_series/equipment_model from
the PDF filename using deterministic pattern matching. This is the minimum
viable identity source given no existing corpus manifest.
"""

from __future__ import annotations

import re
from pathlib import Path

# Known manufacturer patterns (deterministic, not fuzzy).
MANUFACTURER_PATTERNS = [
    (r"siemens", "siemens"),
    (r"\babb\b", "abb"),
    (r"schneider", "schneider electric"),
    (r"mitsubishi", "mitsubishi"),
    (r"danfoss", "danfoss"),
    (r"\bweg\b", "weg"),
    (r"rockwell", "rockwell"),
    (r"powerflex", "rockwell"),
    (r"omron", "omron"),
]

# Product series/model extraction from filename.
SERIES_PATTERNS = [
    # Siemens
    (r"s7[-_]?(\d{3,4})", "siemens", "s7-\\1"),
    (r"sinamics[\s_-]*g(\d{3}[a-z]?)", "siemens", "sinamics g\\1"),
    (r"sinamics[\s_-]*v(\d{2})", "siemens", "sinamics v\\1"),
    (r"sinamics[\s_-]*s(\d{2})", "siemens", "sinamics s\\1"),
    # ABB
    (r"acs(\d{3})", "abb", "acs\\1"),
    (r"acs(\d{3})[-_]?(\w+)", "abb", "acs\\1"),
    # Schneider
    (r"atv(\d{3})", "schneider electric", "altivar atv\\1"),
    (r"m221", "schneider electric", "modicon m221"),
    (r"altivar[\s_-]*(\w+)", "schneider electric", "altivar \\1"),
    # Mitsubishi
    (r"fr[-_]?e(\d{3})", "mitsubishi", "fr-e\\1"),
    (r"fr[-_]?a(\d{3})", "mitsubishi", "fr-a\\1"),
    # Danfoss
    (r"fc(\d{2,3})", "danfoss", "fc\\1"),
    (r"vlt[\s_-]*(\w+)", "danfoss", "vlt \\1"),
    # Rockwell / PowerFlex
    (r"powerflex[\s_-]*(\d{3})", "rockwell", "powerflex \\1"),
    (r"powerflex[\s_-]*(\d+)", "rockwell", "powerflex \\1"),
]


def resolve_document_identity(file_path: str) -> dict:
    """Extract structured identity from PDF filename.

    Returns dict with keys matching ProductIdentity schema:
    manufacturer, product_family, product_series, equipment_model.
    """
    filename = Path(file_path).stem.lower()
    result = {
        "manufacturer": "",
        "product_family": "",
        "product_series": "",
        "equipment_model": "",
        "identity_source": "filename_resolver_v373",
    }

    # Manufacturer detection.
    for pattern, mfr in MANUFACTURER_PATTERNS:
        if re.search(pattern, filename):
            result["manufacturer"] = mfr
            break

    # Series/model extraction.
    for pattern, expected_mfr, series_template in SERIES_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            if not result["manufacturer"]:
                result["manufacturer"] = expected_mfr
            elif result["manufacturer"] != expected_mfr:
                continue
            group_val = ""
            if match.groups():
                group_val = match.group(1) or ""
                series = series_template.replace("\\1", group_val)
            else:
                series = series_template
            result["product_series"] = series
            result["equipment_model"] = series
            break

    # Fallback: use full stem as model if nothing matched.
    if not result["product_series"] and result["manufacturer"]:
        result["product_series"] = filename.replace("_", " ").replace("-", " ").strip()
        result["equipment_model"] = result["product_series"]

    return result


def enrich_chunk_metadata(chunks: list, file_path: str) -> list:
    """Attach document-level identity to each chunk's metadata."""
    identity = resolve_document_identity(file_path)
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None)
        if meta is None:
            continue
        # Don't overwrite existing more-specific values.
        for key in ("manufacturer", "product_family", "product_series",
                    "equipment_model"):
            if key in identity and identity[key]:
                meta.setdefault(key, identity[key])
        meta.setdefault("identity_source", identity.get("identity_source", ""))
    return chunks
