"""Generic industrial technical normalization shared by Evidence and Support gates.

This module holds data-driven, verifiable registries (protocols, vendor/equipment
names, parameter references) and term-matching helpers. It never performs
semantic inference and never encodes rules for a specific frozen query.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Version tag for the Evidence/Support generalization rules. Bump whenever the
# cross-corpus normalization / matching rules change; the calibration freeze
# records this so later runs can be attributed.
EVIDENCE_SUPPORT_RULE_VERSION = "evidence-v323.1-candidate"

# Industrial fieldbus / communication protocols, mapped to their acceptable
# written forms. Distinct protocols are deliberately separate: a device that
# speaks PROFINET is never treated as supporting EtherNet/IP just because both
# are industrial networks.
PROTOCOL_ALIASES: dict[str, tuple[str, ...]] = {
    "profinet": ("profinet",),
    "ethernet_ip": ("ethernet/ip", "ethernet ip", "ethernet/ip™"),
    "modbus": ("modbus", "modbus rtu", "modbus/tcp", "modbus tcp"),
    "profibus": ("profibus", "profibus dp"),
    "devicenet": ("devicenet", "device net"),
    "ethercat": ("ethercat", "ether cat"),
    "opc_ua": ("opc ua", "opc-ua", "opcua"),
    "cip_sync": ("cip sync", "cip sync™"),
    "dlr": ("device-level ring", "device level ring", "dlr"),
}

# Vendor names and their recognizable equipment tokens. Each value maps to a
# vendor; the vendor key is the canonical manufacturer. Used only to detect a
# query that references equipment outside the current corpus (cross-equipment).
VENDOR_EQUIPMENT_TERMS: dict[str, tuple[str, ...]] = {
    "siemens": ("siemens", "simatic", "sinamics", "s7-1200", "s7-1500", "s7-300", "s7-400", "step 7", "tia portal"),
    "schneider electric": ("schneider", "modicon", "m241", "m258", "m340", "telemecanique"),
    "mitsubishi electric": ("mitsubishi", "melsec", "fx3u", "fx5u"),
    "rockwell automation": ("rockwell", "allen-bradley", "allen bradley", "compactlogix", "controllogix", "powerflex", "kinetix", "stratix"),
    "abb": ("acs580", "acs880", "acs550", "acs800", "fpno-21", "fpno-22"),
    "omron": ("cx-programmer", "cj2m", "cp1l", "cp1h", "nj-series", "nx-series"),
    "beckhoff": ("beckhoff", "twincat"),
    "yaskawa": ("yaskawa",),
    "danfoss": ("danfoss", "vlt"),
}

# Parameter/register literals used by the current industrial corpora. The
# concept word is intentionally excluded from the literal so a query such as
# "parameter 04.16" can be checked against a manual table containing "04.16".
PARAMETER_IDENTIFIER_LITERAL = (
    r"(?:\d{1,2}\.\d{1,2}|pr\.?\s*\d{1,4}|mw\s*\d{1,5}|"
    r"a\d{1,2}\s*[-/]\s*\d{1,4}|[acdeprst]\s*\d{2,5})"
)
PARAMETER_REFERENCE_PATTERN = re.compile(
    rf"(?<![a-z0-9])(?:(?P<concept>parameter|param|par\.?|register)"
    rf"\s*(?:number|no\.?)?\s*)?(?P<identifier>{PARAMETER_IDENTIFIER_LITERAL})(?![a-z0-9])",
    re.IGNORECASE,
)
_EXPLICIT_PARAMETER_REFERENCE_PATTERN = re.compile(
    rf"(?<![a-z0-9])(?P<concept>parameter|param|par\.?|register)"
    rf"\s*(?:number|no\.?)?\s*(?P<identifier>{PARAMETER_IDENTIFIER_LITERAL})(?![a-z0-9])",
    re.IGNORECASE,
)
_PARAMETER_IDENTIFIER_PATTERN = re.compile(
    rf"(?<![a-z0-9.])(?P<identifier>{PARAMETER_IDENTIFIER_LITERAL})(?![a-z0-9.])",
    re.IGNORECASE,
)
_BARE_PARAMETER_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9.])(?P<identifier>"
    r"(?:\d{1,2}\.\d{1,2}|pr\.?\s*\d{1,4}|mw\s*\d{1,5}|"
    r"a\d{1,2}\s*[-/]\s*\d{1,4}|[aprt]\s*\d{2,5})"
    r")(?![a-z0-9.])",
    re.IGNORECASE,
)
_BARE_NUMERIC_QUERY_CUE = re.compile(
    r"\b(?:what\s+(?:is|does)|explain|meaning\s+of|default\s+value\s+of)\b",
    re.IGNORECASE,
)

HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\u058a"


@dataclass(frozen=True)
class ParameterReference:
    concept: str
    identifier: str


def normalize_parameter_identifier(value: object) -> str:
    """Return the canonical literal without a leading concept word."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"^(?:parameter|param|par\.?|register)\s*(?:number|no\.?)?\s*", "", text)
    for hyphen in HYPHENS:
        text = text.replace(hyphen, "-")
    text = re.sub(r"\s+", "", text)
    if text.startswith("pr."):
        text = "pr" + text[3:]
    return text.upper()


def extract_parameter_references(query: str) -> tuple[ParameterReference, ...]:
    """Extract parameter concepts and literals without coupling their spelling.

    Bare alphanumeric forms are distinctive enough to accept directly. Bare
    dotted-numeric forms require question wording such as "What is 30.11?" so
    ordinary decimal measurements are not treated as parameter references.
    """
    text = str(query or "")
    references: list[ParameterReference] = []
    explicit_spans: list[tuple[int, int]] = []
    for match in _EXPLICIT_PARAMETER_REFERENCE_PATTERN.finditer(text):
        concept = "register" if match.group("concept").casefold() == "register" else "parameter"
        references.append(ParameterReference(concept, normalize_parameter_identifier(match.group("identifier"))))
        explicit_spans.append(match.span())

    for match in _BARE_PARAMETER_IDENTIFIER_PATTERN.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in explicit_spans):
            continue
        raw = match.group("identifier")
        if re.fullmatch(r"\d{1,2}\.\d{1,2}", raw) and not _BARE_NUMERIC_QUERY_CUE.search(text):
            continue
        references.append(ParameterReference("parameter", normalize_parameter_identifier(raw)))

    unique: dict[tuple[str, str], ParameterReference] = {}
    for reference in references:
        unique[(reference.concept, reference.identifier)] = reference
    return tuple(unique.values())


def contains_parameter_identifier(text: object, identifier: str) -> bool:
    """Check a canonical identifier as a standalone industrial token."""
    expected = normalize_parameter_identifier(identifier)
    return any(
        normalize_parameter_identifier(match.group("identifier")) == expected
        for match in _PARAMETER_IDENTIFIER_PATTERN.finditer(str(text or ""))
    )


def normalize_technical_text(value: object) -> str:
    """NFKC + casefold + whitespace collapse, preserving protocol hyphens."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/uni00a0", " ").replace("™", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_hyphen_insensitive(value: object) -> str:
    """Collapse hyphens and whitespace so 'start-up' == 'startup' == 'start up'.

    Used for concept/action/attribute matching, not for model identities where a
    hyphen can be significant (for example FPNO-21).
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("™", "")
    for hyphen in HYPHENS:
        text = text.replace(hyphen, "")
    text = text.replace("-", "")
    return re.sub(r"\s+", "", text).strip()


def contains_term(text: str, aliases: tuple[str, ...]) -> bool:
    """Return True if any alias (hyphen/whitespace-insensitive) occurs in text."""
    haystack = normalize_hyphen_insensitive(text)
    return any(normalize_hyphen_insensitive(alias) in haystack for alias in aliases)


def matched_terms(text: str, groups: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(name for name, aliases in groups.items() if contains_term(text, aliases))


def mentioned_term(query: str, term: str) -> bool:
    """Whole-token mention of a term in a query (hyphen preserved for models)."""
    haystack = normalize_technical_text(query)
    needle = normalize_technical_text(term)
    if not needle:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack, re.IGNORECASE) is not None


def corpus_manufacturers(documents: list) -> set[str]:
    manufacturers: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        value = normalize_technical_text(metadata.get("manufacturer", ""))
        if value:
            manufacturers.add(value)
    return manufacturers


def foreign_equipment_signal(query: str, documents: list) -> tuple[str, str] | None:
    """Return (vendor, term) when a query references equipment outside the corpus.

    Cross-equipment requests (for example configuring an Allen-Bradley drive
    with Omron software) are a strong unsupported signal. Only known vendor or
    equipment tokens are considered, so ordinary nouns never trigger it.
    """
    corpus = {normalize_technical_text(item) for item in corpus_manufacturers(documents)}
    if not corpus:
        return None
    for vendor, terms in VENDOR_EQUIPMENT_TERMS.items():
        if normalize_technical_text(vendor) in corpus:
            continue
        for term in terms:
            if mentioned_term(query, term):
                return vendor, term
    return None
