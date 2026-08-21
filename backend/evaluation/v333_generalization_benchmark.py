"""V3.33 Independent Mixed Candidate Generalization Benchmark — framework.

Commttable, deterministic, self-validating schema/validation helpers used to
author, freeze, and (in the gitignored runner) execute Corpus K against the
current integrated mixed evidence candidate.  This module carries NO private
data — it only defines the contract that the (gitignored) benchmark_private
Corpus K artifacts must satisfy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

CORPUS_K_ID = "K"
CORPUS_K_NAME = "V3.33 Independent Mixed Candidate Generalization Benchmark"
MIN_MANUAL_COUNT = 6
MIN_QUERY_COUNT = 96
MAX_QUERY_COUNT = 120
MIN_HARD_QUERY_RATIO = 0.90

GENERALIZATION_STATUSES = (
    "GENERALIZES",
    "PARTIAL_GENERALIZATION",
    "GENERALIZATION_FAILED",
    "RUNTIME_INVALID",
)

# Reuse the V3.31 pre-registered gate without lowering any threshold.  V3.33
# applies it to K-CHECK, with K-TRAIN retained as a diagnostic slice.
GENERALIZATION_POLICY = {
    "generalizes": {
        "accuracy_min": 0.70,
        "abstain_recall_min": 0.60,
        "answerable_recall_min": 0.55,
        "false_answer_rate_max": 0.20,
        "false_refusal_rate_max": 0.45,
        "verification_accuracy_min": 0.68,
        "open_accuracy_min": 0.65,
    },
    "partial": {
        "accuracy_min": 0.60,
        "abstain_recall_min": 0.45,
        "answerable_recall_min": 0.40,
    },
}

SPLITS = ("K-TRAIN", "K-CHECK")

# Non-negotiable annotation vocabulary.
QUERY_TYPES = ("VERIFICATION", "OPEN")
GROUND_TRUTHS = ("ANSWER", "ABSTAIN")
DIFFICULTIES = ("L1", "L2", "L3", "L4", "L5")
CONFIDENCES = ("HIGH", "MEDIUM", "AMBIGUOUS")
TABLE_RECOVERABLE = ("YES", "NO")

# §12 product categories (>= 3 required; all optional in validation).
EQUIPMENT_CATEGORIES = (
    "plc_controller",
    "servo_drive",
    "drive",
    "frequency_inverter",
    "industrial_communication",
    "remote_io_fieldbus",
    "safety",
    "motion",
    "industrial_ethernet",
    "machine_vision",
)

# §24 open relation types.
RELATION_TYPES = (
    "HAS_IDENTIFIER",
    "HAS_VALUE",
    "HAS_DEFAULT_VALUE",
    "HAS_RANGE",
    "HAS_SETTING",
    "USES_TERMINAL",
    "USES_CHANNEL",
    "HAS_ATTRIBUTE",
    "REQUIRES_ACTION",
    "LOCATED_AT",
    "USES_PROTOCOL",
    "HAS_PROCEDURE",
)

# §21 verification coverage facets.
VERIFICATION_FACETS = (
    "identity",
    "identifier",
    "protocol",
    "value",
    "attribute",
    "action",
    "condition",
    "relation",
)

# §46 failure attribution vocabulary.
FAILURE_ATTRIBUTIONS = frozenset({
    "RETRIEVAL_MISSING_EVIDENCE",
    "PARSER_STRUCTURE_LOSS",
    "PRODUCT_IDENTITY_ERROR",
    "BASE_RULE_FAILURE",
    "NLI_ROUTER_MISS",
    "NLI_JUDGE_FAILURE",
    "OPEN_REQUIREMENT_PARSE_FAILURE",
    "OPEN_RELATION_MATCH_FAILURE",
    "OPEN_HARD_GATE_REFUSAL",
    "OPEN_FALSE_RELAXATION",
    "BOUNDARY_ADAPTER",
    "ANNOTATION_AMBIGUITY",
    "OTHER",
})

# §41 diagnostics required on every query row.
RUNTIME_DIAGNOSTIC_KEYS = (
    "query_path",
    "base_rule_decision",
    "nli_router_triggered",
    "nli_decision",
    "open_sufficiency_invoked",
    "open_sufficiency_status",
    "final_decision_source",
    "grounding_status",
)

# Required annotation fields (§30) — every query object must contain all of them.
REQUIRED_ANNOTATION_FIELDS = (
    "query_id",
    "query",
    "split",
    "ground_truth",
    "query_type",
    "manufacturer",
    "product",
    "document",
    "difficulty",
    "document_style",
    "confidence",
    "target",
    "relation",
    "critical_requirements",
    "expected_evidence",
    "scope",
    "relation_annotation",
    "table_structure_recoverable",
    "annotation_rationale",
)
OPTIONAL_ANNOTATION_FIELDS = ("requested_slot", "answer_shape")


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warning"
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [{"code": i.code, "message": i.message} for i in self.errors],
            "warnings": [{"code": i.code, "message": i.message} for i in self.warnings],
        }


def _normalize(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower())


def fingerprint(obj: Any) -> str:
    """Deterministic SHA256 over a JSON-canonical representation."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_manifests(manifests: list[dict]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    docs_by_id: dict[str, dict] = {}
    seen_url: dict[str, str] = {}
    seen_sha: dict[str, str] = {}
    required = ("document_id", "file", "official_url", "manufacturer", "product_family",
                "product_series", "equipment_type", "language")
    if len(manifests) < MIN_MANUAL_COUNT:
        issues.append(ValidationIssue("error", "INSUFFICIENT_MANUALS",
                                      f"need >={MIN_MANUAL_COUNT} manuals, got {len(manifests)}"))
    for m in manifests:
        for k in required:
            if not m.get(k):
                issues.append(ValidationIssue("error", "MANIFEST_MISSING_FIELD",
                                              f"{m.get('document_id', '?')} missing {k}"))
        did = m.get("document_id", "")
        if did in docs_by_id:
            issues.append(ValidationIssue("error", "DUPLICATE_DOCUMENT_ID", did))
        docs_by_id[did] = m
        if m.get("equipment_type") not in EQUIPMENT_CATEGORIES:
            issues.append(ValidationIssue("error", "BAD_EQUIPMENT_TYPE",
                                          f"{did}: {m.get('equipment_type')}"))
        url = (m.get("official_url") or "").strip().lower()
        if url and url in seen_url:
            issues.append(ValidationIssue("error", "URL_OVERLAP",
                                          f"{did} shares URL with {seen_url[url]}"))
        seen_url[url] = did
        sha = (m.get("sha256") or "").strip().upper()
        if sha and sha in seen_sha:
            issues.append(ValidationIssue("error", "HASH_OVERLAP",
                                          f"{did} shares SHA256 with {seen_sha[sha]}"))
        seen_sha[sha] = did
    manufacturers = {m.get("manufacturer") for m in manifests}
    if len(manufacturers) < 4:
        issues.append(ValidationIssue("error", "INSUFFICIENT_MANUFACTURERS",
                                      f"need >=4 distinct manufacturers, got {len(manufacturers)}"))
    categories = {m.get("equipment_type") for m in manifests}
    if len(categories) < 3:
        issues.append(ValidationIssue("error", "INSUFFICIENT_CATEGORIES",
                                      f"need >=3 product categories, got {len(categories)}"))
    return ValidationReport(tuple(issues))


def validate_annotations(annotations: list[dict]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    ids: set[str] = set()
    for a in annotations:
        qid = a.get("query_id", "?")
        for k in REQUIRED_ANNOTATION_FIELDS:
            if k not in a:
                issues.append(ValidationIssue("error", "ANNOTATION_MISSING_FIELD",
                                              f"{qid} missing {k}"))
        if qid in ids:
            issues.append(ValidationIssue("error", "DUPLICATE_QUERY_ID", qid))
        ids.add(qid)
        if a.get("query_type") not in QUERY_TYPES:
            issues.append(ValidationIssue("error", "BAD_QUERY_TYPE", f"{qid}: {a.get('query_type')}"))
        if a.get("ground_truth") not in GROUND_TRUTHS:
            issues.append(ValidationIssue("error", "BAD_GROUND_TRUTH", f"{qid}: {a.get('ground_truth')}"))
        if a.get("split") not in SPLITS:
            issues.append(ValidationIssue("error", "BAD_SPLIT", f"{qid}: {a.get('split')}"))
        if a.get("difficulty") not in DIFFICULTIES:
            issues.append(ValidationIssue("error", "BAD_DIFFICULTY", f"{qid}: {a.get('difficulty')}"))
        if a.get("confidence") not in CONFIDENCES:
            issues.append(ValidationIssue("error", "BAD_CONFIDENCE", f"{qid}: {a.get('confidence')}"))
        if a.get("table_structure_recoverable") not in TABLE_RECOVERABLE:
            issues.append(ValidationIssue("error", "BAD_TABLE_RECOVERABLE", f"{qid}: {a.get('table_structure_recoverable')}"))
        if a.get("relation") not in RELATION_TYPES:
            issues.append(ValidationIssue("error", "BAD_RELATION_TYPE", f"{qid}: {a.get('relation')}"))
        if a["query_type"] == "OPEN" and not a.get("requested_slot"):
            issues.append(ValidationIssue("warning", "OPEN_MISSING_REQUESTED_SLOT",
                                          f"{qid}: open query without requested_slot"))
    if not annotations:
        issues.append(ValidationIssue("error", "EMPTY_ANNOTATIONS", "no annotations"))
    return ValidationReport(tuple(issues))


def validate_distribution(annotations: list[dict], split: str) -> ValidationReport:
    issues: list[ValidationIssue] = []
    rows = [a for a in annotations if a.get("split") == split]
    n = len(rows)
    if n == 0:
        issues.append(ValidationIssue("error", "EMPTY_SPLIT", split))
        return ValidationReport(tuple(issues))
    ver = sum(1 for a in rows if a.get("query_type") == "VERIFICATION")
    open_ = sum(1 for a in rows if a.get("query_type") == "OPEN")
    ans = sum(1 for a in rows if a.get("ground_truth") == "ANSWER")
    hard = sum(1 for a in rows if a.get("difficulty") in ("L3", "L4", "L5"))
    ver_ratio = ver / n
    ans_ratio = ans / n
    hard_ratio = hard / n
    if not (0.45 <= ver_ratio <= 0.55):
        issues.append(ValidationIssue("error", "QUERY_TYPE_IMBALANCE",
                                      f"{split}: verification {ver}/{n} = {ver_ratio:.3f} (need 0.45-0.55)"))
    if not (0.40 <= ans_ratio <= 0.60):
        issues.append(ValidationIssue("error", "ANSWER_BALANCE_VIOLATION",
                                      f"{split}: ANSWER {ans}/{n} = {ans_ratio:.3f} (need 0.40-0.60)"))
    if hard_ratio < MIN_HARD_QUERY_RATIO:
        issues.append(ValidationIssue("error", "DIFFICULTY_VIOLATION",
                                      f"{split}: L3-L5 {hard}/{n} = {hard_ratio:.3f} "
                                      f"(need >= {MIN_HARD_QUERY_RATIO:.2f})"))
    # Per-split VERIFICATION/OPEN cannot be near-single-type even if ratio valid.
    if ver == 0 or open_ == 0:
        issues.append(ValidationIssue("error", "SINGLE_TYPE_SPLIT",
                                      f"{split}: verification={ver}, open={open_} (both must be > 0)"))
    return ValidationReport(tuple(issues))


def validate_document_disjoint(annotations: list[dict], manifests: list[dict]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    doc_to_split: dict[str, str] = {}
    for a in annotations:
        doc = a.get("document")
        split = a.get("split")
        if doc and split:
            if doc in doc_to_split and doc_to_split[doc] != split:
                issues.append(ValidationIssue("error", "DOCUMENT_SPLIT_OVERLAP",
                                              f"{doc} assigned to {doc_to_split[doc]} and {split}"))
            doc_to_split[doc] = split
    train_docs = {d for d, s in doc_to_split.items() if s == "K-TRAIN"}
    check_docs = {d for d, s in doc_to_split.items() if s == "K-CHECK"}
    overlap = train_docs & check_docs
    if overlap:
        issues.append(ValidationIssue("error", "DOCUMENT_OVERLAP", ",".join(sorted(overlap))))
    manifest_docs = {m.get("document_id") for m in manifests}
    unknown = (train_docs | check_docs) - manifest_docs
    if unknown:
        issues.append(ValidationIssue("error", "ANNOTATION_REFERENCES_UNKNOWN_DOC", ",".join(sorted(unknown))))
    return ValidationReport(tuple(issues))


def validate_identifier_quota(annotations: list[dict]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    identifiers = [a for a in annotations if a.get("relation") == "HAS_IDENTIFIER"]
    positive = sum(1 for a in identifiers if a.get("ground_truth") == "ANSWER")
    negative = sum(1 for a in identifiers if a.get("ground_truth") == "ABSTAIN")
    for split in SPLITS:
        has = [a for a in identifiers if a.get("split") == split]
        if not has:
            issues.append(ValidationIssue("error", "IDENTIFIER_MISSING_IN_SPLIT",
                                          f"{split}: no identifier queries"))
    if positive < 12 or negative < 12:
        issues.append(ValidationIssue("error", "IDENTIFIER_QUOTA",
                                      f"identifier positive={positive}, negative={negative} (need >=12 each)"))
    return ValidationReport(tuple(issues))


def validate_table_quota(annotations: list[dict]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    tables = [a for a in annotations if a.get("table_structure_recoverable") == "YES" or
              a.get("document_style") == "PARAMETER_TABLE" or a.get("document_style") == "CONFIGURATION_TABLE"]
    if len(tables) < 12:
        issues.append(ValidationIssue("error", "TABLE_QUOTA",
                                      f"need >=12 table-derived cases, got {len(tables)}"))
    for a in tables:
        if a.get("table_structure_recoverable") not in TABLE_RECOVERABLE:
            issues.append(ValidationIssue("error", "TABLE_RECOVERABLE_UNSET", a.get("query_id", "?")))
    return ValidationReport(tuple(issues))


def validate_benchmark_scope(annotations: list[dict], manifests: list[dict]) -> ValidationReport:
    """Validate V3.33 size, confidence, and split-independence gates."""
    issues: list[ValidationIssue] = []
    count = len(annotations)
    if not (MIN_QUERY_COUNT <= count <= MAX_QUERY_COUNT):
        issues.append(ValidationIssue(
            "error", "QUERY_COUNT_VIOLATION",
            f"need {MIN_QUERY_COUNT}-{MAX_QUERY_COUNT} queries, got {count}",
        ))

    non_high = [a.get("query_id", "?") for a in annotations if a.get("confidence") != "HIGH"]
    if non_high:
        issues.append(ValidationIssue(
            "error", "NON_HIGH_CONFIDENCE_QUERY", ",".join(non_high),
        ))

    normalized_queries: dict[str, str] = {}
    for annotation in annotations:
        query_id = annotation.get("query_id", "?")
        normalized = _normalize(annotation.get("query", ""))
        if normalized in normalized_queries:
            issues.append(ValidationIssue(
                "error", "QUERY_SPLIT_OVERLAP",
                f"{query_id} duplicates {normalized_queries[normalized]}",
            ))
        normalized_queries[normalized] = query_id

    document_to_manufacturer = {
        m.get("document_id"): m.get("manufacturer") for m in manifests
    }
    split_documents = {
        split: {a.get("document") for a in annotations if a.get("split") == split}
        for split in SPLITS
    }
    represented = set().union(*split_documents.values())
    missing_documents = set(document_to_manufacturer) - represented
    if missing_documents:
        issues.append(ValidationIssue(
            "error", "UNUSED_CORPUS_DOCUMENT", ",".join(sorted(missing_documents)),
        ))

    split_manufacturers = {
        split: {document_to_manufacturer.get(doc) for doc in documents}
        for split, documents in split_documents.items()
    }
    manufacturer_overlap = split_manufacturers["K-TRAIN"] & split_manufacturers["K-CHECK"]
    manufacturer_overlap.discard(None)
    if manufacturer_overlap:
        issues.append(ValidationIssue(
            "error", "MANUFACTURER_SPLIT_OVERLAP", ",".join(sorted(manufacturer_overlap)),
        ))

    split_candidates = {
        split: {
            candidate
            for a in annotations if a.get("split") == split
            for candidate in (a.get("evidence_chunk_ids") or [])
        }
        for split in SPLITS
    }
    candidate_overlap = split_candidates["K-TRAIN"] & split_candidates["K-CHECK"]
    if candidate_overlap:
        issues.append(ValidationIssue(
            "error", "CANDIDATE_SPLIT_OVERLAP", ",".join(sorted(candidate_overlap)),
        ))
    return ValidationReport(tuple(issues))


def validate_failure_attribution(attributions: Iterable[str]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for att in attributions:
        if att not in FAILURE_ATTRIBUTIONS:
            issues.append(ValidationIssue("error", "BAD_FAILURE_ATTRIBUTION", att))
    return ValidationReport(tuple(issues))


def validate_runtime_diagnostics(row: dict) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for k in RUNTIME_DIAGNOSTIC_KEYS:
        if k not in row:
            issues.append(ValidationIssue("error", "MISSING_DIAGNOSTIC", k))
    return ValidationReport(tuple(issues))


def classify_generalization(check_overall: dict, check_verification: dict,
                             check_open: dict, *, runtime_valid: bool) -> str:
    """Classify the frozen K-CHECK result using the pre-registered policy."""
    if not runtime_valid:
        return "RUNTIME_INVALID"

    strict = GENERALIZATION_POLICY["generalizes"]
    if (
        check_overall.get("accuracy", 0) >= strict["accuracy_min"]
        and check_overall.get("abstain_recall", 0) >= strict["abstain_recall_min"]
        and check_overall.get("answerable_recall", 0) >= strict["answerable_recall_min"]
        and check_overall.get("false_answer_rate", 1) <= strict["false_answer_rate_max"]
        and check_overall.get("false_refusal_rate", 1) <= strict["false_refusal_rate_max"]
        and check_verification.get("accuracy", 0) >= strict["verification_accuracy_min"]
        and check_open.get("accuracy", 0) >= strict["open_accuracy_min"]
    ):
        return "GENERALIZES"

    partial = GENERALIZATION_POLICY["partial"]
    if (
        check_overall.get("accuracy", 0) >= partial["accuracy_min"]
        and check_overall.get("abstain_recall", 0) >= partial["abstain_recall_min"]
        and check_overall.get("answerable_recall", 0) >= partial["answerable_recall_min"]
    ):
        return "PARTIAL_GENERALIZATION"
    return "GENERALIZATION_FAILED"


def validate_independence(k_manifests: list[dict], forbidden_manifests: list[dict]) -> ValidationReport:
    """§10/§33/§34: Corpus K must share ZERO documents (official_url, hash, or
    normalized title) with any prior corpus (A-H, J).  The ``forbidden_manifests``
    list is caller-supplied (compiled once from the prior corpora source
    manifests); regularizing titles/product-lines makes the check robust to
    punctuation/whitespace differences."""
    issues: list[ValidationIssue] = []

    def norm(s: str) -> str:
        return _normalize(s)

    forbidden_urls = {norm(m.get("official_url", "")) for m in forbidden_manifests if m.get("official_url")}
    forbidden_shas = {m.get("sha256", "").strip().upper() for m in forbidden_manifests if m.get("sha256")}
    forbidden_titles = {norm(m.get("source_name", "")) for m in forbidden_manifests if m.get("source_name")}
    forbidden_triples = {
        (norm(m.get("manufacturer", "")), norm(m.get("product_family", "")), norm(m.get("product_series", "")))
        for m in forbidden_manifests if m.get("manufacturer")
    }
    forbidden_families = {
        (norm(m.get("manufacturer", "")), norm(m.get("product_family", "")))
        for m in forbidden_manifests if m.get("manufacturer")
    }
    for m in k_manifests:
        did = m.get("document_id", "?")
        if norm(m.get("official_url", "")) in forbidden_urls:
            issues.append(ValidationIssue("error", "INDEPENDENCE_URL_OVERLAP", did))
        if m.get("sha256", "").strip().upper() in forbidden_shas:
            issues.append(ValidationIssue("error", "INDEPENDENCE_HASH_OVERLAP", did))
        if norm(m.get("source_name", "")) in forbidden_titles:
            issues.append(ValidationIssue("error", "INDEPENDENCE_TITLE_OVERLAP", did))
        triple = (norm(m.get("manufacturer", "")), norm(m.get("product_family", "")), norm(m.get("product_series", "")))
        family = (norm(m.get("manufacturer", "")), norm(m.get("product_family", "")))
        if triple in forbidden_triples:
            issues.append(ValidationIssue("error", "INDEPENDENCE_PRODUCTLINE_OVERLAP", did))
        elif family in forbidden_families:
            issues.append(ValidationIssue("warning", "INDEPENDENCE_PRODUCT_FAMILY_OVERLAP",
                                          f"{did}: {m.get('manufacturer')}/{m.get('product_family')}"))
    return ValidationReport(tuple(issues))


def validate_query_leakage(k_queries: Iterable[str], dev_queries: Iterable[str],
                           threshold: float = 0.8) -> ValidationReport:
    """§34 query-level leakage vs allowed development queries (token-overlap)."""
    issues: list[ValidationIssue] = []

    def tokens(s: str) -> set[str]:
        return set(_normalize(s).split())

    dev = [(i, tokens(q)) for i, q in enumerate(dev_queries) if q]
    for qi, k in enumerate(k_queries):
        kt = tokens(k)
        if not kt:
            continue
        for di, dt in dev:
            overlap = len(kt & dt) / max(1, min(len(kt), len(dt)))
            if overlap >= threshold:
                issues.append(ValidationIssue("warning", "QUERY_LEAKAGE_OVERLAP",
                                              f"k[{qi}] ~ dev[{di}] overlap={overlap:.3f}"))
    return ValidationReport(tuple(issues))


def validate_benchmark(manifests: list[dict], annotations: list[dict]) -> ValidationReport:
    """Aggregate every V3.33 validation gate into one report."""
    parts = [
        validate_manifests(manifests),
        validate_annotations(annotations),
        validate_benchmark_scope(annotations, manifests),
        validate_document_disjoint(annotations, manifests),
        validate_distribution(annotations, "K-TRAIN"),
        validate_distribution(annotations, "K-CHECK"),
        validate_identifier_quota(annotations),
        validate_table_quota(annotations),
    ]
    issues = tuple(i for report in parts for i in report.issues)
    return ValidationReport(issues)
