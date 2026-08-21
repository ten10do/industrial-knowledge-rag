"""Public contract for the V3.35 sealed identity validation gate.

Private manuals, queries, annotations, ledgers, and results are deliberately
excluded. This module only freezes validation and decision policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any


CORPUS_ID = "L"
GATE_VERSION = "v335-sealed-identity-gate-v1"
BASELINE_VERSION = "evidence-v332-integrated-candidate"
CANDIDATE_VERSION = "identity-aware-evidence-v334-candidate"

MIN_MANUALS = 6
MIN_MANUFACTURERS = 6
MIN_QUERIES = 50
MAX_QUERIES = 80
MIN_L4_L5_RATIO = 0.80
MAX_SLICE_FA = 0.20
MIN_FA_REDUCTION = 0.20
MAX_FR_INCREASE = 0.05

IDENTITY_SLICES = (
    "FAMILY_MODEL",
    "MODULE_CONTROLLER",
    "OPTION_ACCESSORY",
    "FIRMWARE_VERSION",
    "SAME_NAME_DIFFERENT_PRODUCT",
    "PARAMETER_SCOPE",
)
DECISIONS = ("SEALED_IDENTITY_READY", "SEALED_IDENTITY_FAIL")


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_corpus(manifests: list[dict], forbidden: list[dict]) -> ValidationReport:
    errors: list[str] = []
    if len(manifests) < MIN_MANUALS:
        errors.append(f"MANUAL_COUNT:{len(manifests)}")
    manufacturers = {str(item.get("manufacturer", "")).casefold() for item in manifests if item.get("manufacturer")}
    if len(manufacturers) < MIN_MANUFACTURERS:
        errors.append(f"MANUFACTURER_COUNT:{len(manufacturers)}")

    required = {
        "document_id", "file", "source_name", "document_type", "official_url",
        "manufacturer", "equipment_type", "equipment_model", "product_family",
        "product_series", "language", "sha256", "pages", "download_timestamp",
    }
    ids: set[str] = set()
    urls: set[str] = set()
    shas: set[str] = set()
    for item in manifests:
        document_id = str(item.get("document_id", "?"))
        missing = sorted(required - {key for key, value in item.items() if value not in (None, "")})
        if missing:
            errors.append(f"MANIFEST_FIELDS:{document_id}:{','.join(missing)}")
        if item.get("language") != "English":
            errors.append(f"NOT_ENGLISH:{document_id}")
        if item.get("source_type") != "official_vendor_publication":
            errors.append(f"NOT_OFFICIAL:{document_id}")
        if document_id in ids:
            errors.append(f"DUPLICATE_DOCUMENT:{document_id}")
        ids.add(document_id)
        url = str(item.get("official_url", "")).casefold().strip()
        sha = str(item.get("sha256", "")).casefold().strip()
        if url in urls:
            errors.append(f"DUPLICATE_URL:{document_id}")
        if sha in shas:
            errors.append(f"DUPLICATE_SHA:{document_id}")
        urls.add(url)
        shas.add(sha)

    forbidden_urls = {str(item.get("official_url", "")).casefold().strip() for item in forbidden}
    forbidden_shas = {str(item.get("sha256", "")).casefold().strip() for item in forbidden if item.get("sha256")}
    forbidden_titles = {" ".join(str(item.get("source_name", "")).casefold().split()) for item in forbidden}
    for item in manifests:
        document_id = item.get("document_id", "?")
        if str(item.get("official_url", "")).casefold().strip() in forbidden_urls:
            errors.append(f"FORBIDDEN_URL:{document_id}")
        if str(item.get("sha256", "")).casefold().strip() in forbidden_shas:
            errors.append(f"FORBIDDEN_SHA:{document_id}")
        title = " ".join(str(item.get("source_name", "")).casefold().split())
        if title in forbidden_titles:
            errors.append(f"FORBIDDEN_TITLE:{document_id}")
    return ValidationReport(tuple(errors))


def validate_queries(queries: list[dict], document_ids: set[str]) -> ValidationReport:
    errors: list[str] = []
    count = len(queries)
    if not MIN_QUERIES <= count <= MAX_QUERIES:
        errors.append(f"QUERY_COUNT:{count}")
    answer_count = sum(item.get("expected") == "ANSWER" for item in queries)
    if answer_count * 2 != count:
        errors.append(f"ANSWER_BALANCE:{answer_count}/{count}")
    hard_count = sum(item.get("difficulty") in {"L4", "L5"} for item in queries)
    if count and hard_count / count < MIN_L4_L5_RATIO:
        errors.append(f"L4_L5_RATIO:{hard_count}/{count}")
    if any(item.get("confidence") != "HIGH" for item in queries):
        errors.append("NON_HIGH_CONFIDENCE")

    ids = [str(item.get("query_id", "")) for item in queries]
    normalized = [" ".join(str(item.get("query", "")).casefold().split()) for item in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_ID_DUPLICATE_OR_MISSING")
    if not all(normalized) or len(normalized) != len(set(normalized)):
        errors.append("QUERY_DUPLICATE_OR_MISSING")
    if any(item.get("expected") not in {"ANSWER", "ABSTAIN"} for item in queries):
        errors.append("BAD_EXPECTED_LABEL")
    if any(item.get("document_id") not in document_ids for item in queries):
        errors.append("UNKNOWN_DOCUMENT")
    if any(item.get("identity_slice") not in IDENTITY_SLICES for item in queries):
        errors.append("BAD_IDENTITY_SLICE")
    if any(not item.get("evidence_chunk_id") for item in queries):
        errors.append("MISSING_EVIDENCE_CHUNK")
    if any(item.get("expected_identity") not in {"COMPATIBLE", "INCOMPATIBLE", "UNKNOWN"} for item in queries):
        errors.append("BAD_EXPECTED_IDENTITY")
    if any(
        (item.get("expected") == "ABSTAIN") != (item.get("expected_identity") == "INCOMPATIBLE")
        for item in queries
    ):
        errors.append("IDENTITY_LABEL_MISMATCH")
    counts = Counter(item.get("identity_slice") for item in queries)
    for identity_slice in IDENTITY_SLICES:
        if counts[identity_slice] == 0:
            errors.append(f"MISSING_SLICE:{identity_slice}")
    return ValidationReport(tuple(errors))


def score(records: list[dict]) -> dict[str, float | int]:
    answerable = [item for item in records if item["expected"] == "ANSWER"]
    abstainable = [item for item in records if item["expected"] == "ABSTAIN"]
    false_answers = sum(item["predicted"] == "ANSWER" for item in abstainable)
    false_refusals = sum(item["predicted"] == "ABSTAIN" for item in answerable)
    correct = sum(item["predicted"] == item["expected"] for item in records)
    return {
        "n": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "answerable_recall": 1 - false_refusals / len(answerable) if answerable else 0.0,
        "abstain_recall": 1 - false_answers / len(abstainable) if abstainable else 0.0,
        "false_answer_rate": false_answers / len(abstainable) if abstainable else 0.0,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else 0.0,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
    }


def decide(baseline: dict, candidate: dict, slice_metrics: dict[str, dict], *, runtime_valid: bool) -> dict:
    fa_reduction = baseline["false_answer_rate"] - candidate["false_answer_rate"]
    fr_increase = candidate["false_refusal_rate"] - baseline["false_refusal_rate"]
    missing_slices = sorted(set(IDENTITY_SLICES) - set(slice_metrics))
    bad_slices = sorted(
        name for name, metrics in slice_metrics.items()
        if metrics.get("false_answer_rate", 1.0) > MAX_SLICE_FA
    )
    ready = (
        fa_reduction >= MIN_FA_REDUCTION
        and fr_increase <= MAX_FR_INCREASE
        and not missing_slices
        and not bad_slices
        and runtime_valid
    )
    return {
        "decision": "SEALED_IDENTITY_READY" if ready else "SEALED_IDENTITY_FAIL",
        "false_answer_reduction": fa_reduction,
        "false_refusal_increase": fr_increase,
        "bad_identity_slices": bad_slices,
        "missing_identity_slices": missing_slices,
        "runtime_valid": runtime_valid,
    }
