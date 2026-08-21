"""Public policy contract for the V3.37 sealed identity re-validation gate.

The corpus, queries, annotations, ledger, and results are private and
gitignored.  This module contains validation, scoring, and the pre-registered
decision policy only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any


CORPUS_ID = "M"
GATE_VERSION = "v337-sealed-identity-revalidation-v1"
BASELINE_VERSION = "identity-aware-evidence-v334-candidate"
CANDIDATE_VERSION = "identity-v336-candidate"

MANUAL_COUNT = 6
MANUFACTURER_COUNT = 6
QUERY_COUNT = 60
ANSWER_COUNT = 30
MIN_L4_L5_RATIO = 0.80
MAX_FA = 0.10
MIN_FR_REDUCTION = 0.50
MAX_HARD_NEGATIVE_FA = 0.10
MIN_ACCURACY = 0.80

IDENTITY_SLICES = (
    "FAMILY_MISMATCH",
    "MODEL_MISMATCH",
    "MODULE_MISMATCH",
    "FIRMWARE_MISMATCH",
    "OPTION_MISMATCH",
)
SCENARIOS = (
    "FAMILY_MODEL_CONFUSION",
    "CONTROLLER_MODULE_CONFUSION",
    "OPTION_ACCESSORY_CONFUSION",
    "FIRMWARE_VERSION_CONFUSION",
    "SAME_NAME_COMPONENT",
    "PARAMETER_OWNERSHIP",
    "CROSS_DOCUMENT_IDENTITY",
)


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


def validate_corpus(
    manifests: list[dict], forbidden_documents: list[dict], forbidden_manufacturers: set[str]
) -> ValidationReport:
    errors: list[str] = []
    if len(manifests) < MANUAL_COUNT:
        errors.append(f"MANUAL_COUNT:{len(manifests)}")
    manufacturers = {
        str(item.get("manufacturer", "")).casefold().strip()
        for item in manifests if item.get("manufacturer")
    }
    if len(manufacturers) < MANUFACTURER_COUNT:
        errors.append(f"MANUFACTURER_COUNT:{len(manufacturers)}")
    reused = sorted(manufacturers & {item.casefold().strip() for item in forbidden_manufacturers})
    if reused:
        errors.append(f"FORBIDDEN_MANUFACTURERS:{','.join(reused)}")

    required = {
        "document_id", "file", "source_name", "source_type", "document_type",
        "official_url", "manufacturer", "equipment_type", "equipment_model",
        "product_family", "product_series", "language", "sha256", "pages",
        "download_timestamp",
    }
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    seen_shas: set[str] = set()
    forbidden_urls = {str(item.get("official_url", "")).casefold().strip() for item in forbidden_documents}
    forbidden_shas = {str(item.get("sha256", "")).casefold().strip() for item in forbidden_documents}
    forbidden_titles = {" ".join(str(item.get("source_name", "")).casefold().split()) for item in forbidden_documents}
    for item in manifests:
        document_id = str(item.get("document_id", "?"))
        missing = sorted(key for key in required if item.get(key) in (None, ""))
        if missing:
            errors.append(f"MANIFEST_FIELDS:{document_id}:{','.join(missing)}")
        if item.get("language") != "English":
            errors.append(f"NOT_ENGLISH:{document_id}")
        if item.get("source_type") != "official_vendor_publication":
            errors.append(f"NOT_OFFICIAL:{document_id}")
        url = str(item.get("official_url", "")).casefold().strip()
        sha = str(item.get("sha256", "")).casefold().strip()
        title = " ".join(str(item.get("source_name", "")).casefold().split())
        if document_id in seen_ids or url in seen_urls or sha in seen_shas:
            errors.append(f"DUPLICATE_DOCUMENT_IDENTITY:{document_id}")
        if url in forbidden_urls or sha in forbidden_shas or title in forbidden_titles:
            errors.append(f"FORBIDDEN_DOCUMENT:{document_id}")
        seen_ids.add(document_id)
        seen_urls.add(url)
        seen_shas.add(sha)
    return ValidationReport(tuple(errors))


def validate_queries(queries: list[dict], document_ids: set[str]) -> ValidationReport:
    errors: list[str] = []
    if len(queries) != QUERY_COUNT:
        errors.append(f"QUERY_COUNT:{len(queries)}")
    answer_count = sum(item.get("expected") == "ANSWER" for item in queries)
    if answer_count != ANSWER_COUNT:
        errors.append(f"ANSWER_COUNT:{answer_count}")
    if queries and sum(item.get("difficulty") in {"L4", "L5"} for item in queries) / len(queries) < MIN_L4_L5_RATIO:
        errors.append("L4_L5_RATIO")
    if any(item.get("confidence") != "HIGH" for item in queries):
        errors.append("NON_HIGH_CONFIDENCE")
    ids = [str(item.get("query_id", "")) for item in queries]
    normalized = [" ".join(str(item.get("query", "")).casefold().split()) for item in queries]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("QUERY_ID_DUPLICATE_OR_MISSING")
    if not all(normalized) or len(normalized) != len(set(normalized)):
        errors.append("QUERY_DUPLICATE_OR_MISSING")
    if any(item.get("document_id") not in document_ids for item in queries):
        errors.append("UNKNOWN_DOCUMENT")
    if any(item.get("identity_slice") not in IDENTITY_SLICES for item in queries):
        errors.append("BAD_IDENTITY_SLICE")
    if any(item.get("scenario") not in SCENARIOS for item in queries):
        errors.append("BAD_SCENARIO")
    if any(not item.get("evidence_chunk_id") for item in queries):
        errors.append("MISSING_EVIDENCE_CHUNK")
    if any(item.get("expected") not in {"ANSWER", "ABSTAIN"} for item in queries):
        errors.append("BAD_EXPECTED_LABEL")
    if any(item.get("expected_identity") not in {"COMPATIBLE", "INCOMPATIBLE"} for item in queries):
        errors.append("BAD_EXPECTED_IDENTITY")
    if any((item.get("expected") == "ABSTAIN") != (item.get("expected_identity") == "INCOMPATIBLE") for item in queries):
        errors.append("IDENTITY_LABEL_MISMATCH")
    slice_counts = Counter(item.get("identity_slice") for item in queries)
    scenario_counts = Counter(item.get("scenario") for item in queries)
    errors.extend(f"MISSING_SLICE:{name}" for name in IDENTITY_SLICES if not slice_counts[name])
    errors.extend(f"MISSING_SCENARIO:{name}" for name in SCENARIOS if not scenario_counts[name])
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


def decide(baseline: dict, candidate: dict, hard_negative: dict, *, runtime_valid: bool) -> dict:
    baseline_fr = float(baseline["false_refusal_rate"])
    candidate_fr = float(candidate["false_refusal_rate"])
    fr_reduction = (baseline_fr - candidate_fr) / baseline_fr if baseline_fr else 0.0
    passed = (
        candidate["false_answer_rate"] <= MAX_FA
        and fr_reduction >= MIN_FR_REDUCTION
        and hard_negative["false_answer_rate"] <= MAX_HARD_NEGATIVE_FA
        and candidate["accuracy"] >= MIN_ACCURACY
        and runtime_valid
    )
    return {
        "decision": "SEALED_IDENTITY_PASS" if passed else "SEALED_IDENTITY_FAIL",
        "false_refusal_reduction": fr_reduction,
        "runtime_valid": runtime_valid,
    }
