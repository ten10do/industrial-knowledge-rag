"""V3.54 merged-cell coverage proof-contract candidate.

This contract composes the frozen V3.51 validator. It can recover only a
single COLUMN_BOUND model-row mismatch when explicit merged-cell coverage
proves stable row membership. It is not integrated into Evidence runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .evidence_table_ownership import TableOwnershipRelation
from .table_structure_proof_contract_v351 import (
    FORBIDDEN_CONCLUSION_FIELDS,
    ProofValidationResult,
    TableStructureClaim,
    TableStructureProof,
    validate_table_structure_proof,
)


MERGED_CELL_COVERAGE_VERSION = "merged-cell-coverage-v354-candidate"
MERGED_CELL_COVERAGE_STATUS = "CONTRACT_CANDIDATE_ONLY"


@dataclass(frozen=True)
class MergedCellCoverage:
    coverage_id: str
    document_id: str
    table_region_id: str
    coverage_owner_scope_id: str
    coverage_anchor_cell_id: str
    covered_row_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    coverage_version: str = MERGED_CELL_COVERAGE_VERSION
    coverage_kind: str = "VERTICAL_MERGED_CELL"
    conflicting_coverage_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageProofValidationResult:
    valid: bool
    reason_codes: tuple[str, ...]
    base_reason_codes: tuple[str, ...]
    coverage_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, (TableStructureProof, MergedCellCoverage)):
        return value.as_dict()
    return dict(value)


def _coverage_reasons(
    proof: dict[str, Any], claim: TableStructureClaim, coverage: dict[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    forbidden = sorted(FORBIDDEN_CONCLUSION_FIELDS & coverage.keys())
    if forbidden:
        reasons.append("COVERAGE_CONCLUSION_FIELD_FORBIDDEN:" + ",".join(forbidden))
    if coverage.get("coverage_version") != MERGED_CELL_COVERAGE_VERSION:
        reasons.append("COVERAGE_VERSION_MISMATCH")
    if coverage.get("coverage_kind") != "VERTICAL_MERGED_CELL":
        reasons.append("COVERAGE_KIND_MISMATCH")
    for field in (
        "coverage_id", "document_id", "table_region_id", "coverage_owner_scope_id",
        "coverage_anchor_cell_id", "covered_row_ids", "chunk_ids",
    ):
        if coverage.get(field) in (None, "", (), []):
            reasons.append(f"COVERAGE_FIELD_MISSING:{field}")
    if proof.get("relation") != TableOwnershipRelation.COLUMN_BOUND.value:
        reasons.append("COVERAGE_RELATION_NOT_COLUMN_BOUND")
    if coverage.get("document_id") != proof.get("document_id"):
        reasons.append("COVERAGE_DOCUMENT_SCOPE_MISMATCH")
    if coverage.get("table_region_id") != proof.get("table_region_id"):
        reasons.append("COVERAGE_TABLE_REGION_MISMATCH")
    owner = coverage.get("coverage_owner_scope_id")
    if owner != proof.get("model_scope_id") or owner != claim.model_scope_id:
        reasons.append("COVERAGE_OWNER_SCOPE_MISMATCH")
    if coverage.get("conflicting_coverage_ids"):
        reasons.append("CONFLICTING_COVERAGE")

    covered = coverage.get("covered_row_ids", ())
    if not isinstance(covered, (list, tuple)):
        reasons.append("COVERED_ROWS_INVALID")
        covered_rows: tuple[Any, ...] = ()
    else:
        covered_rows = tuple(covered)
        if any(not isinstance(row_id, str) or not row_id for row_id in covered_rows):
            reasons.append("COVERED_ROWS_INVALID")
        if len(covered_rows) != len(set(covered_rows)):
            reasons.append("DUPLICATE_COVERED_ROW_ID")
    if proof.get("model_row_id") not in covered_rows:
        reasons.append("COVERAGE_ANCHOR_ROW_MISSING")
    if proof.get("value_row_id") not in covered_rows:
        reasons.append("VALUE_ROW_OUTSIDE_COVERAGE")

    span = proof.get("merged_cell_span", (1, 1))
    if (
        isinstance(span, (list, tuple)) and len(span) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in span)
    ):
        if span[0] < 2:
            reasons.append("VERTICAL_COVERAGE_REQUIRES_ROW_SPAN")
    else:
        reasons.append("COVERAGE_SPAN_INVALID")

    coverage_chunks = coverage.get("chunk_ids", ())
    if not isinstance(coverage_chunks, (list, tuple)):
        reasons.append("COVERAGE_CHUNKS_INVALID")
        coverage_chunks = ()
    elif len(coverage_chunks) != len(set(coverage_chunks)):
        reasons.append("DUPLICATE_COVERAGE_CHUNK_ID")
    if not set(coverage_chunks) & set(proof.get("chunk_ids", ())):
        reasons.append("COVERAGE_CHUNK_SCOPE_MISMATCH")
    return tuple(dict.fromkeys(reasons))


def validate_merged_cell_coverage_proof(
    proof: TableStructureProof | dict[str, Any], claim: TableStructureClaim,
    coverage: MergedCellCoverage | dict[str, Any] | None = None,
) -> CoverageProofValidationResult:
    proof_payload = _payload(proof)
    base: ProofValidationResult = validate_table_structure_proof(proof_payload, claim)
    if coverage is None:
        return CoverageProofValidationResult(
            base.valid, base.reason_codes, base.reason_codes, False,
        )
    coverage_payload = _payload(coverage)
    coverage_reasons = _coverage_reasons(proof_payload, claim, coverage_payload)
    recoverable_base = set(base.reason_codes) == {"MODEL_ROW_OWNERSHIP_MISMATCH"}
    if base.valid and not coverage_reasons:
        return CoverageProofValidationResult(True, (), (), True)
    if recoverable_base and not coverage_reasons:
        return CoverageProofValidationResult(
            True, (), base.reason_codes, True,
        )
    reasons = tuple(dict.fromkeys((*base.reason_codes, *coverage_reasons)))
    return CoverageProofValidationResult(False, reasons, base.reason_codes, False)


def validate_coverage_lineage(
    coverages: Iterable[MergedCellCoverage | dict[str, Any]],
) -> tuple[str, ...]:
    structural_fields = tuple(
        field for field in MergedCellCoverage.__dataclass_fields__
        if field != "chunk_ids"
    )
    signatures: dict[str, tuple[Any, ...]] = {}
    errors: list[str] = []
    for coverage in coverages:
        payload = _payload(coverage)
        coverage_id = str(payload.get("coverage_id", ""))
        signature = tuple(
            tuple(payload.get(field, ())) if isinstance(payload.get(field), list)
            else payload.get(field)
            for field in structural_fields
        )
        if coverage_id in signatures and signatures[coverage_id] != signature:
            errors.append(f"COVERAGE_LINEAGE_DRIFT:{coverage_id}")
        signatures.setdefault(coverage_id, signature)
    return tuple(dict.fromkeys(errors))
