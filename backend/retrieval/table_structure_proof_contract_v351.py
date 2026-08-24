"""V3.51 typed table-structure proof contract.

The contract carries provenance only. It does not parse tables, resolve identity,
retrieve evidence, or make an ANSWER/ABSTAIN decision.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .evidence_table_ownership import TableOwnershipRelation


TABLE_STRUCTURE_PROOF_VERSION = "table-structure-proof-v351-contract"
TABLE_STRUCTURE_PROOF_STATUS = "CONTRACT_ONLY"

SUPPORTED_RELATIONS = frozenset({
    TableOwnershipRelation.DIRECT_ROW.value,
    TableOwnershipRelation.COLUMN_BOUND.value,
    TableOwnershipRelation.HEADER_INHERITED.value,
    TableOwnershipRelation.SECTION_INHERITED.value,
    TableOwnershipRelation.CROSS_REFERENCE.value,
})
FORBIDDEN_CONCLUSION_FIELDS = frozenset({
    "answer", "decision", "identity_compatible", "supported", "abstain",
})
COMMON_REQUIRED_FIELDS = frozenset({
    "proof_id", "proof_version", "document_id", "table_region_id", "relation",
    "model_scope_id", "parameter_scope_id", "value_scope_id", "model_text",
    "parameter_text", "value_text", "cell_role", "chunk_ids",
})
REQUIRED_FIELDS_BY_RELATION = {
    TableOwnershipRelation.DIRECT_ROW.value: frozenset({
        "parameter_row_id", "value_row_id", "parameter_column_id", "value_column_id",
    }),
    TableOwnershipRelation.COLUMN_BOUND.value: frozenset({
        "model_row_id", "value_row_id", "parameter_column_id", "value_column_id",
        "header_path",
    }),
    TableOwnershipRelation.HEADER_INHERITED.value: frozenset({
        "value_row_id", "value_column_id", "header_path",
    }),
    TableOwnershipRelation.SECTION_INHERITED.value: frozenset({
        "parameter_row_id", "value_row_id", "section_scope_id",
    }),
    TableOwnershipRelation.CROSS_REFERENCE.value: frozenset({
        "parameter_row_id", "value_row_id", "reference_target",
    }),
    TableOwnershipRelation.UNSUPPORTED.value: frozenset(),
}


@dataclass(frozen=True)
class TableStructureClaim:
    document_id: str
    ownership_relation: str
    model_scope_id: str
    parameter_scope_id: str
    value_or_action: str
    qualifier_scope_id: str = ""
    section_scope_id: str = ""


@dataclass(frozen=True)
class TableStructureProof:
    proof_id: str
    document_id: str
    table_region_id: str
    relation: str
    model_scope_id: str
    parameter_scope_id: str
    value_scope_id: str
    model_text: str
    parameter_text: str
    value_text: str
    cell_role: str
    chunk_ids: tuple[str, ...]
    proof_version: str = TABLE_STRUCTURE_PROOF_VERSION
    model_row_id: str = ""
    parameter_row_id: str = ""
    value_row_id: str = ""
    model_column_id: str = ""
    parameter_column_id: str = ""
    value_column_id: str = ""
    header_path: tuple[str, ...] = ()
    qualifier_scope_id: str = ""
    section_scope_id: str = ""
    merged_cell_span: tuple[int, int] = (1, 1)
    reference_target: str = ""
    conflicting_owner_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProofValidationResult:
    valid: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reason_codes": list(self.reason_codes)}


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9.%+/-]+", " ", str(value or "").casefold()).split())


def _payload(proof: TableStructureProof | dict[str, Any]) -> dict[str, Any]:
    return proof.as_dict() if isinstance(proof, TableStructureProof) else dict(proof)


def _missing(payload: dict[str, Any], fields: Iterable[str]) -> list[str]:
    return sorted(field for field in fields if payload.get(field) in (None, "", (), []))


def validate_table_structure_proof(
    proof: TableStructureProof | dict[str, Any], claim: TableStructureClaim,
) -> ProofValidationResult:
    payload = _payload(proof)
    reasons: list[str] = []
    forbidden = sorted(FORBIDDEN_CONCLUSION_FIELDS & payload.keys())
    if forbidden:
        reasons.append("CONCLUSION_FIELD_FORBIDDEN:" + ",".join(forbidden))
    if payload.get("proof_version") != TABLE_STRUCTURE_PROOF_VERSION:
        reasons.append("PROOF_VERSION_MISMATCH")
    relation = str(payload.get("relation", ""))
    if relation == TableOwnershipRelation.UNSUPPORTED.value:
        reasons.append("UNSUPPORTED_NOT_A_PROOF")
    elif relation not in SUPPORTED_RELATIONS:
        reasons.append("UNKNOWN_OWNERSHIP_RELATION")
    required = COMMON_REQUIRED_FIELDS | REQUIRED_FIELDS_BY_RELATION.get(relation, frozenset())
    missing = _missing(payload, required)
    if missing:
        reasons.append("MISSING_REQUIRED_FIELDS:" + ",".join(missing))
    if payload.get("document_id") != claim.document_id:
        reasons.append("DOCUMENT_SCOPE_MISMATCH")
    if relation != claim.ownership_relation:
        reasons.append("RELATION_SCOPE_MISMATCH")
    if payload.get("model_scope_id") != claim.model_scope_id:
        reasons.append("MODEL_SCOPE_MISMATCH")
    if payload.get("parameter_scope_id") != claim.parameter_scope_id:
        reasons.append("PARAMETER_SCOPE_MISMATCH")
    if claim.qualifier_scope_id and payload.get("qualifier_scope_id") != claim.qualifier_scope_id:
        reasons.append("QUALIFIER_SCOPE_MISMATCH")
    if claim.section_scope_id and payload.get("section_scope_id") != claim.section_scope_id:
        reasons.append("SECTION_SCOPE_MISMATCH")
    if payload.get("conflicting_owner_ids"):
        reasons.append("CONFLICTING_OWNER")

    span = payload.get("merged_cell_span", (1, 1))
    if (
        not isinstance(span, (list, tuple)) or len(span) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in span)
    ):
        reasons.append("INVALID_MERGED_CELL_SPAN")
    chunk_ids = payload.get("chunk_ids", ())
    if isinstance(chunk_ids, (list, tuple)) and len(chunk_ids) != len(set(chunk_ids)):
        reasons.append("DUPLICATE_CHUNK_ID")

    if relation == TableOwnershipRelation.DIRECT_ROW.value:
        if payload.get("parameter_row_id") != payload.get("value_row_id"):
            reasons.append("ROW_OWNERSHIP_MISMATCH")
        if _norm(payload.get("value_text")) != _norm(claim.value_or_action):
            reasons.append("VALUE_SCOPE_MISMATCH")
    elif relation == TableOwnershipRelation.COLUMN_BOUND.value:
        if payload.get("model_row_id") != payload.get("value_row_id"):
            reasons.append("MODEL_ROW_OWNERSHIP_MISMATCH")
        if payload.get("parameter_column_id") != payload.get("value_column_id"):
            reasons.append("COLUMN_OWNERSHIP_MISMATCH")
        if payload.get("parameter_scope_id") not in set(payload.get("header_path", ())):
            reasons.append("HEADER_LINEAGE_MISMATCH")
        if _norm(payload.get("value_text")) != _norm(claim.value_or_action):
            reasons.append("VALUE_SCOPE_MISMATCH")
    elif relation == TableOwnershipRelation.HEADER_INHERITED.value:
        if payload.get("parameter_scope_id") not in set(payload.get("header_path", ())):
            reasons.append("HEADER_LINEAGE_MISMATCH")
        if _norm(payload.get("value_text")) != _norm(claim.value_or_action):
            reasons.append("VALUE_SCOPE_MISMATCH")
    elif relation == TableOwnershipRelation.SECTION_INHERITED.value:
        if payload.get("parameter_row_id") != payload.get("value_row_id"):
            reasons.append("ROW_OWNERSHIP_MISMATCH")
        if not payload.get("section_scope_id"):
            reasons.append("SECTION_OWNERSHIP_MISSING")
        if _norm(payload.get("value_text")) != _norm(claim.value_or_action):
            reasons.append("VALUE_SCOPE_MISMATCH")
    elif relation == TableOwnershipRelation.CROSS_REFERENCE.value:
        if payload.get("parameter_row_id") != payload.get("value_row_id"):
            reasons.append("REFERENCE_SOURCE_OWNERSHIP_MISMATCH")
        if payload.get("cell_role") != "REFERENCE":
            reasons.append("REFERENCE_CELL_ROLE_REQUIRED")
        if _norm(payload.get("reference_target")) != _norm(claim.value_or_action):
            reasons.append("REFERENCE_TARGET_MISMATCH")
    if relation != TableOwnershipRelation.CROSS_REFERENCE.value and payload.get("cell_role") not in {
        "VALUE", "ACTION", "QUALIFIER",
    }:
        reasons.append("VALUE_CELL_ROLE_REQUIRED")
    return ProofValidationResult(not reasons, tuple(dict.fromkeys(reasons)))


def validate_proof_lineage(
    proofs: Iterable[TableStructureProof | dict[str, Any]],
) -> tuple[str, ...]:
    structural_fields = tuple(
        field for field in TableStructureProof.__dataclass_fields__
        if field != "chunk_ids"
    )
    signatures: dict[str, tuple[Any, ...]] = {}
    errors: list[str] = []
    for proof in proofs:
        payload = _payload(proof)
        proof_id = str(payload.get("proof_id", ""))
        signature = tuple(
            tuple(payload.get(field, ())) if isinstance(payload.get(field), list)
            else payload.get(field)
            for field in structural_fields
        )
        if proof_id in signatures and signatures[proof_id] != signature:
            errors.append(f"LINEAGE_DRIFT:{proof_id}")
        signatures.setdefault(proof_id, signature)
    return tuple(dict.fromkeys(errors))
