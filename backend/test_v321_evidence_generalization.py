"""Public validation checks for the private V3.21 development gate."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from backend.evaluation import v321_evidence_generalization as gate
from backend.evaluation.v311_resume import hash_json


CLASSES = (
    "identifier", "protocol", "attribute", "value", "action", "requirement",
    "semantic", "multi_chunk", "cross_scope", "qualifier",
)


def _manifest(split: str, document_ids: tuple[str, str, str]) -> dict:
    pair_count = 24 if split == "DEV-TRAIN" else 12
    documents = [
        {"document_id": document_id, "manufacturer": f"Vendor-{index}", "model": f"Model-{index}"}
        for index, document_id in enumerate(document_ids)
    ]
    candidates = {
        f"chunk-{index}": {
            "document_id": document_id,
            "metadata": {"document_id": document_id},
            "content": "fixed candidate",
        }
        for index, document_id in enumerate(document_ids)
    }
    queries = []
    for index in range(pair_count):
        document_id = document_ids[index % len(document_ids)]
        chunk_id = f"chunk-{index % len(document_ids)}"
        for answerable, suffix in ((True, "p"), (False, "n")):
            queries.append({
                "query_id": f"{split.lower()}-{index:02}{suffix}",
                "pair_id": f"{split.lower()}-{index:02}",
                "query": f"independent {split} query {index} {suffix}",
                "answerable": answerable,
                "document_id": document_id,
                "manufacturer": f"Vendor-{index % len(document_ids)}",
                "failure_class": CLASSES[index % len(CLASSES)],
                "focus": ("identifier", "action", "value", "protocol")[index % 4],
                "confidence": "HIGH",
                "claim_type": "SEMANTIC_EQUIVALENT" if answerable else "RELATED_ONLY",
                "candidate_chunk_ids": [chunk_id],
                "semantic_positive": answerable and index < (12 if split == "DEV-TRAIN" else 6),
                "multi_chunk_positive": answerable and index < (6 if split == "DEV-TRAIN" else 3),
                "unsafe_multi_chunk_negative": not answerable and index < (6 if split == "DEV-TRAIN" else 3),
                "cross_document_negative": not answerable and index == pair_count - 1,
            })
    manifest = {
        "development_set_id": f"synthetic-{split.lower()}",
        "split": split,
        "documents": documents,
        "queries": queries,
        "candidates": candidates,
    }
    manifest["freeze"] = {
        "query_sha256": hash_json([{"query_id": row["query_id"], "query": row["query"]} for row in queries]),
        "annotation_sha256": hash_json(queries),
        "manifest_sha256": hash_json(manifest),
    }
    return manifest


def test_train_check_are_paired_frozen_and_document_disjoint():
    train = _manifest("DEV-TRAIN", ("train-a", "train-b", "train-c"))
    check = _manifest("DEV-CHECK", ("check-a", "check-b", "check-c"))
    assert gate.validate_manifest(train)["pairs"] == 24
    assert gate.validate_manifest(check)["pairs"] == 12
    assert gate.validate_independence(train, check)["document_disjoint"]

    overlap = deepcopy(check)
    overlap["documents"][0]["document_id"] = "train-a"
    with pytest.raises(ValueError, match="DOCUMENT_OVERLAP"):
        gate.validate_independence(train, overlap)


def test_frozen_d_and_e_products_are_excluded():
    train = _manifest("DEV-TRAIN", ("train-a", "train-b", "train-c"))
    train["documents"][0]["model"] = "S7-1200"
    with pytest.raises(ValueError, match="FROZEN_PRODUCT_LEAK"):
        gate.validate_manifest(train)


def test_dev_check_one_shot_lock(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    freeze = tmp_path / "candidate-freeze.json"
    rule_hashes = {"evidence_contract_sha256": "a", "technical_sha256": "b"}
    freeze.write_text(json.dumps({
        "rule_hashes": rule_hashes,
        "check_manifest_sha256": "check-hash",
    }), encoding="utf-8")
    ledger.write_text(json.dumps({"phases": {"candidate": {"consumed_at": "once"}}}), encoding="utf-8")
    monkeypatch.setattr(gate, "CHECK_LEDGER", ledger)
    monkeypatch.setattr(gate, "CANDIDATE_FREEZE", freeze)
    monkeypatch.setattr(gate, "ensure_private_path", lambda path: path)
    monkeypatch.setattr(gate, "candidate_rule_hashes", lambda: rule_hashes)
    monkeypatch.setattr(gate, "EVIDENCE_SUPPORT_RULE_VERSION", gate.CANDIDATE_VERSION)
    with pytest.raises(RuntimeError, match="ALREADY_CONSUMED"):
        gate._check_phase_allowed("candidate", {"freeze": {"manifest_sha256": "check-hash"}})
