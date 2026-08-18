"""Unit coverage for V3.17's frozen independent-holdout helpers."""

import unittest
from unittest.mock import patch

from backend.evaluation.private_benchmark import annotation_hash
from backend.evaluation.v317_independent_holdout import (
    REQUIRED_CONTRACT_TAGS,
    failure_attribution,
    manifest_hash,
    query_hash,
    validate_holdout_manifest,
)


def _documents():
    return [{
        "document_id": f"doc-{index}", "manufacturer": manufacturer,
        "source_type": "official_vendor_publication", "official_url": "https://example.test/manual.pdf",
        "commit_allowed": False,
    } for index, manufacturer in enumerate(("Schneider Electric", "Mitsubishi Electric", "Siemens"), 1)]


def _manifest():
    queries = []
    value_kinds = ("default", "range", "maximum", "minimum", "rated_nominal")
    for index in range(36):
        supported = index < 20
        query = {
            "query_id": f"d{index + 1:02d}", "query": f"independent holdout query {index}",
            "support_gate_truth": "SUPPORTED" if supported else "INSUFFICIENT",
            "manufacturer": ("Schneider Electric", "Mitsubishi Electric", "Siemens")[index % 3],
            "requirement_category": "semantic", "confidence": "HIGH",
            "contract_tags": ["semantic_equivalence"],
            "expected_evidence": "official evidence", "annotation_rationale": "independent annotation",
            "failure_class": "OTHER", "location_expectation": "none",
            "relevant_chunk_ids": [f"chunk-{index}"], "relevant_document_ids": [f"doc-{index % 3 + 1}"],
            "requested": {key: [] for key in (
                "identity", "identifier", "protocol", "action", "attribute", "value", "unit",
                "value_kind", "requirement_type", "location", "qualifier",
            )},
        }
        if index == 0:
            query["contract_tags"] = sorted(REQUIRED_CONTRACT_TAGS)
        if index < 5:
            query["requested"]["value_kind"] = [value_kinds[index]]
        if index < 4:
            query["semantic_hard_positive"] = True
        if index < 3:
            query["multi_chunk_positive"] = True
            if "action_procedure" not in query["contract_tags"]:
                query["contract_tags"].append("action_procedure")
        if 20 <= index < 25:
            query["partial_support_negative"] = True
        if 20 <= index < 22:
            query["cross_scope_negative"] = True
        queries.append(query)
    manifest = {"documents": _documents(), "queries": queries, "freeze": {}}
    manifest["freeze"] = {
        "query_sha256": query_hash(manifest),
        "annotation_sha256": annotation_hash(manifest),
        "manifest_sha256": manifest_hash(manifest),
    }
    return manifest


class IndependentHoldoutTests(unittest.TestCase):
    def test_complete_independent_holdout_validates_and_tamper_fails(self):
        manifest = _manifest()
        with patch(
            "backend.evaluation.v317_independent_holdout.EVIDENCE_SUPPORT_RULE_VERSION",
            "v311.2",
        ):
            distribution = validate_holdout_manifest(manifest)
            self.assertEqual(distribution["support"], {"INSUFFICIENT": 16, "SUPPORTED": 20})
            manifest["queries"][0]["query"] = "tampered after freeze"
            with self.assertRaisesRegex(ValueError, "QUERY_HASH"):
                validate_holdout_manifest(manifest)

    def test_parser_metadata_failure_has_priority_in_failure_attribution(self):
        manifest = _manifest()
        replay = {"rows": [{
            "query_id": "d01", "expected_supported": True, "predicted_supported": False,
            "base_decision": "ANSWER", "candidate_ids": {"final_context": ["chunk-0"]}, "support": {},
        }]}
        artifact = {"source": {"parser_audit": {"production_ingestion_audit": {
            "doc-1": {"issues": ["missing_section"]},
        }}}}
        report = failure_attribution(replay, manifest, artifact)
        self.assertEqual(report[0]["attribution"], "PARSER_METADATA_FAILURE")
        self.assertEqual(report[0]["post_freeze_status"], "POST_FREEZE_DISCOVERED_ISSUE")


if __name__ == "__main__":
    unittest.main()
