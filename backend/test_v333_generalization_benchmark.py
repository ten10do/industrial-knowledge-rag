"""Tests for the V3.33 generalization benchmark framework (no private data)."""

from backend.evaluation import v333_generalization_benchmark as v


def _manifest(overrides=None):
    base = {
        "document_id": "doc-0",
        "file": "documents/doc0.pdf",
        "official_url": "https://example.com/doc0",
        "manufacturer": "Acme",
        "product_family": "PF",
        "product_series": "PS",
        "equipment_type": "plc_controller",
        "language": "English",
    }
    if overrides:
        base.update(overrides)
    return base


def _annotation(overrides=None):
    base = {
        "query_id": "k01",
        "query": "Is parameter P01 default 5s?",
        "split": "K-TRAIN",
        "ground_truth": "ANSWER",
        "query_type": "VERIFICATION",
        "manufacturer": "Acme",
        "product": "AcmePLC",
        "document": "doc-0",
        "difficulty": "L4",
        "document_style": "PARAMETER_TABLE",
        "confidence": "HIGH",
        "target": "P01",
        "relation": "HAS_DEFAULT_VALUE",
        "requested_slot": "",
        "critical_requirements": ["identity"],
        "expected_evidence": ["doc-0"],
        "scope": "EXACT_MODEL_SCOPE",
        "relation_annotation": {"relation": "HAS_DEFAULT_VALUE", "support": "SUPPORTED"},
        "answer_shape": "",
        "table_structure_recoverable": "YES",
        "annotation_rationale": "grounded in parameter table",
    }
    if overrides:
        base.update(overrides)
    return base


def test_fingerprint_deterministic():
    assert v.fingerprint({"b": 1, "a": 2}) == v.fingerprint({"a": 2, "b": 1})
    assert len(v.fingerprint({"x": 1})) == 64


def test_manifest_validation_ok_and_categories():
    # A single well-formed manifest has field-level validity; only the corpus-wide
    # quota gates (>=4 manufacturers, >=3 categories) fire on a 1-doc list.
    report = v.validate_manifests([_manifest()])
    field_errors = {i.code for i in report.errors} - {
        "INSUFFICIENT_MANUALS", "INSUFFICIENT_MANUFACTURERS", "INSUFFICIENT_CATEGORIES",
    }
    assert not field_errors, report.errors
    # 4 manufacturers / 3 categories gates
    ms = [
        _manifest({"document_id": f"d{i}", "official_url": f"https://e.com/{i}",
                   "manufacturer": f"M{i}", "equipment_type": cat})
        for i, cat in enumerate([
            "plc_controller", "servo_drive", "safety", "plc_controller",
            "drive", "industrial_communication",
        ])
    ]
    assert v.validate_manifests(ms).ok
    # <4 manufacturers -> error
    assert not v.validate_manifests(ms[:3]).ok
    # duplicate URL/hash -> error
    dup = [_manifest({"document_id": "a"}), _manifest({"document_id": "b"})]
    assert any(i.code in ("URL_OVERLAP", "HASH_OVERLAP") for i in v.validate_manifests(dup).issues)


def test_manifest_validation_requires_six_manuals():
    rows = [
        _manifest({"document_id": f"d{i}", "official_url": f"https://e.com/{i}",
                   "manufacturer": f"M{i}", "equipment_type": "plc_controller"})
        for i in range(5)
    ]
    assert any(i.code == "INSUFFICIENT_MANUALS" for i in v.validate_manifests(rows).errors)


def test_annotation_validation():
    assert v.validate_annotations([_annotation()]).ok
    bad = _annotation({"query_type": "BANANA"})
    assert any(i.code == "BAD_QUERY_TYPE" for i in v.validate_annotations([bad]).errors)
    missing = _annotation(); del missing["relation_annotation"]
    assert any(i.code == "ANNOTATION_MISSING_FIELD" for i in v.validate_annotations([missing]).errors)


def test_distribution_gates():
    # Build a synthetic balanced split of 40: 20 VERIFICATION (12 ANSWER/8 ABSTAIN),
    # 20 OPEN (8 ANSWER/12 ABSTAIN), all L3-L5.
    rows = []
    for i in range(40):
        qt = "VERIFICATION" if i < 20 else "OPEN"
        ans = "ANSWER" if (i % 20) < (12 if qt == "VERIFICATION" else 8) else "ABSTAIN"
        rows.append(_annotation({
            "query_id": f"k{i:02d}", "query_type": qt, "ground_truth": ans, "difficulty": "L4",
        }))
    report = v.validate_distribution(rows, "K-TRAIN")
    assert report.ok, report.errors
    # skew verification ratio -> error
    skewed = [r for r in rows if r["query_type"] == "OPEN"] + rows[:1]
    assert any(i.code == "QUERY_TYPE_IMBALANCE" for i in v.validate_distribution(skewed, "K-TRAIN").errors)


def test_distribution_requires_ninety_percent_l3_l5():
    rows = []
    for i in range(20):
        rows.append(_annotation({
            "query_id": f"hard{i:02d}",
            "query_type": "VERIFICATION" if i < 10 else "OPEN",
            "ground_truth": "ANSWER" if i % 2 == 0 else "ABSTAIN",
            "difficulty": "L4" if i < 17 else "L2",
        }))
    assert any(i.code == "DIFFICULTY_VIOLATION"
               for i in v.validate_distribution(rows, "K-TRAIN").errors)


def test_document_disjoint():
    manifests = [_manifest({"document_id": "train-doc"}), _manifest({"document_id": "check-doc"})]
    train = [_annotation({"document": "train-doc", "split": "K-TRAIN"})]
    check = [_annotation({"query_id": "k99", "document": "check-doc", "split": "K-CHECK"})]
    assert v.validate_document_disjoint(train + check, manifests).ok
    overlap = train + [_annotation({"document": "train-doc", "split": "K-CHECK"})]
    assert any(i.code in ("DOCUMENT_SPLIT_OVERLAP", "DOCUMENT_OVERLAP")
               for i in v.validate_document_disjoint(overlap, manifests).errors)
    unknown = [_annotation({"document": "nope", "split": "K-TRAIN"})]
    assert any(i.code == "ANNOTATION_REFERENCES_UNKNOWN_DOC"
               for i in v.validate_document_disjoint(unknown, manifests).errors)


def test_identifier_quota():
    rows = []
    for i in range(24):
        rows.append(_annotation({
            "query_id": f"id{i:02d}", "relation": "HAS_IDENTIFIER",
            "ground_truth": "ANSWER" if i < 12 else "ABSTAIN",
            "split": "K-TRAIN" if i < 12 else "K-CHECK",
        }))
    assert v.validate_identifier_quota(rows).ok
    assert not v.validate_identifier_quota(rows[:10]).ok


def test_table_quota():
    rows = [_annotation({"query_id": f"t{i:02d}", "document_style": "PARAMETER_TABLE"}) for i in range(13)]
    assert v.validate_table_quota(rows).ok
    assert not v.validate_table_quota(rows[:5]).ok


def test_failure_attribution_vocabulary():
    good = ["RETRIEVAL_MISSING_EVIDENCE", "NLI_JUDGE_FAILURE", "OTHER"]
    assert v.validate_failure_attribution(good).ok
    assert not v.validate_failure_attribution(["WHATEVER"]).ok


def test_runtime_diagnostics_required():
    good = {k: "x" for k in v.RUNTIME_DIAGNOSTIC_KEYS}
    assert v.validate_runtime_diagnostics(good).ok
    missing = dict(good); del missing["final_decision_source"]
    assert any(i.code == "MISSING_DIAGNOSTIC" for i in v.validate_runtime_diagnostics(missing).errors)


def test_generalization_classification_uses_frozen_policy():
    overall = {
        "accuracy": 0.75, "abstain_recall": 0.75, "answerable_recall": 0.75,
        "false_answer_rate": 0.15, "false_refusal_rate": 0.20,
    }
    assert v.classify_generalization(
        overall, {"accuracy": 0.75}, {"accuracy": 0.75}, runtime_valid=True,
    ) == "GENERALIZES"
    assert v.classify_generalization(
        {**overall, "accuracy": 0.65, "false_answer_rate": 0.30},
        {"accuracy": 0.70}, {"accuracy": 0.60}, runtime_valid=True,
    ) == "PARTIAL_GENERALIZATION"
    assert v.classify_generalization(
        overall, {"accuracy": 0.75}, {"accuracy": 0.75}, runtime_valid=False,
    ) == "RUNTIME_INVALID"


def test_vocabulary_constants():
    assert set(v.SPLITS) == {"K-TRAIN", "K-CHECK"}
    assert "OPEN_FALSE_RELAXATION" in v.FAILURE_ATTRIBUTIONS
    assert "HAS_DEFAULT_VALUE" in v.RELATION_TYPES
    assert set(v.RUNTIME_DIAGNOSTIC_KEYS) >= {
        "query_path", "nli_router_triggered", "open_sufficiency_invoked",
        "final_decision_source", "grounding_status",
    }


def test_independence_validation():
    forbidden = [{
        "document_id": "old-doc", "official_url": "https://example.com/old.pdf",
        "manufacturer": "Acme", "product_family": "FamilyX", "product_series": "S1",
        "source_name": "Old Manual", "sha256": "AA".ljust(64, "0"),
    }]
    clean = [_manifest({
        "document_id": "k-doc", "official_url": "https://newvendor.com/new.pdf",
        "manufacturer": "NewVendor", "product_family": "FamilyK", "product_series": "K1",
        "source_name": "New Manual", "sha256": "BB".ljust(64, "0"),
    })]
    assert v.validate_independence(clean, forbidden).ok
    # URL overlap -> hard error
    bad_url = [_manifest({"official_url": "https://example.com/old.pdf", "document_id": "k-doc",
                          "manufacturer": "NewVendor"})]
    assert any(i.code == "INDEPENDENCE_URL_OVERLAP" for i in v.validate_independence(bad_url, forbidden).errors)
    # same manufacturer+family (different series) -> warning only
    same_family = [_manifest({"document_id": "k-doc", "official_url": "https://newvendor.com/k.pdf",
                              "manufacturer": "Acme", "product_family": "FamilyX", "product_series": "S9",
                              "source_name": "New S9 Manual"})]
    issues = v.validate_independence(same_family, forbidden).issues
    assert any(i.code == "INDEPENDENCE_PRODUCT_FAMILY_OVERLAP" for i in issues)
    assert not any(i.code == "INDEPENDENCE_PRODUCTLINE_OVERLAP" for i in issues)


def test_query_leakage():
    dev = ["What is the default value of parameter P01 on the AC500-S module?"]
    k_ok = ["Which terminal is the EtherCAT master port on the X20 module?"]
    k_dup = ["what is default value of parameter p01 on ac500-s module"]
    assert v.validate_query_leakage(k_ok, dev).ok
    assert any(i.code == "QUERY_LEAKAGE_OVERLAP" for i in v.validate_query_leakage(k_dup, dev).issues)


def test_benchmark_scope_size_confidence_and_independence():
    manifests = [
        _manifest({"document_id": f"d{i}", "official_url": f"https://e.com/{i}",
                   "manufacturer": f"M{i}", "equipment_type": "plc_controller"})
        for i in range(6)
    ]
    rows = []
    for i in range(96):
        doc = f"d{i % 6}"
        split = "K-TRAIN" if i % 6 < 3 else "K-CHECK"
        rows.append(_annotation({
            "query_id": f"scope{i:03d}", "query": f"Unique query {i}",
            "document": doc, "split": split, "confidence": "HIGH",
            "evidence_chunk_ids": [f"chunk-{i % 6}"],
        }))
    assert v.validate_benchmark_scope(rows, manifests).ok

    bad = list(rows)
    bad[0] = {**bad[0], "confidence": "MEDIUM"}
    assert any(i.code == "NON_HIGH_CONFIDENCE_QUERY"
               for i in v.validate_benchmark_scope(bad, manifests).errors)
