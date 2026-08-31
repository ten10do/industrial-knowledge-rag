"""Unit tests for backend.retrieval.coverage_gate (V3.85 Phase C).

The coverage gate is a pure string/dict module (document_identity_v373,
product_identity, technical are all lightweight), so it is imported
directly without mocks. These tests lock the detection contract:

* pattern-covered model mentions are extracted (series patterns + concrete
  vendor terms, brand words excluded);
* corpus-known models are never rejected (dual key forms);
* out-of-corpus pattern-covered models are rejected with
  OUT_OF_CORPUS_MODEL:<mention>;
* the verdict is independent of the retrieval window when the build-time
  index is supplied;
* devices outside the vendor pattern tables (e.g. Anybus CompactCom M40)
  produce no signal - the documented pattern-coverage blind spot.
"""

import unittest

from backend.retrieval.coverage_gate import (
    corpus_known_models,
    coverage_gate_verdict,
    detect_model_mentions,
    out_of_corpus_models,
)

# Frozen V377 corpus identities (coverage_index_v385.json, V385_COVERAGE_INDEX_V1)
GLOBAL_KNOWN_MODELS = {
    "acs580": {"manufacturer": "abb", "product_series": "acs580", "equipment_model": "acs580"},
    "modicon m221": {"manufacturer": "schneider electric",
                     "product_series": "modicon m221", "equipment_model": "modicon m221"},
    "s7-1200": {"manufacturer": "siemens", "product_series": "s7-1200", "equipment_model": "s7-1200"},
}


class _FakeDoc:
    def __init__(self, metadata):
        self.metadata = metadata


class DetectModelMentionsTests(unittest.TestCase):
    def test_sinamics_g120_detected(self):
        self.assertEqual(
            detect_model_mentions("Tell me about the SINAMICS G120 drive."),
            ["sinamics g120"],
        )

    def test_s7_1200_detected_in_hyphen_and_space_forms(self):
        self.assertEqual(detect_model_mentions("What is the CPU of the S7-1200?"), ["s7-1200"])
        self.assertEqual(detect_model_mentions("Does the s7 1200 support PROFINET?"), ["s7-1200"])

    def test_acs580_and_m221_detected(self):
        self.assertEqual(detect_model_mentions("ACS580 drive manual"), ["acs580"])
        self.assertEqual(detect_model_mentions("Modicon M221 wiring"), ["modicon m221"])
        self.assertEqual(detect_model_mentions("the m221 Ethernet port"), ["modicon m221"])

    def test_brand_only_queries_produce_no_mention(self):
        # "siemens" / "tia portal" are brand-level, not concrete models.
        self.assertEqual(detect_model_mentions("Siemens TIA Portal project"), [])
        self.assertEqual(detect_model_mentions("ABB drive catalogue"), [])

    def test_out_of_pattern_device_produces_no_mention(self):
        # Anybus CompactCom M40 is not covered by the vendor pattern tables.
        # This is the documented blind spot: no signal is conservative.
        self.assertEqual(detect_model_mentions("Anybus CompactCom M40 supply voltage"), [])


class CorpusKnownModelsTests(unittest.TestCase):
    def test_dual_key_forms_from_index(self):
        known = corpus_known_models([], known_models=GLOBAL_KNOWN_MODELS)
        # normalize_identity_text collapses hyphens to spaces, series patterns
        # keep them: both forms must resolve.
        self.assertIn("s7 1200", known)
        self.assertIn("s7-1200", known)
        self.assertIn("acs580", known)
        self.assertIn("modicon m221", known)

    def test_window_fallback_from_document_metadata(self):
        docs = [_FakeDoc({"equipment_model": "s7-1200", "product_series": "s7-1200"})]
        known = corpus_known_models(docs)
        self.assertIn("s7 1200", known)


class OutOfCorpusTests(unittest.TestCase):
    def test_out_of_corpus_model_flagged(self):
        unknown = out_of_corpus_models(
            "Tell me about the SINAMICS G120 drive.", [], known_models=GLOBAL_KNOWN_MODELS)
        self.assertEqual(unknown, ["sinamics g120"])

    def test_other_vendor_models_out_of_corpus(self):
        for text, expected in [
            ("S7-1500 controller", "s7-1500"),
            ("ACS880 drive", "acs880"),
            ("M241 PLC", "m241"),
            ("PowerFlex 520", "powerflex 520"),
        ]:
            self.assertEqual(
                out_of_corpus_models(text, [], known_models=GLOBAL_KNOWN_MODELS),
                [expected],
                msg=text,
            )

    def test_in_corpus_models_never_flagged(self):
        for text in ["S7-1200", "ACS580", "Modicon M221", "the s7 1200 CPU", "acs580 drive"]:
            self.assertEqual(
                out_of_corpus_models(text, [], known_models=GLOBAL_KNOWN_MODELS),
                [],
                msg=text,
            )


class CoverageGateVerdictTests(unittest.TestCase):
    def test_verdict_fires_out_of_corpus(self):
        ok, reason = coverage_gate_verdict(
            "Tell me about the SINAMICS G120 drive.", [], known_models=GLOBAL_KNOWN_MODELS)
        self.assertFalse(ok)
        self.assertEqual(reason, "OUT_OF_CORPUS_MODEL:sinamics g120")

    def test_verdict_silent_in_corpus(self):
        ok, reason = coverage_gate_verdict(
            "What is the CPU variant of the S7-1200?", [], known_models=GLOBAL_KNOWN_MODELS)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_verdict_silent_no_mention(self):
        ok, _ = coverage_gate_verdict(
            "What is the maximum operating temperature for plastic housing?",
            [], known_models=GLOBAL_KNOWN_MODELS)
        self.assertTrue(ok)

    def test_window_independent_given_index(self):
        text = "How do I configure the ABB ACS880 drive?"
        empty = coverage_gate_verdict(text, [], known_models=GLOBAL_KNOWN_MODELS)
        window = coverage_gate_verdict(
            text,
            [_FakeDoc({"equipment_model": "s7-1200", "product_series": "s7-1200"})],
            known_models=GLOBAL_KNOWN_MODELS,
        )
        self.assertEqual(empty, window)
        self.assertFalse(empty[0])


if __name__ == "__main__":
    unittest.main()
