"""Focused public checks for the evidence-v323.1 candidate matching policy.

Covers the new behaviours introduced on top of evidence-v321.1:
  - partial / non-digit identity resolution ("ArmorBlock" -> "ArmorBlock 5000")
  - open-question keyword fallback with majority coverage and light stemming
  - expanded causal-attribute semantic equivalence
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.retrieval import RetrievalCandidate, analyze_query
from backend.retrieval.evidence_contract import (
    evaluate_evidence_contract,
    build_typed_requirement,
    _mentioned_document_identities,
)


def _doc(chunk_id: str, text: str, *, document_id: str = "manual-a", subsection: str = "",
         manufacturer: str = "Example Automation", family: str = "DriveX",
         model: str = "DriveX 100"):
    return SimpleNamespace(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "manufacturer": manufacturer,
            "product_family": family,
            "equipment_model": model,
            "section": "Parameters",
            "subsection": subsection,
            "page": 10,
        },
    )


def _contract(query: str, selected: list, corpus: list | None = None):
    corpus = corpus or selected
    candidates = [
        RetrievalCandidate(document=document, retrieval_source="fixture", final_rank=index)
        for index, document in enumerate(selected, 1)
    ]
    return evaluate_evidence_contract(query, candidates, corpus, analyze_query(query, corpus))


def test_partial_identity_mention_resolves_to_corpus_document():
    doc = _doc("c", "Functional Earth B grounds the module I/O power circuitry.",
               family="ArmorBlock 5000", model="ArmorBlock 5000")
    identities = _mentioned_document_identities(
        "Which ArmorBlock circuit is grounded by Functional Earth B?", [doc],
    )
    assert identities, "partial 'ArmorBlock' should resolve to 'ArmorBlock 5000'"
    assert identities[0]["product_family"] == "ArmorBlock 5000"


def test_non_digit_identity_mention_resolves():
    doc = _doc("c", "remedy: restart the function block.",
               manufacturer="Bosch Rexroth", family="ctrlX CORE", model="ctrlX CORE")
    identities = _mentioned_document_identities(
        "What remedies are given for ctrlX code 0C85009D?", [doc],
    )
    assert identities, "non-digit 'ctrlX' should resolve to 'ctrlX CORE'"


def test_open_question_keyword_fallback_covers_salient_terms():
    doc = _doc("c", "The reset selection parameter chooses the trip reset behaviour.")
    result = _contract("Which parameter selects reset behavior?", [doc])
    assert result.sufficient


def test_keyword_fallback_tolerates_framing_gap_with_majority():
    doc = _doc("c", "The channel mode shows power supply mode when Class B is enabled.")
    result = _contract(
        "What channel mode is displayed when IO-Link Class B is enabled?", [doc],
    )
    assert result.sufficient


def test_keyword_stemming_matches_inflection():
    doc = _doc("c", "Output frequency and acceleration time are defined here.")
    result = _contract(
        "Between which two frequencies is the acceleration time defined?", [doc],
    )
    assert result.sufficient


def test_cause_attribute_accepts_resulting_in():
    doc = _doc("c", "Cards from other makers may fail, resulting in data loss.")
    result = _contract(
        "Why does the manual require industrial CFast cards?", [doc],
    )
    assert result.sufficient


def test_related_only_identifier_still_refuses():
    current = _doc("current", "Index (hex) Object name 1011 Restore default parameters",
                   subsection="Restore default parameters")
    related = _doc("related", "Related parameters: 0x1011")
    assert not _contract(
        "What does object 0x1011 specify?", [related], [related, current],
    ).sufficient


def test_wrong_value_row_still_refuses():
    target = _doc("target", "P30 Restart delay. Default: 1 second.", subsection="P30 Restart delay")
    other = _doc("other", "P31 Stop delay. Default: 5 seconds.", subsection="P31 Stop delay")
    assert not _contract("Is parameter P30 default 5 seconds?", [target, other]).sufficient


def test_cross_document_aggregation_still_unsafe():
    attempts = _doc("attempts", "Restart tries sets the number of restart attempts.", subsection="Restart")
    other_manual = _doc(
        "foreign-delay", "Restart delay sets the time between restart attempts.",
        document_id="manual-b", subsection="Delay",
    )
    query = "Which parameters define restart attempts and the delay between attempts?"
    assert not _contract(query, [attempts, other_manual]).sufficient


def test_unseen_manufacturer_identity_does_not_override_content():
    doc = _doc("c", "Only industrial CFast cards may be used.", manufacturer="OtherVendor")
    identities = _mentioned_document_identities("Which CFast cards are required?", [doc])
    assert identities == (), "a generic query must not resolve an unrelated vendor"
