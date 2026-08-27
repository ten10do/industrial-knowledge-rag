"""V3.81-LAT public tests: spec-existence lattice pre-feasibility probe.

Pins the PROBE's measured behavior (including its failure modes) so any future
lattice attempt starts from a characterized baseline. No manual text embedded;
all fixtures synthetic. Read-only over the runtime (probe is analysis tooling).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import v381_lat_lattice_probe as probe


def _meta(**kw):
    base = {"source": "Manual_X.pdf", "page": 7, "equipment_model": "device x", "chunk_id": "c1"}
    base.update(kw)
    return base


# --- extraction behavior -------------------------------------------------------


def test_prose_rated_value_extracts_when_family_precedes_head():
    """MEASURED shape: family lookup looks LEFT of the rating adjective only;
    corpus-dominant captures therefore have the noun phrase first."""
    entries = probe.extract_entries(
        "Encoder supply module Output current, rated value 200 mA.", _meta())
    fixed = [(e["family"], e["value"]) for e in entries if e["kind"] == "FIXED_RATING"]
    assert ("current", "200 mA") in [(f, v) for f, v in fixed]


def test_rating_adjective_first_sentences_are_missed():
    """Characterized blind spot: family is searched BEFORE the adjective, so
    'The rated voltage is 400 V' extracts NOTHING - counted inside the
    INVALID/missing mass of the audit."""
    entries = probe.extract_entries("The rated voltage is 400 V nominal.", _meta())
    assert entries == []


def test_range_extraction_pairs_numbers_and_unit():
    entries = probe.extract_entries(
        "The output frequency setting range is 0.0 to 599.0 Hz in firmware mode.",
        _meta())
    ranges = [e for e in entries
              if e["kind"] == "RANGE" and e["family"] in {"frequency", "output_frequency"}]
    assert ranges and "599" in ranges[0]["value"]


def test_ip_named_option_extracted():
    entries = probe.extract_entries("Enclosure provides IP 21 protection class.",
                                    _meta())
    assert any(e["kind"] == "NAMED_OPTION" and e["value"] == "IP21" for e in entries)


def test_scale_factor_noise_is_captured_but_flagged_by_audit_only():
    """KNOWN failure mode: '1000 = 1 s' scaling text yields a FIXED_RATING whose
    unit belongs to scaling, not the rating - characterization of measured
    invalid-capture family (see audit verdict failure modes)."""
    entries = probe.extract_entries(
        "Signal filter time Real 0.000 … 30.000 s 1000 = 1 s default 1 min",
        _meta())
    assert entries  # extractor is greedy by design; audit labels handle quality
    assert any(e["kind"] == "RANGE" for e in entries)


def test_table_row_family_currently_yields_nothing():
    """Structural gap (measured): flattened tables defeat P_ROW on stored chunks."""
    entries = probe.extract_entries(
        "23.12 acceleration time 1 Defines ramp Real 0.0 6000.0 s", _meta())
    assert not [e for e in entries if e["kind"] == "PARAM_ROW"]


def test_unit_family_mismatch_capture_is_possible_by_design():
    """Documents the audit's dominant failure class at the pattern level:
    when a DIFFERENT family noun sits before the rating adjective, the emitted
    entry inherits that family while the captured value's UNIT may belong
    elsewhere - extractor emits it, audit labels it INVALID."""
    entries = probe.extract_entries(
        "Supply voltage group Rated input current 4.5 mA typical",
        _meta())
    fixed = [(e["family"], e["value"]) for e in entries if e["kind"] == "FIXED_RATING"]
    assert ("voltage", "4.5 mA") in fixed


def test_rating_head_without_prior_family_is_dropped_entirely():
    """Companion blind spot: no family term BEFORE the adjective -> no entry."""
    entries = probe.extract_entries(
        "Digital input Rated supply voltage 24 V with input current 7 mA limits",
        _meta())
    assert entries == []


# --- identity / privacy ---------------------------------------------------------


def test_subject_from_metadata_not_from_text_guessing():
    entries = probe.extract_entries(
        "Encoder supply Output current, rated value 200 mA.", _meta(equipment_model="dev-x"))
    assert entries and all(e["subject"] == "dev-x" for e in entries)


def test_no_manual_text_or_model_literals_in_module_source():
    source = Path("v381_lat_lattice_probe.py").read_text(encoding="utf-8").casefold()
    for banned in ("acs580", "s7-1200", "m221", "sinamics", "powerflex"):
        assert banned not in source


# --- gate & determinism constants -----------------------------------------------


def test_gate_threshold_pinned():
    import v381_lat_audit as audit
    assert audit.GATE_PRECISION == 0.95
    assert set(audit.LABELS) == set(range(100))


def test_probe_seed_and_sample_size_pinned():
    assert probe.SAMPLE_SEED == 380471
    assert probe.SAMPLE_N == 100
