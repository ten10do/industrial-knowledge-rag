"""V3.77 corpus-support mechanical probes.

Runs deterministic full-corpus lexical probes over ALL PAGES of the frozen
3-document corpus (not Top-K retrieval) so benchmark answerability can be
decided against actual corpus coverage.

Prediction-blind by construction: this script's inputs are probe term lists and
the corpus itself. It never reads runtime predictions, Evidence decisions,
evaluation results, or retrieval traces. All detailed hits are written to the
private alignment directory.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402

BASE = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private"
CORPUS_PDFS = [
    BASE / "v364_generalization" / "documents" / "Siemens_S7_1200_System_Manual_EN.pdf",
    BASE / "v364_generalization" / "documents" / "ABB_ACS580_Firmware_Manual.pdf",
    BASE / "v364_generalization" / "documents" / "Schneider_M221_Hardware_Guide_EN.pdf",
]
OUT_DIR = BASE / "v377_alignment"
PROBES_OUT = OUT_DIR / "probe_results.json"

_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")


def normalize_text(text: str) -> str:
    lowered = (text or "").casefold()
    lowered = _DASH_RE.sub("-", lowered)
    lowered = lowered.replace("\u00a0", " ").replace("’", "'")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def load_page_index() -> dict[str, dict[int, str]]:
    index: dict[str, dict[int, str]] = {}
    for pdf in CORPUS_PDFS:
        pages = rag_core.load_pdf(str(pdf))
        index[pdf.name] = {
            int(doc.metadata.get("page", i)): normalize_text(doc.page_content)
            for i, doc in enumerate(pages)
        }
    return index


def find_pages(index, doc, pattern: re.Pattern) -> list[int]:
    return sorted(page for page, text in index.get(doc, {}).items() if pattern.search(text))


def probe(index, label: str, regex: str, docs=None) -> dict:
    pattern = re.compile(regex, re.IGNORECASE)
    result = {"label": label, "regex": regex, "total_hits": 0, "pages": {}}
    for doc in index:
        if docs and doc not in docs:
            continue
        pages = find_pages(index, doc, pattern)
        if pages:
            result["pages"][doc] = pages[:40]
            result["total_hits"] += len(pages)
    return result


def main() -> None:
    index = load_page_index()
    siemens, abb, schneider = (
        "Siemens_S7_1200_System_Manual_EN.pdf",
        "ABB_ACS580_Firmware_Manual.pdf",
        "Schneider_M221_Hardware_Guide_EN.pdf",
    )

    groups: dict[str, list] = defaultdict(list)

    # --- Known five absent models: absence must be proven over FULL text ----
    groups["known_absent_models"] = [
        probe(index, "SINAMICS G120 (family)", r"(?<![a-z0-9])g120(?![a-z0-9])|sinamics\s+g120"),
        probe(index, "ATV320 / Altivar ATV320", r"(?<![a-z0-9])atv ?320(?![a-z0-9])|altivar"),
        probe(index, "PowerFlex 520 (520 family)", r"powerflex(?![a-z0-9])|(?<![a-z0-9])pf ?52\d(?![a-z0-9])"),
        probe(index, "FR-E800", r"(?<![a-z0-9])(fr[\s-]*)?e800(?![a-z0-9])|fr-e\s*800"),
        probe(index, "Danfoss FC51", r"(?<![a-z0-9])fc ?51(?![a-z0-9])|danfoss"),
        # control group: models that MUST be present
        probe(index, "[control] S7-1200", r"(?<![a-z0-9])s7[\s-]*1200(?![a-z0-9])"),
        probe(index, "[control] ACS580", r"(?<![a-z0-9])acs ?580(?![a-z0-9])"),
        probe(index, "[control] M221 / TM221", r"(?<![a-z0-9])m221(?![a-z0-9])|tm2\d{2}ce\w*t\w*(?![a-z0-9])|modicon\s+m221"),
    ]

    # --- DIRECT_FACT subjects and V1 claim values ---------------------------
    groups["direct_fact"] = [
        probe(index, "acceleration time topic", r"acceleration time|acc?el(?:eration)? (?:ramp|time)|ramp[- ]up time"),
        probe(index, "deceleration time topic", r"deceleration time|ramp[- ]down time"),
        probe(index, "value '5.0 seconds/5 s' near accel", r"(?:5\.0|5)\s*(?:s\b|sec\b|second)"),
        probe(index, "motor overload / class 10", r"motor overload|overload protection|(?<![a-z0-9])class 10(?![a-z0-9])"),
        probe(index, "braking resistor topic", r"braking resistor|brake resistor|resistor brake"),
        probe(index, "integrated braking unit/ chopper", r"integrated brake|braking chopper|brake chopper"),
        probe(index, "Modbus RTU", r"modbus(\s+(rtu|rtu rs485|ascii))?"),
        probe(index, "rated current / frame size", r"rated current|frame size"),
        probe(index, "analog input ranges 0-10V / 4-20mA", r"0[\s\-–to]*10\s*v|(?<!\d)4[\s\-–to]*20\s*ma"),
        probe(index, "digital input count 6/8", r"\b6 (?:digital )?inputs\b|\b8 (?:digital )?inputs\b|(?:six|eight) (?:digital )?inputs"),
        probe(index, "PID control", r"\bpid\b|pid control|pi ?d"),
        probe(index, "safety function STO", r"\bsto\b|safe torque|safe stop|safety function|safety integrated"),
    ]

    # --- VALUE slice claim values -------------------------------------------
    groups["value_slice"] = [
        probe(index, "supply voltage 380-480 V", r"380[\s\-–to]*480\s*v(?:olts?)?\b"),
        probe(index, "output frequency 0-590 Hz", r"(?<!\d)590\s*hz"),
        probe(index, "any output freq ceiling with Hz range", r"\b(?:0[\s\-–to]+)\d{2,4}\s*hz"),
        probe(index, "ambient temperature -10..50 C", r"50\s*°?\s*c\b|−?10\s*(?:to|–|-)?\s*\+?50"),
        probe(index, "IP20", r"(?<![a-z0-9])ip20(?![a-z0-9])"),
        probe(index, "IP ratings generally", r"(?<![a-z0-9])ip\d{2}(?![a-z0-9])"),
        probe(index, "vector control", r"vector control|scalar control|flux vector"),
        probe(index, "overload capacity 150%/60s", r"150\s*%\s*(?:for\s*)?(?:of\s*)?(?:60\s*s|one minute)?|overload capacity"),
        probe(index, "response time <2 ms", r"(?:response|\bscan\b)[^.]{0,40}?\bms\b|(?<!\d)2\s*ms"),
        probe(index, "power rating kW", r"\bkw\b|kilowatt"),
        probe(index, "efficiency class IE2", r"(?<![a-z0-9])ie2(?![a-z0-9])|efficiency class"),
        probe(index, "noise level dB", r"\bdba?\b|sound (?:level|pressure)|noise level"),
    ]

    # --- NON_TABLE general industrial concepts ------------------------------
    groups["non_table"] = [
        probe(index, "variable frequency drive concept", r"variable (?:frequency|speed) drive|\bvfd\b|variable speed driv"),
        probe(index, "PWM concept", r"pulse[- ]width modulation|\bpwm\b"),
        probe(index, "electrical safety precautions", r"safety precautions|electric(al)? shock|danger[^.]{0,30}voltag"),
        probe(index, "overload relay", r"overload relay|thermal overload"),
        probe(index, "PROFINET basics", r"profinet|profibus"),
        probe(index, "sourcing/sinking inputs", r"sourcing|sinking|sink/source|source/sink|pnp|npn"),
        probe(index, "Modbus RTU protocol (general)", r"modbus rtu|modbus protocol|\bmodbus\b"),
        probe(index, "emergency stop circuit", r"emergency stop|e[\s-]?stop"),
        probe(index, "wire gauge selection", r"wire gauge|cable (?:cross[- ]section|size)|conductor size|awg"),
        probe(index, "AC vs DC motors", r"\bac motor|\bdc motor|ac drive"),
        probe(index, "three-phase power", r"three[- ]phase|3[- ]phase"),
        probe(index, "regenerative braking", r"regenerativ|regen braking|braking energy"),
    ]

    # --- HARD_NEGATIVE identifiers ------------------------------------------
    groups["hard_negative"] = [
        probe(index, "PowerFlex 755", r"powerflex 755|(?<![a-z0-9])pf ?755"),
        probe(index, "ACS355", r"acs ?355"),
        probe(index, "PowerFlex 525", r"powerflex 525|(?<![a-z0-9])pf ?525"),
        probe(index, "Siemens G120 install", r"g120"),
        probe(index, "Schneider ATV320 compatibility", r"atv ?320|altivar"),
    ]
    groups["hard_negative"].extend(groups["known_absent_models"])

    serialized = {group: items for group, items in groups.items()}
    PROBES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROBES_OUT, "w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=1, ensure_ascii=False)

    for group, items in serialized.items():
        print(f"\n== {group} ==")
        for item in items:
            hit_summary = ", ".join(
                f"{doc.split('_')[0]}:{len(pages)}p" for doc, pages in item["pages"].items()
            )
            print(f"  {item['label']:<44} hits={item['total_hits']:<5} {hit_summary}")

    print(f"\nsaved: {PROBES_OUT}")


if __name__ == "__main__":
    main()
