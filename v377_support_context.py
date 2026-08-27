"""V3.77 support-context dumper.

Dumps surrounding normalized text snippets for targeted probe patterns over ALL
pages of the frozen corpus so every benchmark-support judgment can be grounded
in verifiable quoted evidence. Prediction-blind: reads only the corpus.

Usage: python v377_support_context.py            # runs the built-in inspection set
       python v377_support_context.py PATTERN    # ad-hoc regex probe
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402
from v377_support_probes import normalize_text, load_page_index  # noqa: E402

BASE = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private"
OUT_DIR = BASE / "v377_alignment" / "context"


def show(index, label, regex, max_snippets=6, window=170):
    pattern = re.compile(regex, re.IGNORECASE)
    print(f"\n### {label}   /{regex}/")
    dumped = 0
    for doc, pages in index.items():
        for page in sorted(pages):
            text = pages[page]
            match = pattern.search(text)
            if not match:
                continue
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = text[start:end].strip()
            print(f"[{doc} p{page}] …{snippet}…")
            dumped += 1
            if dumped >= max_snippets:
                return


def main() -> None:
    index = load_page_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        show(index, "adhoc", sys.argv[1], max_snippets=25, window=250)
        return

    # Known absent models — full-text absence proof (incl. the G120 near-hit).
    show(index, "G120 occurrence", r"(?<![a-z0-9])g ?120(?![a-z0-9])|sinamics")
    show(index, "ATV320/Altivar", r"altivar|atv ?\d{3}")
    show(index, "PowerFlex any", r"powerflex|(?<![a-z0-9])pf ?5\d\d")
    show(index, "FR-E800 / E800", r"e[\s-]?800|fr-e")
    show(index, "FC51/Danfoss", r"(?<![a-z0-9])fc ?51|danfoss")

    # Corpus identity confirmation.
    show(index, "ACS580 title context", r"acs ?580", max_snippets=3, window=200)

    # DIRECT_FACT grounding.
    show(index, "acceleration time default (ABB)", r"acceleration time", max_snippets=8)
    show(index, "deceleration time default (ABB)", r"deceleration time", max_snippets=5)
    show(index, "motor overload protection", r"motor overload", max_snippets=5)
    show(index, "class 10 overload", r"(?<![a-z0-9])class\s*10(?![a-z0-9])", max_snippets=5)
    show(index, "brake chopper/resistor integrated", r"brak(e|ing) (chopper|resistor)", max_snippets=6)
    show(index, "M221 serial/Modbus RTU", r"modbus rtu", max_snippets=4)
    show(index, "M221 communication ports", r"(embedded|serial) (link|port)|com port|line.?(protocol|configuration)", max_snippets=4)
    show(index, "analog input ranges siemens", r"0[\s-]*10\s*v|4[\s-]*20\s*ma", max_snippets=5)
    show(index, "digital inputs counts", r"\b(?:6|8)\s+digital inputs|(?:six|eight)\s+(?:digital\s+)?inputs", max_snippets=5)
    show(index, "PID built-in", r"\bpid\b", max_snippets=5)
    show(index, "STO safety", r"\bsto\b|safe torque off", max_snippets=5)

    # VALUE slice grounding.
    show(index, "mains/supply voltage range ABB", r"(?:mains|supply|input) voltag", max_snippets=5)
    show(index, "voltage 380…480-ish ranges", r"3\s*~?\s*(?:ac)?[\s\d.\-–to]{0,12}48\d\s*v|u\d\s*(?:mains)?\s*voltage", max_snippets=5)
    show(index, "frequency ceiling Hz", r"frequen[^.]{0,60}?hz|\bhz\b.{0,40}", max_snippets=5)
    show(index, "ambient temp specification", r"ambient (?:air )?temperature", max_snippets=5)
    show(index, "IP ratings", r"(?<![a-z0-9])ip\d{2}(?![a-z0-9])", max_snippets=6)
    show(index, "vector/scalar control modes", r"vector control|scalar", max_snippets=4)
    show(index, "overload 110%/150%", r"(?:110|150)\s*% overload|overload[^.]{0,50}?(?:110|150)\s*%", max_snippets=5)
    show(index, "response/filter ms", r"response time|\bms\b", max_snippets=4)
    show(index, "power/kW ratings", r"motor (?:rated )?power| rated power", max_snippets=4)
    show(index, "IE efficiency classes", r"(?<![a-z0-9])ie\d(?![a-z0-9])|efficienc", max_snippets=5)
    show(index, "noise dB", r"d?ba\b|sound pressure|noise level", max_snippets=4)

    # NON_TABLE grounding.
    show(index, "variable frequency drive definitional", r"variable (?:frequency|speed) (?:driv|ac)|\bvfd\b", max_snippets=4)
    show(index, "PWM explanation", r"pulse[- ]width modulation|\bpwm\b", max_snippets=4)
    show(index, "electrical work precautions", r"safety precautions|electric(al)? shock", max_snippets=5)
    show(index, "overload relay purpose", r"overload relay", max_snippets=4)
    show(index, "PROFINET explanation", r"profinet", max_snippets=4)
    show(index, "sink/source wiring", r"sink/source|source/sink|sourcing input|sinking input|sinking/sourcing", max_snippets=5)
    show(index, "emergency stop circuit", r"emergency stop", max_snippets=5)
    show(index, "wire gauge/cross-section guidance", r"wire gauge|conductor size|cross[- ]section", max_snippets=5)
    show(index, "AC/DC motor difference", r"\bac motors?\b|\bdc motors?\b", max_snippets=4)
    show(index, "three-phase power", r"three[- ]phase|3[- ]phase", max_snippets=4)
    show(index, "regenerative braking", r"regenerativ", max_snippets=5)


if __name__ == "__main__":
    main()
