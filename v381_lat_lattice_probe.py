"""V3.81-LAT Spec-Existence Lattice Pre-Feasibility Probe (READ-ONLY).

Question (owner-approved path): does the FROZEN corpus admit a cheap,
verifiable offline SPEC-ENTRY table (entity-attribute-value) precise enough
(anchor >=95% on a hand-audited random sample) to justify building the
spec-existence evidence asset that all prior phases found missing?

SHADOW/ANALYSIS ONLY:
 - reads vector_db_v369 read-only via Chroma ``get``;
 - extracts deterministic spec-shaped entries (prose defaults/ratings, ranges,
   IP/class named options, parameter-table rows);
 - saves ALL entries privately + a seeded random sample for human audit;
 - two full runs must produce identical canonical digests.

NO runtime file changes. Private output: results/v381_lat/
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")

from langchain_chroma import Chroma  # noqa: E402

OUT_DIR = _REPO_ROOT / "results" / "v381_lat"
LATTICE_PROBE_VERSION = "lat-prefeasibility-v381-r0"
SAMPLE_N = 100
SAMPLE_SEED = 380471

# --- generic industrial attribute heads (family normalization only) ------------
ATTRIBUTE_FAMILIES: tuple[tuple[str, str], ...] = (
    ("output_frequency", r"output frequency"),
    ("frequency", r"\bfrequency\b"),
    ("voltage", r"\bvoltage\b|\bsupply voltage\b"),
    ("current", r"\bcurrent\b"),
    ("power", r"\bpower\b"),
    ("torque", r"\btorque\b"),
    ("time", r"\btime\b|\btime-out\b|\bramp\b"),
    ("temperature", r"\btemperature\b"),
    ("protection", r"\bprotection\b|\bingress\b"),
    ("efficiency", r"\befficiency\b"),
    ("noise", r"\bnoise\b|\bacoustic\b|\bsound pressure\b"),
    ("overload", r"\boverload\b"),
    ("control_method", r"\bcontrol (?:method|mode|principle)\b"),
    ("cable_length", r"\bcable length\b|\btransmission distance\b"),
    ("ip_rating", r"\bIP\s?\d{2}",),
)


def attribute_family(context_left: str) -> str:
    for name, pattern in ATTRIBUTE_FAMILIES:
        if re.search(pattern, context_left, re.IGNORECASE):
            return name
    return ""


NUMBER = r"-?\d+(?:[.,]\d+)?"
UNIT = r"(?:hz|khz|mhz|kw|kva|w|v(?:ac|dc)?|mv|a|ma|nm|n\s*[·⋅-]?\s*m|s|sec|seconds?|ms|min(?:utes?)?|h(?:ours?)?|%|°?c|°?f|bar|psi|mm|cm|m|ft|lb|kg|g|byte|kb|mb|ip\d{2}|j)"

# P1: <rating-adjective> ... NUM UNIT   (defaults / rated values in prose)
P_RATING = re.compile(
    rf"(?P<head>\b(?:factory default|factory setting|default value|default setting|default|rated|nominal|maximum|max\.?|minimum|min\.?)\b)"
    rf"[^.\n]{{0,60}}?(?P<num>{NUMBER})\s*(?P<unit>{UNIT})\b",
    re.IGNORECASE,
)
# P2: NUM UNIT .. to .. NUM UNIT   (ranges)
P_RANGE = re.compile(
    rf"(?P<num1>{NUMBER})\s*(?P<u1>{UNIT})?\s*(?:to|–|—|…|\.\.\.)\s*(?P<num2>{NUMBER})\s*(?P<u2>{UNIT})\b",
    re.IGNORECASE,
)
# P3: named protection options
P_IP = re.compile(r"(?<![\w-])IP\s?(?P<num>\d{2})(?P<suffix>[A-K])?(?![\w])")
# P4: parameter-table style row: "<id> <name words> ... numbers [unit]"
P_ROW = re.compile(
    rf"(?<![a-z0-9.])(?P<pid>\d{{1,2}}\.\d{{1,3}})\s(?P<name>[A-Za-z][A-Za-z0-9 ()/\-,]{{2,60}}?)(?:\s{{2,}}|\t|:\s*)(?P<tail>[^\n]{{0,120}})",
)
TAIL_NUM = re.compile(rf"(?P<num>{NUMBER})(?:\.\.){0,1}\s*(?P<unit>{UNIT})?", re.IGNORECASE)

SENT_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")


def _identity(meta: dict) -> str:
    parts = [str(meta.get("equipment_model") or ""), ]
    return parts[0].casefold()


def _quote_around(text: str, start: int, end: int, radius: int = 70) -> str:
    lo, hi = max(0, start - radius), min(len(text), end + radius)
    return " ".join(text[lo:hi].split())


def extract_entries(chunk_text: str, meta: dict) -> list[dict]:
    flat = " ".join(chunk_text.split())
    # table rows behave better on raw newlines; prose patterns on flattened text
    entries: list[dict] = []
    subject = _identity(meta)
    doc = str(meta.get("source") or "")
    page = meta.get("page")
    chunk = str(meta.get("chunk_id") or hashlib.sha1(flat[:200].encode()).hexdigest()[:12])

    def add(kind: str, family: str, value: str, quote: str, span: tuple[int, int], source_text: str):
        span_sig = hashlib.sha1(source_text[span[0]:span[1]].encode("utf-8", "ignore")).hexdigest()[:8]
        entries.append({
            "kind": kind, "family": family, "subject": subject, "value": value,
            "quote": quote, "source_document": doc, "page": page,
            "chunk_hint": int(span_sig[:6], 16),
        })

    for match in P_RATING.finditer(flat):
        fam = attribute_family(flat[max(0, match.start() - 90):match.start()])
        if not fam:
            continue
        value = f"{match.group('num')} {match.group('unit')}"
        add("FIXED_RATING", fam, value, _quote_around(flat, match.start(), match.end()), (match.start(), match.end()), flat)

    for match in P_RANGE.finditer(flat):
        fam_ctx = flat[max(0, match.start() - 110):match.start()]
        fam = attribute_family(fam_ctx)
        if not fam:
            continue
        u = match.group("u2") or ""
        value = f"{match.group('num1')}..{match.group('num2')} {u}".strip()
        add("RANGE", fam, value, _quote_around(flat, match.start(), match.end()), (match.start(), match.end()), flat)

    for match in P_IP.finditer(flat):
        suffix = match.group("suffix") or ""
        add("NAMED_OPTION", "protection", f"IP{match.group('num')}{suffix}",
            _quote_around(flat, match.start(), match.end()), (match.start(), match.end()), flat)

    seen_rows = set()
    for match in P_ROW.finditer(chunk_text):
        pid_norm = match.group("pid")
        tail_nums = TAIL_NUM.findall(match.group("tail"))
        if not tail_nums:
            continue
        first_num, unit = tail_nums[0]
        try:
            float(first_num.replace(",", "."))
        except ValueError:
            continue
        if float(first_num.replace(",", ".")) > 100000:
            continue
        dedup_key = (chunk, pid_norm)
        if dedup_key in seen_rows:
            continue
        seen_rows.add(dedup_key)
        cleaned_name = " ".join(match.group("name").split())
        entries.append({
            "kind": "PARAM_ROW", "family": "", "subject": subject,
            "value": f"{first_num}{' ' + unit if unit else ''}",
            "parameter": pid_norm,
            "attribute_name": cleaned_name,
            "quote": _quote_around(chunk_text, match.start(), min(match.end(), match.start() + 150)),
            "source_document": doc, "page": page, "chunk_hint": 0,
        })
    # NOTE (structural gap, measured): parameter-TABLE rows in this corpus are
    # flattened to single spaces during ingestion, so the P_ROW family above
    # matches nothing on the stored chunks. Table-structured spec extraction
    # would require re-parsing PDFs with layout awareness - recorded as the
    # main unknown of this pre-feasibility probe.
    return entries


def main() -> None:
    started_all = time.perf_counter()
    db = Chroma(persist_directory=rag_core.PERSIST_DIR,
                embedding_function=rag_core.get_embedding_model())
    data = db._collection.get(limit=10000, include=["documents", "metadatas"])
    documents, metas = data["documents"], data["metadatas"]
    assert len(documents) == 7333, f"unexpected corpus size {len(documents)}"

    entries: list[dict] = []
    t0 = time.perf_counter()
    for text, meta in zip(documents, metas):
        entries.extend(extract_entries(str(text), dict(meta)))
    extract_s = time.perf_counter() - t0

    digest_blob = json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(digest_blob).hexdigest()

    by_kind = Counter(e["kind"] for e in entries)
    by_family = Counter(e["family"] for e in entries if e["kind"] != "PARAM_ROW")
    by_doc = Counter(e["source_document"] for e in entries)

    rng = random.Random(SAMPLE_SEED)
    ordered = sorted(entries, key=lambda e: (e["source_document"], e["page"], e["kind"],
                                             e["quote"][:40]))
    sample = rng.sample(ordered, min(SAMPLE_N, len(ordered)))

    wall = time.perf_counter() - started_all
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_version": LATTICE_PROBE_VERSION,
        "n_chunks": len(documents),
        "n_entries": len(entries),
        "digest": digest,
        "by_kind": dict(by_kind),
        "by_family": dict(by_family),
        "by_document": dict(by_doc),
        "extract_seconds": round(extract_s, 2),
        "wall_seconds": round(wall, 2),
        "entries": entries,
        "sample": sample,
    }
    json.dump(payload, open(OUT_DIR / "lattice_probe.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print(f"entries={len(entries)} extract={extract_s:.2f}s digest={digest}")
    print("by_kind:", dict(by_kind))
    print("top families:", dict(by_family.most_common(10)))
    print("\n== SAMPLE FOR HAND AUDIT ==")
    for i, e in enumerate(sample):
        src = e["source_document"].replace("_User_Manual_EN.pdf", "").replace(".pdf", "")[:14]
        attr = e.get("attribute_name") or e["family"]
        print(f"[{i:03d}] {e['kind']:12s} {e['subject']:8s} {attr[:30]:30s} | {e['value'][:18]:18s} "
              f"| p{e['page']} {src} :: {e['quote'][:130]}")
    print(f"\nsaved: {OUT_DIR / 'lattice_probe.json'}")


if __name__ == "__main__":
    main()
