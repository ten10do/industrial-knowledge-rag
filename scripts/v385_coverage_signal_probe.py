#!/usr/bin/env python3
"""V3.85 coverage signal probe: attribute-word coverage discriminability.

The ANSWERABLE queries "What is the acceleration time setting?" (Q0001) and the
CORPUS_UNSUPPORTED false answers "What is the output frequency?" (Q0012) are
syntactically isomorphic; the only difference is corpus coverage. This probe
measures whether a build-time, deterministic signal separates them:

    coverage(attribute) = number of corpus chunks whose text mentions the
                          attribute's content words (excluding stopwords)

It scans the frozen 69-case benchmark, deduplicates, extracts the requested
attribute phrase for "What is the X?"-style queries, and reports the coverage
distribution per slice. This is the evidence basis for the parameter/attribute
dimension of the coverage gate. No runtime decision logic is changed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "corpus"
BENCH = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "aligned_benchmark_v2.json"
CATALOG = CORPUS_DIR / "chunk_catalog_v385_fulltext.json"

DUPLICATE_IDS = {
    "V369-Q0027", "V369-Q0028", "V369-Q0029",
    "V369-Q0030", "V369-Q0031", "V369-Q0032",
}

STOPWORDS = frozenset(
    "a an the of in for to is are does do what which when where how under on with by that this and or can could may will should must be setting settings value values range time method mode type level capacity frequency class rating current protection safety precaution purpose function basics protocol circuit".split()
)

WHAT_IS_RE = re.compile(
    r"^\s*(?:what|how)\s+(?:is|are|does|do)\s+(?:the\s+)?(?P<attr>[a-z][a-z0-9 /.-]*?)\s*\??$",
    re.IGNORECASE,
)


def attribute_words(query: str) -> list[str]:
    match = WHAT_IS_RE.match(query)
    if not match:
        return []
    phrase = match.group("attr")
    words = [
        w for w in re.findall(r"[a-z0-9]+", phrase.lower())
        if w not in STOPWORDS and len(w) >= 3
    ]
    return words


def main() -> int:
    catalog = json.load(open(CATALOG, encoding="utf-8"))
    bench = json.load(open(BENCH, encoding="utf-8"))

    # build a per-chunk bag of words once
    chunk_texts = [c.get("text", "") or "" for c in catalog["chunks"]]
    print(f"chunks scanned: {len(chunk_texts)}")

    results = []
    for case in bench["cases"]:
        qid = case["query_id"]
        if qid in DUPLICATE_IDS:
            continue
        words = attribute_words(case["query_text"])
        if not words:
            continue
        hits = 0
        any_word_chunks = 0
        for text in chunk_texts:
            low = text.lower()
            if all(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low) for w in words):
                hits += 1
            if any(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low) for w in words):
                any_word_chunks += 1
        results.append({
            "query_id": qid,
            "slice": case["slice_labels"][0],
            "attr": " ".join(words),
            "all_words_chunks": hits,
            "any_word_chunks": any_word_chunks,
        })

    print(f"\n{'query':<10} {'slice':<20} {'attr':<30} {'allWordsChunks':>14} {'anyWordChunks':>14}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["query_id"]):
        print(f"{r['query_id']:<10} {r['slice']:<20} {r['attr']:<30} {r['all_words_chunks']:>14} {r['any_word_chunks']:>14}")

    # separation stats
    ans = [r for r in results if r["slice"] == "ANSWERABLE"]
    uns = [r for r in results if r["slice"] == "CORPUS_UNSUPPORTED"]
    for name, group in (("ANSWERABLE", ans), ("CORPUS_UNSUPPORTED", uns)):
        if not group:
            continue
        vals = sorted(r["all_words_chunks"] for r in group)
        print(f"\n{name}: n={len(group)} allWordsChunks min={vals[0]} median={vals[len(vals)//2]} max={vals[-1]}")
        print("  " + ", ".join(f"{r['query_id']}:{r['all_words_chunks']}" for r in sorted(group, key=lambda x: x['all_words_chunks'])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
