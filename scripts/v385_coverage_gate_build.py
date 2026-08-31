#!/usr/bin/env python3
"""V3.85 coverage index builder (build-time parameter/device coverage).

Consumes the full-text catalog produced by v385_coverage_index_build.py and
emits the build-time coverage index that the coverage gate consults at
decision time:

* known_models     - corpus-confirmed equipment models (from chunk identity
                     metadata) with their identity fields.
* parameters       - parameter identifiers observed in corpus chunk text,
                     grouped by model, with occurrence counts. This is a
                     build-time FACT registry: it records whether the corpus
                     states a parameter at all, and in which model scope.
* model_parameters - per-model parameter sets for the whitelist product:
                     (equipment_model, parameter) pairs.

The gate (backend/retrieval/coverage_gate.py) reads this index to turn
"out-of-corpus device / parameter" from an LLM guess into a table lookup.

Usage
-----
    python scripts/v385_coverage_gate_build.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment" / "corpus"
CATALOG = CORPUS_DIR / "chunk_catalog_v385_fulltext.json"
OUTPUT = CORPUS_DIR / "coverage_index_v385.json"

sys.path.insert(0, str(ROOT))
from backend.retrieval.technical import _PARAMETER_IDENTIFIER_PATTERN, normalize_parameter_identifier  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    catalog = json.load(open(CATALOG, encoding="utf-8"))
    chunks = catalog["chunks"]
    catalog_sha = sha256_file(CATALOG)

    known_models: dict[str, dict] = {}
    model_aliases: dict[str, list[str]] = {}
    params: dict[str, dict] = {}       # identifier -> {"chunks": n, "models": {model: n_chunks}}
    model_parameters: dict[str, set] = {}

    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        model = str(meta.get("equipment_model", "") or "").strip().casefold()
        if not model:
            model = str(chunk.get("source", "") or "").split("_")[0].casefold()
        if not model:
            continue
        if model not in known_models:
            known_models[model] = {
                "manufacturer": meta.get("manufacturer", ""),
                "product_family": meta.get("product_family", ""),
                "product_series": meta.get("product_series", ""),
                "equipment_model": meta.get("equipment_model", ""),
            }
            model_aliases[model] = []
            model_parameters[model] = set()

        text = chunk.get("text", "") or ""
        seen = set()
        for match in _PARAMETER_IDENTIFIER_PATTERN.finditer(text):
            ident = normalize_parameter_identifier(match.group("identifier"))
            if not ident:
                continue
            if ident in seen:
                continue
            seen.add(ident)
            entry = params.setdefault(ident, {"chunks": 0, "models": {}})
            entry["chunks"] += 1
            entry["models"][model] = entry["models"].get(model, 0) + 1
            model_parameters[model].add(ident)

    payload = {
        "index_version": "V385_COVERAGE_INDEX_V1",
        "catalog_sha256": catalog_sha,
        "catalog_n_chunks": len(chunks),
        "known_models": {k: known_models[k] for k in sorted(known_models)},
        "model_aliases": {k: sorted(v) for k, v in sorted(model_aliases.items())},
        "parameters": {
            k: {"chunks": v["chunks"], "models": v["models"]}
            for k, v in sorted(params.items())
        },
        "model_parameters": {k: sorted(v) for k, v in sorted(model_parameters.items())},
        "stats": {
            "n_chunks": len(chunks),
            "n_models": len(known_models),
            "n_parameters": len(params),
        },
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("=== V385 coverage index (parameters) build ===")
    print(f"catalog sha256   : {catalog_sha}")
    print(f"known models     : {sorted(known_models)}")
    print(f"n parameters     : {len(params)}")
    for model in sorted(model_parameters):
        print(f"  {model}: {len(model_parameters[model])} parameters")
    print(f"output           : {OUTPUT}")
    print(f"output sha256    : {sha256_file(OUTPUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
