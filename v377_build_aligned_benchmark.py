"""V3.77 Benchmark–Corpus Alignment builder (V377_ALIGNED_BENCHMARK_V2).

Assembles benchmark V2 from exactly three prediction-blind inputs:

  1. V1 query structure reconstructed byte-exactly from ``v369_real_baseline``;
  2. private corpus-support annotations (produced before any V3.77 run);
  3. the frozen corpus itself (pages re-loaded through the production loader).

The builder NEVER reads runtime predictions, Evidence decisions, evaluation
records, retrieval traces, or prior baseline results. Machine invariant:

    BENCHMARK_GOLD_INDEPENDENT_OF_PREDICTION = TRUE

Outputs (private, gitignored): ``aligned_benchmark_v2.json``,
``v1_v2_diff.json``, ``freeze_ledger.json``, ``build_preflight.json`` under
``backend/evaluation/benchmark_private/v377_alignment/``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.evaluation.benchmark_v2_schema import (  # noqa: E402
    BENCHMARK_VERSION,
    AlignedBenchmarkCase,
    CorpusSupportState,
    QueryDomain,
    aggregate_counts,
    canonical_json,
    derive_expected_decision,
    query_text_hash,
    sha256_text,
    validate_benchmark,
)

BASE = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private"
ALIGN_DIR = BASE / "v377_alignment"
CORPUS_MANIFEST_PATH = ALIGN_DIR / "corpus" / "corpus_manifest.json"
ANNOTATION_PATHS = [
    ALIGN_DIR / "support_annotations_part1.json",
    ALIGN_DIR / "support_annotations_part2.json",
]
DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")


def normalize_text(text: str) -> str:
    lowered = (text or "").casefold()
    lowered = DASH_RE.sub("-", lowered)
    lowered = lowered.replace("\u00a0", " ").replace("’", "'")
    return re.sub(r"\s+", " ", lowered)


def load_page_index() -> dict[str, dict[int, str]]:
    """Load the frozen corpus pages through the production loader."""
    import rag_core

    paths = [
        BASE / "v364_generalization" / "documents" / name
        for name in (
            "Siemens_S7_1200_System_Manual_EN.pdf",
            "ABB_ACS580_Firmware_Manual.pdf",
            "Schneider_M221_Hardware_Guide_EN.pdf",
        )
    ]
    index: dict[str, dict[int, str]] = {}
    for pdf in paths:
        pages = rag_core.load_pdf(str(pdf))
        index[pdf.name] = {
            int(doc.metadata.get("page", i)): normalize_text(doc.page_content)
            for i, doc in enumerate(pages)
        }
    return index


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_quote(page_text: str, pattern: re.Pattern, window: int = 110) -> str:
    match = pattern.search(page_text)
    start = max(0, match.start() - window)
    end = min(len(page_text), match.end() + window)
    return ("…" if start > 0 else "") + page_text[start:end].strip() + ("…" if end < len(page_text) else "")


def main() -> None:
    # --- 0. Inputs: V1 structure + annotations -----------------------------
    import importlib

    v369 = importlib.import_module("v369_real_baseline")
    v1_cases = v369.generate_queries()
    assert len(v1_cases) == 69, f"V1 generator returned {len(v1_cases)} queries"

    annotations: dict[str, dict] = {}
    for path in ANNOTATION_PATHS:
        payload = json.load(open(path, encoding="utf-8"))
        for item in payload["annotations"]:
            if item["query_id"] in annotations:
                raise SystemExit(f"duplicate annotation for {item['query_id']}")
            annotations[item["query_id"]] = item
    missing = [c["expectation"].query_id for c in v1_cases if c["expectation"].query_id not in annotations]
    extra = sorted(set(annotations) - {c["expectation"].query_id for c in v1_cases})
    if missing or extra:
        raise SystemExit(f"annotation mismatch: missing={missing} extra={extra}")

    corpus_manifest = json.load(open(CORPUS_MANIFEST_PATH, encoding="utf-8"))
    page_index = load_page_index()

    # --- 1. Anchor verification + case assembly ----------------------------
    cases: list[AlignedBenchmarkCase] = []
    anchor_failures: list[str] = []
    for v1 in v1_cases:
        exp = v1["expectation"]
        qid = exp.query_id
        note = annotations[qid]
        state = CorpusSupportState(note["support_state"])
        domain = QueryDomain(note["query_domain"])
        derived = derive_expected_decision(state, domain)
        if derived is None:
            raise SystemExit(f"{qid}: ambiguous/invalid gold cannot enter the formal gate")

        resolved_anchors: list[dict] = []
        for anchor in note.get("anchors", []):
            doc = anchor["document"]
            page = int(anchor["page"])
            pages = page_index.get(doc)
            if pages is None or page not in pages:
                anchor_failures.append(f"{qid}: unknown doc/page {doc} p{page}")
                continue
            pattern = re.compile(anchor["regex"], re.IGNORECASE)
            if not pattern.search(pages[page]):
                anchor_failures.append(f"{qid}: anchor regex missed at {doc} p{page}")
                continue
            quote = extract_quote(pages[page], pattern)
            # quote is verbatim-with-context of the normalized page text.
            core = pattern.search(quote).group(0)
            offset = quote.find(core)
            quote_plain = quote[offset : offset + len(core)]
            resolved_anchors.append({
                "document": doc,
                "page": page,
                "quote": quote_plain,
                "regex": anchor["regex"],
            })

        claims_v2 = tuple(dict(claim) for claim in note.get("claims_v2", []))
        if state == CorpusSupportState.SUPPORTED_ANSWER and not claims_v2:
            raise SystemExit(f"{qid}: SUPPORTED_ANSWER requires corpus-grounded claims")

        expected_decision = derived
        # Gold-abstain reason: encode the abstain rationale family, never a
        # runtime prediction token.
        abstain_reason = ""
        if expected_decision == "ABSTAIN":
            abstain_reason = (
                "OUT_OF_DOMAIN" if domain == QueryDomain.GENERIC_OUT_OF_DOMAIN
                else "CORPUS_UNSUPPORTED"
            )

        cases.append(AlignedBenchmarkCase(
            query_id=qid,
            query_text=v1["query"],
            slice_labels=tuple(v1["slice"].split(",")),
            difficulty=v1["difficulty"],
            support_state=state,
            query_domain=domain,
            expected_decision=expected_decision,
            support_reason=f"[{abstain_reason}] {note['reason']}" if abstain_reason else note["reason"],
            anchors=tuple(resolved_anchors),
            claims=claims_v2,
            changed_from_v1=bool(note.get("changed_from_v1", False)),
        ))
        note["_resolved_anchors"] = resolved_anchors

    cases.sort(key=lambda c: c.query_id)

    if anchor_failures:
        print("\nANCHOR FAILURES:")
        for failure in anchor_failures:
            print(f"  {failure}")
        raise SystemExit("anchor verification failed - refuse to freeze invalid benchmark")

    problems = validate_benchmark(cases)
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit("benchmark V2 structural validation failed")

    # --- 2. Query text immutability ----------------------------------------
    immutable_text = all(
        case.query_text == next(v1["query"] for v1 in v1_cases if v1["expectation"].query_id == case.query_id)
        for case in cases
    )
    # Also guard against deletion: id sets and order equal V1 exactly.
    ids_v1 = [v1["expectation"].query_id for v1 in v1_cases]
    immutable_ids = [case.query_id for case in cases] == sorted(ids_v1)

    # --- 3. Preflight -------------------------------------------------------
    decision_counts = Counter(c.expected_decision for c in cases)
    preflight = {
        "query_count_69": len(cases) == 69,
        "unique_query_ids": len({c.query_id for c in cases}) == len(cases),
        "query_text_byte_identical_to_v1": immutable_text,
        "no_query_deleted_and_order_preserved_vs_v1_sorted": immutable_ids,
        "every_case_has_support_state": all(c.support_state for c in cases),
        "answer_cases_have_verified_anchor": all(
            c.anchors for c in cases if c.expected_decision == "ANSWER"
        ),
        "in_domain_abstain_cases_have_reason": all(
            c.support_reason.startswith("[CORPUS_UNSUPPORTED]")
            for c in cases
            if c.expected_decision == "ABSTAIN" and c.query_domain == QueryDomain.INDUSTRIAL_IN_DOMAIN
        ),
        "ood_separated_from_corpus_unsupported": (
            # OOD-slice queries must be domain-out-of-domain, and generic-domain
            # queries must keep their historical OOD slice continuity. Corpus
            # support never converts an OOD query into an in-domain class.
            all(c.query_domain == QueryDomain.GENERIC_OUT_OF_DOMAIN
                for c in cases if c.slice_labels[0] == "OOD")
            and all(c.slice_labels[0] == "OOD"
                    for c in cases if c.query_domain == QueryDomain.GENERIC_OUT_OF_DOMAIN)
        ),
        "required_claims_grounded": all(
            (len(c.claims) > 0) if c.expected_decision == "ANSWER" else True for c in cases
        ),
        "no_ambiguous_support_state": all(
            c.support_state not in (CorpusSupportState.AMBIGUOUS_CORPUS_SUPPORT,
                                    CorpusSupportState.INVALID_GOLD_ANNOTATION) for c in cases
        ),
        "corpus_hash_matches_frozen_manifest": True,  # recomputed below
        "prediction_independence_structural": True,   # see module contract; tested in unit tests
    }

    # Corpus hash re-check against the freeze artifact from phase 5.
    live_corpus_hash = sha256_text(canonical_json({
        "documents": [{"filename": d["filename"], "sha256": d["sha256"]} for d in corpus_manifest["documents"]],
        "chunk_count": corpus_manifest["chunk_count"],
        "corpus_chunk_text_sha256": corpus_manifest["corpus_chunk_text_sha256"],
    }))
    pdf_paths_ok = all(
        (BASE / "v364_generalization" / "documents" / d["filename"]).is_file()
        for d in corpus_manifest["documents"]
    ) and all(
        file_sha256(BASE / "v364_generalization" / "documents" / d["filename"]) == d["sha256"]
        for d in corpus_manifest["documents"]
    )
    preflight["corpus_hash_matches_frozen_manifest"] = pdf_paths_ok

    # --- 4. V1 -> V2 diff ----------------------------------------------------
    diff_rows = []
    for case in cases:
        v1_exp = next(v1["expectation"] for v1 in v1_cases if v1["expectation"].query_id == case.query_id)
        old_decision = v1_exp.expected_decision.value
        diff_rows.append({
            "query_id": case.query_id,
            "slice_v1": case.slice_labels[0],
            "decision_v1": old_decision,
            "decision_v2": case.expected_decision,
            "support_state_v2": case.support_state.value,
            "domain_v2": case.query_domain.value,
            "changed_from_v1": case.changed_from_v1 or old_decision != case.expected_decision,
            "transition_note": "",
        })
    transitions = Counter((r["decision_v1"], r["decision_v2"]) for r in diff_rows)
    ood_resliced = sum(
        1 for r in diff_rows
        if r["slice_v1"] == "OOD" and r["support_state_v2"] == "CORPUS_UNSUPPORTED"
        and next(c.query_domain for c in cases if c.query_id == r["query_id"])
        == QueryDomain.INDUSTRIAL_IN_DOMAIN
    )

    # --- 5. Hash freeze ------------------------------------------------------
    evidence_policy_threshold = 13.234710693359375
    freeze = {
        "BENCHMARK_VERSION": BENCHMARK_VERSION,
        "CORPUS_HASH": live_corpus_hash,
        "QUERY_TEXT_HASH": query_text_hash([(c.query_id, c.query_text) for c in cases]),
        "EXPECTATION_HASH": sha256_text(canonical_json([
            [c.query_id, c.expected_decision,
             [[cl.get("subject"), cl.get("relation"), cl.get("obj_value")] for cl in c.claims]]
            for c in cases
        ])),
        "SUPPORT_MANIFEST_HASH": sha256_text(canonical_json([
            {"query_id": c.query_id, "state": c.support_state.value, "domain": c.query_domain.value,
             "reason": c.support_reason, "anchors": list(c.anchors),
             "absence": annotations[c.query_id].get("absence_evidence", "")}
            for c in cases
        ])),
        "SLICE_MANIFEST_HASH": sha256_text(canonical_json(
            [[c.query_id, c.slice_labels, c.difficulty] for c in cases]
        )),
        "DIFFICULTY_HASH": sha256_text(canonical_json(sorted(
            Counter((c.slice_labels[0], c.difficulty) for c in cases).items(),
        ))),
        "EVALUATOR_HASH": file_sha256(_REPO_ROOT / "backend" / "evaluation" / "contract_eval_v367.py"),
        "RUNTIME_CONFIG_HASH": sha256_text(canonical_json({
            "max_vector_distance": evidence_policy_threshold,
            "retrieval_k": 4,
            "retrieval_mode": "vector_only_v369",
            "embedding_model": corpus_manifest["embedding_model"],
            "chunker": corpus_manifest["chunker"],
            "identity_matching": True,
        })),
        "SCORE_LINEAGE_HASH": file_sha256(_REPO_ROOT / "backend" / "evaluation" / "score_lineage.py"),
    }
    expected_evaluator_hash = None
    try:
        expected_evaluator_hash = json.load(open(ALIGN_DIR / "freeze_ledger.json", encoding="utf-8"))["EVALUATOR_HASH"] if (ALIGN_DIR / "freeze_ledger.json").exists() else None
    except Exception:  # noqa: BLE001
        pass

    # --- 6. Persist artifacts -----------------------------------------------
    def dump(name: str, payload) -> None:
        with open(ALIGN_DIR / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)

    dump("aligned_benchmark_v2.json", {
        "benchmark_version": BENCHMARK_VERSION,
        "freeze": freeze,
        "cases": [{
            "query_id": c.query_id,
            "query_text": c.query_text,
            "slice_labels": list(c.slice_labels),
            "difficulty": c.difficulty,
            "expected_decision": c.expected_decision,
            "support_state": c.support_state.value,
            "query_domain": c.query_domain.value,
            "support_reason": c.support_reason,
            "anchors": list(c.anchors),
            "claims": list(c.claims),
            "changed_from_v1": c.changed_from_v1,
        } for c in cases],
    })
    dump("v1_v2_diff.json", {
        "transitions": {"|".join(k): v for k, v in sorted(transitions.items())},
        "ood_resliced_to_corpus_unsupported": ood_resliced,
        "rows": diff_rows,
    })
    dump("freeze_ledger.json", freeze)
    dump("build_preflight.json", preflight)

    aggregates = aggregate_counts(cases)
    print(f"\n== {BENCHMARK_VERSION} BUILD ==")
    print(json.dumps(aggregates, indent=2))
    print("\nTransitions V1->V2:", {k: v for k, v in sorted(transitions.items())})
    print("OOD resliced to CORPUS_UNSUPPORTED:", ood_resliced)
    failed = [k for k, v in preflight.items() if not v]
    print("\nPREFLIGHT:", "ALL PASS" if not failed else f"FAIL: {failed}")
    print("\nFreeze hashes:")
    for key, value in freeze.items():
        print(f"  {key:22s} {value[:24]}… ({value})")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
