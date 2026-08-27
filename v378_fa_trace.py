"""V3.78 False-Answer acceptance-path audit (prediction-side only).

Replays the real chain over all 69 aligned cases and produces a full runtime
trace with an acceptance-branch index for EVERY case:

    FA cohort          gold ABSTAIN  -> runtime ANSWER   (9 expected)
    POSITIVE cohort    gold ANSWER   -> runtime ANSWER correct (20 expected)
    SAFE_ABSTAIN       gold ABSTAIN  -> runtime ABSTAIN correct

Branch index comes from ``traced_analyze`` - a faithful instrumented replica of
the frozen ``backend.retrieval.evidence.analyze_retrieval_evidence`` decision
procedure that additionally records WHICH accept/refuse branch fired. Validity
is machine-enforced: the replica must reproduce the real runtime's
decision AND reason exactly for all 69/69 cases or the audit aborts
(BRANCH_MIRROR_DIVERGENCE). No runtime file is modified.

All output is private under ``results/v378_audit/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core  # noqa: E402
rag_core.PERSIST_DIR = str(_REPO_ROOT / "vector_db_v369")

from langchain_chroma import Chroma  # noqa: E402
from backend.evaluation.score_lineage import build_retrieval_result  # noqa: E402
from backend.retrieval import evidence as ev  # noqa: E402
from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult  # noqa: E402
from backend.retrieval.filters import QueryAnalysis, analyze_query  # noqa: E402
from backend.retrieval.evidence_contract import evaluate_evidence_contract  # noqa: E402
from backend.retrieval.product_identity import (  # noqa: E402
    IdentityRelation,
    ProductIdentity,
    identities_from_documents,
    identity_from_metadata,
    identity_is_compatible,
    identity_relation,
    normalize_identity_text,
)
from backend.retrieval.technical import (  # noqa: E402
    contains_parameter_identifier,
    extract_parameter_references,
    foreign_equipment_signal,
)

ALIGN_DIR = _REPO_ROOT / "backend" / "evaluation" / "benchmark_private" / "v377_alignment"
OUT_DIR = _REPO_ROOT / "results" / "v378_audit"

BRANCH_NAMES = {
    1: "UNKNOWN_IDENTIFIER_EARLY",
    2: "MODEL_MISMATCH_KNOWN_IDENTITY",
    3: "NO_CANDIDATE",
    4: "MODEL_MISMATCH_COMPAT",
    5: "UNKNOWN_IDENTIFIER_LATE",
    6: "CROSS_EQUIPMENT",
    7: "UNKNOWN_PARAMETER",
    8: "UNSUPPORTED_PROCEDURE",
    9: "CONTRACT_INSUFFICIENT_ABSTAIN",
    10: "DETAIL_UNSUPPORTED_ABSTAIN",
    11: "CONTRACT_SUFFICIENT_ACCEPT",        # <- hypothesis H1 target
    12: "EXACT_IDENTIFIER_ACCEPT",
    13: "VECTOR_GATE_ACCEPT",
    14: "LEXICAL_PATH_ACCEPT_OR_ABSTAIN",
    15: "WEAK_EVIDENCE_ABSTAIN",
    16: "FINAL_FALLBACK_ABSTAIN",
}


def _mirror_values(vector_distance):
    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol
    return close


def traced_analyze(query, result, documents, retrieval_mode, *, policy=None, identity_matching=True):
    """Instrumented replica of evidence.analyze_retrieval_evidence (frozen logic).

    Returns ``(evidence, trace_dict)`` where trace_dict carries the raw internal
    predicates and the index/name of the branch that decided the case.
    Transcribed verbatim from backend/retrieval/evidence.py (HEAD 3616b38).
    """
    policy = policy or ev.default_policy()
    candidates = list(getattr(result, "candidates", []) or [])
    analysis_factory = ev._evidence_analysis if identity_matching else ev._legacy_evidence_analysis
    analysis = analysis_factory(query, documents, getattr(result, "query_analysis", None))
    identifiers = ev._document_identifiers(documents)
    top = candidates[0] if candidates else None
    top_metadata = top.metadata if top else {}
    query_identity = ProductIdentity(
        manufacturer=analysis.manufacturer,
        product_family=analysis.product_family,
        product_series=analysis.product_series,
        equipment_model=analysis.equipment_model,
        aliases=((analysis.equipment_model,) if analysis.equipment_model else ()),
    )
    candidate_identity = identity_from_metadata(top_metadata)
    candidate_identities = [identity_from_metadata(candidate.metadata) for candidate in candidates]
    explicit_corpus_identities: dict[tuple[str, str, str, str], ProductIdentity] = {}
    normalized_query = normalize_identity_text(query)
    query_tokens = {token for token in __import__("re").findall(r"[a-z0-9]+", normalized_query) if len(token) >= 3 and token not in ev._IDENTITY_GENERIC_WORDS}
    query_numbers = set(__import__("re").findall(r"\d+", normalized_query))
    for document in documents:
        identity = identity_from_metadata(getattr(document, "metadata", {}) or {})
        terms = tuple(filter(None, (
            identity.equipment_model, identity.product_series, identity.product_family, *identity.aliases,
        )))
        normalized_terms = {
            variant
            for term in terms
            for variant in (
                normalize_identity_text(term),
                __import__("re").sub(r"(?:-|\s*)series$", "", normalize_identity_text(term)),
            )
            if len(variant) >= 3
            and (any(character.isdigit() for character in variant)
                 or __import__("re").search(r"^[a-z][a-z0-9]*[A-Z]", term) is not None)
        }
        if any(
            __import__("re").search(rf"(?<![a-z0-9]){__import__('re').escape(term)}(?![a-z0-9])", normalized_query)
            or any(
                term.startswith(token)
                and len(token) >= max(3, len(term) // 2)
                and (not (term_numbers := set(__import__("re").findall(r"\d+", term))) or query_numbers <= term_numbers)
                for token in query_tokens
            )
            for term in normalized_terms
        ):
            key = tuple(normalize_identity_text(getattr(identity, field)) for field in (
                "manufacturer", "product_family", "product_series", "equipment_model"
            ))
            explicit_corpus_identities[key] = identity
    analyzed_identities = analysis.product_identities or (query_identity,)
    explicit_identities = tuple(explicit_corpus_identities.values())
    analysis_matches_explicit = any(
        identity_is_compatible(analyzed, explicit)
        for analyzed in analyzed_identities
        for explicit in explicit_identities
    )
    query_identities = analyzed_identities if analysis_matches_explicit or not explicit_identities else explicit_identities
    has_query_identity = any(
        identity.product_family or identity.product_series or identity.equipment_model
        for identity in query_identities
    )
    relation = IdentityRelation.UNKNOWN
    compatible_identity = True
    known_identity = True
    corpus_identities = ()
    if identity_matching:
        relations = {
            identity_relation(identity, candidate_item)
            for identity in query_identities
            for candidate_item in candidate_identities
        }
        relation = next(
            (
                value for value in (
                    IdentityRelation.EXACT_MODEL,
                    IdentityRelation.SAME_SERIES,
                    IdentityRelation.SAME_FAMILY,
                    IdentityRelation.MISMATCH,
                )
                if value in relations
            ),
            IdentityRelation.UNKNOWN,
        )
        compatible_identity = any(
            identity_is_compatible(identity, candidate_item)
            for identity in query_identities
            for candidate_item in candidate_identities
        )
        corpus_identities = identities_from_documents(documents)
        known_identity = all(
            any(identity_is_compatible(query_item, candidate_item) for candidate_item in corpus_identities)
            for query_item in query_identities
        ) if has_query_identity else True
    exact_identifier = bool(
        analysis.error_code
        and any(
            str(candidate.metadata.get("error_code", "")).casefold() == analysis.error_code.casefold()
            or ev.identifier_supported(
                analysis.error_code,
                ev.normalize_technical_text(str(getattr(candidate.document, "page_content", "") or "")),
            )
            for candidate in candidates
        )
    )
    exact_model = relation == IdentityRelation.EXACT_MODEL
    lexical_scores = ev._candidate_values(candidates, "lexical_score")
    vector_distances = ev._candidate_values(candidates, "vector_score")
    vector_distance = float(top.vector_score) if top and top.vector_score is not None else None
    lexical_score = float(top.lexical_score) if top and top.lexical_score is not None else None
    unsupported_detail = ev._detail_request_lacks_support(query, candidates)
    foreign_equipment = foreign_equipment_signal(query, documents)
    unknown_parameter = ev._unknown_parameter(query, candidates, documents, analysis)
    contract = evaluate_evidence_contract(query, candidates, documents, analysis)
    contract_reason_map = {
        "IDENTIFIER": ev.DecisionReason.IDENTIFIER_NOT_IN_EVIDENCE,
        "PROTOCOL": ev.DecisionReason.PROTOCOL_MISMATCH,
        "ATTRIBUTE": ev.DecisionReason.MISSING_ATTRIBUTE_EVIDENCE,
        "ACTION": ev.DecisionReason.MISSING_ACTION_EVIDENCE,
        "VALUE": ev.DecisionReason.MISSING_VALUE_EVIDENCE,
        "UNIT": ev.DecisionReason.MISSING_VALUE_EVIDENCE,
        "VALUE_KIND": ev.DecisionReason.MISSING_VALUE_EVIDENCE,
        "POLARITY": ev.DecisionReason.MISSING_VALUE_EVIDENCE,
        "REQUIREMENT_TYPE": ev.DecisionReason.MISSING_REQUIREMENT_EVIDENCE,
    }

    trace = {
        "has_candidates": bool(candidates),
        "n_candidates": len(candidates),
        "error_code": analysis.error_code,
        "exact_identifier": exact_identifier,
        "has_query_identity": has_query_identity,
        "known_identity": bool(known_identity),
        "compatible_identity": bool(compatible_identity),
        "identity_relation": relation.value,
        "corpus_identities": sorted({(i.product_series or i.equipment_model or "") for i in corpus_identities}),
        "foreign_equipment_signal": None if foreign_equipment is None else str(foreign_equipment)[:60],
        "unknown_parameter": unknown_parameter,
        "unsupported_detail": unsupported_detail,
        "contract_has_critical_requirements": bool(contract.has_critical_requirements),
        "contract_sufficient": bool(contract.sufficient),
        "contract_missing": list(contract.missing)[:6],
        "vector_distance_top1": vector_distance,
        "vector_distances_all": vector_distances,
        "lexical_score_top1": lexical_score,
        "threshold": policy.max_vector_distance,
        "vector_quality_admissible": (
            None if vector_distance is None else bool(vector_distance <= policy.max_vector_distance)
        ),
        "branch": None,
        "branch_name": None,
    }

    def finish(branch: int, decision, reason):
        trace["branch"] = branch
        trace["branch_name"] = BRANCH_NAMES[branch]
        return (
            ev.RetrievalEvidence(
                has_candidates=bool(candidates),
                exact_identifier_match=exact_identifier,
                exact_model_match=exact_model,
                lexical_score=lexical_score,
                lexical_margin=ev._margin(lexical_scores, higher_is_better=True),
                vector_distance=vector_distance,
                vector_margin=ev._margin(vector_distances, higher_is_better=False),
                top1_top2_margin=ev._margin(vector_distances, higher_is_better=False)
                or ev._margin(lexical_scores, higher_is_better=True),
                metadata_consistency=bool(
                    (not analysis.error_code or exact_identifier)
                    and (not has_query_identity or compatible_identity)
                ),
                retrieval_mode=retrieval_mode,
                effective_mode=retrieval_mode,
                decision=decision.value,
                reason=reason.value,
                query_identity=(
                    {"identities": [identity.as_dict() for identity in query_identities]}
                    if len(query_identities) > 1 else query_identity.as_dict()
                ),
                candidate_identity=candidate_identity.as_dict(),
                identity_relation=relation.value,
                contract=contract.as_dict(),
            ),
            trace,
        )

    # --- decision ladder (verbatim order from evidence.py) ------------------
    if analysis.error_code and not exact_identifier:
        return finish(1, ev.Decision.ABSTAIN, ev.DecisionReason.UNKNOWN_IDENTIFIER)
    if has_query_identity and not known_identity:
        return finish(2, ev.Decision.ABSTAIN, ev.DecisionReason.MODEL_MISMATCH)
    if not candidates:
        return finish(3, ev.Decision.ABSTAIN, ev.DecisionReason.NO_CANDIDATE)
    if has_query_identity and not compatible_identity:
        return finish(4, ev.Decision.ABSTAIN, ev.DecisionReason.MODEL_MISMATCH)
    if analysis.error_code and not exact_identifier:
        return finish(5, ev.Decision.ABSTAIN, ev.DecisionReason.UNKNOWN_IDENTIFIER)
    if foreign_equipment is not None:
        return finish(6, ev.Decision.ABSTAIN, ev.DecisionReason.CROSS_EQUIPMENT)
    if unknown_parameter is not None:
        return finish(7, ev.Decision.ABSTAIN, ev.DecisionReason.UNKNOWN_PARAMETER)
    if ev._security_bypass_signal(query):
        return finish(8, ev.Decision.ABSTAIN, ev.DecisionReason.UNSUPPORTED_PROCEDURE)
    if contract.has_critical_requirements and not contract.sufficient:
        missing_kind = contract.missing[0].split(":", 1)[0].upper() if contract.missing else ""
        return finish(9, ev.Decision.ABSTAIN, contract_reason_map.get(missing_kind, ev.DecisionReason.PARTIAL_EVIDENCE_ONLY))
    if unsupported_detail:
        return finish(10, ev.Decision.ABSTAIN, ev.DecisionReason.INSUFFICIENT_EVIDENCE)
    if contract.has_critical_requirements:
        # H1 TARGET: structural sufficiency accepted without quality gate.
        if exact_identifier:
            return finish(11, ev.Decision.ANSWER, ev.DecisionReason.EXACT_IDENTIFIER_EVIDENCE)
        if identity_matching and relation == IdentityRelation.EXACT_MODEL:
            return finish(11, ev.Decision.ANSWER, ev.DecisionReason.EXACT_MODEL_EVIDENCE)
        if identity_matching and relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            return finish(11, ev.Decision.ANSWER, ev.DecisionReason.FAMILY_COMPATIBLE_EVIDENCE)
        return finish(11, ev.Decision.ANSWER, ev.DecisionReason.CONTRACT_REQUIREMENTS_COVERED)
    if exact_identifier:
        return finish(12, ev.Decision.ANSWER, ev.DecisionReason.EXACT_IDENTIFIER_EVIDENCE)
    if vector_distance is not None and vector_distance <= policy.max_vector_distance:
        if identity_matching and relation == IdentityRelation.EXACT_MODEL:
            return finish(13, ev.Decision.ANSWER, ev.DecisionReason.EXACT_MODEL_EVIDENCE)
        if identity_matching and relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            return finish(13, ev.Decision.ANSWER, ev.DecisionReason.FAMILY_COMPATIBLE_EVIDENCE)
        return finish(13, ev.Decision.ANSWER, ev.DecisionReason.STRONG_VECTOR_EVIDENCE)
    if lexical_score is not None and compatible_identity:
        if not identity_matching:
            return finish(14, ev.Decision.ANSWER, ev.DecisionReason.STRONG_LEXICAL_EVIDENCE)
        if relation == IdentityRelation.EXACT_MODEL:
            return finish(14, ev.Decision.ANSWER, ev.DecisionReason.EXACT_MODEL_EVIDENCE)
        if relation in {IdentityRelation.SAME_SERIES, IdentityRelation.SAME_FAMILY}:
            return finish(14, ev.Decision.ANSWER, ev.DecisionReason.FAMILY_COMPATIBLE_EVIDENCE)
        return finish(14, ev.Decision.ABSTAIN, ev.DecisionReason.INSUFFICIENT_EVIDENCE)
    if lexical_score is not None and vector_distance is not None:
        return finish(15, ev.Decision.ABSTAIN, ev.DecisionReason.WEAK_RETRIEVAL_EVIDENCE)
    return finish(16, ev.Decision.ABSTAIN, ev.DecisionReason.INSUFFICIENT_EVIDENCE)


def main() -> None:
    payload = json.load(open(ALIGN_DIR / "aligned_benchmark_v2.json", encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) == 69

    search_db = Chroma(persist_directory=rag_core.PERSIST_DIR, embedding_function=rag_core.get_embedding_model())
    rows = []
    divergences = []
    for case in cases:
        qid = case["query_id"]
        text = case["query_text"]
        scored_docs = search_db.similarity_search_with_score(text, k=4)
        rr = build_retrieval_result(scored_docs)
        documents = [d for d, _s in scored_docs]

        real_ev = ev.analyze_retrieval_evidence(
            text, rr, documents, retrieval_mode="vector_only_v369", identity_matching=True,
        )
        mirrored_ev, trace = traced_analyze(
            text, rr, documents, retrieval_mode="vector_only_v369", identity_matching=True,
        )
        if (real_ev.decision, real_ev.reason) != (mirrored_ev.decision, mirrored_ev.reason):
            divergences.append({
                "query_id": qid,
                "runtime": [real_ev.decision, real_ev.reason],
                "mirror": [mirrored_ev.decision, mirrored_ev.reason],
            })
            continue

        expected = case["expected_decision"]
        support_slice = "OOD" if case["query_domain"] == "GENERIC_OUT_OF_DOMAIN" else (
            "HARD_NEGATIVE" if case["slice_labels"][0] == "HARD_NEGATIVE"
            else "CORPUS_UNSUPPORTED" if case["support_state"] == "CORPUS_UNSUPPORTED" else "ANSWERABLE"
        )
        answered = real_ev.decision == "ANSWER"
        cohort = (
            "FA" if (expected == "ABSTAIN" and answered)
            else "POSITIVE_ANSWERED" if (expected == "ANSWER" and answered)
            else "SAFE_ABSTAIN" if expected == "ABSTAIN"
            else "FALSE_REFUSAL"
        )
        rows.append({
            "query_id": qid,
            "benchmark_slice": case["slice_labels"][0],
            "support_slice": support_slice,
            "expected_decision": expected,
            "support_state": case["support_state"],
            "cohort": cohort,
            "decision": real_ev.decision,
            "reason": real_ev.reason,
            **trace,
        })

    if divergences:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(divergences, open(OUT_DIR / "MIRROR_DIVERGENCES.json", "w"), indent=1)
        print(f"BRANCH_MIRROR_DIVERGENCE: {len(divergences)} cases")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "full_traces.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1, ensure_ascii=False, default=str)

    fa_rows = [r for r in rows if r["cohort"] == "FA"]
    pos_rows = [r for r in rows if r["cohort"] == "POSITIVE_ANSWERED"]
    safe_rows = [r for r in rows if r["cohort"] == "SAFE_ABSTAIN"]
    fr_rows = [r for r in rows if r["cohort"] == "FALSE_REFUSAL"]

    def dist_stats(subset):
        vals = sorted(r["vector_distance_top1"] for r in subset if r["vector_distance_top1"] is not None)
        if not vals:
            return {"n": 0}
        def pct(p):
            idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
            return round(vals[idx], 4)
        return {"n": len(vals), "min": round(vals[0], 4), "p10": pct(0.10), "median": pct(0.5),
                "p90": pct(0.90), "max": round(vals[-1], 4)}

    print(f"\n== MIRROR VALIDATION: 69/69 decision+reason exact match ==")
    print(f"cohorts: FA={len(fa_rows)} POSITIVE={len(pos_rows)} SAFE_ABSTAIN={len(safe_rows)} FR={len(fr_rows)}")

    print("\n== FA BRANCH DISTRIBUTION ==")
    for r in fa_rows:
        print(f"  {r['query_id']} [{r['benchmark_slice']}/{r['support_slice']}] branch#{r['branch']} "
              f"{r['branch_name']} reason={r['reason']} vdist={round(r['vector_distance_top1'],4) if r['vector_distance_top1'] is not None else None} "
              f"vq_adm={r['vector_quality_admissible']}")
    print("  counts:", dict(Counter(f"{r['branch_name']}" for r in fa_rows)))

    print("\n== ALL ANSWER ROWS BY BRANCH ==")
    answer_rows = [r for r in rows if r["decision"] == "ANSWER"]
    for name, n in sorted(Counter(f"#{r['branch']} {r['branch_name']}" for r in answer_rows).items()):
        print(f"  {name}: {n}")

    print("\n== DISTANCE STATS ==")
    print(f"  FA (gold abstain/answered):      {dist_stats(fa_rows)}")
    print(f"  POSITIVE (gold answer/answered): {dist_stats(pos_rows)}")
    print(f"  SAFE_ABSTAIN:                    {dist_stats(safe_rows)}")

    print("\n== PREDICATE VECTORS: FA cohort ==")
    fields = ("error_code", "exact_identifier", "has_query_identity", "known_identity",
              "compatible_identity", "identity_relation", "contract_has_critical_requirements",
              "contract_sufficient", "vector_quality_admissible", "unsupported_detail",
              "unknown_parameter")
    for r in fa_rows:
        vec = {k: r[k] for k in fields}
        print(f"  {r['query_id']}: {json.dumps(vec, default=str)}")

    print("\n== CONTRACT-BRANCH CENSUS (branch#11 across all rows) ==")
    b11 = [r for r in rows if r["branch"] == 11]
    for r in b11:
        print(f"  {r['query_id']} [{r['cohort']}] vdist={round(r['vector_distance_top1'],4) if r['vector_distance_top1'] is not None else None} "
              f"vq_adm={r['vector_quality_admissible']} reason={r['reason']}")
    b11_fa_over_thr = sum(1 for r in b11 if r["cohort"] == "FA" and r["vector_quality_admissible"] is False)
    b11_pos_over_thr = sum(1 for r in b11 if r["cohort"] == "POSITIVE_ANSWERED" and r["vector_quality_admissible"] is False)
    print(f"  branch#11 total={len(b11)} | FA@>threshold={b11_fa_over_thr} | POSITIVE@>threshold={b11_pos_over_thr}")

    print(f"\nsaved: {OUT_DIR / 'full_traces.json'}")


if __name__ == "__main__":
    main()
