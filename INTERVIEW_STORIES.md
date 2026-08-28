# Interview Stories

These stories are phrased as evidence-backed talking points. They should be adapted to the speaker's real role and never presented as team or production impact beyond what the repository proves.

## Story 1 — the evaluator was the bug

**Situation:** A benchmark reported 21 false answers, suggesting a serious Evidence-runtime regression.

**Task:** Determine whether to tune the safety policy without making the system less useful.

**Action:** Traced structured fields from runtime contract through the evaluation adapter. Found the adapter read a non-existent requirements_covered key and omitted covered claim keys. Added fidelity checks and reran the same authority contract.

**Result:** 19 of 21 apparent false answers were measurement defects. The adapter was corrected without changing Evidence semantics. The lesson: establish measurement fidelity before optimizing the measured system.

## Story 2 — refuse to tune an unwired score

**Situation:** Remaining false refusals appeared to cluster around a vector-distance threshold.

**Task:** Decide whether threshold calibration could safely recover recall.

**Action:** Audited the complete score lineage and discovered the benchmark harness never populated vector_score, while production retrieval did. Built a dedicated adapter that preserved raw Chroma distance and ranking, with None-safe tests.

**Result:** Score fidelity became 69/69 and five refusals resolved as an honest propagation side effect, with zero new false answers and no threshold change. The project avoided hiding a wiring defect with a magic number.

## Story 3 — rebuild invalid gold without moving the queries

**Situation:** Several “false refusals” asked about models or specifications absent from the three-document corpus.

**Task:** Restore benchmark validity without leaking predictions into labeling.

**Action:** Kept all 69 query texts byte-stable, froze corpus and query hashes, audited support from source documents prediction-blind, and rebuilt structured expectations.

**Result:** Eighteen unsupported ANSWER labels became ABSTAIN. The aligned benchmark now contains 26 supported-answer and 43 abstainable cases with a reproducible digest.

## Story 4 — a NO-GO is an engineering result

**Situation:** False answers remained, and simple distance vetoes, claim-support logic, local NLI and query extraction all looked promising.

**Task:** Improve safety without collapsing answerable recall.

**Action:** Defined stop gates before implementation, shadow-replayed each candidate, and measured both blocked false answers and killed correct answers.

**Result:** Candidates failed precision or utility gates—one veto would kill at least 13 correct answers; local NLI had zero useful recall; a requirement extractor created unsafe supports. The runtime remained unchanged and the negative evidence was documented.

## Story 5 — table understanding needed a different representation

**Situation:** Industrial specifications often live in borderless tables, but retrieved flat text could not reliably bind a value to its row and column.

**Task:** Decide whether Evidence could infer ownership from proximity or table-region context.

**Action:** Ran synthetic feasibility, real-manual producer probes, merged-cell and perimeter analyses, region metadata integration and independent generalization.

**Result:** Synthetic precision did not generalize; flat extraction lost decisive column semantics. The project froze cell ownership authority at NONE and identified layout-aware ingestion as a separate architecture problem.

## Story 6 — close the container gate honestly

**Situation:** Native-process lifecycle tests passed, but the formal Docker Compose topology had not run because the daemon was unavailable.

**Task:** Avoid calling the RC ready until the deployable artifact itself was validated.

**Action:** Kept the phase partial, then resumed when Docker became available. Audited image contents, fixed only the packaging exclusion, rebuilt, ran non-root/readiness/privacy checks, restart and shutdown 3/3, persistence, force recreation and load profiles.

**Result:** All internal container gates passed. Optional provider generation and remote CI remained explicitly pending instead of being inferred from local success.

## Concise project pitch

“I built an industrial RAG system where retrieval is not treated as proof. The central design is a deterministic Evidence contract that can refuse before a model call. I also built the evaluation and release discipline around it: a structured evaluator, frozen corpus/gold hashes, score-lineage checks, explicit NO-GO research decisions, 948 backend regressions, and real Docker lifecycle/persistence/privacy validation. The current 9 false answers and unsupported table-cell ownership are documented, not hidden.”
