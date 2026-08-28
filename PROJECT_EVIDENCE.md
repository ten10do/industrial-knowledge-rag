# Project Evidence

## Portfolio summary

Industrial Knowledge RAG is a complete engineering case study in building a safety-oriented RAG system under evidence constraints. Its strongest evidence is not a single accuracy number; it is the chain from data boundary, structured evaluation and failure attribution to reproducible deployment gates and honest stop decisions.

## Evidence map

| Claim | Artifact | Verifiable evidence |
|---|---|---|
| Evidence decisions are separate from generation | ARCHITECTURE.md; EVIDENCE_RESPONSIBILITY_BOUNDARY.md | explicit ownership contract and runtime ABSTAIN path |
| Correctness is evaluated structurally | EVALUATION.md; V3.67 report | 151-case authority suite, 0% evaluator FA |
| Benchmark labels match the corpus | V3.77 report | 69/69 query-text preservation; prediction-blind expectation rebuild |
| Current quality is disclosed | V382/V383 manifests | 54/69, 9 FA, 3 FR, HN 10/10, frozen digest |
| Retrieval score meaning is traceable | V3.75/V3.76 reports | score-lineage defect isolated and fixed 69/69 without threshold tuning |
| Unsafe research candidates were stopped | RESEARCH_DECISIONS.md | V3.78–V3.81 explicit NO-GO/PARTIAL decisions |
| Runtime is regression protected | V3.83 report | 948 passed, 2 skipped; frontend 34/34 and build |
| Release topology was actually executed | DEPLOYMENT_VALIDATION.md | image, non-root, lifecycle, persistence, load and privacy gates |
| Public behavior is reproducible | scripts/v384_public_demo.py | keyless supported decision, wrong-model and unknown-fault refusal |
| Private assets remain private | .gitignore, .dockerignore, release guard | tracked private files 0; image content audit passed |

## Engineering decisions with demonstrated value

### Challenge/action/evidence matrix

| Engineering challenge | Action | Evidence / metric | Trade-off |
|---|---|---|---|
| Hybrid retrieval across exact identifiers and paraphrases | Combined BM25 with light TF-IDF or full dense candidates through RRF while preserving source scores | retrieval regression fixtures; score lineage 69/69 | RRF ranks but never becomes a confidence probability |
| Formal Evidence correctness | Replaced prose similarity with typed decisions and required/forbidden claims | V3.67 authority suite: 0% evaluator FA | more annotation/schema work for defensible verdicts |
| Benchmark validity | Rebuilt gold prediction-blind against the frozen corpus, preserving 69/69 query texts | 26 supported-answer, 43 abstainable | historical labels changed, but source truth became authoritative |
| Score-lineage fidelity | Propagated the real vector distance at the harness boundary | five honest FR recoveries, delta FA=0, threshold unchanged | no opportunistic calibration |
| Safe abstention research | Shadow-tested vetoes and counted killed correct answers | 9 residual FA/3 FR retained; unsafe candidate stopped | accepts known residual risk instead of destroying utility |
| Research NO-GO discipline | Used preregistered precision/recall gates | spec lattice 0.61 versus 0.95; STOP | foregoes incremental rule patches |
| Docker deployment | Tested the built non-root artifact and persistent volumes | restart 3/3, shutdown 3/3, recreation/persistence PASS | single-service topology, not HA |
| Observability | Added request IDs, privacy-safe structured logs, readiness and bounded metrics | request-ID/log/privacy/metrics E2E PASS | process-local metrics require external retention |
| Failure injection/recovery | Removed/malformed required indexes and exercised optional failure states | readiness 200->503->200 recovery | provider restoration still external |
| Soak/load | Ran a native 600.12-second soak and bounded container concurrency profiles | 11,678/11,678; container 24/24 at 1/4/8 | no SLO or enterprise-scale claim |
| Privacy/security boundary | Ignored private assets, release-guarded protected files, audited image contents | 28/28 protected; tracked private files 0; non-root uid 100 | not a formal security audit |

### 1. Replace plausible-text evaluation with a contract

Problem: a legacy evaluator accepted most adversarially mutated predictions.

Action: define typed expectations and runtime records, then make decisions, required/forbidden claims, identity, relation and reason codes authoritative.

Result: the mutation false-accept rate dropped from 88% in the legacy evaluator to 0% in the V3.67 authority suite. Later work could distinguish runtime errors from evaluator errors.

### 2. Fix measurement fidelity before tuning

Problem: headline false-answer and false-refusal counts suggested Evidence thresholds were wrong.

Action: trace adapter fields and vector score lineage end-to-end.

Result: 19 of 21 apparent false answers were an adapter mapping defect; the vector distance was also unwired in one harness. Both were corrected at evaluation boundaries without changing the runtime threshold. This prevented “improvements” aimed at broken measurement.

### 3. Align benchmark gold to evidence

Problem: the benchmark expected answers that the frozen corpus did not contain.

Action: preserve all 69 query texts, audit source support prediction-blind, and rebuild expectations.

Result: 18 invalid ANSWER labels became ABSTAIN; the current 54/69 result now measures the actual closed-corpus contract.

### 4. Stop unsafe refinements

Problem: simple vetoes, claim-support signals, NLI and query-pattern extraction looked capable of blocking false answers.

Action: preregister precision/recall and regression gates; inspect killed correct answers and unsupported false supports.

Result: V3.78–V3.81 stopped without runtime changes when candidates failed. The repository preserves negative results and avoids an always-reject system masquerading as safe.

### 5. Validate the operational artifact, not only localhost

Problem: native-process tests cannot establish image contents, non-root execution, container lifecycle or named-volume persistence.

Action: run actual Docker Compose build, image audit, readiness, restart 3/3, force recreation, shutdown 3/3, bounded load and cleanup.

Result: all internal container gates passed; the image identity and evidence are frozen in V383_RELEASE_CANDIDATE_MANIFEST.json.

## What this project does not claim

- certified safety or domain-wide production accuracy;
- correct cell-level ownership for industrial tables;
- validated real-provider answer generation in the certified container;
- remote CI success;
- HA, Kubernetes or multi-region readiness;
- permission to publish private manuals or annotations.

## Review path

For a short review: README -> ARCHITECTURE -> EVALUATION -> DEPLOYMENT_VALIDATION -> DEMO_SCENARIOS.

For a research review: RESEARCH_DECISIONS -> docs/research/README.md -> V3.65, V3.71, V3.76, V3.77, V3.78, V3.79, V3.80 and V3.81 source reports.

For an operational review: DEPLOYMENT -> OPERATIONS_RUNBOOK -> CONFIGURATION_MATRIX -> V383_RELEASE_CANDIDATE_MANIFEST.json.

## Resume bullet candidates

- Built a safety-oriented industrial RAG platform that separates hybrid retrieval from a deterministic ANSWER/ABSTAIN Evidence contract, with model/identifier scoping, versioned knowledge bases and optional cited generation.
- Rebuilt evaluation around a contract-native 151-case authority suite with 0% evaluator false accepts, then repaired benchmark/corpus alignment while preserving all 69 query texts and freezing reproducibility hashes.
- Diagnosed two evaluation-infrastructure defects—adapter field mapping and unwired vector-score lineage—that masqueraded as model failures; corrected both without threshold or Evidence-semantic tuning.
- Validated a non-root Docker Compose RC across image-content privacy, readiness, restart/recreation/persistence/shutdown and 1/4/8 concurrency load, backed by a separate 11,678-request native soak with zero errors.
- Maintained a 28/28 protected semantic freeze and zero tracked private artifacts while documenting the remaining 9 false answers, 3 false refusals and unsupported table-cell ownership.
