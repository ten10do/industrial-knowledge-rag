# Research Decisions

This document records the decisions that define the V3.84 freeze. A failed experiment is retained as evidence, not hidden as unfinished work.

## Decision ledger

| Phase | Question | Evidence | Decision |
|---|---|---|---|
| V3.65–V3.67 | Can prose-oriented evaluation be trusted? | Legacy evaluator false-accepted 88% of mutations; contract-native 151-case authority suite reached 0% FA | Use V3.67 structured evaluator as authority |
| V3.70–V3.71 | Were 19 reported false answers runtime defects? | Adapter read a non-existent requirements_covered field instead of sufficient and omitted covered keys | Correct evaluator adapter; do not tune Evidence for evaluator defects |
| V3.75–V3.76 | Should the vector threshold be retuned? | Harness failed to propagate vector_score; real score lineage was restored 69/69 with threshold unchanged | Fix lineage fidelity, do not disguise wiring defects as calibration |
| V3.77 | Did benchmark gold match the actual corpus? | 18 ANSWER expectations lacked corpus support; query text remained 69/69 unchanged | Rebuild expectations prediction-blind and freeze aligned V2 |
| V3.78 | Can a simple abstention boundary remove unsafe answers? | Candidate vetoes killed 13 or more correct answers while failing utility gates | NO-GO; no runtime change |
| V3.79 | Can current claim-support or local NLI safely gate answers? | Structured signal had discrimination potential but low precision; local NLI recall was 0 and nearly always rejected | PARTIAL/STOP; no production gate |
| V3.80 | Can query-only requirement extraction close the gap? | Extraction coverage improved, but merged support precision worsened and created unsafe supports | PARTIAL/STOP; pattern accumulation is not the missing semantics |
| V3.49–V3.64 | Can flat/layout text prove table-cell ownership? | Real-manual ownership precision/recall and structure capture gates failed or were inconclusive; flat text lost column semantics | Structured producer FROZEN_PARTIAL; cell ownership authority NONE |
| V3.81 | Can a lightweight attribute/value lattice prove fixed specification existence? | Hand-audited strict precision 0.61 versus required 0.95 | NO-GO/STOP |
| V3.82–V3.83-C | Is the frozen system operationally deployable? | Release guards, tests, lifecycle, failure/recovery, soak, real image/container and privacy gates passed | Internal deployment-validation READY; package RC evidence |

## Evaluator fidelity before model tuning

V3.70 traced 19 of 21 apparent false answers to a field-mapping defect in the evaluation adapter. V3.71 fixed the adapter. V3.75 then found that the evaluation harness never wired vector distance into the runtime field consumed by Evidence. V3.76 fixed only that lineage boundary and preserved ranking, threshold and Evidence semantics.

The general decision is: when a benchmark result changes, verify schema mapping, score meaning, field propagation and corpus/gold alignment before tuning model or policy behavior.

## Safety before recall

The system treats unsupported confirmation as more severe than a conservative refusal. That does not justify indiscriminate abstention. V3.78 evaluated plausible veto rules and rejected them because they destroyed supported-answer utility. V3.79 and V3.80 likewise stopped when a signal looked promising but could not meet precision and false-refusal gates.

The current 9 false answers remain disclosed. They are not “fixed” through an unvalidated threshold or an always-reject classifier.

## Why table ownership is unsupported

Industrial parameter tables frequently use borderless, whitespace-aligned layouts with wrapped and stacked cells. The current text extraction path discards enough geometry that same-region or nearby text cannot prove row/column ownership. Synthetic feasibility was strong, but real-manual candidates failed precision/recall gates; independent generalization confirmed the representation mismatch.

Therefore:

- same table region does not imply a supported relation;
- flat-text ownership inference is forbidden;
- TABLE_CELL_OWNERSHIP_AUTHORITY is NONE;
- table-context and structure-producer code remains research-only and OFF by default.

The future independent direction is layout-aware ingestion with trustworthy cell coordinates and ownership provenance. It requires a new architecture decision and is not V3.84 work.

## Why local NLI is not a production gate

The frozen local NLI wrapper was evaluated without threshold tuning. On the aligned cases it produced zero useful supported-case recall and rejected almost everything through neutral-dominant behavior. A model being called “NLI” does not make its chunk-versus-query output a safe answerability signal. The wrapper remains research evidence, not decision authority.

## Release freeze decision

V3.84 changes documentation, public demo tooling and release evidence only. It does not change retrieval, Evidence, identity, threshold, score lineage, benchmark gold, generation semantics or experimental defaults. Any proposal to resume those areas must begin as a separate phase with explicit success gates.

See [Historical Research Index](docs/research/README.md) for the source reports.
