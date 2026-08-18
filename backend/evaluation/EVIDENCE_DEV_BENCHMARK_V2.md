# Evidence Development Benchmark V2

## Status and boundary

V3.22 replaces the consumed V3.21 development gate with two new development
splits under benchmark schema `v322-dev-v1`:

- `DEV-TRAIN-V2` is the rule-development split.
- `DEV-TUNE-V2` is a document-disjoint development tuning split.
- Neither split is a sealed or one-shot gate.

The V3.21 `DEV-CHECK` data and results remain historical artifacts. They must
not be edited, relabeled, copied into V3.22, or presented as a fresh check.
`evidence-v321.1` is therefore recorded as `ARCHIVED_NOT_READY_CANDIDATE`.
Corpus D and Corpus E remain permanent holdouts and are excluded from all V3.22
development activity.

V3.22 changes benchmark data, validation, evaluation tooling, tests, and this
protocol only. It does not change Evidence, Support, Retrieval, parsing,
product identity, or reranking behavior. Support remains `support-v316.1`.

## Corpus G design

Corpus G contains eight official English vendor manuals from five manufacturers.
The manuals cover PLC/controller, drive/servo, industrial communication, and
remote I/O/fieldbus equipment. The two splits use different manuals and product
lines. The tuning split includes Bosch Rexroth as an explicitly unseen
manufacturer; its other manufacturers are seen only through different product
lines.

The private PDFs, production chunks, annotations, frozen manifests, and result
rows live below `backend/evaluation/benchmark_private/` and are git-ignored.
Only the schema validator, evaluator, tests, and aggregate protocol are public.

| Split | Manufacturer | Manual / product line | Category |
| --- | --- | --- | --- |
| TRAIN | Beckhoff | EL6751 CANopen terminal | industrial communication |
| TRAIN | Omron | CJ2 CPU Unit software | PLC/controller |
| TRAIN | Rockwell Automation | Kinetix 5700 | servo drive |
| TRAIN | ABB | ACS880 primary control program | drive |
| TUNE | Beckhoff | CX51x0 | PLC/controller |
| TUNE | Omron | MX2 Series Type V2 | drive |
| TUNE | Rockwell Automation | ArmorBlock 5000 IO-Link master | remote I/O/fieldbus |
| TUNE | Bosch Rexroth | ctrlX CORE diagnostics | PLC/controller |

## Annotation contract

Every hard pair contains one `ANSWER` case and one `ABSTAIN` near miss grounded
in fixed candidates from the same one-time production ingestion. The pair
families are:

`identifier`, `protocol`, `attribute`, `value`, `action`, `requirement`,
`semantic`, `multi_chunk`, `cross_scope`, and `qualifier`.

Each case records:

- a document style from the V2 style taxonomy;
- difficulty from `L1_EXPLICIT` through `L5_HARD_NEAR_MISS`;
- a surface-form type;
- explicit critical requirements and non-critical cues;
- expected evidence scope;
- confidence and claim type;
- negative hardness plus a forbidden-scope reason for every negative;
- fixed candidate chunk identifiers and an annotation rationale.

The core set excludes ambiguous annotations. The validator enforces the split
size and 45/55–55/45 label balance, all ten families in both splits, quotas for
semantic positives and safe/unsafe multi-chunk cases, focus and style diversity,
and a majority of L3–L5 and N3–N5 cases.

## Leakage and freeze protocol

Before freeze, the validator performs these non-embedding audits:

1. exact document ID, official URL, and exact source-path comparison against D/E;
2. strict TRAIN/TUNE document and source-path disjointness;
3. product-line overlap reporting;
4. exact, normalized, and token-Jaccard query comparison against A–E, the V3.20
   calibration, and both V3.21 development splits.

Each split freezes three independent SHA-256 values: ordered query text,
annotations, and the complete manifest payload excluding the freeze block.
After freeze, changes require a new benchmark version; the V2 manifests are not
edited in place.

## Baseline protocol

Both `evidence-v320.1` and `evidence-v321.1` are evaluated on both V2 splits
against exactly the same candidate-fixed manifests. The older rule is executed
from an isolated source archive, rather than by changing the working tree.
Reports include safety and utility metrics plus slices by difficulty, document
style, seen/unseen manufacturer, and failure class. No Evidence rule is tuned or
created during V3.22.

### Frozen baseline aggregates

| Split / rule | Accuracy | Answerable recall | Abstain recall | False-answer rate | False-refusal rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN / evidence-v320.1 | 58.33% | 41.67% | 75.00% | 25.00% | 58.33% |
| TRAIN / evidence-v321.1 | 58.33% | 47.22% | 69.44% | 30.56% | 52.78% |
| TUNE / evidence-v320.1 | 57.50% | 45.00% | 70.00% | 30.00% | 55.00% |
| TUNE / evidence-v321.1 | 62.50% | 50.00% | 75.00% | 25.00% | 50.00% |

On TRAIN, v321 exchanges 5.56 percentage points of abstain recall for the same
amount of answerable recall and leaves accuracy unchanged. On TUNE, all three
headline measures improve by 5 points. The archived candidate is still not a
ready Evidence rule: the V2 results are starting measurements for V3.23, not a
promotion decision.

## V3.23 rule-development protocol

V3.23 may start only when both V2 manifests validate and both historical
baselines have reproducible reports. Rule development must then follow this
loop:

1. inspect failures on `DEV-TRAIN-V2`;
2. implement one narrow Evidence hypothesis and run targeted tests;
3. measure TRAIN safety and utility, including all required slices;
4. evaluate `DEV-TUNE-V2` only after the hypothesis is fixed;
5. retain the change only when safety does not regress and utility improves
   across the intended slices;
6. record every attempted rule version, including rejected attempts.

V3.23 must not read or run Corpus D/E and must not create or claim a sealed
gate. A later V3.24 task may independently design a new sealed final gate from
unseen documents and queries.

## Engineering status after V3.22

```text
EVIDENCE_GENERALIZATION_STATUS =
DEV_CHECK_FAILED;
CANDIDATE_NOT_READY

EVIDENCE_DEV_BENCHMARK_STATUS = READY

SUPPORT_GENERALIZATION_STATUS =
INDEPENDENT_HOLDOUT_FAILED;
POST_FREEZE_DISCOVERED_ISSUES_RECORDED
```

`READY` applies only to the development benchmark. It does not promote an
Evidence rule or change the recorded Support status.
