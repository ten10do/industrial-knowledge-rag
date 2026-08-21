# V3.33 Independent Mixed Evidence Generalization Benchmark

## Final status

**PARTIAL_GENERALIZATION**

The one permitted `V333_UNTOUCHED_BASELINE` run completed successfully through
`backend/retrieval/evidence_mixed.py::analyze_mixed_evidence`. Runtime validity
passed, but the independent K-CHECK split met only the pre-registered PARTIAL
floor, not the GENERALIZES gate.

No retrieval, router, evidence rule, NLI model or threshold, open-sufficiency
logic, support gate, or production default was changed for V3.33. No baseline
replay was performed.

## 1. Corpus K

Corpus K contains six English technical manuals from six manufacturers and five
equipment categories. Every file came from an official manufacturer source.
Brochures, catalogs, marketing PDFs, third-party mirrors, and documents already
used by corpora A-H or J were excluded.

| Manufacturer | Manual | Product line | Type | Physical pages | Production parsed pages |
| --- | --- | --- | --- | ---: | ---: |
| SICK | Operating Instructions — microScan3 Pro I/O / EFI-pro | microScan3 Pro | safety | 236 | 236 |
| Festo | Manual — CPX-FB33 / CPX-M-FB34 / CPX-M-FB35 PROFINET IO | CPX-FB | remote I/O / fieldbus | 158 | 157 |
| SEW-EURODRIVE | Operating Instructions — MOVIDRIVE system, 27786641/EN | MOVIDRIVE system | drive | 304 | 300 |
| Balluff | User's Guide — BNI EIP-502/508-105-Z015 | BNI EIP | remote I/O / fieldbus | 53 | 53 |
| Danfoss | Operating Guide — VLT AQUA Drive FC 202, MG21A502 | VLT AQUA Drive FC 202 | frequency inverter | 162 | 161 |
| Banner Engineering | VE Series Smart Camera Instruction Manual, 191666 Rev. L | VE Series | machine vision | 274 | 274 |

The difference between physical and parsed page counts is caused by the
production parser omitting blank pages: one Danfoss page, one Festo page, and
four SEW pages. The physical PDF counts and manifest counts agree exactly.

### Acquisition record

The recovery phase attempted nine manufacturers. It accepted Danfoss and Banner
Engineering and recorded the following unsuccessful official-source attempts:

| Manufacturer | Official entry result | HTTP / access result | Anti-bot or access gate |
| --- | --- | --- | --- |
| Delta Electronics | ASDA-B2 official manual | 403 | yes |
| Lenze | i550 official product/support page | 200; download target dynamically injected | no explicit block |
| KEYENCE | official English PLC manual index | 200; registration workflow required | access-gated |
| KEBA | official documentation portal | 403; login-restricted portal | yes |
| Murrelektronik | official download center | 200; dynamic index, no stable PDF resolved | no explicit block |
| Kollmorgen | official AKD User Guide | 403 | yes |
| ifm electronic | official AL1352 manual URL | 410 Gone | no |

The accepted additions were downloaded before corpus construction:

- Danfoss VLT AQUA Drive FC 202 Operating Guide:
  `https://files.danfoss.com/download/Drives/DrivesMG21A502.pdf`
- Banner VE Series Smart Camera Instruction Manual:
  `https://info.bannerengineering.com/cs/groups/public/documents/literature/191666.pdf`

Per-document download time, complete SHA-256, URL, page count, product line, and
document type are retained in the gitignored private manifest fragments.

## 2. Integrity and ingestion audits

All six documents passed:

- `%PDF-` file signature check;
- byte-for-byte SHA-256 agreement with the private manifest;
- physical PDF page-count agreement;
- non-empty production text extraction;
- English signal check;
- CJK gate with a measured CJK ratio of `0.0` in every sampled manual.

Production ingestion generated 3,655 chunks:

| Document | Chunks |
| --- | ---: |
| Balluff BNI EIP | 61 |
| Banner VE Series | 965 |
| Danfoss FC 202 | 545 |
| Festo CPX-FB | 362 |
| SEW MOVIDRIVE system | 750 |
| SICK microScan3 Pro | 972 |

Chunk audit: zero empty chunks, zero duplicate chunk IDs, content lengths from
10 to 1,600 characters. Metadata audit: zero missing values across document ID,
chunk ID, source, page, section, manufacturer, equipment type/model, product
family, and product series. Document-identity audit: all 3,655 chunks exactly
matched their source manifest identity metadata.

The spread-sample parser audit had 100% page and section metadata coverage for
all manuals. It emitted `LOW_PRODUCT_IDENTITY_COVERAGE` warnings for Banner and
Festo because sampled body text did not repeatedly restate the full model name.
This is not a document-identity failure: identity metadata remained exact on
100% of their chunks. The parser was not changed.

## 3. Independence and leakage audit

Corpus K was compared with 35 forbidden prior-corpus manifests. It had zero
overlap by official URL, SHA-256, normalized title, or manufacturer/product-line
triple.

The frozen split is fully independent:

| Property | K-TRAIN | K-CHECK | Overlap |
| --- | --- | --- | ---: |
| Documents | Balluff, Danfoss, SEW | Banner, Festo, SICK | 0 |
| Manufacturers | Balluff, Danfoss, SEW-EURODRIVE | Banner Engineering, Festo, SICK | 0 |
| Queries | 48 | 48 | 0 |
| Annotated evidence candidates | document-local | document-local | 0 |

Construction did not load D/E/H/J plaintext or queries. The permitted F/G
development-query leakage check found zero normalized overlap. No old benchmark
query was copied, and no A-E or H replay was performed.

## 4. Benchmark design

The benchmark contains 96 manual-first queries from 24 independently reviewed
fact families. Each fact family contributes a positive verification, negative
verification, positive open question, and relation near-miss open question.

| Design property | K-TRAIN | K-CHECK | Total |
| --- | ---: | ---: | ---: |
| Queries | 48 | 48 | 96 |
| Verification | 24 | 24 | 48 |
| Open | 24 | 24 | 48 |
| ANSWER | 24 | 24 | 48 |
| ABSTAIN | 24 | 24 | 48 |
| HIGH confidence | 48 | 48 | 96 |
| L4-L5 | 48 | 48 | 96 |
| Identifier | 12 | 12 | 24 |
| Table reasoning | 36 | 24 | 60 |

Additional coverage: action/procedure 12, value/default/range/setting 52,
protocol 4, multi-chunk 28, and cross-scope 16. Every annotated evidence chunk
exists, belongs to the declared document, and contains at least 361 characters.

## 5. Freeze and one-shot execution

The following hashes were frozen before any prediction was observed:

| Artifact | SHA-256 |
| --- | --- |
| Corpus manifest | `b156a4f745191a2d2d6b7849704011fa72458d721784b105f9b9189f7619a81b` |
| Queries | `237804e5ae3a6ceecbceee2d952d3a317be49260e52db18df04d118841727e21` |
| Annotations | `ec994469019fb3591b7cea53d0a1852ac9c02ca7d2de1b533bdce23581d8da52` |
| Runtime/config | `f2056890c206bb9a34a7f4ae9294f121b06d591a046699040137cceb4cd311e2` |

The ledger shows `official_baseline_runs = 1`, started at
`2026-08-21T02:22:16Z` and completed at `2026-08-21T02:36:09Z`.

Frozen runtime configuration:

- official entry: `evidence_mixed.py::analyze_mixed_evidence`;
- retrieval: production hybrid, candidate `k=5`, reranker off;
- embedding model: `paraphrase-multilingual-MiniLM-L12-v2`;
- NLI: `cross-encoder/nli-deberta-v3-xsmall`;
- entailment/contradiction thresholds: `0.5 / 0.5`, unknown floor `0.33`;
- production semantic judge, open sufficiency, grounding enrichment, and support
  gate defaults remain off.

## 6. Baseline metrics

| Slice | N | Accuracy | Answerable recall | Abstain recall | FA rate | FR rate | FA / FR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 96 | 0.6250 | 0.7917 | 0.4583 | 0.5417 | 0.2083 | 26 / 10 |
| K-TRAIN | 48 | 0.6458 | 0.8333 | 0.4583 | 0.5417 | 0.1667 | 13 / 4 |
| K-CHECK | 48 | 0.6042 | 0.7500 | 0.4583 | 0.5417 | 0.2500 | 13 / 6 |
| Verification overall | 48 | 0.6042 | 0.7917 | 0.4167 | 0.5833 | 0.2083 | 14 / 5 |
| Open overall | 48 | 0.6458 | 0.7917 | 0.5000 | 0.5000 | 0.2083 | 12 / 5 |
| K-CHECK verification | 24 | 0.6250 | 0.7500 | 0.5000 | 0.5000 | 0.2500 | 6 / 3 |
| K-CHECK open | 24 | 0.5833 | 0.7500 | 0.4167 | 0.5833 | 0.2500 | 7 / 3 |

### Manufacturer slices

| Manufacturer | Accuracy | FA rate | FR rate |
| --- | ---: | ---: | ---: |
| Danfoss | 0.7500 | 0.3750 | 0.1250 |
| Festo | 0.6875 | 0.1250 | 0.5000 |
| Balluff | 0.6250 | 0.6250 | 0.1250 |
| Banner Engineering | 0.6250 | 0.6250 | 0.1250 |
| SEW-EURODRIVE | 0.5625 | 0.6250 | 0.2500 |
| SICK | 0.5000 | 0.8750 | 0.1250 |

### Relation and structural slices

| Relation | N | Accuracy | FA rate | FR rate |
| --- | ---: | ---: | ---: | ---: |
| USES_PROTOCOL | 4 | 1.0000 | 0.0000 | 0.0000 |
| HAS_DEFAULT_VALUE | 28 | 0.6786 | 0.5714 | 0.0714 |
| HAS_IDENTIFIER | 24 | 0.6250 | 0.7500 | 0.0000 |
| REQUIRES_ACTION | 8 | 0.6250 | 0.2500 | 0.5000 |
| HAS_RANGE | 12 | 0.5833 | 0.3333 | 0.5000 |
| HAS_PROCEDURE | 4 | 0.5000 | 0.0000 | 1.0000 |
| HAS_SETTING | 8 | 0.5000 | 1.0000 | 0.0000 |
| HAS_VALUE | 4 | 0.5000 | 0.5000 | 0.5000 |
| USES_TERMINAL | 4 | 0.5000 | 0.5000 | 0.5000 |

L4 accuracy was 0.6029 and L5 accuracy was 0.6786. Table-derived cases scored
0.6333 versus 0.6111 for non-table cases. Multi-chunk cross-scope cases scored
0.6875; multi-chunk product-scope cases scored 0.6667; exact-model cases scored
0.6029.

## 7. Runtime diagnostics

The runtime was valid: benchmark validation passed, failure-attribution vocabulary
validation passed, and every required per-query diagnostic field was present.

| Diagnostic | K-TRAIN | K-CHECK | Overall |
| --- | ---: | ---: | ---: |
| Verification query path | 22 | 24 | 46 |
| Open query path | 24 | 24 | 48 |
| Fallback path | 2 | 0 | 2 |
| NLI router triggers | 0 | 1 | 1 |
| Open-sufficiency invocations | 9 | 3 | 12 |

The single NLI call returned `CONTRADICTS` and was correct. Open sufficiency was
reachable but returned `INSUFFICIENT` for all 12 invocations; relation-support
recall was therefore 0 on both splits. Final decision authority was RULE for 91
queries, HARD_GATE for 4, and NLI for 1. Grounding had no decision authority.

Mean per-query latency was 7,745 ms, median 7,810 ms, and maximum 9,619 ms,
excluding one-time model/index construction.

## 8. Failure attribution

The frozen runner's raw taxonomy reported:

- `BASE_RULE_FAILURE`: 18;
- `PARSER_STRUCTURE_LOSS`: 13;
- `RETRIEVAL_MISSING_EVIDENCE`: 5.

There is a post-run diagnostic-label issue: the runner treated
`table_structure_recoverable = NO` (a non-table case) as evidence of parser loss.
The parser audit does not support that conclusion. This bug does not affect any
decision, metric, hash, or final status, and the baseline was not replayed.

A read-only re-attribution from the frozen per-query diagnostics gives:

| Audited attribution | Count |
| --- | ---: |
| Base-rule failure | 27 |
| Retrieval missing annotated evidence | 5 |
| Open relation-match failure | 2 |
| Open hard-gate refusal | 1 |
| Verification hard-gate refusal / other | 1 |

The dominant error is false acceptance of relation near-misses. Annotated
evidence entered the retrieved candidate set for 77/96 queries, yet that slice
scored 0.5974 and contained 23 false answers. Retrieval misses explain only five
of the 36 errors. NLI routing was too sparse to materially correct verification
near-misses, and open sufficiency never reached `SUPPORTED` on an answerable open
case.

## 9. Decision policy and status rationale

V3.33 reused the V3.31 pre-registered gate without lowering thresholds.
`GENERALIZES` requires:

- accuracy at least 0.70;
- abstain recall at least 0.60;
- answerable recall at least 0.55;
- false-answer rate at most 0.20;
- false-refusal rate at most 0.45;
- verification accuracy at least 0.68;
- open accuracy at least 0.65.

`PARTIAL_GENERALIZATION` requires accuracy at least 0.60, abstain recall at
least 0.45, and answerable recall at least 0.40.

K-CHECK passed the PARTIAL floor (`0.6042 / 0.4583 / 0.7500`) but failed the
GENERALIZES accuracy, abstain-recall, false-answer, verification, and open gates.
Therefore the only valid status is **PARTIAL_GENERALIZATION**.

## 10. Engineering status

- Corpus recovery: **COMPLETE**.
- Corpus K production ingestion: **COMPLETE**.
- Parser/chunk/metadata/document-identity audits: **PASS**, with two documented
  body-text identity-coverage warnings.
- Independence and leakage audit: **PASS**.
- K-TRAIN/K-CHECK construction and freeze: **COMPLETE**.
- Official baseline runs: **1**, permanently consumed.
- Runtime validity: **VALID**.
- Mixed-evidence path reachability: **CONFIRMED**.
- Independent evidence generalization: **PARTIAL_GENERALIZATION**.
- Production readiness claim: **NOT UPGRADED**.

All PDFs, chunks, queries, annotations, raw results, ledgers, model weights, and
recovery logs remain under `backend/evaluation/benchmark_private/` and are
gitignored. Only the public framework, public tests, and this aggregate report
are eligible for commit.
