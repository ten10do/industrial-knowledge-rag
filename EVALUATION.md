# Evaluation

## Correctness authority

The current authority is the V3.67 contract-native evaluator running against V377_ALIGNED_BENCHMARK_V2. It evaluates structured runtime records: ANSWER/ABSTAIN decisions, required and forbidden claims, identity, scope, relation and reason fields. It does not accept an answer merely because its prose looks plausible.

V3.65 demonstrated why this matters: the legacy evaluator accepted 7 of 8 adversarial mutations, an 88% false-accept rate. The typed replacement reduced that to 1 of 8. V3.67 then established a 151-case authority suite with 0% false accepts.

## Evaluation layers

| Layer | Data | Purpose | May support a production-quality claim? |
|---|---|---|---|
| Unit/regression | committed synthetic fixtures | deterministic code and contract regression | No |
| Public challenge | committed authored cases | failure discovery, retrieval and refusal stress | No |
| Public demo | generated synthetic text | reproducible operational and safety behavior | No |
| Private aligned benchmark | local ignored three-document corpus and annotations | frozen correctness comparison | Only as the disclosed aggregate result, not broad production accuracy |
| Deployment validation | synthetic indexes plus real process/container execution | lifecycle, recovery, privacy, persistence and load | Operational claims only |

## Frozen V377 result

| Metric | Result |
|---|---:|
| Total cases | 69 |
| Correct | 54 |
| False answers | 9 |
| False refusals | 3 |
| Accuracy | 78.26% |
| Answerable recall | 76.92% |
| Abstention recall | 79.07% |
| Hard negatives | 10/10 correct |
| Score-lineage fidelity | 69/69 |
| Replay digest | dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d |

The 69 query texts were preserved during V377 alignment. Gold expectations were rebuilt prediction-blind against what the frozen corpus actually contains: 26 supported-answer cases and 43 abstainable cases. Eighteen former ANSWER labels became ABSTAIN because the corpus did not support them. This made the benchmark stricter about evidence rather than preserving invalid labels.

## Benchmark privacy and reproducibility

The private corpus, annotations, indexes and per-query outputs stay in ignored local paths. Public artifacts contain only schemas, aggregate metrics, tool/version contracts and cryptographic hashes. A contributor with authorized source documents and the matching local manifest can reproduce the private gate; a public contributor cannot reconstruct the documents from this repository.

Important frozen identities:

- evaluator: backend/evaluation/contract_eval_v367.py
- evaluator SHA-256: db838edf3f4bb7b56d7c1f2f48175de0492e55b7050644eb7c21ce71cb01b96d
- score lineage SHA-256: 80a5e4d98b38d1dd274b3e76046332aef3255b011264c0e30c768174fa1c7ca1
- corpus hash: 1c4b2d78cfe5cde01d4cddc419a27f5a28a36a6287e0709c2a23c2283ebf2fe1
- query-text hash: a7de41a7db76f160d15b990bb4fba0f3fe772e9f4e7413202299b4f13872d145
- expectation hash: 7734c28c7470bc94ffb286dfc8dad84cc5de1d4cf57c6c45eace44a8b898bf3a

## Validation commands

Public, keyless:

    python -m backend.v382_release_guard
    python -m pytest backend -v
    python scripts/v384_public_demo.py
    python -m backend.evaluation.run

Private, only with authorized local inputs:

    python v377_aligned_baseline.py

Do not publish private output or treat public synthetic scores as a substitute for the V377 result.

## Interpreting the result

The benchmark establishes a frozen, reproducible engineering baseline. It does not establish domain-wide accuracy, vendor coverage, table-cell understanding, generation quality, or safety certification. The remaining 9 false answers are the highest-severity correctness risk. Any future runtime change to protected semantics requires an explicit new phase, new replay evidence and a new release decision.

Research history and rejected improvement paths are summarized in [Research Decisions](RESEARCH_DECISIONS.md). Operational evidence is in [Deployment Validation](DEPLOYMENT_VALIDATION.md).
