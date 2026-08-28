# Demo Scenarios

## Public demo contract

The V3.84 demo is deterministic, keyless, corpus-free and safe to run in a public clone. It creates a temporary two-chunk synthetic AX-100 pump index, runs real retrieval/Evidence/API paths, verifies the index hash, then removes the file.

It never loads a private PDF, private annotation, private query, API key, model cache or network service.

Run from the repository root:

    python scripts/v384_public_demo.py

Expected top-level output:

    "status": "PASS"

## Scenario 1 — supported answerability

Question: What is the maximum operating pressure of the AX-100 cooling pump?

Synthetic evidence explicitly states that AX-100 has a maximum operating pressure of 8 bar.

Expected:

- retrieval returns candidates;
- Evidence decision is ANSWER;
- reason is EXACT_MODEL_EVIDENCE;
- provider_generation_executed is false.

This proves the deterministic answerability boundary, not generated-answer quality. A real provider response is intentionally not faked.

## Scenario 2 — wrong-model hard negative

Question: What is the maximum operating pressure of the AX-200 cooling pump?

The corpus contains AX-100 only.

Expected:

- Evidence decision is ABSTAIN;
- reason is MODEL_MISMATCH.

This demonstrates that semantically similar pump text is not enough to authorize an answer for another model.

## Scenario 3 — unknown fault via API

Question: What does fault code F999 mean?

The synthetic corpus contains F101 but not F999.

Expected:

- POST /ask returns HTTP 200 with the stable refusal contract;
- is_refused is true;
- Evidence decision is ABSTAIN;
- sources is empty;
- no model call is made.

## Scenario 4 — operations

Expected:

- /live is HTTP 200 and process status is ok;
- /ready is HTTP 200 with ready=true;
- readiness status is degraded when no optional provider is configured;
- /metrics includes HTTP and RAG abstention counters;
- the supplied request ID is returned;
- the temporary index digest is unchanged after requests;
- experimental reranking and support defaults are OFF.

## Demo interpretation

The public demo is intentionally small. It validates wiring, refusal semantics, identity safety, observability and reproducibility. It must not be presented as the V377 accuracy benchmark, production load evidence, broad vendor coverage or a model-generation evaluation. Those boundaries are documented in [Evaluation](EVALUATION.md) and [Deployment Validation](DEPLOYMENT_VALIDATION.md).
