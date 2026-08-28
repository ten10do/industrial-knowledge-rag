# V3.84 RC — Deployment-Validated Industrial Knowledge RAG

> Draft release notes. The MIT-licensed repository is public and exact-SHA CI has passed. Do not create a tag or GitHub Release without separate owner authorization.

## Overview

This release candidate packages a safety-oriented industrial knowledge retrieval and question-answering system. It combines industrial document ingestion, hybrid retrieval, model/identifier-aware scoping, a deterministic Evidence ANSWER/ABSTAIN contract, optional cited generation, operational observability and a validated Docker Compose backend.

It is a deployment-validated development RC, not an enterprise-scale production, SLA, security-audit or high-availability claim.

## Capabilities

- Text-based industrial PDF ingestion with stable chunk/document identity and page provenance.
- BM25 plus light TF-IDF or full HuggingFace/Chroma retrieval, with RRF candidate fusion.
- Equipment identity and technical-identifier scoping.
- Evidence-based safe refusal before any optional model call.
- Knowledge-base draft/build/publish/history/rollback and task lifecycle.
- FastAPI liveness, readiness, metrics, request IDs and privacy-safe structured logging.
- React/Vite frontend and a non-root Docker Compose backend with persistent named volumes.
- Keyless public synthetic demo covering supported answerability and safe hard negatives.

## Frozen evaluation

The formal authority is CONTRACT_NATIVE_V367 on V377_ALIGNED_BENCHMARK_V2:

| Metric | Result |
|---|---:|
| Correct | 54/69 |
| Accuracy | 78.26% |
| False answers | 9 |
| False refusals | 3 |
| Answerable recall | 76.92% |
| Abstention recall | 79.07% |
| Hard negatives | 10/10 |

Canonical correctness digest:

    dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d

These numbers describe one frozen, corpus-aligned private benchmark. They are not a universal industrial-RAG accuracy claim.

## Deployment validation

- Docker image build and content/privacy audit passed.
- Formal image ran as uid 100(app), gid 101(app).
- Liveness and required readiness checks passed; the keyless formal container was degraded only because the optional provider was absent.
- Restart 3/3, graceful shutdown 3/3, force recreation and persistent-volume digest checks passed.
- Container load completed 24/24 requests at concurrency 1, 4 and 8.
- A separate production-like native soak completed 11,678 requests over 600.12 seconds with zero errors.
- Protected semantic hashes passed 28/28; tracked private files were zero.

## Public demo

Run:

    python scripts/v384_public_demo.py

The demo creates and removes a synthetic local index. It requires no private PDF, private benchmark, provider credential or network model call.

## Known limitations

- The frozen benchmark retains 9 false answers and 3 false refusals.
- Exact table-cell ownership is unsupported; layout-aware ingestion is deferred.
- Claim support, local NLI, requirement extraction upgrades, reranking and table-context paths are not production-ready and remain OFF by default.
- Optional real-provider generation was not validated in the certified keyless container.
- The container P95 observation remains a non-blocking investigation item.
- The topology is not Kubernetes, HA, multi-region or SLA certified.

## Provider and CI state

Provider state: OPTIONAL_PROVIDER_CAPABILITY_NOT_VALIDATED.

CI state: `REMOTE_CI_EXECUTED_PASS` for exact SHA `5df4ef42efc0d9e9922087a763aa65ba92dd8c4a` in GitHub Actions run `33156355678`. Every later public HEAD requires its own completed successful run.

## Publication prerequisite

The project LICENSE is MIT, Copyright (c) 2026 ten10do. Third-party dependencies and external data/model weights retain their own licenses.

The public repository, `origin/main`, MIT license and real CI evidence are present. Tag creation and GitHub Release publication remain separate owner decisions.
