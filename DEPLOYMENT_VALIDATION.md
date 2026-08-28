# Deployment Validation

## Certified topology

The V3.83-C internal release topology is a Docker Compose light-mode FastAPI backend with four persistent named volumes and a separately hosted frontend. The image runs as non-root uid 100. Provider generation is optional; the certified container was intentionally keyless.

Status: **DEPLOYMENT_FAILURE_MODE_VALIDATION_READY**

## Image and container evidence

| Gate | Evidence | Result |
|---|---|---|
| Docker environment | Docker Desktop 4.85.0; client/server 29.6.2; Compose 5.3.1; linux/amd64 | PASS |
| Compose config | docker compose config --quiet | PASS |
| Image build | docker compose build; final rebuild 5.446 s | PASS |
| Image identity | sha256:183cc4017252d50ecfdf49f946d7e52cff5d4047161b9ce3c11e38be741bcc99 | FROZEN |
| Image size | 162,451,362 bytes | RECORDED |
| Runtime user | uid=100(app) gid=101(app) | PASS |
| Content/privacy audit | tests, evaluation tree, PDFs, private data, results and developer paths absent | PASS |
| Liveness | /live HTTP 200, process-only semantics | PASS |
| Readiness | /ready HTTP 200 degraded; all required checks pass; optional provider absent | PASS |
| Request ID/logging/metrics | round trip, structured allowlist, privacy sentinel, metrics smoke | PASS |

## Lifecycle and persistence

| Gate | Evidence | Result |
|---|---|---|
| Restart | docker compose restart api, 3/3 | PASS |
| Restart recovery | live/ready in 1.27–1.50 seconds | PASS |
| Synthetic index persistence | SHA-256 b83bb64ea9c8bb79a3f5060b0125d746a62c02ad0216c92177cff8dd60366b83 before/after | PASS |
| Force recreation | container ID changed; volume and index digest retained | PASS |
| Graceful stop | 3/3, exit code 0, 0.77–0.85 seconds, no OOM | PASS |
| Compose cleanup | containers/network removed; named volumes retained | PASS |

## Load and resource evidence

Each container profile executed 24 requests with zero errors:

| Concurrency | Median | P95 | Max |
|---:|---:|---:|---:|
| 1 | 1.91 ms | 17.25 ms | 23.73 ms |
| 4 | 5.79 ms | 23.28 ms | 23.30 ms |
| 8 | 6.79 ms | 23.22 ms | 23.23 ms |

A confirmation concurrency-1 run measured median 1.64 ms and P95 9.11 ms. The first profile's P95 variation is retained as CONTAINER_PERFORMANCE_INVESTIGATION_REQUIRED, a non-blocking follow-up. It is not hidden or promoted into an invented service-level objective.

Observed container resources: CPU 0.11%, memory 112 MiB, 61 PIDs, not OOM-killed.

The earlier production-like native soak remains valid reused evidence: 600.12 seconds, concurrency 4, 11,678 requests, 100% success, no unexplained 5xx, crash, deadlock, index mutation or trace contamination. RSS moved from 145.89 MiB to 150.58 MiB with a 156.21 MiB peak.

## Correctness and regressions

- V377 aligned replay: 54/69 correct, 9 FA, 3 FR, hard negatives 10/10.
- Replay digest: dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d.
- Protected release guard: 28/28 PASS; tracked private files: 0.
- Backend: 948 passed, 2 skipped, plus 5 subtests.
- Frontend: 34/34 passed; production build and configured-origin CORS connectivity passed.
- RAG semantic diff across the deployment-validation closure: zero.

The full backend and frontend suites were run in V3.83. V3.83-C reused those results because its only code fix was image packaging in .dockerignore; it reran the real image/container gates, formal V377 replay and release guard.

## Failure modes validated

Validated in the production-like native process and/or the real Compose container:

- missing and malformed required index: readiness 503, then recovery to 200;
- absent optional provider: degraded readiness without false failure;
- invalid request: stable safe error contract;
- unknown evidence: safe ABSTAIN without model call or sources;
- upload validation, idempotent job submission and worker execution;
- restart, stop, recreation, port release and index persistence;
- request-ID isolation, bounded metrics labels and privacy-safe structured logs.

## Explicitly pending

- OPTIONAL_PROVIDER_CAPABILITY_NOT_VALIDATED: no authorized provider credential was used in the certified keyless container.
- CI_EXTERNAL_EXECUTION_PENDING: the workflow is configured, but V3.84 performs no remote, push or release action.
- Kubernetes, HA, multi-region, external metrics retention and production SLO certification are outside this topology.

For commands and incident handling see [Deployment](DEPLOYMENT.md) and [Operations Runbook](OPERATIONS_RUNBOOK.md). The immutable machine-readable source is [V383 Release Candidate Manifest](V383_RELEASE_CANDIDATE_MANIFEST.json).
