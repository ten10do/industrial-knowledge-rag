# V3.84 Local Release Candidate Checklist

RC identifier: **industrial-knowledge-rag V3.84 local RC**

This checklist packages the deployment-validated development RC. It is not a Git tag, public release, enterprise production certification, SLA, security audit or HA claim.

This document is historical V3.84 evidence. Later V3.85 changes and dependency
hardening have local verification only; they require a new exact-SHA remote CI
run and container build before this checklist can be used for a new release.

| Gate | State | Evidence |
|---|---|---|
| V3.83 RC manifest present and internally consistent | PASS | V383_RELEASE_CANDIDATE_MANIFEST.json; source input hashes rechecked |
| V3.84 freeze manifest | PASS | V384_RC_FREEZE_MANIFEST.json |
| Correctness digest preserved | PASS | dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d |
| Benchmark identity | PASS | V377_ALIGNED_BENCHMARK_V2 |
| Protected semantic hashes | PASS | backend.v382_release_guard, 28/28 |
| RAG runtime semantic diff | PASS | zero runtime/protected-file diff from f981ac6 RC source |
| Private artifact guard | PASS | tracked private files 0 |
| Secret/developer-path scan | PASS | no secret token pattern or absolute developer path in public repository files |
| Backend tests | PASS / REUSED | V3.83 formal result: 948 passed, 2 skipped, 0 failed, plus 5 subtests |
| Frontend tests/build | PASS / REUSED | V3.83 formal result: 34/34 and production build PASS |
| Docker image build | PASS | V3.83-C full gate; V3.84 cache-only command validation also completed |
| Image content audit | PASS | V3.83-C private/test/path exclusion audit |
| Non-root runtime | PASS | uid=100(app), gid=101(app) |
| Liveness/readiness | PASS | formal V3.83-C gate plus V3.84 Compose command smoke |
| Restart/persistence/recreation | PASS | 3/3 restart; unchanged persistence digest; force-recreate PASS |
| Graceful shutdown | PASS | 3/3, exit 0, no OOM |
| Privacy/request ID/metrics | PASS | formal container sentinel and public demo |
| Native soak | PASS / REUSED | 600.12 s, 11,678 requests, 0 errors |
| Container bounded load | PASS / REUSED | 24/24 at concurrency 1/4/8 |
| Public demo | PASS | scripts/v384_public_demo.py, fresh synthetic run |
| Documentation links | PASS | local case-sensitive target audit |
| Quick start / commands | PASS | Compose config/build/start, /live, header-scoped /ready, public demo |
| Known limitations current | PASS | KNOWN_LIMITATIONS.md |
| Deployment/runbook docs current | PASS | DEPLOYMENT.md, DEPLOYMENT_VALIDATION.md, OPERATIONS_RUNBOOK.md |
| Optional provider capability | OPTIONAL_NOT_VALIDATED | authorized real provider unavailable in formal certified container |
| Remote CI execution | PASS / HISTORICAL | V3.84 run `33156355678`, exact SHA `5df4ef42efc0d9e9922087a763aa65ba92dd8c4a`, all four jobs successful; not evidence for the current working tree |
| Container P95 follow-up | NON_BLOCKING_KNOWN_LIMITATION | CONTAINER_PERFORMANCE_INVESTIGATION_REQUIRED |
| Git tag / remote release | NOT_PERFORMED | owner authorization required |

Release-candidate package gate: **PASS**.

Project-evidence package gate: **PASS**.

Final V3.84 package status: **V3_84_RC_PROJECT_EVIDENCE_READY**.

Current working-tree release status: **EXACT_SHA_REMOTE_CI_REQUIRED**.
