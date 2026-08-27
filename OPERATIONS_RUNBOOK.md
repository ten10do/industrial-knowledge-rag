# Operations Runbook — V3.82

## Install and start

Supported baseline: Python 3.11+, Node.js 20+, or Docker Compose.

```powershell
Copy-Item .env.example .env
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Production-like container start:

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

The API may be live but not ready until a compatible index is mounted or built. That is expected and must not be hidden.

## Stop and restart

Local Uvicorn: send `Ctrl+C` and wait for shutdown completion. Container:

```powershell
docker compose stop -t 30
docker compose start
```

The lifespan shutdown stops the public-version synchronizer. The memory queue executor performs normal interpreter shutdown; RQ workers should be stopped with their normal process supervisor before Redis maintenance. Do not terminate during an index replace unless an operator has verified the active version and snapshot.

## Health, readiness, and metrics

```powershell
Invoke-RestMethod http://127.0.0.1:8000/live
Invoke-WebRequest http://127.0.0.1:8000/ready -Headers @{'X-Knowledge-Base-ID'='kb-public-shared-00000001'}
Invoke-WebRequest http://127.0.0.1:8000/metrics
```

- `/live`: process-only liveness. Dependency/provider failures do not make it fail.
- `/health`: legacy dashboard health and version-governance detail; HTTP 200 can be `degraded`.
- `/ready`: mode-aware required dependency and index integrity gate. HTTP 503 means do not route query traffic.
- `/metrics`: low-cardinality Prometheus text. It contains request/error counts, request/retrieval/Evidence latency, answer/abstain counts, abstain families, dependency failures, and queue gauges.

Never add query, chunk text, path, request ID, chunk ID, or query hashes as metric labels.

## Logs and request IDs

Set `RUNTIME_ENV=production` and `LOG_FORMAT=json`. Every HTTP response returns `X-Request-ID`; a safe caller-provided ID is preserved. Use that ID to correlate request completion, retrieval/Evidence outcome, error response, and submitted background job trace. Logs use an allowlist and do not contain raw query, PDF/chunk content, secrets, benchmark gold, private paths, or tracebacks.

## Index verification

Readiness never rebuilds an index. Missing, empty, unreadable, or incompatible indexes return 503.

Full indexes can include `index_manifest.json`:

```json
{
  "manifest_version": 1,
  "index_version": "operator-release-id",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dimension": 384,
  "metadata_schema_version": 2,
  "chunk_count": 123
}
```

When present, all fields are mandatory and enforced. Legacy full indexes without a manifest receive fact-only validation and readiness `degraded`; migrate them before a production release. Light JSON indexes are validated for existence, non-empty content, and document shape.

## Common failures and recovery

| Signal | Expected behavior | Recovery |
|---|---|---|
| `index_index_exists` false | `/ready` 503; `/live` 200; no rebuild | Mount/restore the intended index, verify ownership, call `/ready` again. |
| Index parse/open failure | `/ready` 503 with safe error type | Restore a known-good snapshot; do not edit an active index in place. |
| Embedding stack/model mismatch | Full-mode `/ready` 503 | Install `requirements-full.txt`, mount model cache, use the frozen model identity. |
| Redis unavailable | Redis-selected readiness 503; safe API failures | Restore Redis/network/TLS, then verify ping, `/ready`, worker health. |
| Worker unavailable | Redis queue readiness 503 | Start the configured RQ worker on `TASK_QUEUE_NAME`; confirm heartbeat. |
| LLM key missing | Readiness `degraded`; retrieval remains serviceable | Configure one provider key or accept retrieval-only degraded mode. |
| LLM remote timeout/outage | Safe 5xx with request ID; no fabricated answer | Retry only when error says retryable; check provider status and quotas. |
| Invalid config | Startup exits with named setting | Correct `.env`; never depend on silent fallback. |
| Upload rejected | Safe 4xx; staged directory cleanup | Verify PDF extension/magic/parseability, limits, duplicate names, and sanitized basename. |
| Job failed | Task center shows failed stage and trace | Fix dependency, use the retry endpoint with a new idempotency key. |

## Failure validation commands

```powershell
.\venv\Scripts\python.exe -m pytest backend\test_v382_production_readiness.py -v
.\venv\Scripts\python.exe scripts\v382_public_smoke.py
.\venv\Scripts\python.exe scripts\v382_load_smoke.py
.\venv\Scripts\python.exe -m backend.v382_release_guard
```

These public gates use a synthetic tiny index and no private corpus or API key.

## Rollback

Use the authenticated version history/rollback API or restore the last known-good persistent volume/snapshot. After rollback, verify active version consistency, `/ready`, a public smoke query, and logs. Do not use Git rollback as a substitute for restoring runtime indexes/data. Do not silently rebuild an incompatible index.

## Security and privacy checks

- Production debug is rejected; wildcard CORS is not configured.
- `.env`, PDFs, indexes, results, model caches, and private benchmarks are ignored and excluded from Docker context.
- Run `python -m backend.v382_release_guard`; `TRACKED_PRIVATE_FILES` must be zero.
- Management endpoints require `ADMIN_TOKEN`; management rate limiting fails closed by default.

See `KNOWN_LIMITATIONS.md`, `CONFIGURATION_MATRIX.md`, and `DEPLOYMENT.md` before release approval.
