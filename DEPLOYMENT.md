# Deployment — V3.83

## Supported topology

```text
Browser / frontend
        |
        v
FastAPI API (:8000)
  |-- light JSON index or full Chroma index (persistent mount)
  |-- local version storage or S3-compatible storage
  |-- optional Groq / DeepSeek generation
  `-- memory backend (single process) or Redis + RQ worker (shared deployment)
```

The repository provides a minimal non-root backend image and Compose deployment. The existing frontend can be built with `npm run build` and hosted separately; set `VITE_API_BASE_URL` at frontend build time and `FRONTEND_ORIGIN` on the API. No secret may use a `VITE_*` variable.

## Container deployment

```powershell
Copy-Item .env.example .env
# Replace ADMIN_TOKEN and optional provider credentials outside source control.
docker compose config --quiet
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/live
Invoke-WebRequest http://127.0.0.1:8000/ready -Headers @{'X-Knowledge-Base-ID'='kb-public-shared-00000001'}
```

Port 8000 is published. Named volumes persist uploaded data, light indexes, public versions, and runtime state. Full mode additionally requires `backend/requirements-full.txt` in a derived image plus persistent Chroma and model-cache mounts; the supplied image intentionally defaults to light mode and does not download model weights during build.

Private PDFs, benchmarks, `.env`, vector databases, results, `.git`, and model caches are excluded from the image context. Mount authorized private data at runtime; never bake it into an image.

## Runtime modes

- `light`: base dependencies, JSON index, BM25 + character TF-IDF hybrid. Suitable for keyless public operational smoke.
- `full`: Chroma + frozen embedding model. Full dependencies, compatible index manifest, and model cache are required. An invalid full installation must fail readiness; it must not be represented as full/hybrid success.
- Memory queue/rate limits: one API process only.
- Redis/RQ: required for shared/multi-process queues and limiters; Redis URL and live worker become readiness requirements.

## Health checks and routing

Container health uses `/live`. Traffic routing must additionally require `/ready` for the target knowledge-base ID. A `degraded` 200 is routable when only optional providers or legacy-manifest warnings are present. An `unavailable` 503 is not routable.

An unavailable `/ready` response follows the safe operational error contract (`error_code=DEPENDENCY_UNAVAILABLE`, `message`, `request_id`, `retryable=true`) while retaining the per-check readiness details. Each failed required check also increments the low-cardinality dependency-failure metric.

## Persistence and permissions

The image runs as the `app` system user. Ensure bind mounts are writable by that UID/group or use named volumes. Persist indexes and version stores across restarts. Index replace/build operations use staging and atomic replacement where supported; do not share a writable local index between unrelated releases without an operator-controlled version boundary.

## Configuration and secrets

See `CONFIGURATION_MATRIX.md`. Use the deployment platform's secret store for `ADMIN_TOKEN`, provider keys, AWS credentials, and Redis credentials. Readiness, errors, metrics, logs, reports, and frontend bundles must not expose them.

## Validation boundary

`PUBLIC_CI_GATE` runs backend public tests, offline public evaluation, frontend tests/build, research/private guards, public operational smoke, Compose validation, and image build. It requires no private PDF, index, or external API key.

`PRIVATE_EVALUATION_GATE` is local and ignored by Git. It owns the V377 aligned corpus replay and must remain exactly 54 correct / 9 false answers / 3 false refusals with hard negatives 10/10. Public synthetic tests cannot replace that gate.

V3.83 validated the native production-like process lifecycle, restart/persistence, safe failure/recovery, observability, frontend/API connectivity, bounded load, and a 10-minute soak. The supported release topology is still Docker Compose; image build and container lifecycle remain unvalidated until a Docker daemon is available. This is not a claim of Kubernetes hardening or multi-region failover.
