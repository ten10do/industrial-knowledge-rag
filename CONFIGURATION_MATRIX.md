# Configuration Matrix — V3.83

Startup validation is implemented in `backend/runtime_config.py`. Invalid enums, booleans, numbers, required companion values, URLs, and configured storage paths fail fast with an actionable error. Empty optional providers degrade deterministically; they do not fabricate availability.

| Setting | Class | Default / allowed | Operational meaning |
|---|---|---|---|
| `RUNTIME_ENV` | REQUIRED in production | `development`; `development/test/production` | Selects production CORS/logging posture. |
| `APP_DEBUG` | OPTIONAL | `false` | Explicit development-only debug; rejected in production. |
| `LOG_FORMAT` | OPTIONAL | text in dev, JSON in production | Privacy-safe allowlisted structured logging in production. |
| `RAG_MODE` | REQUIRED | `light`; `light/full` | Selects index/runtime implementation. Full dependencies are required for `full`. |
| `RETRIEVAL_MODE` | REQUIRED | `hybrid`; lexical/vector/hybrid | Existing frozen retrieval mode; aliases `bm25/tfidf` remain compatible. |
| `PUBLIC_KNOWLEDGE_BASE_ID` | REQUIRED | stable `kb-*` ID | Public read-only knowledge-base scope. |
| `FRONTEND_ORIGIN` | OPTIONAL | local dev origin | Only configured origin is allowed in production; wildcard is not used. |
| `VITE_API_BASE_URL` | OPTIONAL (frontend) | `/api` production | Frontend API origin; never place secrets in `VITE_*`. |
| `ADMIN_TOKEN` | SECRET | none | Required at request time for upload, publish, rollback, and job management. |
| `GROQ_API_KEY`, `DEEPSEEK_API_KEY` | SECRET / OPTIONAL | none | Optional generation providers; missing both yields readiness `degraded`, not false readiness failure. |
| `EMBEDDING_MODEL_NAME` | OPTIONAL | frozen MiniLM ID | Full-mode override must match the index/runtime model identity. |
| `PUBLIC_VERSION_STORAGE_BACKEND` | REQUIRED | `local`; local/s3 | Version storage implementation. |
| `PUBLIC_VERSION_STORAGE_DIR` | OPTIONAL PATH | backend runtime directory | Existing value must be a directory; relative values resolve below backend. |
| `PUBLIC_VERSION_S3_BUCKET` | REQUIRED for S3 | none | Required with S3 storage. |
| `PUBLIC_VERSION_S3_PREFIX`, `PUBLIC_VERSION_S3_ENDPOINT_URL`, `PUBLIC_VERSION_S3_REGION` | OPTIONAL | documented `.env.example` values | S3-compatible routing. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | SECRET / OPTIONAL | provider chain | S3 credentials; never committed or exposed. |
| `TASK_QUEUE_BACKEND` | REQUIRED | `memory`; memory/redis | Background-job backend. |
| `RATE_LIMIT_BACKEND` | REQUIRED | `memory`; memory/redis | Traffic/quota backend. |
| `REDIS_URL` | REQUIRED when Redis selected | none; redis/rediss URL | Shared queue, worker, limiter, and event dependency. |
| `TASK_QUEUE_NAME` | OPTIONAL | `knowledge` | RQ queue name. |
| `TASK_QUEUE_WORKERS`, `TASK_JOB_TIMEOUT_SECONDS`, `TASK_RETENTION_SECONDS`, `TASK_INPUT_RETENTION_SECONDS`, `TASK_STALLED_SECONDS` | OPTIONAL NUMERIC | positive values | Worker sizing, timeouts, and retention. |
| `PUBLIC_VERSION_SYNC_INTERVAL_SECONDS`, `PUBLIC_VERSION_EVENT_CHANNEL` | OPTIONAL | 5 seconds / namespaced channel | Public-version convergence. |
| `RATE_LIMIT_PUBLIC_FAIL_OPEN` | OPTIONAL | `true` | Explicit public rate-limit degradation policy. |
| `RATE_LIMIT_MANAGEMENT_FAIL_OPEN` | OPTIONAL | `false` | Management endpoints fail closed. |
| `MODEL_REQUEST_TIMEOUT_SECONDS`, `MODEL_MAX_RETRIES`, `MODEL_MAX_OUTPUT_TOKENS` | OPTIONAL NUMERIC | 60 / 1 / 2048 | Provider request bounds. |
| `MODEL_DAILY_TOKEN_LIMIT`, `MODEL_MAX_CONCURRENT_PER_USER`, `MODEL_CONCURRENCY_SLOT_TTL_SECONDS` | OPTIONAL NUMERIC | see `.env.example` | Model governance. |
| `HEALTH_RATE_LIMIT`, `READY_RATE_LIMIT`, `METRICS_RATE_LIMIT`, `ASK_RATE_LIMIT`, `STUDY_RATE_LIMIT`, `UPLOAD_RATE_LIMIT`, `RESET_RATE_LIMIT`, `PUBLISH_RATE_LIMIT`, `VERSION_LIST_RATE_LIMIT`, `ROLLBACK_RATE_LIMIT`, `JOB_STATUS_RATE_LIMIT`, `JOB_RETRY_RATE_LIMIT` | OPTIONAL NUMERIC | see `.env.example` | Endpoint rate limits; all must be positive. |
| `MAX_UPLOAD_FILES`, `MAX_UPLOAD_FILE_BYTES`, `MAX_UPLOAD_TOTAL_BYTES`, `MAX_PDF_PAGES`, `MAX_KNOWLEDGE_BASE_CHUNKS`, `MAX_INDEX_SNAPSHOT_BYTES` | OPTIONAL NUMERIC | bounded defaults | Upload/index safety limits. |
| `LEXICAL_TOP_K`, `VECTOR_TOP_K`, `HYBRID_TOP_K`, `RRF_K` | FROZEN RUNTIME | 10 / 10 / 5 / 60 | Existing correctness semantics; V3.83 validates deployment without retuning them. |
| `LIGHT_MAX_RELEVANT_DISTANCE`, `FULL_MAX_RELEVANT_DISTANCE`, `EVIDENCE_MAX_VECTOR_DISTANCE`, `EVIDENCE_MIN_VECTOR_MARGIN` | FROZEN RUNTIME | existing values | Correctness thresholds; validated, never changed by V3.83. |
| `RERANK_ENABLED` | EXPERIMENTAL | `false` | Reranker, default OFF; explicit opt-in warning. |
| `SECTION_EXPANSION_ENABLED` | EXPERIMENTAL | `false` | Section expansion, default OFF. |
| `SUPPORT_GATE_ENABLED` | EXPERIMENTAL | `false` | Claim-support gate, not production-ready. |
| `TABLE_REGION_CONTEXT_ENABLED` | EXPERIMENTAL | `false` | TableContextBundle, default OFF. |
| `CLAIM_SUPPORT_EXPERIMENT_ENABLED` | EXPERIMENTAL | `false` | Reserved experimental claim-support path, default OFF. |
| `RETRIEVAL_TRACE_ENABLED` | OPTIONAL | `false` | Diagnostic trace; raw query/content still must not be logged. |
| `RERANK_MODEL`, `RERANK_CANDIDATE_K`, `RERANK_TOP_K`, `RERANK_DEVICE` | EXPERIMENTAL | existing model / 5 / 3 / cpu | Reranker configuration; top K cannot exceed candidate K. |
| `SECTION_NEIGHBOR_WINDOW`, `SECTION_CANDIDATE_K`, `SECTION_MAX_EXPANDED` | EXPERIMENTAL | existing bounded defaults | Section-expansion bounds. |

Deprecated settings: none are newly introduced in V3.83. `bm25` and `tfidf` retrieval-mode aliases are compatibility aliases, not recommended production names.
