"""Fail-fast validation for production runtime configuration.

This module validates the environment already consumed by the application. It
does not select retrieval policy or change any RAG defaults; it only prevents
invalid values from being silently replaced by defaults during startup.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

ENUM_SETTINGS = {
    "RUNTIME_ENV": {"development", "test", "production"},
    "RAG_MODE": {"light", "full"},
    "RETRIEVAL_MODE": {"lexical", "vector", "hybrid", "bm25", "tfidf"},
    "TASK_QUEUE_BACKEND": {"memory", "redis"},
    "RATE_LIMIT_BACKEND": {"memory", "redis"},
    "PUBLIC_VERSION_STORAGE_BACKEND": {"local", "s3"},
    "LOG_FORMAT": {"text", "json"},
    "RERANK_DEVICE": {"cpu", "cuda", "mps"},
}

BOOLEAN_SETTINGS = {
    "APP_DEBUG",
    "RERANK_ENABLED",
    "SECTION_EXPANSION_ENABLED",
    "RETRIEVAL_TRACE_ENABLED",
    "SUPPORT_GATE_ENABLED",
    "TABLE_REGION_CONTEXT_ENABLED",
    "CLAIM_SUPPORT_EXPERIMENT_ENABLED",
    "RATE_LIMIT_PUBLIC_FAIL_OPEN",
    "RATE_LIMIT_MANAGEMENT_FAIL_OPEN",
}

POSITIVE_INTEGER_SETTINGS = {
    "LEXICAL_TOP_K",
    "VECTOR_TOP_K",
    "HYBRID_TOP_K",
    "RRF_K",
    "RERANK_CANDIDATE_K",
    "RERANK_TOP_K",
    "SECTION_NEIGHBOR_WINDOW",
    "SECTION_CANDIDATE_K",
    "SECTION_MAX_EXPANDED",
    "TASK_QUEUE_WORKERS",
    "TASK_JOB_TIMEOUT_SECONDS",
    "TASK_RETENTION_SECONDS",
    "TASK_INPUT_RETENTION_SECONDS",
    "TASK_STALLED_SECONDS",
    "PUBLIC_VERSION_SYNC_INTERVAL_SECONDS",
    "MODEL_MAX_OUTPUT_TOKENS",
    "MODEL_DAILY_TOKEN_LIMIT",
    "MODEL_MAX_CONCURRENT_PER_USER",
    "MODEL_CONCURRENCY_SLOT_TTL_SECONDS",
    "HEALTH_RATE_LIMIT",
    "READY_RATE_LIMIT",
    "METRICS_RATE_LIMIT",
    "ASK_RATE_LIMIT",
    "STUDY_RATE_LIMIT",
    "UPLOAD_RATE_LIMIT",
    "RESET_RATE_LIMIT",
    "PUBLISH_RATE_LIMIT",
    "VERSION_LIST_RATE_LIMIT",
    "ROLLBACK_RATE_LIMIT",
    "JOB_STATUS_RATE_LIMIT",
    "JOB_RETRY_RATE_LIMIT",
    "MAX_UPLOAD_FILES",
    "MAX_UPLOAD_FILE_BYTES",
    "MAX_UPLOAD_TOTAL_BYTES",
    "MAX_PDF_PAGES",
    "MAX_KNOWLEDGE_BASE_CHUNKS",
    "MAX_INDEX_SNAPSHOT_BYTES",
    "LEARNING_BATCH_CHARS",
    "LEARNING_MAX_BATCHES",
}

NONNEGATIVE_INTEGER_SETTINGS = {"MODEL_MAX_RETRIES"}
POSITIVE_FLOAT_SETTINGS = {
    "MODEL_REQUEST_TIMEOUT_SECONDS",
    "LIGHT_MAX_RELEVANT_DISTANCE",
    "FULL_MAX_RELEVANT_DISTANCE",
    "EVIDENCE_MAX_VECTOR_DISTANCE",
}
NONNEGATIVE_FLOAT_SETTINGS = {"EVIDENCE_MIN_VECTOR_MARGIN"}

EXPERIMENTAL_FLAGS = {
    "RERANK_ENABLED": "experimental reranker",
    "SECTION_EXPANSION_ENABLED": "experimental section expansion",
    "SUPPORT_GATE_ENABLED": "claim-support gate not production-ready",
    "TABLE_REGION_CONTEXT_ENABLED": "TableContextBundle experimental/default OFF",
    "CLAIM_SUPPORT_EXPERIMENT_ENABLED": "claim-support experimental path",
}


class RuntimeConfigurationError(RuntimeError):
    """Actionable startup error caused by invalid operator configuration."""


def bool_value(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise RuntimeConfigurationError(
        f"{name} must be one of true/false/1/0/yes/no/on/off; got an invalid value."
    )


def _validate_url(name: str, *, schemes: set[str]) -> None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return
    parsed = urlparse(raw)
    if parsed.scheme not in schemes or not parsed.hostname:
        allowed = ", ".join(sorted(schemes))
        raise RuntimeConfigurationError(f"{name} must be an absolute {allowed} URL.")


def _validate_numeric(name: str, *, integer: bool, minimum: float) -> None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        kind = "integer" if integer else "number"
        raise RuntimeConfigurationError(f"{name} must be a valid {kind}.") from exc
    if value < minimum:
        comparator = ">= 0" if minimum == 0 else "> 0"
        raise RuntimeConfigurationError(f"{name} must be {comparator}.")


def validate_runtime_environment(*, load_environment: bool = True) -> tuple[str, ...]:
    """Validate startup settings and return explicit non-fatal warnings."""
    if load_environment:
        load_dotenv(ROOT_DIR / ".env")

    for name, choices in ENUM_SETTINGS.items():
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        normalized = raw.strip().lower()
        if normalized not in choices:
            allowed = ", ".join(sorted(choices))
            raise RuntimeConfigurationError(f"{name} must be one of: {allowed}.")

    for name in BOOLEAN_SETTINGS:
        bool_value(name)
    for name in POSITIVE_INTEGER_SETTINGS:
        _validate_numeric(name, integer=True, minimum=1)
    for name in NONNEGATIVE_INTEGER_SETTINGS:
        _validate_numeric(name, integer=True, minimum=0)
    for name in POSITIVE_FLOAT_SETTINGS:
        _validate_numeric(name, integer=False, minimum=1e-300)
    for name in NONNEGATIVE_FLOAT_SETTINGS:
        _validate_numeric(name, integer=False, minimum=0)

    runtime_env = os.getenv("RUNTIME_ENV", "development").strip().lower()
    if runtime_env == "production" and bool_value("APP_DEBUG", False):
        raise RuntimeConfigurationError("APP_DEBUG=true is incompatible with RUNTIME_ENV=production.")

    queue_backend = os.getenv("TASK_QUEUE_BACKEND", "memory").strip().lower()
    limiter_backend = os.getenv("RATE_LIMIT_BACKEND", "memory").strip().lower()
    if "redis" in {queue_backend, limiter_backend}:
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            raise RuntimeConfigurationError(
                "REDIS_URL is required when TASK_QUEUE_BACKEND or RATE_LIMIT_BACKEND is redis."
            )
        _validate_url("REDIS_URL", schemes={"redis", "rediss"})

    storage_backend = os.getenv("PUBLIC_VERSION_STORAGE_BACKEND", "local").strip().lower()
    if storage_backend == "s3" and not os.getenv("PUBLIC_VERSION_S3_BUCKET", "").strip():
        raise RuntimeConfigurationError(
            "PUBLIC_VERSION_S3_BUCKET is required when PUBLIC_VERSION_STORAGE_BACKEND=s3."
        )

    configured_storage = os.getenv("PUBLIC_VERSION_STORAGE_DIR", "").strip()
    if configured_storage:
        storage_path = Path(configured_storage)
        if storage_path.exists() and not storage_path.is_dir():
            raise RuntimeConfigurationError(
                "PUBLIC_VERSION_STORAGE_DIR must point to a directory, not a file."
            )

    _validate_url("FRONTEND_ORIGIN", schemes={"http", "https"})
    if os.getenv("FRONTEND_ORIGIN", "").strip().endswith("/"):
        raise RuntimeConfigurationError("FRONTEND_ORIGIN must not include a trailing slash.")

    rerank_candidate = os.getenv("RERANK_CANDIDATE_K")
    rerank_top = os.getenv("RERANK_TOP_K")
    if rerank_candidate and rerank_top and int(rerank_top) > int(rerank_candidate):
        raise RuntimeConfigurationError("RERANK_TOP_K cannot exceed RERANK_CANDIDATE_K.")

    warnings = []
    for flag, description in sorted(EXPERIMENTAL_FLAGS.items()):
        if bool_value(flag, False):
            warnings.append(f"{flag}=true explicitly enables {description}.")
    return tuple(warnings)


def runtime_environment() -> str:
    return os.getenv("RUNTIME_ENV", "development").strip().lower()


def app_debug_enabled() -> bool:
    return bool_value("APP_DEBUG", False)
