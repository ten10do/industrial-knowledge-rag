"""Mode-aware readiness evaluation.

Readiness answers: "can THIS process, in ITS current configuration, serve
queries right now?" - unlike liveness (/health) it MAY return 503.

Rules encoded here (V3.82):
 - the RAG query path is the only hard dependency unless the operator
   explicitly selects Redis-backed queueing or rate limiting;
 - a selected-but-broken optional infra failure is degraded, not dead;
 - index incompatibility NEVER triggers an automatic rebuild; it surfaces
   as an explicit readiness failure for operator action;
 - response details are safe by construction: no absolute local paths,
   no raw stack traces, no secrets.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

try:
    from .observability import sanitize_label
except ImportError:  # direct-script context
    from observability import sanitize_label  # type: ignore
try:
    from .index_integrity import (
        FULL_EMBEDDING_MODEL,
        validate_full_index,
        validate_light_index,
    )
except ImportError:  # direct-script context
    from index_integrity import (  # type: ignore
        FULL_EMBEDDING_MODEL,
        validate_full_index,
        validate_light_index,
    )

BACKEND_ENV_VARS = {
    "queue": "TASK_QUEUE_BACKEND",
    "rate_limit": "RATE_LIMIT_BACKEND",
}


def _selected_backends() -> dict[str, str]:
    return {
        capability: os.getenv(var, "").strip().lower() or "memory"
        for capability, var in BACKEND_ENV_VARS.items()
    }


def _redis_reachable(redis_url: str) -> tuple[bool, str]:
    try:
        from redis import Redis

        client = Redis.from_url(redis_url, socket_connect_timeout=1.5)
        client.ping()
        return True, "ping_ok"
    except ImportError:
        return False, "redis_dependency_missing"
    except Exception as exc:
        return False, f"ping_failed:{type(exc).__name__}"


def _embedding_available() -> tuple[bool, str]:
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is None:
        return False, "sentence_transformers_missing"
    configured_override = os.getenv("EMBEDDING_MODEL_NAME", "").strip()
    if configured_override and configured_override != FULL_EMBEDDING_MODEL:
        return False, "embedding_model_identity_mismatch"
    return True, "importable"


def evaluate_readiness(
    *,
    effective_rag_mode: str,
    knowledge_base_id: str,
    get_index_storage_path,
    task_queue_health=None,
) -> dict:
    checks: list[dict] = []
    warnings: list[str] = []

    def add(name: str, *, required: bool, ok: bool, detail: str) -> None:
        checks.append({
            "check": name,
            "required": required,
            "ok": bool(ok),
            "detail": sanitize_label(detail),
        })

    rag_mode = effective_rag_mode
    add("rag_mode_known", required=True, ok=rag_mode in {"full", "light"}, detail=f"mode={rag_mode}")

    if rag_mode == "full":
        report = validate_full_index(Path(get_index_storage_path(knowledge_base_id)))
    else:
        report = validate_light_index(Path(get_index_storage_path(knowledge_base_id)))
    for entry in report.as_dict()["checks"]:
        add(f"index_{entry['check']}", required=True, ok=entry["ok"], detail=entry["detail"])
    warnings.extend(report.warnings)

    if rag_mode == "full":
        embed_ok, embed_detail = _embedding_available()
        add("embedding_stack", required=True, ok=embed_ok, detail=embed_detail)

    configured_providers = {
        "groq": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "deepseek": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
    }
    add(
        "llm_provider",
        required=False,
        ok=any(configured_providers.values()),
        detail="configured" if any(configured_providers.values()) else "no_provider_configured",
    )

    for capability, backend in _selected_backends().items():
        if backend != "redis":
            continue
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            add(f"{capability}_redis", required=True, ok=False, detail="redis_selected_but_url_missing")
            continue
        reachable, ping_detail = _redis_reachable(redis_url)
        add(f"{capability}_redis", required=True, ok=reachable, detail=ping_detail)

    queue_backend = _selected_backends()["queue"]
    if task_queue_health is not None:
        try:
            queue_health = task_queue_health()
            queue_ok = bool(queue_health.get("healthy"))
            detail = "worker_available" if queue_ok else "worker_unavailable"
        except Exception as exc:
            queue_ok = False
            detail = f"health_failed:{type(exc).__name__}"
        add(
            "task_worker",
            required=queue_backend == "redis",
            ok=queue_ok,
            detail=detail,
        )

    any_required_fail = any(
        (not check["ok"]) and check["required"] for check in checks
    )
    optional_warn = [
        check["check"] for check in checks if not check["required"] and not check["ok"]
    ]
    if any_required_fail:
        status = "unavailable"
    elif optional_warn or warnings:
        status = "degraded"
    else:
        status = "ok"

    return {
        "ready": not any_required_fail,
        "status": status,
        "runtime_mode": rag_mode,
        "checks": checks,
        "warnings": warnings[:10],
        "degraded": status == "degraded",
    }
