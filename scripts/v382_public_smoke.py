"""Keyless, corpus-free operational smoke for local runs and public CI."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.update({
    "RUNTIME_ENV": "test",
    "APP_DEBUG": "false",
    "RAG_MODE": "light",
    "RETRIEVAL_MODE": "hybrid",
    "TASK_QUEUE_BACKEND": "memory",
    "RATE_LIMIT_BACKEND": "memory",
    "RERANK_ENABLED": "false",
    "SECTION_EXPANSION_ENABLED": "false",
    "SUPPORT_GATE_ENABLED": "false",
    "TABLE_REGION_CONTEXT_ENABLED": "false",
    "CLAIM_SUPPORT_EXPERIMENT_ENABLED": "false",
})

from fastapi.testclient import TestClient  # noqa: E402
from backend import main  # noqa: E402
from backend.observability import REQUEST_ID_HEADER  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


KB_ID = "kb-v382-public-smoke-00000001"


def main_smoke() -> dict:
    index_path = Path(main.get_index_storage_path(KB_ID))
    if index_path.exists():
        raise RuntimeError("Synthetic smoke index already exists; refusing to overwrite it.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{
        "page_content": "Synthetic public operational smoke document.",
        "metadata": {
            "source": "synthetic-public.pdf",
            "chunk_id": "synthetic-public-1",
            "document_id": "synthetic-public-document",
        },
    }]
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    digest_before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    headers = {
        "X-Knowledge-Base-ID": KB_ID,
        REQUEST_ID_HEADER: "req-v382-public-smoke-0001",
    }
    try:
        with TestClient(main.app) as client:
            live = client.get("/live", headers=headers)
            ready = client.get("/ready", headers=headers)
            metrics = client.get("/metrics", headers=headers)
            invalid = client.post("/ask", headers=headers, json={"question": ""})
        digest_after = hashlib.sha256(index_path.read_bytes()).hexdigest()
        checks = {
            "liveness_200": live.status_code == 200 and live.json().get("status") == "ok",
            "readiness_200": ready.status_code == 200 and ready.json().get("ready") is True,
            "metrics_200": metrics.status_code == 200 and "http_requests_total" in metrics.text,
            "safe_error_422": invalid.status_code == 422 and invalid.json().get("error_code") == "VALIDATION_ERROR",
            "request_id_round_trip": live.headers.get(REQUEST_ID_HEADER) == headers[REQUEST_ID_HEADER],
            "graceful_shutdown_index_unchanged": digest_before == digest_after,
            "experimental_defaults_off": not any((
                main.reranker.requested,
                main.support_gate_enabled(),
            )),
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "readiness_status": ready.json().get("status"),
        }
    finally:
        index_path.unlink(missing_ok=True)


if __name__ == "__main__":
    report = main_smoke()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)
