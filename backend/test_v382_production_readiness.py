from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.index_integrity import IndexIntegrityReport, validate_light_index
from backend.observability import (
    AllowlistJsonFormatter,
    MetricsRegistry,
    REQUEST_ID_HEADER,
    request_id_from,
)
from backend.readiness import evaluate_readiness
from backend.runtime_config import (
    EXPERIMENTAL_FLAGS,
    RuntimeConfigurationError,
    validate_runtime_environment,
)
from backend.security import FixedWindowRateLimiter
from backend.v382_release_guard import (
    _matches_frozen_file,
    audit_tracked_private_files,
    verify_research_freeze,
)
from backend.task_queue import MemoryTaskQueue


def _write_light_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{
            "page_content": "synthetic public readiness text",
            "metadata": {"source": "synthetic.pdf", "chunk_id": "synthetic-1"},
        }]),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_MODE", "automatic"),
        ("MAX_UPLOAD_FILES", "many"),
        ("MAX_UPLOAD_FILES", "0"),
        ("RERANK_ENABLED", "sometimes"),
        ("FRONTEND_ORIGIN", "localhost:5173"),
    ],
)
def test_runtime_configuration_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeConfigurationError, match=name):
        validate_runtime_environment(load_environment=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        # positive floats: NaN, +Infinity, -Infinity (all three spellings)
        ("LIGHT_MAX_RELEVANT_DISTANCE", "NaN"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "Infinity"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "-Infinity"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "nan"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "inf"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "-inf"),
        ("FULL_MAX_RELEVANT_DISTANCE", "NaN"),
        ("EVIDENCE_MAX_VECTOR_DISTANCE", "Infinity"),
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "-Infinity"),
        # contract-branch ratio is optional but must be finite when set
        ("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", "NaN"),
        ("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", "Infinity"),
        ("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", "-Infinity"),
        # non-negative floats
        ("EVIDENCE_MIN_VECTOR_MARGIN", "NaN"),
        ("EVIDENCE_MIN_VECTOR_MARGIN", "Infinity"),
        ("EVIDENCE_MIN_VECTOR_MARGIN", "-Infinity"),
    ],
)
def test_runtime_configuration_rejects_non_finite_floats(monkeypatch, name, value):
    """NaN / +/-Infinity must never reach retrieval or evidence thresholds."""
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeConfigurationError, match="finite"):
        validate_runtime_environment(load_environment=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIGHT_MAX_RELEVANT_DISTANCE", "0.5"),
        ("LIGHT_MAX_RELEVANT_DISTANCE", "1.2"),
        ("EVIDENCE_MIN_VECTOR_MARGIN", "0"),
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "30.0"),
        ("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", "1.20"),
        ("EVIDENCE_CONTRACT_MAX_DISTANCE_RATIO", ""),
    ],
)
def test_runtime_configuration_accepts_finite_floats(monkeypatch, name, value):
    """Finite values keep the existing behavior; empty optional stays valid."""
    monkeypatch.setenv(name, value)
    validate_runtime_environment(load_environment=False)


def test_runtime_configuration_rejects_missing_redis_url(monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "redis")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeConfigurationError, match="REDIS_URL"):
        validate_runtime_environment(load_environment=False)


def test_runtime_configuration_rejects_production_debug(monkeypatch):
    monkeypatch.setenv("RUNTIME_ENV", "production")
    monkeypatch.setenv("APP_DEBUG", "true")
    with pytest.raises(RuntimeConfigurationError, match="APP_DEBUG"):
        validate_runtime_environment(load_environment=False)


def test_runtime_configuration_rejects_storage_file(monkeypatch, tmp_path):
    configured = tmp_path / "not-a-directory"
    configured.write_text("x", encoding="utf-8")
    monkeypatch.setenv("PUBLIC_VERSION_STORAGE_DIR", str(configured))
    with pytest.raises(RuntimeConfigurationError, match="PUBLIC_VERSION_STORAGE_DIR"):
        validate_runtime_environment(load_environment=False)


def test_experimental_feature_flags_default_off(monkeypatch):
    for flag in EXPERIMENTAL_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    assert validate_runtime_environment(load_environment=False) == ()


def test_explicit_experimental_feature_is_reported(monkeypatch):
    for flag in EXPERIMENTAL_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("SUPPORT_GATE_ENABLED", "true")
    assert "not production-ready" in validate_runtime_environment(load_environment=False)[0]


def test_liveness_is_process_only_even_if_dependency_probe_would_fail():
    with patch.object(main, "get_knowledge_base_status", side_effect=RuntimeError("private path")):
        response = TestClient(main.app).get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "private path" not in response.text


def test_ready_enforces_rate_limit_contract(monkeypatch):
    """READY_RATE_LIMIT must actually gate /ready (config contract).

    Regression: /ready previously skipped enforce_rate_limit(), so the
    configured bucket was dead config and repeated probes could replay the
    full light-index read/parse. With READY_RATE_LIMIT=1 the second request
    from the same client+knowledge base must be rejected with 429, proving
    the limit is enforced before any readiness evaluation.
    """
    fresh_limiter = FixedWindowRateLimiter()
    monkeypatch.setattr(main, "rate_limiter", fresh_limiter)
    monkeypatch.setitem(main.RATE_LIMITS, "ready", (1, 60))
    knowledge_base_id = "kb-rate-limit-ready-00000001"
    client = TestClient(main.app)

    first = client.get("/ready", headers={"X-Knowledge-Base-ID": knowledge_base_id})
    # dependency state is irrelevant: readiness may be 200 or 503, but the
    # request must consume one token and return the governance headers.
    # httpx normalizes header names to lowercase, so match case-insensitively.
    assert first.status_code in (200, 503)
    assert any(key.lower().startswith("x-ratelimit") for key in first.headers)

    second = client.get("/ready", headers={"X-Knowledge-Base-ID": knowledge_base_id})
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert any(key.lower().startswith("x-ratelimit") for key in second.headers)


def test_readiness_fails_for_missing_index_without_leaking_absolute_path(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    missing = tmp_path / "private" / "secret-index.json"
    report = evaluate_readiness(
        effective_rag_mode="light",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: missing,
        task_queue_health=lambda: {"healthy": True},
    )
    assert report["ready"] is False
    assert report["status"] == "unavailable"
    assert str(tmp_path) not in json.dumps(report)


def test_readiness_http_503_uses_safe_error_contract_and_dependency_metric():
    knowledge_base_id = "kb-v383-missing-00000001"
    index_path = Path(main.get_index_storage_path(knowledge_base_id))
    assert not index_path.exists()
    before = main.METRICS.snapshot()["counters"].get(
        "dependency_failures_total",
        [],
    )
    before_total = sum(entry["value"] for entry in before)

    response = TestClient(main.app).get(
        "/ready",
        headers={
            "X-Knowledge-Base-ID": knowledge_base_id,
            REQUEST_ID_HEADER: "req-v383-missing-00000001",
        },
    )

    payload = response.json()
    assert response.status_code == 503
    assert payload["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert payload["request_id"] == "req-v383-missing-00000001"
    assert payload["retryable"] is True
    assert payload["message"] == payload["detail"]
    assert str(index_path.parent) not in response.text
    after = main.METRICS.snapshot()["counters"]["dependency_failures_total"]
    assert sum(entry["value"] for entry in after) == before_total + 1
    assert any(entry["labels"].get("dependency") == "index" for entry in after)


def test_readiness_is_degraded_not_failed_when_optional_llm_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    index = tmp_path / "index.json"
    _write_light_index(index)
    report = evaluate_readiness(
        effective_rag_mode="light",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: index,
        task_queue_health=lambda: {"healthy": True},
    )
    assert report["ready"] is True
    assert report["status"] == "degraded"
    assert any(check["check"] == "llm_provider" and not check["required"] for check in report["checks"])


def test_readiness_requires_selected_redis_worker(monkeypatch, tmp_path):
    index = tmp_path / "index.json"
    _write_light_index(index)
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr("backend.readiness._redis_reachable", lambda _url: (True, "ping_ok"))
    report = evaluate_readiness(
        effective_rag_mode="light",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: index,
        task_queue_health=lambda: {"healthy": False},
    )
    assert report["ready"] is False
    assert any(check["check"] == "task_worker" and check["required"] for check in report["checks"])


def test_readiness_fails_when_selected_redis_is_unavailable(monkeypatch, tmp_path):
    index = tmp_path / "index.json"
    _write_light_index(index)
    monkeypatch.setenv("RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr("backend.readiness._redis_reachable", lambda _url: (False, "ping_failed:TimeoutError"))
    report = evaluate_readiness(
        effective_rag_mode="light",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: index,
        task_queue_health=lambda: {"healthy": True},
    )
    assert report["ready"] is False
    assert any(check["check"] == "rate_limit_redis" and not check["ok"] for check in report["checks"])


def test_embedding_failure_is_required_in_full_mode(monkeypatch):
    report = IndexIntegrityReport(ok=True, index_kind="full")
    report.add("index_exists", True, "present")
    monkeypatch.setattr("backend.readiness.validate_full_index", lambda _path: report)
    monkeypatch.setattr("backend.readiness._embedding_available", lambda: (False, "init_failed"))
    result = evaluate_readiness(
        effective_rag_mode="full",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: Path("unused"),
        task_queue_health=lambda: {"healthy": True},
    )
    assert result["ready"] is False
    assert any(check["check"] == "embedding_stack" and not check["ok"] for check in result["checks"])


def test_vector_store_unavailable_fails_full_readiness(monkeypatch):
    report = IndexIntegrityReport(ok=False, index_kind="full")
    report.add("index_openable", False, "index_unreadable:ConnectionError")
    monkeypatch.setattr("backend.readiness.validate_full_index", lambda _path: report)
    monkeypatch.setattr("backend.readiness._embedding_available", lambda: (True, "importable"))
    result = evaluate_readiness(
        effective_rag_mode="full",
        knowledge_base_id="kb-v382-test-00000001",
        get_index_storage_path=lambda _kb: Path("unused"),
        task_queue_health=lambda: {"healthy": True},
    )
    assert result["ready"] is False
    assert any(check["check"] == "index_index_openable" and not check["ok"] for check in result["checks"])


def test_invalid_light_index_fails_integrity(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("not-json", encoding="utf-8")
    report = validate_light_index(path).as_dict()
    assert report["ok"] is False
    assert "not-json" not in json.dumps(report)


def test_request_id_validation_and_response_contract():
    assert request_id_from("req-valid-12345678") == "req-valid-12345678"
    assert request_id_from("unsafe query text") is None
    client = TestClient(main.app)
    response = client.get("/live", headers={REQUEST_ID_HEADER: "req-valid-12345678"})
    assert response.headers[REQUEST_ID_HEADER] == "req-valid-12345678"


def test_validation_error_is_structured_and_does_not_echo_input():
    private_input = "TOP-SECRET-QUERY-" + "x" * 5000
    response = TestClient(main.app).post(
        "/ask",
        headers={"X-Knowledge-Base-ID": main.PUBLIC_KNOWLEDGE_BASE_ID},
        json={"question": private_input},
    )
    payload = response.json()
    assert response.status_code == 422
    assert payload["error_code"] == "VALIDATION_ERROR"
    assert payload["retryable"] is False
    assert payload["request_id"]
    assert private_input not in response.text


def test_invalid_provider_config_is_safe_validation_error():
    response = TestClient(main.app).post(
        "/ask",
        headers={"X-Knowledge-Base-ID": main.PUBLIC_KNOWLEDGE_BASE_ID},
        json={"question": "synthetic question", "model_provider": "unknown-provider"},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
    assert "unknown-provider" not in response.text


def test_llm_provider_failure_returns_safe_error_with_request_id():
    docs = [(
        type("Doc", (), {
            "page_content": "synthetic content",
            "metadata": {"source": "synthetic.pdf", "page": 0},
        })(),
        0.1,
    )]
    with patch.object(main, "retrieve_docs", return_value=docs):
        with patch.object(main, "has_relevant_docs", return_value=True):
            with patch.object(main, "generate_answer", side_effect=ConnectionError("provider secret detail")):
                response = TestClient(main.app).post(
                    "/ask",
                    headers={"X-Knowledge-Base-ID": main.PUBLIC_KNOWLEDGE_BASE_ID},
                    json={"question": "synthetic question", "model_provider": "Groq"},
                )
    assert response.status_code == 500
    assert response.json()["error_code"] == "HTTP_500"
    assert response.json()["request_id"]
    assert "provider secret detail" not in response.text


def test_request_id_is_forwarded_as_background_trace_id():
    client = TestClient(main.app)
    record = {
        "job_id": "job-" + "1" * 32,
        "task_type": "build_draft",
        "status": "pending",
        "progress": 0,
        "message": "pending",
        "trace_id": "req-v382-trace-12345678",
    }
    headers = {
        "X-Knowledge-Base-ID": "kb-v382-draft-00000001",
        "X-Admin-Token": "replace-with-a-long-random-secret",
        "Idempotency-Key": "v382-upload-smoke-00000001",
        REQUEST_ID_HEADER: "req-v382-trace-12345678",
    }
    with patch.object(main, "require_admin_token", return_value=None):
        with patch.object(main, "prepare_draft_task_input", return_value={}):
            with patch.object(main.task_queue, "submit", return_value=(record, True)) as submit:
                response = client.post(
                    "/upload",
                    headers=headers,
                    files={"files": ("safe.pdf", b"%PDF-safe", "application/pdf")},
                )
    assert response.status_code == 202
    assert submit.call_args.kwargs["trace_id"] == response.headers[REQUEST_ID_HEADER]


def test_structured_logging_redacts_message_and_unlisted_fields():
    formatter = AllowlistJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw private query text",
        args=(),
        exc_info=None,
    )
    record.raw_query = "do-not-log-this"
    payload = json.loads(formatter.format(record))
    assert payload["event"] == "redacted_event"
    assert "raw_query" not in payload
    assert "do-not-log-this" not in json.dumps(payload)


def test_metrics_reject_high_cardinality_or_private_labels():
    registry = MetricsRegistry()
    with pytest.raises(ValueError, match="allowlist"):
        registry.inc("requests_total", {"request_id": "req-12345678"})
    with pytest.raises(ValueError, match="allowlist"):
        registry.inc("requests_total", {"query": "private query"})


def test_metrics_endpoint_contains_operational_metrics_without_request_ids():
    client = TestClient(main.app)
    client.get("/live")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "queue_depth" in response.text
    assert "request_id=" not in response.text


def test_upload_filename_and_magic_validation_are_safe():
    assert main.sanitize_pdf_filename("../../safe.pdf") == "safe.pdf"
    with TemporaryDirectory() as temp_dir:
        from starlette.datastructures import UploadFile

        path = Path(temp_dir)
        upload = UploadFile(filename="bad.pdf", file=__import__("io").BytesIO(b"not a pdf"))
        with pytest.raises(ValueError, match="有效的 PDF"):
            main.save_validated_uploads([upload], path)


def test_research_freeze_and_private_artifact_guards_pass():
    freeze = verify_research_freeze()
    private = audit_tracked_private_files()
    assert freeze["status"] == "PASS"
    assert freeze["baseline"]["correct"] == 54
    assert private == {"status": "PASS", "tracked_private_files": 0, "paths": []}


def test_research_freeze_accepts_only_line_ending_differences(tmp_path):
    path = tmp_path / "frozen.py"
    path.write_bytes(b"first\r\nsecond\r\n")
    frozen = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"first\nsecond\n")
    with patch("backend.v382_release_guard.subprocess.run", return_value=frozen):
        assert _matches_frozen_file(path, "frozen.py", "not-the-raw-hash")

    frozen.stdout = b"first\nchanged\n"
    with patch("backend.v382_release_guard.subprocess.run", return_value=frozen):
        assert not _matches_frozen_file(path, "frozen.py", "not-the-raw-hash")


def test_memory_worker_lifecycle_shuts_down_and_restarts_cleanly():
    queue = MemoryTaskQueue(lambda *_args: {}, max_workers=1)
    queue.close()
    assert queue.executor is None
    queue.start()
    assert queue.executor is not None
    queue.close()
