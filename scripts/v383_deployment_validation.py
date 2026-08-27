"""Real-process V3.83 native deployment lifecycle validation.

This is intentionally a bounded, operator-invoked validation tool.  It uses
only synthetic public data, launches the documented production Uvicorn command,
and never touches the frozen private Chroma index.  Long soak execution stays
out of the normal pytest suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "v383_deployment_validation"
KB_ID = "kb-v383-deploy-00000001"
DRAFT_KB_ID = "kb-v383-draft-00000001"
INDEX_PATH = REPO_ROOT / "backend" / "light_indexes" / f"{KB_ID}.json"
DRAFT_INDEX_PATH = REPO_ROOT / "backend" / "light_indexes" / f"{DRAFT_KB_ID}.json"
DRAFT_DATA_PATH = REPO_ROOT / "backend" / "data" / DRAFT_KB_ID
ADMIN_TOKEN = "v383-synthetic-admin-token-00000001"
REQUEST_ID_HEADER = "X-Request-ID"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def response_header(response: dict, name: str) -> str | None:
    target = name.lower()
    return next(
        (value for key, value in response["headers"].items() if key.lower() == target),
        None,
    )


def request(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict | None = None,
    body: bytes | None = None,
    timeout: float = 15.0,
) -> dict:
    outgoing_headers = dict(headers or {})
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        outgoing_headers["Content-Type"] = "application/json"
    outgoing = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
        headers=outgoing_headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(outgoing, timeout=timeout) as response:
            response_body = response.read()
            status = response.status
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        status = exc.code
        response_headers = dict(exc.headers.items())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = response_body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    return {
        "status": status,
        "headers": response_headers,
        "text": text,
        "json": payload,
        "latency_ms": elapsed_ms,
    }


def port_is_closed(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


class Service:
    def __init__(self, port: int, label: str, *, keyless: bool):
        self.port = port
        self.label = label
        self.keyless = keyless
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.log_path = RESULTS_DIR / f"{label}.log"
        self.started_at = 0.0
        self.server_pid: int | None = None

    def start(self) -> dict:
        if not port_is_closed(self.port):
            raise RuntimeError(f"Port {self.port} is already in use.")
        environment = os.environ.copy()
        environment.update({
            "RUNTIME_ENV": "production",
            "APP_DEBUG": "false",
            "LOG_FORMAT": "json",
            "RAG_MODE": "light",
            "RETRIEVAL_MODE": "hybrid",
            "TASK_QUEUE_BACKEND": "memory",
            "RATE_LIMIT_BACKEND": "memory",
            "PUBLIC_KNOWLEDGE_BASE_ID": KB_ID,
            "FRONTEND_ORIGIN": "http://127.0.0.1:5173",
            "ADMIN_TOKEN": ADMIN_TOKEN,
            "RERANK_ENABLED": "false",
            "SECTION_EXPANSION_ENABLED": "false",
            "SUPPORT_GATE_ENABLED": "false",
            "TABLE_REGION_CONTEXT_ENABLED": "false",
            "CLAIM_SUPPORT_EXPERIMENT_ENABLED": "false",
            "HEALTH_RATE_LIMIT": "100000",
            "READY_RATE_LIMIT": "100000",
            "METRICS_RATE_LIMIT": "100000",
            "ASK_RATE_LIMIT": "100000",
            "UPLOAD_RATE_LIMIT": "1000",
            "JOB_STATUS_RATE_LIMIT": "100000",
        })
        if self.keyless:
            environment["GROQ_API_KEY"] = ""
            environment["DEEPSEEK_API_KEY"] = ""
        else:
            environment.pop("GROQ_API_KEY", None)
            environment.pop("DEEPSEEK_API_KEY", None)
        self.log_handle = self.log_path.open("wb")
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        self.started_at = time.perf_counter()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--no-access-log",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        live = self._wait_for("/live")
        ready = self._wait_for(
            "/ready",
            headers={"X-Knowledge-Base-ID": KB_ID},
            allowed_statuses={200, 503},
        )
        log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Started server process \[(\d+)\]", log_text)
        self.server_pid = int(match.group(1)) if match else self.process.pid
        return {
            "pid": self.server_pid,
            "live_ms": round(live["observed_ms"], 2),
            "ready_ms": round(ready["observed_ms"], 2),
            "live": live["response"]["json"],
            "ready_status_code": ready["response"]["status"],
            "ready": ready["response"]["json"],
        }

    def _wait_for(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        allowed_statuses: set[int] = frozenset({200}),
    ) -> dict:
        deadline = time.monotonic() + 30
        last_error = "no response"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Service {self.label} exited early with {self.process.returncode}."
                )
            try:
                response = request(self.port, "GET", path, headers=headers, timeout=1)
                if response["status"] in allowed_statuses:
                    return {
                        "observed_ms": (time.perf_counter() - self.started_at) * 1000,
                        "response": response,
                    }
                last_error = f"HTTP {response['status']}"
            except (OSError, TimeoutError) as exc:
                last_error = type(exc).__name__
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for {path}: {last_error}")

    def stop_gracefully(self) -> dict:
        if self.process is None:
            raise RuntimeError("Service not started.")
        started = time.perf_counter()
        fallback = False
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            fallback = True
            self.process.terminate()
            self.process.wait(timeout=10)
        duration_ms = (time.perf_counter() - started) * 1000
        self._close_log()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not port_is_closed(self.port):
            time.sleep(0.05)
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return {
            "exit_code": self.process.returncode,
            "duration_ms": round(duration_ms, 2),
            "port_released": port_is_closed(self.port),
            "used_terminate_fallback": fallback,
            "shutdown_log_complete": all(marker in text for marker in (
                "Waiting for application shutdown",
                "Application shutdown complete",
                "Finished server process",
            )),
        }

    def stop_abruptly(self) -> dict:
        if self.process is None:
            raise RuntimeError("Service not started.")
        started = time.perf_counter()
        os.kill(self.server_pid or self.process.pid, signal.SIGTERM)
        self.process.wait(timeout=10)
        self._close_log()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not port_is_closed(self.port):
            time.sleep(0.05)
        return {
            "exit_code": self.process.returncode,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "port_released": port_is_closed(self.port),
        }

    def _close_log(self) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def process_stats(service: Service) -> dict:
    process = service.process
    if process is None or process.poll() is not None:
        return {}
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    process_handle = kernel32.OpenProcess(
        0x0400 | 0x0010,
        False,
        service.server_pid or process.pid,
    )
    if not process_handle:
        return {}
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            counters.cb,
        ):
            return {}
        handle_count = wintypes.DWORD()
        kernel32.GetProcessHandleCount(
            process_handle,
            ctypes.byref(handle_count),
        )
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        kernel32.GetProcessTimes(
            process_handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
    finally:
        kernel32.CloseHandle(process_handle)

    def filetime_seconds(value) -> float:
        ticks = (value.dwHighDateTime << 32) | value.dwLowDateTime
        return ticks / 10_000_000

    return {
        "rss_mb": round(counters.WorkingSetSize / (1024 * 1024), 2),
        "peak_rss_mb": round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
        "handles": int(handle_count.value),
        "cpu_seconds": round(filetime_seconds(kernel) + filetime_seconds(user), 3),
    }


def public_headers(request_id: str | None = None, *, kb_id: str = KB_ID) -> dict[str, str]:
    headers = {"X-Knowledge-Base-ID": kb_id}
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return headers


def run_load(port: int) -> list[dict]:
    profiles = []
    for concurrency in (1, 2, 4, 8):
        count = 24

        def one(number: int) -> dict:
            path = "/ready" if number % 5 == 0 else "/live"
            return request(port, "GET", path, headers=public_headers(), timeout=5)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            responses = list(pool.map(one, range(count)))
        wall_seconds = time.perf_counter() - started
        latencies = [item["latency_ms"] for item in responses]
        success = sum(item["status"] == 200 for item in responses)
        profiles.append({
            "concurrency": concurrency,
            "requests": count,
            "success": success,
            "success_rate": round(success / count, 4),
            "throughput_rps": round(count / wall_seconds, 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(percentile(latencies, 0.95), 2),
            "errors": count - success,
        })
    return profiles


def run_soak(service: Service, seconds: int) -> dict:
    port = service.port
    abstain_probe = request(
        port,
        "POST",
        "/ask",
        headers=public_headers("req-v383-soak-abstain-probe"),
        json_body={
            "question": "Explain bananas in outer space.",
            "model_provider": "Groq",
        },
        timeout=15,
    )
    abstain_supported = (
        abstain_probe["status"] == 200
        and bool((abstain_probe["json"] or {}).get("is_refused"))
    )
    deadline = time.monotonic() + seconds
    lock = threading.Lock()
    latencies: list[float] = []
    errors: list[dict] = []
    route_counts: dict[str, int] = {}
    counter = 0

    def next_number() -> int:
        nonlocal counter
        with lock:
            counter += 1
            return counter

    def worker(worker_id: int) -> None:
        while time.monotonic() < deadline:
            number = next_number()
            if abstain_supported and number % 50 == 0:
                path = "/ask"
                response = request(
                    port,
                    "POST",
                    path,
                    headers=public_headers(f"req-v383-soak-{worker_id}-{number}"),
                    json_body={
                        "question": "Explain bananas in outer space.",
                        "model_provider": "Groq",
                    },
                    timeout=15,
                )
                expected = response["status"] == 200 and bool(
                    (response["json"] or {}).get("is_refused")
                )
            else:
                path = ("/live", "/ready", "/health", "/metrics")[number % 4]
                response = request(
                    port,
                    "GET",
                    path,
                    headers=public_headers(f"req-v383-soak-{worker_id}-{number}"),
                    timeout=10,
                )
                expected = response["status"] == 200
            with lock:
                latencies.append(response["latency_ms"])
                route_counts[path] = route_counts.get(path, 0) + 1
                if not expected and len(errors) < 20:
                    errors.append({"path": path, "status": response["status"]})
            time.sleep(0.2)

    initial_stats = process_stats(service)
    samples = [initial_stats] if initial_stats else []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, worker_id) for worker_id in range(4)]
        next_progress = started + 30
        while time.monotonic() < deadline:
            stats = process_stats(service)
            if stats:
                samples.append(stats)
            now = time.monotonic()
            if now >= next_progress:
                print(json.dumps({
                    "event": "soak_progress",
                    "elapsed_seconds": round(now - started),
                    "requests": len(latencies),
                    "errors": len(errors),
                    "rss_mb": stats.get("rss_mb"),
                }), flush=True)
                next_progress += 30
            time.sleep(min(2, max(0.05, deadline - now)))
        for future in futures:
            future.result()
    duration = time.monotonic() - started
    final_stats = process_stats(service)
    if final_stats:
        samples.append(final_stats)
    rss_values = [sample["rss_mb"] for sample in samples]
    handle_values = [sample["handles"] for sample in samples]
    cpu_values = [sample["cpu_seconds"] for sample in samples]
    rss_growth = (rss_values[-1] - rss_values[0]) if rss_values else None
    handle_growth = (handle_values[-1] - handle_values[0]) if handle_values else None
    memory_runaway = bool(
        rss_growth is not None
        and rss_growth > max(64.0, rss_values[0] * 0.5)
    )
    handle_leak = bool(handle_growth is not None and handle_growth > 64)
    return {
        "duration_seconds": round(duration, 2),
        "concurrency": 4,
        "requests": len(latencies),
        "success_rate": round((len(latencies) - len(errors)) / len(latencies), 4),
        "errors": len(errors),
        "error_samples": errors,
        "route_counts": route_counts,
        "abstain_in_mix": abstain_supported,
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "max_ms": round(max(latencies), 2),
        "rss_start_mb": rss_values[0] if rss_values else None,
        "rss_end_mb": rss_values[-1] if rss_values else None,
        "rss_peak_mb": max(rss_values) if rss_values else None,
        "rss_growth_mb": round(rss_growth, 2) if rss_growth is not None else None,
        "cpu_seconds": round(cpu_values[-1] - cpu_values[0], 3) if cpu_values else None,
        "handles_start": handle_values[0] if handle_values else None,
        "handles_end": handle_values[-1] if handle_values else None,
        "handles_peak": max(handle_values) if handle_values else None,
        "memory_runaway": memory_runaway,
        "handle_leak": handle_leak,
        "pass": not errors and not memory_runaway and not handle_leak,
    }


def multipart_pdf(pdf: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "----v383syntheticboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def create_synthetic_pdf() -> bytes | None:
    try:
        import pymupdf as fitz
    except ImportError:
        return None
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Synthetic V3.83 public upload. Pump pressure is 16 bar.",
    )
    value = document.tobytes()
    document.close()
    return value


def validate_upload_and_queue(port: int) -> dict:
    pdf = create_synthetic_pdf()
    if pdf is None:
        return {"status": "SKIPPED", "reason": "synthetic PDF generator unavailable"}
    body, content_type = multipart_pdf(pdf, "v383-synthetic.pdf")
    headers = {
        **public_headers("req-v383-upload-00000001", kb_id=DRAFT_KB_ID),
        "X-Admin-Token": ADMIN_TOKEN,
        "Idempotency-Key": "v383-upload-idempotency-00000001",
        "Content-Type": content_type,
    }
    submitted = request(port, "POST", "/upload", headers=headers, body=body, timeout=20)
    job_id = (submitted["json"] or {}).get("job_id")
    final = None
    if submitted["status"] == 202 and job_id:
        deadline = time.monotonic() + 30
        job_headers = {
            **public_headers(kb_id=DRAFT_KB_ID),
            "X-Admin-Token": ADMIN_TOKEN,
        }
        while time.monotonic() < deadline:
            final = request(port, "GET", f"/jobs/{job_id}", headers=job_headers)
            if (final["json"] or {}).get("status") in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
    duplicate = request(port, "POST", "/upload", headers=headers, body=body, timeout=20)
    invalid_body, invalid_type = multipart_pdf(b"not a PDF", "../unsafe.pdf")
    invalid_headers = dict(headers)
    invalid_headers.update({
        "Idempotency-Key": "v383-invalid-upload-00000001",
        "Content-Type": invalid_type,
        REQUEST_ID_HEADER: "req-v383-invalid-upload-00000001",
    })
    invalid = request(
        port,
        "POST",
        "/upload",
        headers=invalid_headers,
        body=invalid_body,
        timeout=20,
    )
    return {
        "status": "PASS" if all((
            submitted["status"] == 202,
            (final or {}).get("status") == 200,
            ((final or {}).get("json") or {}).get("status") == "succeeded",
            duplicate["status"] == 202,
            (duplicate["json"] or {}).get("job_id") == job_id,
            invalid["status"] == 400,
            (invalid["json"] or {}).get("error_code") == "INVALID_REQUEST",
        )) else "FAIL",
        "submit_status": submitted["status"],
        "job_id_present": bool(job_id),
        "job_final_status": ((final or {}).get("json") or {}).get("status"),
        "duplicate_same_job": (duplicate["json"] or {}).get("job_id") == job_id,
        "invalid_status": invalid["status"],
        "invalid_error_code": (invalid["json"] or {}).get("error_code"),
    }


def cleanup_synthetic_runtime() -> None:
    INDEX_PATH.unlink(missing_ok=True)
    DRAFT_INDEX_PATH.unlink(missing_ok=True)
    if DRAFT_DATA_PATH.is_dir():
        shutil.rmtree(DRAFT_DATA_PATH)


def run(port: int, soak_seconds: int) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() or DRAFT_INDEX_PATH.exists() or DRAFT_DATA_PATH.exists():
        raise RuntimeError("Synthetic V3.83 runtime state already exists; refusing to overwrite it.")
    payload = [{
        "page_content": (
            "Synthetic V3.83 public deployment document. The synthetic pump model "
            "V383-PUMP-A has a maximum working pressure of 16 bar."
        ),
        "metadata": {
            "source": "v383-synthetic-public.pdf",
            "chunk_id": "v383-synthetic-chunk-1",
            "document_id": "v383-synthetic-document-1",
            "manufacturer": "Synthetic",
            "equipment_type": "pump",
            "equipment_model": "V383-PUMP-A",
            "section": "Specifications",
            "knowledge_type": "parameter",
        },
    }]
    index_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_bytes(index_bytes)
    index_digest = sha256_bytes(index_bytes)
    report: dict = {
        "status": "RUNNING",
        "port": port,
        "startup_command": (
            f"{sys.executable} -m uvicorn backend.main:app --host 127.0.0.1 "
            f"--port {port} --no-access-log"
        ),
        "synthetic_index_digest_start": index_digest,
        "restart_cycles": [],
    }
    services: list[Service] = []
    try:
        first = Service(port, "cycle1_keyless_failure_recovery", keyless=True)
        services.append(first)
        first_start = first.start()
        report["restart_cycles"].append({"start": first_start})

        ready_initial = request(port, "GET", "/ready", headers=public_headers())
        INDEX_PATH.unlink()
        missing = request(
            port,
            "GET",
            "/ready",
            headers=public_headers("req-v383-index-missing-00000001"),
        )
        INDEX_PATH.write_bytes(index_bytes)
        recovered = request(port, "GET", "/ready", headers=public_headers())
        INDEX_PATH.write_text("not-json", encoding="utf-8")
        invalid = request(
            port,
            "GET",
            "/ready",
            headers=public_headers("req-v383-index-invalid-00000001"),
        )
        INDEX_PATH.write_bytes(index_bytes)
        invalid_recovered = request(port, "GET", "/ready", headers=public_headers())
        report["index_failure_recovery"] = {
            "initial": ready_initial["status"],
            "missing": missing["status"],
            "missing_error_code": (missing["json"] or {}).get("error_code"),
            "recovered": recovered["status"],
            "invalid": invalid["status"],
            "invalid_error_code": (invalid["json"] or {}).get("error_code"),
            "invalid_recovered": invalid_recovered["status"],
            "dynamic_recovery": recovered["status"] == invalid_recovered["status"] == 200,
        }

        trace_id = "req-v383-e2e-trace-00000001"
        abstain = request(
            port,
            "POST",
            "/ask",
            headers=public_headers(trace_id),
            json_body={
                "question": "Explain bananas in outer space.",
                "model_provider": "Groq",
            },
        )
        sentinel_secret = "V383_SECRET_SENTINEL_DO_NOT_LOG"
        sentinel_path = "C:/synthetic/private/V383_PATH_SENTINEL.pdf"
        private_marker = "V383_PRIVATE_QUERY_SENTINEL"
        invalid_request = request(
            port,
            "POST",
            "/ask",
            headers=public_headers("req-v383-invalid-00000001"),
            json_body={
                "question": f"{sentinel_secret} {sentinel_path} {private_marker} " + "x" * 5000,
                "model_provider": "Groq",
            },
        )
        provider_failure = request(
            port,
            "POST",
            "/ask",
            headers=public_headers("req-v383-provider-fail-00000001"),
            json_body={
                "question": "What is the maximum working pressure of V383-PUMP-A?",
                "model_provider": "Groq",
            },
            timeout=30,
        )
        concurrent_ids = [f"req-v383-concurrent-{number:08d}" for number in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            concurrent = list(pool.map(
                lambda rid: request(
                    port,
                    "GET",
                    "/ready",
                    headers=public_headers(rid),
                ),
                concurrent_ids,
            ))
        metrics = request(port, "GET", "/metrics")
        report["observability_cycle1"] = {
            "trace_id": trace_id,
            "trace_response_header_match": response_header(abstain, REQUEST_ID_HEADER) == trace_id,
            "abstain_status": abstain["status"],
            "abstain_decision": bool((abstain["json"] or {}).get("is_refused")),
            "invalid_status": invalid_request["status"],
            "invalid_error_code": (invalid_request["json"] or {}).get("error_code"),
            "invalid_request_id_match": (invalid_request["json"] or {}).get("request_id")
            == response_header(invalid_request, REQUEST_ID_HEADER),
            "provider_failure_status": provider_failure["status"],
            "provider_failure_error_contract": all(
                key in (provider_failure["json"] or {})
                for key in ("error_code", "message", "request_id", "retryable")
            ),
            "provider_failure_safe": all(marker not in provider_failure["text"] for marker in (
                sentinel_secret,
                sentinel_path,
                private_marker,
            )),
            "concurrent_response_ids_unique": len({
                response_header(item, REQUEST_ID_HEADER) for item in concurrent
            }) == 8,
            "concurrent_all_200": all(item["status"] == 200 for item in concurrent),
            "metrics_status": metrics["status"],
            "metrics_has_http": "http_requests_total" in metrics["text"],
            "metrics_has_error": "http_request_errors_total" in metrics["text"],
            "metrics_has_dependency_failure": "dependency_failures_total" in metrics["text"],
            "metrics_has_abstain": "rag_answers_total" in metrics["text"],
            "metrics_low_cardinality": not any(marker in metrics["text"] for marker in (
                "request_id=", "query=", "chunk_id=", "document_path=",
                trace_id, sentinel_secret, sentinel_path, private_marker,
            )),
        }
        first_stop = first.stop_gracefully()
        report["restart_cycles"][-1]["shutdown"] = first_stop
        first_log = first.log_path.read_text(encoding="utf-8", errors="replace")
        report["observability_cycle1"].update({
            "trace_in_structured_log": trace_id in first_log and '"event": "ask_outcome"' in first_log,
            "concurrent_ids_in_logs": all(rid in first_log for rid in concurrent_ids),
            "logging_privacy": not any(marker in first_log for marker in (
                sentinel_secret, sentinel_path, private_marker,
            )),
        })

        second = Service(port, "cycle2_provider_upload_queue", keyless=False)
        services.append(second)
        second_start = second.start()
        report["restart_cycles"].append({"start": second_start})
        provider_answer = request(
            port,
            "POST",
            "/ask",
            headers=public_headers("req-v383-provider-answer-00000001"),
            json_body={
                "question": "What is the maximum working pressure of V383-PUMP-A?",
                "model_provider": "DeepSeek",
            },
            timeout=60,
        )
        provider_metrics = request(port, "GET", "/metrics")
        report["provider_recovery"] = {
            "keyless_status": (ready_initial["json"] or {}).get("status"),
            "configured_status": (second_start["ready"] or {}).get("status"),
            "recovery_requires_restart": True,
            "answer_status": provider_answer["status"],
            "answer_is_refused": (provider_answer["json"] or {}).get("is_refused"),
            "real_provider_available": (
                provider_answer["status"] == 200
                and not bool((provider_answer["json"] or {}).get("is_refused"))
            ),
            "answer_metric_observed": 'decision="ANSWER"' in provider_metrics["text"],
        }
        report["provider_recovery"]["status"] = (
            "PASS" if report["provider_recovery"]["real_provider_available"]
            else "REAL_PROVIDER_UNAVAILABLE"
        )
        report["upload_queue"] = validate_upload_and_queue(port)
        report["restart_cycles"][-1]["shutdown"] = second.stop_gracefully()

        third = Service(port, "cycle3_load", keyless=False)
        services.append(third)
        third_start = third.start()
        report["restart_cycles"].append({"start": third_start})
        report["load"] = run_load(port)
        report["load_process_stats"] = process_stats(third)
        report["restart_cycles"][-1]["shutdown"] = third.stop_gracefully()

        abrupt = Service(port, "cycle4_abrupt", keyless=True)
        services.append(abrupt)
        abrupt_start = abrupt.start()
        report["abrupt_termination"] = {
            "start": abrupt_start,
            "termination": abrupt.stop_abruptly(),
            "index_digest_after_termination": sha256_bytes(INDEX_PATH.read_bytes()),
        }

        soak_service = Service(port, "cycle5_recovery_soak", keyless=True)
        services.append(soak_service)
        soak_start = soak_service.start()
        report["abrupt_termination"]["recovery_start"] = soak_start
        report["soak"] = run_soak(soak_service, soak_seconds)
        report["soak_metrics"] = request(port, "GET", "/metrics")["text"]
        report["restart_cycles"].append({
            "start": soak_start,
            "shutdown": soak_service.stop_gracefully(),
            "purpose": "abrupt-recovery-and-soak",
        })

        report["synthetic_index_digest_end"] = sha256_bytes(INDEX_PATH.read_bytes())
        report["index_persistence"] = (
            report["synthetic_index_digest_start"]
            == report["synthetic_index_digest_end"]
        )
        report["restart_success_rate"] = {
            "succeeded": sum(
                cycle["start"]["ready_status_code"] == 200
                and cycle["shutdown"]["port_released"]
                and cycle["shutdown"]["shutdown_log_complete"]
                and not cycle["shutdown"]["used_terminate_fallback"]
                for cycle in report["restart_cycles"][:3]
            ),
            "attempted": 3,
        }
        report["orphan_processes"] = sum(
            service.process is not None and service.process.poll() is None
            for service in services
        )
        report["status"] = "PASS" if all((
            report["index_failure_recovery"]["dynamic_recovery"],
            report["index_persistence"],
            report["restart_success_rate"] == {"succeeded": 3, "attempted": 3},
            report["abrupt_termination"]["termination"]["port_released"],
            report["soak"]["pass"],
            all(profile["success_rate"] == 1 for profile in report["load"]),
            report["observability_cycle1"]["logging_privacy"],
            report["observability_cycle1"]["trace_response_header_match"],
            report["observability_cycle1"]["trace_in_structured_log"],
            report["observability_cycle1"]["concurrent_response_ids_unique"],
            report["observability_cycle1"]["concurrent_ids_in_logs"],
            report["observability_cycle1"]["metrics_has_http"],
            report["observability_cycle1"]["metrics_has_error"],
            report["observability_cycle1"]["metrics_has_dependency_failure"],
            report["observability_cycle1"]["metrics_has_abstain"],
            report["observability_cycle1"]["metrics_low_cardinality"],
            report["observability_cycle1"]["invalid_error_code"] == "VALIDATION_ERROR",
            report["observability_cycle1"]["invalid_request_id_match"],
            report["observability_cycle1"]["provider_failure_error_contract"],
            report["upload_queue"]["status"] == "PASS",
            report["orphan_processes"] == 0,
        )) else "FAIL"
        return report
    finally:
        for service in services:
            if service.process is not None and service.process.poll() is None:
                service.process.terminate()
                service.process.wait(timeout=10)
            service._close_log()
        cleanup_synthetic_runtime()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8038)
    parser.add_argument("--soak-seconds", type=int, default=600)
    args = parser.parse_args()
    report = run(args.port, args.soak_seconds)
    output = RESULTS_DIR / "native_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = dict(report)
    summary.pop("soak_metrics", None)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
