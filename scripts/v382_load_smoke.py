"""Bounded concurrency/load and resource smoke; not a throughput benchmark."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROCESS_STARTED = time.perf_counter()
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.update({
    "RUNTIME_ENV": "test",
    "APP_DEBUG": "false",
    "LOG_FORMAT": "json",
    "RAG_MODE": "light",
    "TASK_QUEUE_BACKEND": "memory",
    "RATE_LIMIT_BACKEND": "memory",
})

from fastapi.testclient import TestClient  # noqa: E402
from backend import main  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


KB_ID = "kb-v382-load-smoke-00000001"


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        if os.name == "nt":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-Process -Id {os.getpid()}).WorkingSet64",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return round(int(result.stdout.strip()) / (1024 * 1024), 2)
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
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

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return round(counters.WorkingSetSize / (1024 * 1024), 2)
            return None
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
            return round(value / divisor, 2)
        except (ImportError, OSError):
            return None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def run() -> dict:
    index_path = Path(main.get_index_storage_path(KB_ID))
    if index_path.exists():
        raise RuntimeError("Synthetic load index already exists; refusing to overwrite it.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([{
        "page_content": "Synthetic load smoke document.",
        "metadata": {"source": "synthetic.pdf", "chunk_id": "load-1"},
    }]), encoding="utf-8")
    digest_before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    startup_ms = (time.perf_counter() - PROCESS_STARTED) * 1000
    start_rss = _rss_mb()
    rss_samples = [start_rss] if start_rss is not None else []
    index_started = time.perf_counter()
    if not main.reload_knowledge_base(KB_ID):
        raise RuntimeError("Synthetic load index could not be opened.")
    index_open_ms = (time.perf_counter() - index_started) * 1000
    rows = []
    errors = []
    try:
        with TestClient(main.app) as client:
            headers = {"X-Knowledge-Base-ID": KB_ID}

            def request_once(number: int) -> float:
                started = time.perf_counter()
                route = "/ready" if number % 5 == 0 else "/live"
                response = client.get(route, headers=headers)
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    errors.append({"route": route, "status": response.status_code})
                return elapsed

            for concurrency in (1, 2, 4, 8):
                count = 24
                wall_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    latencies = list(pool.map(request_once, range(count)))
                wall_ms = (time.perf_counter() - wall_started) * 1000
                rows.append({
                    "concurrency": concurrency,
                    "requests": count,
                    "success_rate": round((count - len(errors)) / count, 4),
                    "median_ms": round(statistics.median(latencies), 2),
                    "p95_ms": round(_percentile(latencies, 0.95), 2),
                    "wall_ms": round(wall_ms, 2),
                })
                sampled_rss = _rss_mb()
                if sampled_rss is not None:
                    rss_samples.append(sampled_rss)

            retrieval_latencies = []
            evidence_latencies = []
            e2e_latencies = []
            query = "What does the synthetic operational smoke document contain?"
            for _ in range(50):
                e2e_started = time.perf_counter()
                retrieval_started = time.perf_counter()
                result = main.retrieve_docs(
                    query,
                    knowledge_base_id=KB_ID,
                    retrieval_mode="hybrid",
                )
                retrieval_latencies.append((time.perf_counter() - retrieval_started) * 1000)
                evidence_started = time.perf_counter()
                main.analyze_evidence(query, result, "hybrid")
                evidence_latencies.append((time.perf_counter() - evidence_started) * 1000)
                e2e_latencies.append((time.perf_counter() - e2e_started) * 1000)
        digest_after = hashlib.sha256(index_path.read_bytes()).hexdigest()
        peak_rss = _rss_mb()
        if peak_rss is not None:
            rss_samples.append(peak_rss)
        return {
            "status": "PASS" if not errors and digest_before == digest_after else "FAIL",
            "profiles": rows,
            "errors": errors[:10],
            "startup_ms": round(startup_ms, 2),
            "startup_rss_mb": start_rss,
            "warm_rss_mb": peak_rss,
            "peak_observed_rss_mb": max(rss_samples) if rss_samples else None,
            "index_open_ms": round(index_open_ms, 2),
            "embedding_load_ms": None,
            "embedding_load_note": "not applicable in light mode",
            "warm_runtime_latency_ms": {
                "retrieval_median": round(statistics.median(retrieval_latencies), 2),
                "retrieval_p95": round(_percentile(retrieval_latencies, 0.95), 2),
                "evidence_median": round(statistics.median(evidence_latencies), 2),
                "evidence_p95": round(_percentile(evidence_latencies, 0.95), 2),
                "e2e_median": round(statistics.median(e2e_latencies), 2),
                "e2e_p95": round(_percentile(e2e_latencies, 0.95), 2),
            },
            "cpu_process_seconds": round(time.process_time(), 3),
            "index_size_bytes": index_path.stat().st_size,
            "shutdown_index_unchanged": digest_before == digest_after,
        }
    finally:
        index_path.unlink(missing_ok=True)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)
