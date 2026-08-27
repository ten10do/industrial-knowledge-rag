"""Production observability primitives: request IDs, structured logging,
low-cardinality in-process metrics.

Design constraints (V3.82):
 - zero research-semantics impact; additive-only runtime support;
 - privacy: log fields are an ALLOWLIST, never raw query/chunk/document text;
 - metrics labels are low cardinality (route templates / code classes /
   decision families), never per-request or per-chunk identifiers;
 - dependency-free (stdlib only).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{7,63}$")

_PROM_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

# Structured-log field allowlist. Anything else on the record is dropped.
_LOG_FIELD_ALLOWLIST = frozenset({
    "request_id", "endpoint", "runtime_mode", "status_code", "latency_ms",
    "error_code", "retrieval_count", "decision", "reason_family",
    "conversation_id", "history_turn_count", "retained_turn_count",
    "compressed_turn_count", "was_compressed", "query_rewrite_status",
    "standalone_query_length", "job_id", "task_type", "created_count",
    "check", "dependency", "version_id",
})
_MAX_LABEL_VALUE_CHARS = 64
_METRIC_LABEL_ALLOWLIST = frozenset({
    "endpoint", "code_class", "error_code", "mode", "decision",
    "reason_family", "dependency", "status",
})
_SAFE_EVENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def sanitize_label(value: str) -> str:
    text = str(value)
    if len(text) > _MAX_LABEL_VALUE_CHARS:
        text = text[:_MAX_LABEL_VALUE_CHARS]
    return re.sub(r"[^A-Za-z0-9_.\-/{}]", "_", text) or "other"


def new_request_id() -> str:
    return f"req-{time.time_ns():x}-{os.getpid():x}"[:63]


def request_id_from(value: str | None) -> str | None:
    if value and REQUEST_ID_PATTERN.fullmatch(value.strip()):
        return value.strip()
    return None


class MetricsRegistry:
    """Thread-safe counters and millisecond histograms with bounded labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            dict[str, float],
        ] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None):
        unexpected = set(labels or {}) - _METRIC_LABEL_ALLOWLIST
        if unexpected:
            raise ValueError("Metric labels must use the production low-cardinality allowlist.")
        frozen = tuple(sorted((k, sanitize_label(v)) for k, v in (labels or {}).items()))
        return name, frozen

    def inc(self, name: str, labels: dict[str, str] | None = None, amount: float = 1.0) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe_latency(self, name: str, ms: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            bucket = self._histograms.setdefault(key, {
                "count": 0, "sum": 0.0,
                **{f"le_{bound}": 0 for bound in _PROM_BUCKETS_MS},
                "le_inf": 0,
            })
            bucket["count"] += 1
            bucket["sum"] += max(0.0, float(ms))
            bucket["le_inf"] += 1
            for bound in _PROM_BUCKETS_MS:
                if ms <= bound:
                    bucket[f"le_{bound}"] += 1

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = float(value)

    def snapshot(self) -> dict:
        with self._lock:
            rendered: dict[str, list[dict]] = {}
            for (name, labels), value in sorted(self._counters.items()):
                rendered.setdefault(name, []).append(
                    {"labels": dict(labels), "value": value}
                )
            hist: dict[str, list[dict]] = {}
            for (name, labels), data in sorted(self._histograms.items()):
                hist.setdefault(name, []).append({"labels": dict(labels), **data})
            gauges: dict[str, list[dict]] = {}
            for (name, labels), value in sorted(self._gauges.items()):
                gauges.setdefault(name, []).append({"labels": dict(labels), "value": value})
            return {"counters": rendered, "histograms": hist, "gauges": gauges}

    def render_prometheus(self) -> str:
        lines: list[str] = []
        snapshot = self.snapshot()
        for name, entries in snapshot["gauges"].items():
            safe = sanitize_label(name)
            lines.append(f"# TYPE {safe} gauge")
            for entry in entries:
                lines.append(f"{safe}{_render_labels(entry['labels'])} {entry['value']}")
        for name, entries in snapshot["counters"].items():
            safe = sanitize_label(name)
            lines.append(f"# TYPE {safe} counter")
            for entry in entries:
                label_text = _render_labels(entry["labels"])
                lines.append(f"{safe}{label_text} {entry['value']}")
        for name, entries in snapshot["histograms"].items():
            safe = sanitize_label(name)
            lines.append(f"# TYPE {safe} histogram")
            for entry in entries:
                base_labels = entry["labels"]
                for bound in _PROM_BUCKETS_MS:
                    label_text = _render_labels(
                        base_labels, extra={"le": str(bound)}
                    )
                    lines.append(f"{safe}_bucket{label_text} {entry[f'le_{bound}']}")
                label_text = _render_labels(base_labels, extra={"le": "+Inf"})
                lines.append(f"{safe}_bucket{label_text} {entry['le_inf']}")
                label_text = _render_labels(base_labels)
                lines.append(f"{safe}_sum{label_text} {entry['sum']}")
                lines.append(f"{safe}_count{label_text} {entry['count']}")
        return "\n".join(lines) + "\n"


def _render_labels(labels: dict[str, str], extra: dict[str, str] | None = None) -> str:
    merged = {**labels, **(extra or {})}
    if not merged:
        return ""
    inner = ",".join(f'{sanitize_label(k)}="{sanitize_label(v)}"' for k, v in sorted(merged.items()))
    return "{" + inner + "}"


METRICS = MetricsRegistry()


class AllowlistJsonFormatter(logging.Formatter):
    """Single-line JSON formatter restricted to allowlisted fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = record.getMessage()
        if not _SAFE_EVENT_PATTERN.fullmatch(event):
            event = "redacted_event"
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
        }
        for field in sorted(_LOG_FIELD_ALLOWLIST):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            # Exception NAME only - never a traceback (privacy + safety).
            payload["exception"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int | None = None) -> None:
    default_format = "json" if os.getenv("RUNTIME_ENV", "development").strip().lower() == "production" else "text"
    format_mode = os.getenv("LOG_FORMAT", default_format).strip().lower()
    resolved_level = level if level is not None else logging.INFO
    handler_spec = None
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_dsh_observability", False):
            handler_spec = existing
    if handler_spec is None:
        import sys

        handler_spec = logging.StreamHandler(sys.stdout)
        setattr(handler_spec, "_dsh_observability", True)
        root.addHandler(handler_spec)
    if format_mode == "json":
        handler_spec.setFormatter(AllowlistJsonFormatter())
    else:
        handler_spec.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.setLevel(resolved_level)


class RequestContextLogger(logging.LoggerAdapter):
    """Attach request-scoped fields without touching call sites."""

    def __init__(self, logger: logging.Logger, context: dict):
        super().__init__(logger, context)

    def process(self, msg, kwargs):  # noqa: D401 - stdlib signature
        extra = dict(kwargs.get("extra") or {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def dependency_failure(dependency: str, error_code: str) -> None:
    METRICS.inc("dependency_failures_total", {"dependency": dependency, "error_code": error_code})
