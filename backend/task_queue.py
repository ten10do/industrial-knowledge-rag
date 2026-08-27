import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock, RLock
from uuid import uuid4


JOB_STATUS_VALUES = {"pending", "running", "succeeded", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job_id() -> str:
    return f"job-{uuid4().hex}"


def create_trace_id() -> str:
    return f"trace-{uuid4().hex}"


def duration_seconds(started_at: str, finished_at: str) -> float | None:
    if not started_at or not finished_at:
        return None
    return round(
        (
            datetime.fromisoformat(finished_at)
            - datetime.fromisoformat(started_at)
        ).total_seconds(),
        3,
    )


def task_metrics(records: list[dict]) -> dict:
    status_counts = {
        status: sum(1 for record in records if record["status"] == status)
        for status in JOB_STATUS_VALUES
    }
    durations = sorted(
        float(record["duration_seconds"])
        for record in records
        if record.get("duration_seconds") is not None
    )
    p95 = (
        durations[max(0, math.ceil(len(durations) * 0.95) - 1)]
        if durations
        else None
    )
    return {
        "total": len(records),
        "status_counts": status_counts,
        "average_duration_seconds": (
            round(sum(durations) / len(durations), 3) if durations else None
        ),
        "p95_duration_seconds": p95,
    }


class MemoryTaskQueue:
    backend_name = "memory"

    def __init__(self, runner, max_workers: int = 2):
        self.runner = runner
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="knowledge-task",
        )
        self.records = {}
        self.idempotency = {}
        self.futures = {}
        self.state_lock = RLock()
        self.task_locks = {}

    def start(self) -> None:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="knowledge-task",
            )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def _copy(self, record):
        return {
            **record,
            "payload": dict(record.get("payload") or {}),
            "result": (
                dict(record["result"])
                if isinstance(record.get("result"), dict)
                else record.get("result")
            ),
        }

    def submit(
        self,
        task_type: str,
        payload: dict,
        *,
        scope: str,
        idempotency_key: str,
        job_id: str | None = None,
        attempt: int = 1,
        retry_of: str = "",
        trace_id: str | None = None,
    ) -> tuple[dict, bool]:
        idempotency_scope = (scope, task_type, idempotency_key)
        with self.state_lock:
            existing_id = self.idempotency.get(idempotency_scope)
            if existing_id and existing_id in self.records:
                return self._copy(self.records[existing_id]), False

            job_id = job_id or create_job_id()
            now = utc_now()
            record = {
                "job_id": job_id,
                "task_type": task_type,
                "status": "pending",
                "progress": 0,
                "message": "任务等待执行。",
                "stage": "queued",
                "failed_stage": "",
                "error": "",
                "result": None,
                "payload": dict(payload),
                "scope": scope,
                "attempt": attempt,
                "retry_of": retry_of,
                "trace_id": trace_id or create_trace_id(),
                "created_at": now,
                "updated_at": now,
                "started_at": "",
                "finished_at": "",
                "duration_seconds": None,
            }
            self.records[job_id] = record
            self.idempotency[idempotency_scope] = job_id
            self.futures[job_id] = self.executor.submit(
                self._execute,
                job_id,
            )
            return self._copy(record), True

    def _update(self, job_id: str, **values) -> None:
        with self.state_lock:
            record = self.records[job_id]
            record.update(values)
            record["updated_at"] = utc_now()

    def _execute(self, job_id: str) -> None:
        with self.state_lock:
            record = self._copy(self.records[job_id])
        self._update(
            job_id,
            status="running",
            message="任务开始执行。",
            stage="starting",
            started_at=utc_now(),
        )

        def report(
            progress: int,
            message: str,
            stage: str | None = None,
        ) -> None:
            values = {
                "progress": max(0, min(99, int(progress))),
                "message": message,
            }
            if stage:
                values["stage"] = stage
            self._update(
                job_id,
                **values,
            )

        try:
            result = self.runner(
                record["task_type"],
                record["payload"],
                report,
            )
            finished_at = utc_now()
            current = self.get(job_id)
            self._update(
                job_id,
                status="succeeded",
                progress=100,
                message="任务执行完成。",
                stage="completed",
                result=result,
                finished_at=finished_at,
                duration_seconds=duration_seconds(
                    current["started_at"],
                    finished_at,
                ),
            )
        except Exception as exc:
            finished_at = utc_now()
            current = self.get(job_id)
            self._update(
                job_id,
                status="failed",
                message="任务执行失败。",
                failed_stage=current.get("stage", ""),
                error=str(exc) or exc.__class__.__name__,
                finished_at=finished_at,
                duration_seconds=duration_seconds(
                    current["started_at"],
                    finished_at,
                ),
            )

    def get(self, job_id: str) -> dict:
        with self.state_lock:
            if job_id not in self.records:
                raise ValueError(f"任务 {job_id} 不存在或已过期。")
            return self._copy(self.records[job_id])

    def wait(self, job_id: str, timeout: float | None = None) -> dict:
        with self.state_lock:
            future = self.futures.get(job_id)
        if future:
            future.result(timeout=timeout)
        return self.get(job_id)

    def list(self, *, scope: str, limit: int = 50) -> list[dict]:
        with self.state_lock:
            records = [
                self._copy(record)
                for record in self.records.values()
                if record["scope"] == scope
            ]
        records.reverse()
        return sorted(
            records,
            key=lambda record: record["created_at"],
            reverse=True,
        )[:limit]

    def retry(
        self,
        job_id: str,
        *,
        idempotency_key: str,
    ) -> tuple[dict, bool]:
        original = self.get(job_id)
        if original["status"] != "failed":
            raise ValueError("只有失败任务可以重试。")
        return self.submit(
            original["task_type"],
            original["payload"],
            scope=original["scope"],
            idempotency_key=idempotency_key,
            attempt=int(original.get("attempt", 1)) + 1,
            retry_of=job_id,
            trace_id=original["trace_id"],
        )

    def metrics(self, *, scope: str) -> dict:
        return task_metrics(self.list(scope=scope, limit=1000))

    def health(self) -> dict:
        with self.state_lock:
            running = sum(
                1
                for record in self.records.values()
                if record["status"] == "running"
            )
        return {
            "backend": self.backend_name,
            "healthy": True,
            "worker_count": self.max_workers,
            "running_jobs": running,
        }

    def publish_event(self, _channel: str, _value: str) -> None:
        return None

    @contextmanager
    def lock(self, key: str):
        with self.state_lock:
            lock = self.task_locks.setdefault(key, Lock())
        with lock:
            yield


class RedisTaskQueue:
    backend_name = "redis"

    def __init__(
        self,
        redis_url: str,
        *,
        queue_name: str = "knowledge",
        job_timeout: int = 1800,
        retention_seconds: int = 86400,
        redis_client=None,
        queue=None,
    ):
        if not redis_url:
            raise RuntimeError("REDIS_URL 不能为空。")
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:
            raise RuntimeError("Redis 任务队列需要安装 redis 和 rq。") from exc

        self.redis = redis_client or Redis.from_url(redis_url)
        self.queue = queue or Queue(queue_name, connection=self.redis)
        self.queue_name = queue_name
        self.job_timeout = job_timeout
        self.retention_seconds = retention_seconds

    def _idempotency_key(
        self,
        scope: str,
        task_type: str,
        idempotency_key: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{scope}:{task_type}:{idempotency_key}".encode("utf-8")
        ).hexdigest()
        return f"industrial-knowledge-rag:task:idempotency:{digest}"

    def submit(
        self,
        task_type: str,
        payload: dict,
        *,
        scope: str,
        idempotency_key: str,
        job_id: str | None = None,
        attempt: int = 1,
        retry_of: str = "",
        trace_id: str | None = None,
    ) -> tuple[dict, bool]:
        job_id = job_id or create_job_id()
        mapping_key = self._idempotency_key(
            scope,
            task_type,
            idempotency_key,
        )
        created = self.redis.set(
            mapping_key,
            job_id,
            nx=True,
            ex=self.retention_seconds,
        )
        if not created:
            existing = self.redis.get(mapping_key)
            existing_id = (
                existing.decode("utf-8")
                if isinstance(existing, bytes)
                else str(existing)
            )
            try:
                return self.get(existing_id), False
            except ValueError:
                self.redis.delete(mapping_key)
                return self.submit(
                    task_type,
                    payload,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    job_id=job_id,
                    attempt=attempt,
                    retry_of=retry_of,
                    trace_id=trace_id,
                )

        now = utc_now()
        try:
            job = self.queue.enqueue(
                "backend.knowledge_tasks.execute_job",
                task_type,
                dict(payload),
                job_id=job_id,
                job_timeout=self.job_timeout,
                result_ttl=self.retention_seconds,
                failure_ttl=self.retention_seconds,
            )
            job.meta.update(
                {
                    "task_type": task_type,
                    "progress": 0,
                    "message": "任务等待执行。",
                    "stage": "queued",
                    "failed_stage": "",
                    "error": "",
                    "scope": scope,
                    "payload": dict(payload),
                    "attempt": attempt,
                    "retry_of": retry_of,
                    "trace_id": trace_id or create_trace_id(),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            job.save_meta()
            self.redis.zadd(
                self._scope_jobs_key(scope),
                {job_id: datetime.now(timezone.utc).timestamp()},
            )
            self.redis.expire(
                self._scope_jobs_key(scope),
                self.retention_seconds,
            )
            return self.get(job_id), True
        except Exception:
            self.redis.delete(mapping_key)
            raise

    def get(self, job_id: str) -> dict:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job
        try:
            job = Job.fetch(job_id, connection=self.redis)
        except NoSuchJobError as exc:
            raise ValueError(f"任务 {job_id} 不存在或已过期。") from exc

        status_value = job.get_status(refresh=True)
        status_value = getattr(status_value, "value", status_value)
        status_map = {
            "queued": "pending",
            "deferred": "pending",
            "scheduled": "pending",
            "started": "running",
            "finished": "succeeded",
            "failed": "failed",
            "stopped": "failed",
            "canceled": "failed",
        }
        status = status_map.get(str(status_value), "pending")
        started_at = job.started_at.isoformat() if job.started_at else ""
        finished_at = job.ended_at.isoformat() if job.ended_at else ""
        updated_at = (
            finished_at
            if status in {"succeeded", "failed"} and finished_at
            else job.meta.get("updated_at", "")
        )
        return {
            "job_id": job.id,
            "task_type": job.meta.get("task_type", ""),
            "status": status,
            "progress": (
                100 if status == "succeeded" else job.meta.get("progress", 0)
            ),
            "message": job.meta.get("message", ""),
            "stage": (
                "completed"
                if status == "succeeded"
                else job.meta.get("stage", "")
            ),
            "failed_stage": job.meta.get("failed_stage", ""),
            "error": job.meta.get("error", ""),
            "result": job.result if status == "succeeded" else None,
            "payload": dict(job.meta.get("payload") or {}),
            "scope": job.meta.get("scope", ""),
            "attempt": int(job.meta.get("attempt", 1)),
            "retry_of": job.meta.get("retry_of", ""),
            "trace_id": job.meta.get("trace_id", ""),
            "created_at": job.meta.get("created_at", ""),
            "updated_at": updated_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds(
                started_at,
                finished_at,
            ),
        }

    def _scope_jobs_key(self, scope: str) -> str:
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"industrial-knowledge-rag:task:scope:{digest}"

    def list(self, *, scope: str, limit: int = 50) -> list[dict]:
        job_ids = self.redis.zrevrange(
            self._scope_jobs_key(scope),
            0,
            max(0, limit - 1),
        )
        records = []
        for value in job_ids:
            job_id = value.decode("utf-8") if isinstance(value, bytes) else value
            try:
                records.append(self.get(job_id))
            except ValueError:
                self.redis.zrem(self._scope_jobs_key(scope), job_id)
        return records

    def retry(
        self,
        job_id: str,
        *,
        idempotency_key: str,
    ) -> tuple[dict, bool]:
        original = self.get(job_id)
        if original["status"] != "failed":
            raise ValueError("只有失败任务可以重试。")
        return self.submit(
            original["task_type"],
            original["payload"],
            scope=original["scope"],
            idempotency_key=idempotency_key,
            attempt=int(original.get("attempt", 1)) + 1,
            retry_of=job_id,
            trace_id=original["trace_id"],
        )

    def metrics(self, *, scope: str) -> dict:
        return task_metrics(self.list(scope=scope, limit=1000))

    def health(self) -> dict:
        from rq import Worker

        workers = [
            worker
            for worker in Worker.all(connection=self.redis)
            if any(
                queue.name == self.queue_name
                for queue in getattr(worker, "queues", [])
            )
        ]
        worker_states = [
            getattr(worker.state, "value", worker.state)
            for worker in workers
        ]
        return {
            "backend": self.backend_name,
            "healthy": bool(workers),
            "worker_count": len(workers),
            "running_jobs": sum(
                1 for state in worker_states if str(state) == "busy"
            ),
            "workers": [
                {
                    "name": worker.name,
                    "state": str(
                        getattr(worker.state, "value", worker.state)
                    ),
                    "last_heartbeat": (
                        worker.last_heartbeat.isoformat()
                        if worker.last_heartbeat
                        else ""
                    ),
                }
                for worker in workers
            ],
        }

    def publish_event(self, channel: str, value: str) -> None:
        self.redis.publish(channel, value)

    def start(self) -> None:
        return None

    def close(self) -> None:
        close = getattr(self.redis, "close", None)
        if callable(close):
            close()
        else:
            self.redis.connection_pool.disconnect()

    def listen_events(self, channel: str, stop_event, callback) -> None:
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
        try:
            while not stop_event.is_set():
                message = pubsub.get_message(timeout=1)
                if not message:
                    continue
                value = message.get("data")
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                callback(str(value))
        finally:
            pubsub.close()

    @contextmanager
    def lock(self, key: str):
        lock = self.redis.lock(
            f"industrial-knowledge-rag:task:lock:{key}",
            timeout=self.job_timeout,
            blocking_timeout=self.job_timeout,
        )
        acquired = lock.acquire(blocking=True)
        if not acquired:
            raise RuntimeError("无法获取知识库任务锁。")
        try:
            yield
        finally:
            lock.release()


def create_task_queue(runner):
    backend = os.getenv("TASK_QUEUE_BACKEND", "memory").strip().lower()
    if backend == "memory":
        try:
            max_workers = int(os.getenv("TASK_QUEUE_WORKERS", "2"))
        except ValueError:
            max_workers = 2
        return MemoryTaskQueue(runner, max_workers=max(1, max_workers))
    if backend == "redis":
        try:
            timeout = int(os.getenv("TASK_JOB_TIMEOUT_SECONDS", "1800"))
            retention = int(os.getenv("TASK_RETENTION_SECONDS", "86400"))
        except ValueError:
            timeout = 1800
            retention = 86400
        return RedisTaskQueue(
            os.getenv("REDIS_URL", "").strip(),
            queue_name=os.getenv("TASK_QUEUE_NAME", "knowledge").strip(),
            job_timeout=max(60, timeout),
            retention_seconds=max(3600, retention),
        )
    raise RuntimeError("TASK_QUEUE_BACKEND 只支持 memory 或 redis。")
