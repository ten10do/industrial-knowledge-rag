from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from backend.task_queue import MemoryTaskQueue, RedisTaskQueue


class TaskQueueTests(unittest.TestCase):
    def test_memory_queue_tracks_progress_and_success(self):
        started = Event()
        release = Event()

        def runner(task_type, payload, report):
            self.assertEqual(task_type, "build")
            started.set()
            report(45, "正在构建索引")
            release.wait(timeout=2)
            return {"value": payload["value"]}

        queue = MemoryTaskQueue(runner, max_workers=1)
        job, created = queue.submit(
            "build",
            {"value": 7},
            scope="kb-test",
            idempotency_key="same-request",
        )

        self.assertTrue(created)
        self.assertTrue(started.wait(timeout=2))
        self.assertEqual(queue.get(job["job_id"])["progress"], 45)
        release.set()
        completed = queue.wait(job["job_id"], timeout=2)
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"], {"value": 7})

    def test_idempotency_key_reuses_the_existing_job(self):
        calls = []

        def runner(_task_type, _payload, _report):
            calls.append(1)
            return {"ok": True}

        queue = MemoryTaskQueue(runner, max_workers=1)
        first, first_created = queue.submit(
            "publish",
            {},
            scope="kb-test",
            idempotency_key="publish-once",
        )
        second, second_created = queue.submit(
            "publish",
            {},
            scope="kb-test",
            idempotency_key="publish-once",
        )
        queue.wait(first["job_id"], timeout=2)

        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(len(calls), 1)

    def test_failed_job_exposes_a_safe_error_message(self):
        def runner(_task_type, _payload, _report):
            raise ValueError("草稿知识库为空")

        queue = MemoryTaskQueue(runner, max_workers=1)
        job, _ = queue.submit(
            "publish",
            {},
            scope="kb-test",
            idempotency_key="failed-job",
        )
        failed = queue.wait(job["job_id"], timeout=2)

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "草稿知识库为空")

    def test_failed_job_can_be_listed_and_retried_with_audit_chain(self):
        calls = []

        def runner(_task_type, _payload, report):
            calls.append(1)
            report(60, "正在构建索引", "indexing")
            if len(calls) == 1:
                raise RuntimeError("index failed")
            return {"ok": True}

        queue = MemoryTaskQueue(runner, max_workers=1)
        original, _ = queue.submit(
            "build_draft",
            {"input_job_id": "job-input"},
            scope="kb-test",
            idempotency_key="build-original",
        )
        failed = queue.wait(original["job_id"], timeout=2)
        retried, created = queue.retry(
            original["job_id"],
            idempotency_key="retry-once",
        )
        succeeded = queue.wait(retried["job_id"], timeout=2)

        self.assertTrue(created)
        self.assertEqual(failed["failed_stage"], "indexing")
        self.assertEqual(succeeded["status"], "succeeded")
        self.assertEqual(succeeded["retry_of"], original["job_id"])
        self.assertEqual(succeeded["attempt"], 2)
        self.assertEqual(succeeded["trace_id"], original["trace_id"])
        self.assertEqual(
            queue.list(scope="kb-test")[0]["job_id"],
            succeeded["job_id"],
        )
        metrics = queue.metrics(scope="kb-test")
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["status_counts"]["failed"], 1)
        self.assertEqual(metrics["status_counts"]["succeeded"], 1)
        self.assertTrue(queue.health()["healthy"])

    def test_redis_version_event_is_published_and_decoded(self):
        stop = Event()
        received = []
        pubsub = SimpleNamespace(
            subscribe=Mock(),
            get_message=Mock(return_value={"data": b"version-2"}),
            close=Mock(),
        )
        redis = SimpleNamespace(
            publish=Mock(),
            pubsub=Mock(return_value=pubsub),
        )
        queue = RedisTaskQueue.__new__(RedisTaskQueue)
        queue.redis = redis

        queue.publish_event("versions", "version-2")

        def receive(value):
            received.append(value)
            stop.set()

        queue.listen_events("versions", stop, receive)

        redis.publish.assert_called_once_with("versions", "version-2")
        pubsub.subscribe.assert_called_once_with("versions")
        self.assertEqual(received, ["version-2"])
        pubsub.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
