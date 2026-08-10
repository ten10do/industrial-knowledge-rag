from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from backend.knowledge_tasks import execute_job
from backend.version_store import sha256_hex


class FakeStore:
    def __init__(self, manifest, bundle):
        self.manifest = manifest
        self.bundle = bundle
        self.saved_draft = None
        self.saved_snapshot = None
        self.saved_cache = None
        self.deleted_job_id = None

    def load_task_input(self, _job_id):
        return self.manifest, self.bundle

    def save_draft(self, knowledge_base_id, manifest, bundle):
        self.saved_draft = (knowledge_base_id, manifest, bundle)

    def load_draft(self, _knowledge_base_id):
        raise ValueError("草稿不存在")

    def delete_task_input(self, job_id):
        self.deleted_job_id = job_id

    def save_draft_snapshot(self, knowledge_base_id, snapshot):
        self.saved_snapshot = (knowledge_base_id, snapshot)

    def save_draft_build_cache(self, knowledge_base_id, cache):
        self.saved_cache = (knowledge_base_id, cache)


class KnowledgeTaskTests(unittest.TestCase):
    def test_build_task_promotes_input_only_after_index_build(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            bundle = b"stored-upload"
            job_id = "job-" + "a" * 32
            knowledge_base_id = "kb-backend-test-00000001"
            manifest = {
                "files": ["course.pdf"],
                "sha256": sha256_hex(bundle),
                "size_bytes": len(bundle),
            }
            store = FakeStore(manifest, bundle)

            def extract(_bundle, _files, target, max_total_bytes):
                self.assertGreater(max_total_bytes, 0)
                target.mkdir()
                (target / "course.pdf").write_bytes(b"%PDF-course")

            def replace(
                source,
                target_id,
                after_build,
                previous_build_cache,
            ):
                self.assertTrue((source / "course.pdf").exists())
                self.assertEqual(target_id, knowledge_base_id)
                self.assertEqual(previous_build_cache, {})
                callback = after_build(
                    3,
                    8,
                    ["course.pdf"],
                    [source / "course.pdf"],
                    {
                        "cache": {"files": {}},
                        "stats": {"reused_file_count": 0},
                    },
                )
                return 3, 8, ["course.pdf"], callback

            fake_main = SimpleNamespace(
                DATA_DIR=data_dir,
                MAX_UPLOAD_TOTAL_BYTES=100,
                version_store=store,
                sha256_hex=sha256_hex,
                extract_pdf_bundle=extract,
                replace_knowledge_base_from_directory=replace,
                create_index_snapshot=Mock(
                    return_value=(
                        {"fingerprint": "test"},
                        b"snapshot",
                    )
                ),
                get_index_storage_path=lambda _key: data_dir / "index.json",
                RAG_MODE="light",
                task_queue=SimpleNamespace(
                    lock=lambda _key: nullcontext()
                ),
                get_knowledge_base_lock=lambda _key: nullcontext(),
                remove_storage_path=lambda path: (
                    shutil.rmtree(path) if path.exists() else None
                ),
                datetime=datetime,
                timezone=timezone,
                logger=Mock(),
            )
            progress = []

            with patch(
                "backend.knowledge_tasks._main_module",
                return_value=fake_main,
            ):
                result = execute_job(
                    "build_draft",
                    {
                        "input_job_id": job_id,
                        "knowledge_base_id": knowledge_base_id,
                    },
                    lambda value, message, _stage=None: progress.append(
                        (value, message)
                    ),
                )

            self.assertEqual(result["chunk_count"], 8)
            self.assertEqual(store.saved_draft[0], knowledge_base_id)
            self.assertTrue(store.saved_draft[1]["ready"])
            self.assertEqual(store.saved_snapshot[1], b"snapshot")
            self.assertEqual(
                store.saved_cache[1],
                {"files": {}},
            )
            self.assertEqual(store.deleted_job_id, job_id)
            self.assertEqual(progress[-1][0], 95)

    def test_failed_build_keeps_task_input_for_retry(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            bundle = b"stored-upload"
            job_id = "job-" + "b" * 32
            manifest = {
                "files": ["course.pdf"],
                "sha256": sha256_hex(bundle),
                "size_bytes": len(bundle),
                "expires_at": "2099-07-28T06:00:00+00:00",
            }
            store = FakeStore(manifest, bundle)

            def extract(_bundle, _files, target, max_total_bytes):
                target.mkdir()
                (target / "course.pdf").write_bytes(b"%PDF-course")

            fake_main = SimpleNamespace(
                DATA_DIR=data_dir,
                MAX_UPLOAD_TOTAL_BYTES=100,
                version_store=store,
                sha256_hex=sha256_hex,
                extract_pdf_bundle=extract,
                replace_knowledge_base_from_directory=Mock(
                    side_effect=RuntimeError("index failed")
                ),
                task_queue=SimpleNamespace(
                    lock=lambda _key: nullcontext()
                ),
                get_knowledge_base_lock=lambda _key: nullcontext(),
                remove_storage_path=lambda path: (
                    shutil.rmtree(path) if path.exists() else None
                ),
                datetime=datetime,
                timezone=timezone,
                logger=Mock(),
            )

            with patch(
                "backend.knowledge_tasks._main_module",
                return_value=fake_main,
            ):
                with self.assertRaisesRegex(RuntimeError, "index failed"):
                    execute_job(
                        "build_draft",
                        {
                            "input_job_id": job_id,
                            "knowledge_base_id": "kb-backend-test-00000001",
                        },
                        lambda _value, _message, _stage=None: None,
                    )

            self.assertIsNone(store.deleted_job_id)

    def test_rollback_notifies_web_instances_after_activation(self):
        version_id = "v-20260728T090000000000Z-aaaaaaaa"
        publish_event = Mock()
        fake_main = SimpleNamespace(
            PUBLIC_KNOWLEDGE_BASE_ID="kb-public-shared-00000001",
            task_queue=SimpleNamespace(
                lock=lambda _key: nullcontext()
            ),
            get_knowledge_base_lock=lambda _key: nullcontext(),
            activate_stored_public_version=Mock(
                return_value=(
                    3,
                    8,
                    ["course.pdf"],
                    {
                        "version_id": version_id,
                        "created_at": "2026-07-28T09:00:00+00:00",
                    },
                )
            ),
            publish_public_version_event=publish_event,
        )

        with patch(
            "backend.knowledge_tasks._main_module",
            return_value=fake_main,
        ):
            result = execute_job(
                "rollback",
                {"version_id": version_id},
                lambda _value, _message, _stage=None: None,
            )

        self.assertEqual(result["version_id"], version_id)
        publish_event.assert_called_once_with(version_id)


if __name__ == "__main__":
    unittest.main()
