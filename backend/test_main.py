import importlib
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import UploadFile
from fastapi.testclient import TestClient

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("langchain_chroma", types.SimpleNamespace(Chroma=object))
sys.modules.setdefault(
    "langchain_community.document_loaders",
    types.SimpleNamespace(PyPDFLoader=object),
)
sys.modules.setdefault(
    "langchain_community.embeddings",
    types.SimpleNamespace(HuggingFaceEmbeddings=object),
)
sys.modules.setdefault(
    "langchain_text_splitters",
    types.SimpleNamespace(RecursiveCharacterTextSplitter=object),
)

from backend.main import app
import backend.main as main_module
from backend.model_governance import ModelQuotaExceeded

TEST_KNOWLEDGE_BASE_ID = "kb-backend-test-00000001"
TEST_PUBLIC_KNOWLEDGE_BASE_ID = main_module.PUBLIC_KNOWLEDGE_BASE_ID
TEST_ADMIN_TOKEN = "test-admin-token"
TEST_HEADERS = {
    "X-Knowledge-Base-ID": TEST_KNOWLEDGE_BASE_ID,
    "X-Admin-Token": TEST_ADMIN_TOKEN,
}


class FastApiBackendTests(unittest.TestCase):
    def setUp(self):
        self.admin_environment = patch.dict(
            os.environ,
            {"ADMIN_TOKEN": TEST_ADMIN_TOKEN},
        )
        self.admin_environment.start()
        main_module.rate_limiter.clear()
        self.client = TestClient(app, headers=TEST_HEADERS)

    def tearDown(self):
        main_module.rate_limiter.clear()
        self.admin_environment.stop()

    def test_health_returns_service_status(self):
        with patch("backend.main.get_knowledge_base_status", return_value=(True, 2)):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "knowledge_base_ready": True,
                "pdf_count": 2,
            },
        )

    def test_scoped_routes_require_a_valid_knowledge_base_id(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 422)

        response = TestClient(
            app,
            headers={"X-Knowledge-Base-ID": "shared"},
        ).get("/health")
        self.assertEqual(response.status_code, 422)

    def test_upload_and_reset_require_the_management_token(self):
        headers = {"X-Knowledge-Base-ID": TEST_KNOWLEDGE_BASE_ID}
        client = TestClient(app, headers=headers)

        upload_response = client.post(
            "/upload",
            files=[("files", ("course.pdf", b"pdf", "application/pdf"))],
        )
        reset_response = client.post("/reset")
        publish_response = client.post("/publish")
        versions_response = client.get("/versions")
        rollback_response = client.post(
            "/versions/v-20260728T010000000000Z-aaaaaaaa/rollback"
        )
        jobs_response = client.get("/jobs")
        retry_response = client.post(
            f"/jobs/job-{'a' * 32}/retry"
        )

        self.assertEqual(upload_response.status_code, 401)
        self.assertEqual(reset_response.status_code, 401)
        self.assertEqual(publish_response.status_code, 401)
        self.assertEqual(versions_response.status_code, 401)
        self.assertEqual(rollback_response.status_code, 401)
        self.assertEqual(jobs_response.status_code, 401)
        self.assertEqual(retry_response.status_code, 401)
        self.assertEqual(
            upload_response.headers["www-authenticate"],
            "X-Admin-Token",
        )

    def test_public_knowledge_base_cannot_be_uploaded_or_reset_directly(self):
        client = TestClient(
            app,
            headers={
                "X-Knowledge-Base-ID": TEST_PUBLIC_KNOWLEDGE_BASE_ID,
                "X-Admin-Token": TEST_ADMIN_TOKEN,
            },
        )

        upload_response = client.post(
            "/upload",
            files=[("files", ("course.pdf", b"pdf", "application/pdf"))],
        )
        reset_response = client.post("/reset")

        self.assertEqual(upload_response.status_code, 409)
        self.assertEqual(reset_response.status_code, 409)

    def test_public_health_exposes_version_consistency(self):
        client = TestClient(
            app,
            headers={
                "X-Knowledge-Base-ID": TEST_PUBLIC_KNOWLEDGE_BASE_ID,
            },
        )
        sync_status = {
            "status": "synchronized",
            "remote_active_version": "version-2",
            "loaded_version": "version-2",
            "last_checked_at": "2026-07-28T08:00:00+00:00",
            "last_success_at": "2026-07-28T08:00:00+00:00",
            "last_error": "",
        }
        with patch.object(
            main_module,
            "ensure_public_version_current",
        ):
            with patch.object(
                main_module,
                "get_knowledge_base_status",
                return_value=(True, 2),
            ):
                with patch.object(
                    main_module.public_version_synchronizer,
                    "status",
                    return_value=sync_status,
                ):
                    response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version_sync"], sync_status)

    def test_publish_endpoint_promotes_the_current_draft(self):
        record = {
            "job_id": "job-" + "a" * 32,
            "task_type": "publish",
            "status": "pending",
            "progress": 0,
            "message": "任务等待执行。",
        }
        with patch.object(
            main_module.task_queue,
            "submit",
            return_value=(record, True),
        ) as submit:
            response = self.client.post("/publish")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["job_id"], record["job_id"])
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[0], "publish")
        self.assertEqual(
            submit.call_args.args[1]["knowledge_base_id"],
            TEST_KNOWLEDGE_BASE_ID,
        )

    def test_publish_rebuilds_public_data_without_removing_the_draft(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            draft_dir = data_root / TEST_KNOWLEDGE_BASE_ID
            public_dir = data_root / TEST_PUBLIC_KNOWLEDGE_BASE_ID
            draft_dir.mkdir(parents=True)
            public_dir.mkdir()
            (draft_dir / "new.pdf").write_bytes(b"%PDF-new")
            (public_dir / "old.pdf").write_bytes(b"%PDF-old")
            index_path = Path(temp_dir) / "public-index.json"
            index_path.write_text("old-index", encoding="utf-8")
            fake_store = SimpleNamespace(
                save_version=Mock(),
                set_active_version=Mock(),
            )

            def scoped_dir(knowledge_base_id):
                return data_root / knowledge_base_id

            def build_public(_paths, knowledge_base_id):
                index_path.write_text("new-index", encoding="utf-8")
                return 5, 14

            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    side_effect=scoped_dir,
                ):
                    with patch.object(
                        main_module,
                        "build_knowledge_base",
                        side_effect=build_public,
                    ) as build:
                        with patch.object(
                            main_module,
                            "get_index_storage_path",
                            return_value=index_path,
                        ):
                            with patch.object(
                                main_module,
                                "version_store",
                                fake_store,
                            ):
                                with patch.object(
                                    main_module,
                                    "write_active_version_marker",
                                ):
                                    result = main_module.publish_knowledge_base(
                                        TEST_KNOWLEDGE_BASE_ID
                                    )

            self.assertEqual(result[:3], (5, 14, ["new.pdf"]))
            self.assertTrue((draft_dir / "new.pdf").exists())
            self.assertTrue((public_dir / "new.pdf").exists())
            self.assertFalse((public_dir / "old.pdf").exists())
            fake_store.save_version.assert_called_once()
            fake_store.set_active_version.assert_called_once_with(
                result[3]["version_id"]
            )
            self.assertEqual(
                build.call_args.kwargs["knowledge_base_id"],
                TEST_PUBLIC_KNOWLEDGE_BASE_ID,
            )

    def test_failed_publish_preserves_previous_public_data(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            draft_dir = data_root / TEST_KNOWLEDGE_BASE_ID
            public_dir = data_root / TEST_PUBLIC_KNOWLEDGE_BASE_ID
            draft_dir.mkdir(parents=True)
            public_dir.mkdir()
            (draft_dir / "new.pdf").write_bytes(b"%PDF-new")
            (public_dir / "old.pdf").write_bytes(b"%PDF-old")
            index_path = Path(temp_dir) / "public-index.json"
            index_path.write_text("old-index", encoding="utf-8")

            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    side_effect=lambda value: data_root / value,
                ):
                    with patch.object(
                        main_module,
                        "build_knowledge_base",
                        side_effect=RuntimeError("publish failed"),
                    ):
                        with patch.object(
                            main_module,
                            "get_index_storage_path",
                            return_value=index_path,
                        ):
                            with patch.object(
                                main_module,
                                "reload_knowledge_base",
                            ):
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    "publish failed",
                                ):
                                    main_module.publish_knowledge_base(
                                        TEST_KNOWLEDGE_BASE_ID
                                    )

            self.assertTrue((draft_dir / "new.pdf").exists())
            self.assertTrue((public_dir / "old.pdf").exists())
            self.assertFalse((public_dir / "new.pdf").exists())
            self.assertEqual(
                index_path.read_text(encoding="utf-8"),
                "old-index",
            )

    def test_failed_active_pointer_update_restores_public_data_and_index(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            draft_dir = data_root / TEST_KNOWLEDGE_BASE_ID
            public_dir = data_root / TEST_PUBLIC_KNOWLEDGE_BASE_ID
            draft_dir.mkdir(parents=True)
            public_dir.mkdir()
            (draft_dir / "new.pdf").write_bytes(b"%PDF-new")
            (public_dir / "old.pdf").write_bytes(b"%PDF-old")
            index_path = Path(temp_dir) / "public-index.json"
            index_path.write_text("old-index", encoding="utf-8")

            def build_new_index(_paths, knowledge_base_id):
                self.assertEqual(
                    knowledge_base_id,
                    TEST_PUBLIC_KNOWLEDGE_BASE_ID,
                )
                index_path.write_text("new-index", encoding="utf-8")
                return 3, 7

            fake_store = SimpleNamespace(
                save_version=Mock(),
                set_active_version=Mock(
                    side_effect=RuntimeError("pointer failed")
                ),
            )
            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    side_effect=lambda value: data_root / value,
                ):
                    with patch.object(
                        main_module,
                        "get_index_storage_path",
                        return_value=index_path,
                    ):
                        with patch.object(
                            main_module,
                            "build_knowledge_base",
                            side_effect=build_new_index,
                        ):
                            with patch.object(
                                main_module,
                                "reload_knowledge_base",
                            ) as reload_index:
                                with patch.object(
                                    main_module,
                                    "version_store",
                                    fake_store,
                                ):
                                    with self.assertRaisesRegex(
                                        RuntimeError,
                                        "pointer failed",
                                    ):
                                        main_module.publish_knowledge_base(
                                            TEST_KNOWLEDGE_BASE_ID
                                        )

            self.assertTrue((public_dir / "old.pdf").exists())
            self.assertFalse((public_dir / "new.pdf").exists())
            self.assertEqual(
                index_path.read_text(encoding="utf-8"),
                "old-index",
            )
            reload_index.assert_called_once_with(
                TEST_PUBLIC_KNOWLEDGE_BASE_ID
            )

    def test_rollback_activates_stored_version(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = root / "source"
            public_dir = data_root / TEST_PUBLIC_KNOWLEDGE_BASE_ID
            source_dir.mkdir()
            public_dir.mkdir(parents=True)
            (source_dir / "restored.pdf").write_bytes(b"%PDF-restored")
            (public_dir / "current.pdf").write_bytes(b"%PDF-current")
            bundle = main_module.create_pdf_bundle(
                [source_dir / "restored.pdf"]
            )
            version_id = "v-20260728T040000000000Z-dddddddd"
            manifest = {
                "version_id": version_id,
                "created_at": "2026-07-28T04:00:00+00:00",
                "page_count": 4,
                "chunk_count": 9,
                "files": ["restored.pdf"],
                "sha256": main_module.sha256_hex(bundle),
                "size_bytes": len(bundle),
                "source_draft_id": TEST_KNOWLEDGE_BASE_ID,
            }
            index_path = root / "public-index.json"
            index_path.write_text("current-index", encoding="utf-8")
            fake_store = SimpleNamespace(
                load_version=Mock(return_value=(manifest, bundle)),
                set_active_version=Mock(),
            )

            def build_restored(_paths, knowledge_base_id):
                self.assertEqual(
                    knowledge_base_id,
                    TEST_PUBLIC_KNOWLEDGE_BASE_ID,
                )
                index_path.write_text("restored-index", encoding="utf-8")
                return 4, 9

            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    return_value=public_dir,
                ):
                    with patch.object(
                        main_module,
                        "get_index_storage_path",
                        return_value=index_path,
                    ):
                        with patch.object(
                            main_module,
                            "build_knowledge_base",
                            side_effect=build_restored,
                        ):
                            with patch.object(
                                main_module,
                                "version_store",
                                fake_store,
                            ):
                                with patch.object(
                                    main_module,
                                    "write_active_version_marker",
                                ):
                                    result = (
                                        main_module.activate_stored_public_version(
                                            version_id
                                        )
                                    )

            self.assertEqual(result[:3], (4, 9, ["restored.pdf"]))
            self.assertTrue((public_dir / "restored.pdf").exists())
            self.assertFalse((public_dir / "current.pdf").exists())
            self.assertEqual(
                index_path.read_text(encoding="utf-8"),
                "restored-index",
            )
            fake_store.set_active_version.assert_called_once_with(version_id)

    def test_rollback_prefers_a_compatible_index_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = root / "source"
            public_dir = data_root / TEST_PUBLIC_KNOWLEDGE_BASE_ID
            source_index = root / "source-index.json"
            target_index = root / "public-index.json"
            source_dir.mkdir()
            public_dir.mkdir(parents=True)
            (source_dir / "restored.pdf").write_bytes(b"%PDF-restored")
            source_index.write_text(
                '[{"page_content":"snapshot","metadata":'
                '{"source":"restored.pdf","page":0}}]',
                encoding="utf-8",
            )
            snapshot_metadata, snapshot_bundle = (
                main_module.create_index_snapshot(
                    source_index,
                    main_module.RAG_MODE,
                )
            )
            bundle = main_module.create_pdf_bundle(
                [source_dir / "restored.pdf"]
            )
            version_id = "v-20260728T041000000000Z-abababab"
            manifest = {
                "version_id": version_id,
                "created_at": "2026-07-28T04:10:00+00:00",
                "page_count": 1,
                "chunk_count": 1,
                "files": ["restored.pdf"],
                "sha256": main_module.sha256_hex(bundle),
                "size_bytes": len(bundle),
                "source_draft_id": TEST_KNOWLEDGE_BASE_ID,
                "index_snapshot": snapshot_metadata,
            }
            fake_store = SimpleNamespace(
                load_version=Mock(return_value=(manifest, bundle)),
                load_version_snapshot=Mock(return_value=snapshot_bundle),
                set_active_version=Mock(),
            )

            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    return_value=public_dir,
                ):
                    with patch.object(
                        main_module,
                        "get_index_storage_path",
                        return_value=target_index,
                    ):
                        with patch.object(
                            main_module,
                            "reload_knowledge_base",
                            return_value=True,
                        ):
                            with patch.object(
                                main_module,
                                "build_knowledge_base",
                            ) as rebuild:
                                with patch.object(
                                    main_module,
                                    "version_store",
                                    fake_store,
                                ):
                                    with patch.object(
                                        main_module,
                                        "write_active_version_marker",
                                    ):
                                        result = (
                                            main_module.activate_stored_public_version(
                                                version_id
                                            )
                                        )

            self.assertEqual(result[:3], (1, 1, ["restored.pdf"]))
            self.assertEqual(
                target_index.read_text(encoding="utf-8"),
                source_index.read_text(encoding="utf-8"),
            )
            rebuild.assert_not_called()
            fake_store.set_active_version.assert_called_once_with(version_id)

    def test_corrupt_index_snapshot_falls_back_to_pdf_rebuild(self):
        with TemporaryDirectory() as temp_dir:
            version_id = "v-20260728T042000000000Z-acacacac"
            pdf_bundle = b"pdf-bundle"
            snapshot_metadata = {
                "schema_version": 1,
                "rag_mode": main_module.RAG_MODE,
                "fingerprint": main_module.build_fingerprint(
                    main_module.RAG_MODE
                ),
                "kind": "file",
                "sha256": "invalid",
                "size_bytes": 7,
            }
            manifest = {
                "version_id": version_id,
                "created_at": "2026-07-28T04:20:00+00:00",
                "page_count": 2,
                "chunk_count": 3,
                "files": ["restored.pdf"],
                "sha256": main_module.sha256_hex(pdf_bundle),
                "size_bytes": len(pdf_bundle),
                "index_snapshot": snapshot_metadata,
            }
            fake_store = SimpleNamespace(
                load_version=Mock(return_value=(manifest, pdf_bundle)),
                load_version_snapshot=Mock(return_value=b"corrupt"),
                set_active_version=Mock(),
            )

            def extract(_bundle, _files, target, max_total_bytes):
                target.mkdir()
                (target / "restored.pdf").write_bytes(b"%PDF")

            def rebuild(_source, after_build):
                after_build(2, 3, ["restored.pdf"], [], {})
                return 2, 3, ["restored.pdf"], None

            with patch.object(
                main_module,
                "DATA_DIR",
                Path(temp_dir),
            ):
                with patch.object(
                    main_module,
                    "version_store",
                    fake_store,
                ):
                    with patch.object(
                        main_module,
                        "extract_pdf_bundle",
                        side_effect=extract,
                    ):
                        with patch.object(
                            main_module,
                            "replace_knowledge_base_from_snapshot",
                            side_effect=ValueError("corrupt snapshot"),
                        ):
                            with patch.object(
                                main_module,
                                "replace_public_knowledge_base",
                                side_effect=rebuild,
                            ) as rebuild_index:
                                with patch.object(
                                    main_module,
                                    "write_active_version_marker",
                                ):
                                    result = (
                                        main_module.activate_stored_public_version(
                                            version_id
                                        )
                                    )

            self.assertEqual(result[:3], (2, 3, ["restored.pdf"]))
            rebuild_index.assert_called_once()
            fake_store.set_active_version.assert_called_once_with(version_id)

    def test_startup_restore_uses_remote_active_version_when_cache_is_stale(self):
        version_id = "v-20260728T050000000000Z-eeeeeeee"
        fake_store = SimpleNamespace(
            get_active_version_id=Mock(return_value=version_id)
        )
        with patch.object(main_module, "version_store", fake_store):
            with patch.object(
                main_module,
                "read_active_version_marker",
                return_value=None,
            ):
                with patch.object(
                    main_module,
                    "activate_stored_public_version",
                ) as activate:
                    restored = main_module.restore_active_public_version()

        self.assertTrue(restored)
        activate.assert_called_once_with(
            version_id,
            update_remote_pointer=False,
        )

    def test_version_history_and_rollback_endpoints(self):
        version_id = "v-20260728T060000000000Z-ffffffff"
        manifest = {
            "version_id": version_id,
            "created_at": "2026-07-28T06:00:00+00:00",
            "page_count": 6,
            "chunk_count": 12,
            "files": ["history.pdf"],
        }
        fake_store = SimpleNamespace(
            get_active_version_id=Mock(return_value=version_id),
            list_versions=Mock(return_value=[manifest]),
        )
        with patch.object(main_module, "version_store", fake_store):
            list_response = self.client.get("/versions")
        rollback_job = {
            "job_id": "job-" + "b" * 32,
            "task_type": "rollback",
            "status": "pending",
            "progress": 0,
            "message": "任务等待执行。",
        }
        with patch.object(
            main_module.task_queue,
            "submit",
            return_value=(rollback_job, True),
        ) as submit:
            rollback_response = self.client.post(
                f"/versions/{version_id}/rollback"
            )

        self.assertEqual(list_response.status_code, 200)
        self.assertTrue(list_response.json()["versions"][0]["active"])
        self.assertEqual(rollback_response.status_code, 202)
        self.assertEqual(
            rollback_response.json()["job_id"],
            rollback_job["job_id"],
        )
        self.assertEqual(submit.call_args.args[0], "rollback")
        self.assertEqual(submit.call_args.args[1]["version_id"], version_id)

    def test_rate_limit_returns_retry_after_header(self):
        with patch.dict(
            main_module.RATE_LIMITS,
            {"health": (1, 60)},
        ):
            first = self.client.get("/health")
            second = self.client.get("/health")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertGreaterEqual(int(second.headers["retry-after"]), 1)
        self.assertEqual(first.headers["x-ratelimit-limit"], "1")
        self.assertEqual(first.headers["x-ratelimit-remaining"], "0")
        self.assertEqual(second.headers["x-ratelimit-remaining"], "0")

    def test_rate_limit_backend_failure_opens_public_and_closes_management(self):
        with patch.object(
            main_module.rate_limiter,
            "consume_result",
            side_effect=RuntimeError("redis unavailable"),
        ):
            with patch.dict(
                os.environ,
                {
                    "RATE_LIMIT_PUBLIC_FAIL_OPEN": "true",
                    "RATE_LIMIT_MANAGEMENT_FAIL_OPEN": "false",
                },
            ):
                public_response = self.client.get("/health")
                management_response = self.client.get("/versions")

        self.assertEqual(public_response.status_code, 200)
        self.assertEqual(
            public_response.headers["x-ratelimit-policy"],
            "degraded-open",
        )
        self.assertEqual(management_response.status_code, 503)

    def test_job_status_endpoint_returns_progress_and_result(self):
        job_id = "job-" + "d" * 32
        record = {
            "job_id": job_id,
            "task_type": "build_draft",
            "status": "succeeded",
            "progress": 100,
            "message": "任务执行完成。",
            "error": "",
            "result": {"page_count": 3, "chunk_count": 8, "files": ["a.pdf"]},
            "scope": TEST_KNOWLEDGE_BASE_ID,
            "trace_id": "trace-" + "d" * 32,
            "attempt": 1,
            "retry_of": "",
            "stage": "completed",
            "failed_stage": "",
            "started_at": "2026-07-28T06:00:10+00:00",
            "finished_at": "2026-07-28T06:01:00+00:00",
            "duration_seconds": 50.0,
            "created_at": "2026-07-28T06:00:00+00:00",
            "updated_at": "2026-07-28T06:01:00+00:00",
        }
        with patch.object(
            main_module.task_queue,
            "get",
            return_value=record,
        ):
            response = self.client.get(f"/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "succeeded")
        self.assertEqual(response.json()["result"]["chunk_count"], 8)

    def test_task_center_lists_metrics_and_retries_failed_jobs(self):
        failed_id = "job-" + "e" * 32
        retried_id = "job-" + "f" * 32
        failed_record = {
            "job_id": failed_id,
            "task_type": "build_draft",
            "status": "failed",
            "progress": 60,
            "message": "任务执行失败。",
            "stage": "indexing",
            "failed_stage": "indexing",
            "error": "index failed",
            "result": None,
            "payload": {"input_job_id": "job-input"},
            "scope": TEST_KNOWLEDGE_BASE_ID,
            "trace_id": "trace-" + "e" * 32,
            "attempt": 1,
            "retry_of": "",
            "created_at": "2026-07-28T06:00:00+00:00",
            "updated_at": "2026-07-28T06:01:00+00:00",
            "started_at": "2026-07-28T06:00:10+00:00",
            "finished_at": "2026-07-28T06:01:00+00:00",
            "duration_seconds": 50.0,
        }
        retried_record = {
            **failed_record,
            "job_id": retried_id,
            "status": "pending",
            "progress": 0,
            "message": "任务等待执行。",
            "attempt": 2,
            "retry_of": failed_id,
        }
        health = {
            "backend": "memory",
            "healthy": True,
            "worker_count": 2,
            "running_jobs": 0,
        }
        metrics = {
            "total": 1,
            "status_counts": {"failed": 1},
            "average_duration_seconds": 50.0,
            "p95_duration_seconds": 50.0,
        }
        with patch.object(
            main_module.task_queue,
            "health",
            return_value=health,
        ):
            with patch.object(
                main_module.task_queue,
                "list",
                return_value=[failed_record],
            ):
                with patch.object(
                    main_module.task_queue,
                    "metrics",
                    return_value=metrics,
                ):
                    center_response = self.client.get("/jobs")
        with patch.object(
            main_module.task_queue,
            "get",
            return_value=failed_record,
        ):
            with patch.object(
                main_module.task_queue,
                "retry",
                return_value=(retried_record, True),
            ):
                retry_response = self.client.post(
                    f"/jobs/{failed_id}/retry"
                )

        self.assertEqual(center_response.status_code, 200)
        self.assertEqual(
            center_response.json()["jobs"][0]["failed_stage"],
            "indexing",
        )
        self.assertTrue(center_response.json()["worker"]["healthy"])
        self.assertEqual(retry_response.status_code, 202)
        self.assertEqual(retry_response.json()["job_id"], retried_id)

    def test_upload_rejects_non_pdf_content_before_building(self):
        upload = UploadFile(
            filename="fake.pdf",
            file=BytesIO(b"not-a-real-pdf"),
        )
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "不是有效的 PDF"):
                main_module.save_validated_uploads(
                    [upload],
                    Path(temp_dir),
                )

    def test_upload_enforces_streamed_file_size_limit(self):
        upload = UploadFile(
            filename="large.pdf",
            file=BytesIO(b"%PDF-" + b"x" * 8),
        )
        with TemporaryDirectory() as temp_dir:
            with patch.object(main_module, "MAX_UPLOAD_FILE_BYTES", 8):
                with self.assertRaisesRegex(ValueError, "单文件大小限制"):
                    main_module.save_validated_uploads(
                        [upload],
                        Path(temp_dir),
                    )

    def test_failed_rebuild_preserves_previous_data_directory(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_root.mkdir()
            scoped_data = data_root / TEST_KNOWLEDGE_BASE_ID
            scoped_data.mkdir()
            (scoped_data / "old.pdf").write_bytes(b"%PDF-old")

            def fake_save(_uploads, staging_dir):
                new_path = staging_dir / "new.pdf"
                new_path.write_bytes(b"%PDF-new")
                return [new_path], ["new.pdf"]

            with patch.object(main_module, "DATA_DIR", data_root):
                with patch.object(
                    main_module,
                    "get_data_dir",
                    return_value=scoped_data,
                ):
                    with patch.object(
                        main_module,
                        "save_validated_uploads",
                        side_effect=fake_save,
                    ):
                        with patch.object(
                            main_module,
                            "build_knowledge_base",
                            side_effect=RuntimeError("index failed"),
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError,
                                "index failed",
                            ):
                                main_module.rebuild_knowledge_base(
                                    [object()],
                                    TEST_KNOWLEDGE_BASE_ID,
                                )

            self.assertTrue((scoped_data / "old.pdf").exists())
            self.assertFalse((scoped_data / "new.pdf").exists())

    def test_default_rag_mode_uses_light_backend(self):
        self.assertEqual(getattr(main_module, "RAG_MODE", None), "light")
        self.assertEqual(
            getattr(main_module, "RAG_BACKEND_NAME", None),
            "light_rag_core",
        )

    def test_full_rag_mode_keeps_existing_backend(self):
        with patch.dict(os.environ, {"RAG_MODE": "full"}):
            full_module = importlib.reload(main_module)
            selected_mode = getattr(full_module, "RAG_MODE", None)
            selected_backend = getattr(full_module, "RAG_BACKEND_NAME", None)

        importlib.reload(main_module)
        self.assertEqual(selected_mode, "full")
        self.assertEqual(selected_backend, "rag_core")

    def test_upload_accepts_multiple_pdf_files_and_builds_knowledge_base(self):
        files = [
            ("files", ("course-a.pdf", b"pdf-a", "application/pdf")),
            ("files", ("course-b.pdf", b"pdf-b", "application/pdf")),
        ]

        record = {
            "job_id": "job-" + "c" * 32,
            "task_type": "build_draft",
            "status": "pending",
            "progress": 0,
            "message": "任务等待执行。",
        }
        with patch(
            "backend.main.prepare_draft_task_input",
            return_value={"files": ["course-a.pdf", "course-b.pdf"]},
        ) as prepare:
            with patch.object(
                main_module.task_queue,
                "submit",
                return_value=(record, True),
            ) as submit:
                response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["job_id"], record["job_id"])
        self.assertEqual(len(prepare.call_args.args[0]), 2)
        self.assertEqual(prepare.call_args.args[1], TEST_KNOWLEDGE_BASE_ID)
        self.assertEqual(submit.call_args.args[0], "build_draft")

    def test_ask_returns_answer_and_serialized_sources(self):
        docs = [
            (
                SimpleNamespace(
                    page_content="PLC 扫描周期参考内容",
                    metadata={"source": "course.pdf", "page": 0},
                ),
                0.25,
            )
        ]

        with patch("backend.main.retrieve_docs", return_value=docs):
            with patch("backend.main.has_relevant_docs", return_value=True):
                with patch("backend.main.generate_answer", return_value="模型回答") as generate:
                    response = self.client.post(
                        "/ask",
                        json={
                            "question": "PLC 的扫描周期是什么？",
                            "model_provider": "DeepSeek",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "模型回答")
        self.assertFalse(payload["is_refused"])
        self.assertEqual(
            payload["sources"],
            [
                {
                    "citation_id": "S1",
                    "source": "course.pdf",
                    "page": 1,
                    "score": 0.25,
                    "content": "PLC 扫描周期参考内容",
                }
            ],
        )
        generate.assert_called_once_with(
            "PLC 的扫描周期是什么？",
            docs,
            provider="DeepSeek",
        )

    def test_model_quota_exhaustion_returns_429_and_quota_headers(self):
        docs = [
            (
                SimpleNamespace(
                    page_content="课程资料",
                    metadata={"source": "course.pdf", "page": 0},
                ),
                0.25,
            )
        ]
        quota_error = ModelQuotaExceeded(
            "今日模型 Token 配额已用完。",
            retry_after=3600,
            limit=1000,
            remaining=0,
            quota_reset_after=3600,
        )
        with patch("backend.main.retrieve_docs", return_value=docs):
            with patch("backend.main.has_relevant_docs", return_value=True):
                with patch(
                    "backend.main.generate_answer",
                    side_effect=quota_error,
                ):
                    response = self.client.post(
                        "/ask",
                        json={"question": "课程问题"},
                    )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "3600")
        self.assertEqual(response.headers["x-model-token-limit"], "1000")
        self.assertEqual(response.headers["x-model-token-remaining"], "0")

    def test_ask_refuses_irrelevant_question_without_calling_model(self):
        docs = [(SimpleNamespace(page_content="无关片段", metadata={}), 40.0)]

        with patch("backend.main.retrieve_docs", return_value=docs):
            with patch("backend.main.has_relevant_docs", return_value=False):
                with patch("backend.main.generate_answer") as generate:
                    response = self.client.post(
                        "/ask",
                        json={"question": "今天天气如何？", "model_provider": "Groq"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_refused"])
        generate.assert_not_called()

    def test_study_endpoints_use_expected_task_types(self):
        cases = [
            ("/study/summary", "summary"),
            ("/study/knowledge-points", "knowledge_points"),
            ("/study/quiz", "review_questions"),
        ]

        for path, task_type in cases:
            with self.subTest(path=path):
                with patch(
                    "backend.main.generate_learning_content",
                    return_value="学习辅助结果",
                ) as generate:
                    response = self.client.post(
                        path,
                        json={"model_provider": "DeepSeek"},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["content"], "学习辅助结果")
                generate.assert_called_once_with(
                    task_type,
                    provider="DeepSeek",
                    knowledge_base_id=TEST_KNOWLEDGE_BASE_ID,
                )

    def test_reset_clears_backend_knowledge_base(self):
        with patch("backend.main.clear_knowledge_base") as clear:
            response = self.client.post("/reset")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "知识库已清空。")
        clear.assert_called_once_with(TEST_KNOWLEDGE_BASE_ID)

    def test_cors_allows_localhost_and_loopback_frontend_origins(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/health",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["access-control-allow-origin"],
                    origin,
                )

    def test_cors_allows_frontend_origin_from_environment(self):
        frontend_origin = "https://autocourse-rag.example.com"

        with patch.dict(os.environ, {"FRONTEND_ORIGIN": frontend_origin}):
            deployed_module = importlib.reload(main_module)
            deployed_client = TestClient(
                deployed_module.app,
                headers=TEST_HEADERS,
            )
            response = deployed_client.options(
                "/health",
                headers={
                    "Origin": frontend_origin,
                    "Access-Control-Request-Method": "GET",
                },
            )

        importlib.reload(main_module)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            frontend_origin,
        )

    def test_render_root_directory_can_import_main_app(self):
        backend_dir = Path(__file__).resolve().parent
        import_error = None
        sys.path.insert(0, str(backend_dir))

        try:
            for module_name in ("main", "rag_core", "light_rag_core", "llm_client"):
                sys.modules.pop(module_name, None)
            render_main = importlib.import_module("main")
        except Exception as exc:
            import_error = exc
            render_main = None
        finally:
            sys.path.remove(str(backend_dir))
            for module_name in ("main", "rag_core", "light_rag_core", "llm_client"):
                sys.modules.pop(module_name, None)

        self.assertIsNone(import_error)
        self.assertIsNotNone(getattr(render_main, "app", None))


if __name__ == "__main__":
    unittest.main()
