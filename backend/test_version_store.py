from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.version_store import (
    LocalVersionStore,
    S3VersionStore,
    create_pdf_bundle,
    extract_pdf_bundle,
    sha256_hex,
)


class FakeS3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class FakePaginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, Bucket, Prefix):
        contents = [
            {"Key": key}
            for key in sorted(self.client.objects)
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3Error("404")
        return {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = bytes(Body)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3Error("NoSuchKey")
        return {"Body": FakeBody(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


class FakeBody:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


def version_manifest(version_id, created_at):
    return {
        "version_id": version_id,
        "created_at": created_at,
        "page_count": 2,
        "chunk_count": 4,
        "files": ["course.pdf"],
        "sha256": "checksum",
        "size_bytes": 7,
        "source_draft_id": "kb-backend-test-00000001",
    }


class VersionStoreTests(unittest.TestCase):
    def test_local_store_saves_immutable_versions_and_active_pointer(self):
        with TemporaryDirectory() as temp_dir:
            store = LocalVersionStore(Path(temp_dir))
            older = version_manifest(
                "v-20260728T010000000000Z-aaaaaaaa",
                "2026-07-28T01:00:00+00:00",
            )
            newer = version_manifest(
                "v-20260728T020000000000Z-bbbbbbbb",
                "2026-07-28T02:00:00+00:00",
            )

            store.save_version(older, b"older")
            store.save_version(newer, b"newer")
            store.save_version_snapshot(
                newer["version_id"],
                b"newer-index",
            )
            store.set_active_version(older["version_id"])

            self.assertEqual(
                [item["version_id"] for item in store.list_versions()],
                [newer["version_id"], older["version_id"]],
            )
            self.assertEqual(
                store.load_version(newer["version_id"]),
                (newer, b"newer"),
            )
            self.assertEqual(
                store.load_version_snapshot(newer["version_id"]),
                b"newer-index",
            )
            self.assertEqual(store.get_active_version_id(), older["version_id"])
            with self.assertRaisesRegex(ValueError, "已经存在"):
                store.save_version(older, b"duplicate")

    def test_s3_store_uses_version_objects_and_active_pointer(self):
        client = FakeS3Client()
        store = S3VersionStore(
            bucket="course-bucket",
            prefix="industrial-knowledge-rag/public",
            client=client,
        )
        manifest = version_manifest(
            "v-20260728T030000000000Z-cccccccc",
            "2026-07-28T03:00:00+00:00",
        )

        store.save_version(manifest, b"archive")
        store.save_version_snapshot(
            manifest["version_id"],
            b"index-snapshot",
        )
        store.set_active_version(manifest["version_id"])

        self.assertEqual(store.list_versions(), [manifest])
        self.assertEqual(
            store.load_version(manifest["version_id"]),
            (manifest, b"archive"),
        )
        self.assertEqual(
            store.load_version_snapshot(manifest["version_id"]),
            b"index-snapshot",
        )
        self.assertEqual(store.get_active_version_id(), manifest["version_id"])

        job_id = "job-" + "b" * 32
        draft_id = "kb-backend-test-00000001"
        task_manifest = {"job_id": job_id, "files": ["course.pdf"]}
        draft_manifest = {
            "knowledge_base_id": draft_id,
            "files": ["course.pdf"],
        }
        store.save_task_input(job_id, task_manifest, b"task")
        store.save_draft(draft_id, draft_manifest, b"draft")
        store.save_draft_snapshot(draft_id, b"draft-index")
        store.save_draft_build_cache(
            draft_id,
            {"schema_version": 1, "files": {}},
        )
        self.assertEqual(
            store.load_task_input(job_id),
            (task_manifest, b"task"),
        )
        self.assertEqual(
            store.load_draft(draft_id),
            (draft_manifest, b"draft"),
        )
        self.assertEqual(
            store.load_draft_snapshot(draft_id),
            b"draft-index",
        )
        self.assertEqual(
            store.load_draft_build_cache(draft_id),
            {"schema_version": 1, "files": {}},
        )
        store.delete_task_input(job_id)
        store.delete_draft(draft_id)

    def test_pdf_bundle_round_trip_checks_expected_files_and_size(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "a.pdf").write_bytes(b"%PDF-a")
            (source / "b.pdf").write_bytes(b"%PDF-b")

            bundle = create_pdf_bundle(
                [source / "a.pdf", source / "b.pdf"]
            )
            extract_pdf_bundle(
                bundle,
                ["a.pdf", "b.pdf"],
                target,
                max_total_bytes=100,
            )

            self.assertEqual((target / "a.pdf").read_bytes(), b"%PDF-a")
            self.assertEqual((target / "b.pdf").read_bytes(), b"%PDF-b")
            self.assertEqual(len(sha256_hex(bundle)), 64)

            with self.assertRaisesRegex(ValueError, "文件清单"):
                extract_pdf_bundle(
                    bundle,
                    ["a.pdf"],
                    root / "bad-target",
                    max_total_bytes=100,
                )

    def test_task_inputs_are_immutable_and_drafts_are_replaceable(self):
        with TemporaryDirectory() as temp_dir:
            store = LocalVersionStore(Path(temp_dir))
            job_id = "job-" + "a" * 32
            draft_id = "kb-backend-test-00000001"
            task_manifest = {
                "job_id": job_id,
                "files": ["first.pdf"],
                "expires_at": "2026-07-28T02:00:00+00:00",
            }
            first_draft = {
                "knowledge_base_id": draft_id,
                "files": ["first.pdf"],
            }
            second_draft = {
                "knowledge_base_id": draft_id,
                "files": ["second.pdf"],
            }

            store.save_task_input(job_id, task_manifest, b"task")
            self.assertEqual(
                store.load_task_input(job_id),
                (task_manifest, b"task"),
            )
            with self.assertRaisesRegex(ValueError, "已经存在"):
                store.save_task_input(job_id, task_manifest, b"duplicate")

            store.save_draft(draft_id, first_draft, b"first")
            store.save_draft(draft_id, second_draft, b"second")
            self.assertEqual(
                store.load_draft(draft_id),
                (second_draft, b"second"),
            )

            store.delete_task_input(job_id)
            store.delete_draft(draft_id)
            with self.assertRaisesRegex(ValueError, "不存在"):
                store.load_task_input(job_id)
            with self.assertRaisesRegex(ValueError, "不存在"):
                store.load_draft(draft_id)

    def test_expired_task_inputs_are_cleaned_up(self):
        with TemporaryDirectory() as temp_dir:
            store = LocalVersionStore(Path(temp_dir))
            expired_id = "job-" + "c" * 32
            active_id = "job-" + "d" * 32
            store.save_task_input(
                expired_id,
                {
                    "job_id": expired_id,
                    "expires_at": "2026-07-28T01:00:00+00:00",
                },
                b"expired",
            )
            store.save_task_input(
                active_id,
                {
                    "job_id": active_id,
                    "expires_at": "2026-07-28T03:00:00+00:00",
                },
                b"active",
            )

            removed = store.cleanup_expired_task_inputs(
                "2026-07-28T02:00:00+00:00"
            )

            self.assertEqual(removed, 1)
            with self.assertRaisesRegex(ValueError, "不存在"):
                store.load_task_input(expired_id)
            self.assertEqual(store.load_task_input(active_id)[1], b"active")


if __name__ == "__main__":
    unittest.main()
