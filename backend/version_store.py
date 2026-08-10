import hashlib
import json
import os
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile


VERSION_ID_PATTERN = re.compile(
    r"^v-\d{8}T\d{12}Z-[a-f0-9]{8}$"
)
JOB_ID_PATTERN = re.compile(r"^job-[a-f0-9]{32}$")
KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"^kb-[A-Za-z0-9_-]{16,64}$")


def validate_version_id(version_id: str) -> str:
    if not VERSION_ID_PATTERN.fullmatch(version_id):
        raise ValueError("知识库版本 ID 格式无效。")
    return version_id


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("任务 ID 格式无效。")
    return job_id


def validate_knowledge_base_id(knowledge_base_id: str) -> str:
    if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(knowledge_base_id):
        raise ValueError("知识库 ID 格式无效。")
    return knowledge_base_id


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_replace(source: Path, target: Path) -> None:
    for attempt in range(3):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.01 * (attempt + 1))


def create_pdf_bundle(pdf_paths) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in pdf_paths:
            pdf_path = Path(path)
            archive.write(pdf_path, arcname=pdf_path.name)
    return buffer.getvalue()


def extract_pdf_bundle(
    bundle: bytes,
    expected_filenames: list[str],
    target_dir: Path,
    max_total_bytes: int,
) -> list[Path]:
    expected = sorted(expected_filenames)
    if not expected or any(Path(name).name != name for name in expected):
        raise ValueError("知识库版本中的文件清单无效。")

    try:
        archive = ZipFile(BytesIO(bundle))
    except BadZipFile as exc:
        raise ValueError("知识库版本压缩包已损坏。") from exc

    with archive:
        members = archive.infolist()
        names = sorted(member.filename for member in members if not member.is_dir())
        if names != expected or len(members) != len(expected):
            raise ValueError("知识库版本压缩包与文件清单不一致。")
        total_bytes = sum(member.file_size for member in members)
        if total_bytes > max_total_bytes:
            raise ValueError("知识库版本解压后超过资源限制。")

        target_dir.mkdir(parents=True, exist_ok=False)
        paths = []
        for member in members:
            target = target_dir / member.filename
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            paths.append(target)
    return paths


def _manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_manifest(value: bytes) -> dict:
    manifest = json.loads(value.decode("utf-8"))
    validate_version_id(str(manifest.get("version_id", "")))
    return manifest


class LocalVersionStore:
    backend_name = "local"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.tasks_dir = self.root / "tasks"
        self.drafts_dir = self.root / "drafts"
        self.active_path = self.root / "active.json"

    def _save_artifact(
        self,
        final_dir: Path,
        manifest: dict,
        bundle: bytes,
        *,
        replace: bool,
    ) -> None:
        if final_dir.exists() and not replace:
            raise ValueError(f"任务输入 {final_dir.name} 已经存在。")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = final_dir.parent / f".{final_dir.name}-{uuid4().hex}"
        backup_dir = final_dir.parent / f".{final_dir.name}.backup-{uuid4().hex}"
        staging_dir.mkdir()
        backup_created = False
        try:
            (staging_dir / "bundle.zip").write_bytes(bundle)
            (staging_dir / "manifest.json").write_bytes(
                _manifest_bytes(manifest)
            )
            if final_dir.exists():
                atomic_replace(final_dir, backup_dir)
                backup_created = True
            atomic_replace(staging_dir, final_dir)
            if backup_created:
                shutil.rmtree(backup_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if backup_created and backup_dir.exists():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                atomic_replace(backup_dir, final_dir)
            raise

    def _load_artifact(self, artifact_dir: Path, label: str) -> tuple[dict, bytes]:
        if not artifact_dir.exists():
            raise ValueError(f"{label} {artifact_dir.name} 不存在。")
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return manifest, (artifact_dir / "bundle.zip").read_bytes()

    def save_version(self, manifest: dict, bundle: bytes) -> None:
        version_id = validate_version_id(manifest["version_id"])
        final_dir = self.versions_dir / version_id
        if final_dir.exists():
            raise ValueError(f"知识库版本 {version_id} 已经存在。")

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = self.versions_dir / f".{version_id}-{uuid4().hex}"
        staging_dir.mkdir()
        try:
            (staging_dir / "bundle.zip").write_bytes(bundle)
            (staging_dir / "manifest.json").write_bytes(
                _manifest_bytes(manifest)
            )
            atomic_replace(staging_dir, final_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

    def load_version(self, version_id: str) -> tuple[dict, bytes]:
        version_id = validate_version_id(version_id)
        version_dir = self.versions_dir / version_id
        if not version_dir.exists():
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        manifest = _load_manifest((version_dir / "manifest.json").read_bytes())
        return manifest, (version_dir / "bundle.zip").read_bytes()

    def save_version_snapshot(
        self,
        version_id: str,
        snapshot: bytes,
    ) -> None:
        version_id = validate_version_id(version_id)
        version_dir = self.versions_dir / version_id
        if not version_dir.exists():
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        snapshot_path = version_dir / "index_snapshot.zip"
        if snapshot_path.exists():
            raise ValueError(f"知识库版本 {version_id} 的索引快照已经存在。")
        temporary_path = version_dir / f".index_snapshot-{uuid4().hex}.tmp"
        temporary_path.write_bytes(snapshot)
        atomic_replace(temporary_path, snapshot_path)

    def load_version_snapshot(self, version_id: str) -> bytes:
        version_id = validate_version_id(version_id)
        snapshot_path = (
            self.versions_dir / version_id / "index_snapshot.zip"
        )
        if not snapshot_path.exists():
            raise ValueError(f"知识库版本 {version_id} 没有索引快照。")
        return snapshot_path.read_bytes()

    def list_versions(self) -> list[dict]:
        if not self.versions_dir.exists():
            return []
        manifests = [
            _load_manifest(path.read_bytes())
            for path in self.versions_dir.glob("*/manifest.json")
        ]
        return sorted(
            manifests,
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def get_active_version_id(self) -> str | None:
        if not self.active_path.exists():
            return None
        payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        return validate_version_id(payload["version_id"])

    def set_active_version(self, version_id: str) -> None:
        version_id = validate_version_id(version_id)
        if not (self.versions_dir / version_id).exists():
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        self.root.mkdir(parents=True, exist_ok=True)
        staging_path = self.root / f".active-{uuid4().hex}.json"
        staging_path.write_bytes(_manifest_bytes({"version_id": version_id}))
        atomic_replace(staging_path, self.active_path)

    def save_task_input(
        self,
        job_id: str,
        manifest: dict,
        bundle: bytes,
    ) -> None:
        job_id = validate_job_id(job_id)
        self._save_artifact(
            self.tasks_dir / job_id,
            manifest,
            bundle,
            replace=False,
        )

    def load_task_input(self, job_id: str) -> tuple[dict, bytes]:
        job_id = validate_job_id(job_id)
        return self._load_artifact(self.tasks_dir / job_id, "任务输入")

    def delete_task_input(self, job_id: str) -> None:
        job_id = validate_job_id(job_id)
        path = self.tasks_dir / job_id
        if path.exists():
            shutil.rmtree(path)

    def cleanup_expired_task_inputs(self, now: str) -> int:
        if not self.tasks_dir.exists():
            return 0
        removed = 0
        for manifest_path in self.tasks_dir.glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires_at = manifest.get("expires_at", "")
            if expires_at and expires_at <= now:
                shutil.rmtree(manifest_path.parent)
                removed += 1
        return removed

    def save_draft(
        self,
        knowledge_base_id: str,
        manifest: dict,
        bundle: bytes,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        self._save_artifact(
            self.drafts_dir / knowledge_base_id,
            manifest,
            bundle,
            replace=True,
        )

    def save_draft_snapshot(
        self,
        knowledge_base_id: str,
        snapshot: bytes,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        draft_dir = self.drafts_dir / knowledge_base_id
        if not draft_dir.exists():
            raise ValueError(f"草稿 {knowledge_base_id} 不存在。")
        path = draft_dir / "index_snapshot.zip"
        temporary_path = draft_dir / f".index_snapshot-{uuid4().hex}.tmp"
        temporary_path.write_bytes(snapshot)
        atomic_replace(temporary_path, path)

    def load_draft_snapshot(self, knowledge_base_id: str) -> bytes:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        path = self.drafts_dir / knowledge_base_id / "index_snapshot.zip"
        if not path.exists():
            raise ValueError(f"草稿 {knowledge_base_id} 没有索引快照。")
        return path.read_bytes()

    def save_draft_build_cache(
        self,
        knowledge_base_id: str,
        cache: dict,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        draft_dir = self.drafts_dir / knowledge_base_id
        if not draft_dir.exists():
            raise ValueError(f"草稿 {knowledge_base_id} 不存在。")
        path = draft_dir / "build_cache.json"
        temporary_path = draft_dir / f".build_cache-{uuid4().hex}.tmp"
        temporary_path.write_bytes(_manifest_bytes(cache))
        atomic_replace(temporary_path, path)

    def load_draft_build_cache(self, knowledge_base_id: str) -> dict:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        path = self.drafts_dir / knowledge_base_id / "build_cache.json"
        if not path.exists():
            raise ValueError(f"草稿 {knowledge_base_id} 没有构建缓存。")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_draft(self, knowledge_base_id: str) -> tuple[dict, bytes]:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        return self._load_artifact(
            self.drafts_dir / knowledge_base_id,
            "草稿",
        )

    def delete_draft(self, knowledge_base_id: str) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        path = self.drafts_dir / knowledge_base_id
        if path.exists():
            shutil.rmtree(path)


class S3VersionStore:
    backend_name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "industrial-knowledge-rag/public",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        client=None,
    ):
        if not bucket:
            raise RuntimeError("PUBLIC_VERSION_S3_BUCKET 不能为空。")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3 版本存储需要安装 boto3。"
                ) from exc
            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region_name or None,
            )
        self.client = client

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _manifest_key(self, version_id: str) -> str:
        return self._key(f"versions/{version_id}/manifest.json")

    def _bundle_key(self, version_id: str) -> str:
        return self._key(f"versions/{version_id}/bundle.zip")

    def _exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            code = str(
                getattr(exc, "response", {})
                .get("Error", {})
                .get("Code", "")
            )
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def _read(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def save_version(self, manifest: dict, bundle: bytes) -> None:
        version_id = validate_version_id(manifest["version_id"])
        manifest_key = self._manifest_key(version_id)
        if self._exists(manifest_key):
            raise ValueError(f"知识库版本 {version_id} 已经存在。")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._bundle_key(version_id),
            Body=bundle,
            ContentType="application/zip",
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=_manifest_bytes(manifest),
            ContentType="application/json",
        )

    def load_version(self, version_id: str) -> tuple[dict, bytes]:
        version_id = validate_version_id(version_id)
        if not self._exists(self._manifest_key(version_id)):
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        manifest = _load_manifest(self._read(self._manifest_key(version_id)))
        return manifest, self._read(self._bundle_key(version_id))

    def save_version_snapshot(
        self,
        version_id: str,
        snapshot: bytes,
    ) -> None:
        version_id = validate_version_id(version_id)
        if not self._exists(self._manifest_key(version_id)):
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        key = self._key(
            f"versions/{version_id}/index_snapshot.zip"
        )
        if self._exists(key):
            raise ValueError(f"知识库版本 {version_id} 的索引快照已经存在。")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=snapshot,
            ContentType="application/zip",
        )

    def load_version_snapshot(self, version_id: str) -> bytes:
        version_id = validate_version_id(version_id)
        key = self._key(
            f"versions/{version_id}/index_snapshot.zip"
        )
        if not self._exists(key):
            raise ValueError(f"知识库版本 {version_id} 没有索引快照。")
        return self._read(key)

    def list_versions(self) -> list[dict]:
        versions_prefix = self._key("versions/")
        paginator = self.client.get_paginator("list_objects_v2")
        manifests = []
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=versions_prefix,
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/manifest.json"):
                    manifests.append(_load_manifest(self._read(key)))
        return sorted(
            manifests,
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def get_active_version_id(self) -> str | None:
        key = self._key("active.json")
        if not self._exists(key):
            return None
        payload = json.loads(self._read(key).decode("utf-8"))
        return validate_version_id(payload["version_id"])

    def set_active_version(self, version_id: str) -> None:
        version_id = validate_version_id(version_id)
        if not self._exists(self._manifest_key(version_id)):
            raise ValueError(f"知识库版本 {version_id} 不存在。")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key("active.json"),
            Body=_manifest_bytes({"version_id": version_id}),
            ContentType="application/json",
        )

    def _save_artifact(
        self,
        prefix: str,
        manifest: dict,
        bundle: bytes,
        *,
        immutable: bool,
    ) -> None:
        manifest_key = self._key(f"{prefix}/manifest.json")
        if immutable and self._exists(manifest_key):
            raise ValueError(f"任务输入 {Path(prefix).name} 已经存在。")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(f"{prefix}/bundle.zip"),
            Body=bundle,
            ContentType="application/zip",
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=_manifest_bytes(manifest),
            ContentType="application/json",
        )

    def _load_artifact(self, prefix: str, label: str) -> tuple[dict, bytes]:
        manifest_key = self._key(f"{prefix}/manifest.json")
        if not self._exists(manifest_key):
            raise ValueError(f"{label} {Path(prefix).name} 不存在。")
        manifest = json.loads(self._read(manifest_key).decode("utf-8"))
        bundle = self._read(self._key(f"{prefix}/bundle.zip"))
        return manifest, bundle

    def _delete_artifact(self, prefix: str) -> None:
        for filename in ("manifest.json", "bundle.zip"):
            self.client.delete_object(
                Bucket=self.bucket,
                Key=self._key(f"{prefix}/{filename}"),
            )

    def save_task_input(
        self,
        job_id: str,
        manifest: dict,
        bundle: bytes,
    ) -> None:
        job_id = validate_job_id(job_id)
        self._save_artifact(
            f"tasks/{job_id}",
            manifest,
            bundle,
            immutable=True,
        )

    def load_task_input(self, job_id: str) -> tuple[dict, bytes]:
        job_id = validate_job_id(job_id)
        return self._load_artifact(f"tasks/{job_id}", "任务输入")

    def delete_task_input(self, job_id: str) -> None:
        job_id = validate_job_id(job_id)
        self._delete_artifact(f"tasks/{job_id}")

    def cleanup_expired_task_inputs(self, now: str) -> int:
        tasks_prefix = self._key("tasks/")
        paginator = self.client.get_paginator("list_objects_v2")
        removed = 0
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=tasks_prefix,
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith("/manifest.json"):
                    continue
                manifest = json.loads(self._read(key).decode("utf-8"))
                expires_at = manifest.get("expires_at", "")
                if expires_at and expires_at <= now:
                    job_id = key.split("/")[-2]
                    self.delete_task_input(job_id)
                    removed += 1
        return removed

    def save_draft(
        self,
        knowledge_base_id: str,
        manifest: dict,
        bundle: bytes,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        self._save_artifact(
            f"drafts/{knowledge_base_id}",
            manifest,
            bundle,
            immutable=False,
        )

    def save_draft_snapshot(
        self,
        knowledge_base_id: str,
        snapshot: bytes,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        key = self._key(
            f"drafts/{knowledge_base_id}/index_snapshot.zip"
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=snapshot,
            ContentType="application/zip",
        )

    def load_draft_snapshot(self, knowledge_base_id: str) -> bytes:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        key = self._key(
            f"drafts/{knowledge_base_id}/index_snapshot.zip"
        )
        if not self._exists(key):
            raise ValueError(f"草稿 {knowledge_base_id} 没有索引快照。")
        return self._read(key)

    def save_draft_build_cache(
        self,
        knowledge_base_id: str,
        cache: dict,
    ) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(
                f"drafts/{knowledge_base_id}/build_cache.json"
            ),
            Body=_manifest_bytes(cache),
            ContentType="application/json",
        )

    def load_draft_build_cache(self, knowledge_base_id: str) -> dict:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        key = self._key(
            f"drafts/{knowledge_base_id}/build_cache.json"
        )
        if not self._exists(key):
            raise ValueError(f"草稿 {knowledge_base_id} 没有构建缓存。")
        return json.loads(self._read(key).decode("utf-8"))

    def load_draft(self, knowledge_base_id: str) -> tuple[dict, bytes]:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        return self._load_artifact(
            f"drafts/{knowledge_base_id}",
            "草稿",
        )

    def delete_draft(self, knowledge_base_id: str) -> None:
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        self._delete_artifact(f"drafts/{knowledge_base_id}")
        for filename in ("index_snapshot.zip", "build_cache.json"):
            self.client.delete_object(
                Bucket=self.bucket,
                Key=self._key(
                    f"drafts/{knowledge_base_id}/{filename}"
                ),
            )


def create_version_store(base_dir: Path):
    backend = os.getenv(
        "PUBLIC_VERSION_STORAGE_BACKEND",
        "local",
    ).strip().lower()
    if backend == "local":
        configured = os.getenv("PUBLIC_VERSION_STORAGE_DIR", "").strip()
        root = Path(configured) if configured else base_dir / "public_versions"
        if not root.is_absolute():
            root = base_dir / root
        return LocalVersionStore(root)
    if backend == "s3":
        return S3VersionStore(
            bucket=os.getenv("PUBLIC_VERSION_S3_BUCKET", "").strip(),
            prefix=os.getenv(
                "PUBLIC_VERSION_S3_PREFIX",
                "industrial-knowledge-rag/public",
            ).strip(),
            endpoint_url=os.getenv(
                "PUBLIC_VERSION_S3_ENDPOINT_URL",
                "",
            ).strip(),
            region_name=os.getenv(
                "PUBLIC_VERSION_S3_REGION",
                "",
            ).strip(),
        )
    raise RuntimeError(
        "PUBLIC_VERSION_STORAGE_BACKEND 只支持 local 或 s3。"
    )
