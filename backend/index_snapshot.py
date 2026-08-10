import hashlib
import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile


SNAPSHOT_SCHEMA_VERSION = 1
FULL_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def build_fingerprint(rag_mode: str) -> str:
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "rag_mode": rag_mode,
        "chunk_size": 800,
        "chunk_overlap": 150,
    }
    if rag_mode == "full":
        payload["embedding_model"] = FULL_EMBEDDING_MODEL
    else:
        payload.update(
            {
                "vectorizer": "tfidf-char",
                "ngram_range": [2, 4],
                "max_features": 15000,
            }
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_index_snapshot(index_path: Path, rag_mode: str) -> tuple[dict, bytes]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise ValueError("索引不存在，无法创建快照。")
    kind = "directory" if index_path.is_dir() else "file"
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        if kind == "file":
            archive.write(index_path, arcname="index.file")
        else:
            files = sorted(
                path
                for path in index_path.rglob("*")
                if path.is_file()
            )
            if not files:
                raise ValueError("索引目录为空，无法创建快照。")
            for path in files:
                archive.write(
                    path,
                    arcname=(
                        "index/"
                        + path.relative_to(index_path).as_posix()
                    ),
                )
    bundle = buffer.getvalue()
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "rag_mode": rag_mode,
        "fingerprint": build_fingerprint(rag_mode),
        "kind": kind,
        "sha256": hashlib.sha256(bundle).hexdigest(),
        "size_bytes": len(bundle),
    }, bundle


def snapshot_is_compatible(metadata: dict, rag_mode: str) -> bool:
    return (
        metadata.get("schema_version") == SNAPSHOT_SCHEMA_VERSION
        and metadata.get("rag_mode") == rag_mode
        and metadata.get("fingerprint") == build_fingerprint(rag_mode)
        and metadata.get("kind") in {"file", "directory"}
    )


def extract_index_snapshot(
    bundle: bytes,
    metadata: dict,
    target_path: Path,
    *,
    max_total_bytes: int,
) -> None:
    if len(bundle) > max_total_bytes:
        raise ValueError("索引快照超过资源限制。")
    if hashlib.sha256(bundle).hexdigest() != metadata.get("sha256"):
        raise ValueError("索引快照完整性校验失败。")
    if len(bundle) != metadata.get("size_bytes"):
        raise ValueError("索引快照大小校验失败。")
    try:
        archive = ZipFile(BytesIO(bundle))
    except BadZipFile as exc:
        raise ValueError("索引快照已损坏。") from exc

    target_path = Path(target_path)
    staging_root = target_path.parent / (
        f".{target_path.name}.snapshot-{uuid4().hex}"
    )
    try:
        with archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
            ]
            if not members:
                raise ValueError("索引快照为空。")
            if sum(member.file_size for member in members) > max_total_bytes:
                raise ValueError("索引快照解压后超过资源限制。")
            for member in members:
                parts = Path(member.filename).parts
                if ".." in parts or Path(member.filename).is_absolute():
                    raise ValueError("索引快照路径无效。")

            staging_root.mkdir(parents=True, exist_ok=False)
            if metadata["kind"] == "file":
                if [member.filename for member in members] != ["index.file"]:
                    raise ValueError("索引文件快照结构无效。")
                with archive.open(members[0]) as source:
                    (staging_root / "index.file").write_bytes(source.read())
                os.replace(staging_root / "index.file", target_path)
                staging_root.rmdir()
            else:
                if any(
                    not member.filename.startswith("index/")
                    for member in members
                ):
                    raise ValueError("索引目录快照结构无效。")
                extracted = staging_root / "index"
                for member in members:
                    relative = Path(member.filename).relative_to("index")
                    destination = extracted / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source:
                        destination.write_bytes(source.read())
                os.replace(extracted, target_path)
                staging_root.rmdir()
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
