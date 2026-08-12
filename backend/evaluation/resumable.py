"""Atomic, resumable persistence for long-running private evaluations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_STATUSES = {"PENDING", "RUNNING", "PARTIAL", "COMPLETED", "FAILED"}


class CheckpointCorruptionError(RuntimeError):
    """A checkpoint exists but cannot safely be read."""


class ResumeConfigurationMismatch(RuntimeError):
    """A checkpoint belongs to a different frozen experiment."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably write JSON before replacing the visible checkpoint on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise CheckpointCorruptionError(f"Corrupted checkpoint: {path}") from exc


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    evaluation_version: str
    corpus_id: str
    pipeline_id: str
    manifest_hash: str
    annotation_hash: str
    configuration_hash: str
    started_at: str
    updated_at: str
    status: str = "PENDING"


class CheckpointStore:
    """Persist independent stage results and per-query rows for one evaluation run."""

    def __init__(self, runtime_root: Path, identity: EvaluationRun):
        self.identity = identity
        self.root = runtime_root / identity.run_id
        self.manifest_path = self.root / "run_manifest.json"
        self.progress_path = self.root / "progress.json"
        self.stages_path = self.root / "stages"

    def initialize(self, *, resume: bool = False, restart: bool = False) -> None:
        if restart and self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted((item for item in self.root.rglob("*") if item.is_dir()), reverse=True):
                path.rmdir()
            self.root.rmdir()
        if self.manifest_path.exists():
            existing = read_json(self.manifest_path)
            if not resume:
                raise FileExistsError(f"Checkpoint already exists; use --resume or --restart: {self.root}")
            self._validate_identity(existing)
            return
        if resume:
            raise FileNotFoundError(f"No checkpoint is available for --resume: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, {
            **asdict(self.identity),
            "completed_stages": [], "failed_stages": [], "result_files": {},
            "finished_at": None, "elapsed_seconds": 0.0,
        })
        atomic_write_json(self.progress_path, {
            "current_stage": None, "completed_stages": [], "current_query": None,
            "completed_queries": 0, "total_queries": 0, "elapsed_seconds": 0.0,
            "last_checkpoint": utc_now(),
        })

    def _validate_identity(self, existing: dict[str, Any]) -> None:
        keys = ("evaluation_version", "corpus_id", "pipeline_id", "manifest_hash", "annotation_hash", "configuration_hash")
        if any(existing.get(key) != getattr(self.identity, key) for key in keys):
            raise ResumeConfigurationMismatch("RESUME_REFUSED_CONFIGURATION_MISMATCH")

    def stage_path(self, stage: str) -> Path:
        return self.stages_path / f"{stage.lower()}.json"

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        path = self.stage_path(stage)
        return read_json(path) if path.exists() else None

    def save_stage(self, stage: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.stage_path(stage), payload)
        manifest = read_json(self.manifest_path)
        completed = set(manifest.get("completed_stages", []))
        completed.add(stage)
        manifest["completed_stages"] = sorted(completed)
        manifest["result_files"][stage] = str(self.stage_path(stage).name)
        manifest["updated_at"] = utc_now()
        manifest["status"] = "RUNNING"
        atomic_write_json(self.manifest_path, manifest)
        self.save_progress(stage, None, len(payload.get("rows", [])), len(payload.get("rows", [])), completed)

    def begin_stage(self, stage: str, total_queries: int) -> None:
        """Persist that a stage is active before its first expensive operation."""
        manifest = read_json(self.manifest_path)
        completed = set(manifest.get("completed_stages", []))
        completed.discard(stage)
        manifest["completed_stages"] = sorted(completed)
        manifest["updated_at"] = utc_now()
        manifest["status"] = "RUNNING"
        atomic_write_json(self.manifest_path, manifest)
        self.save_progress(stage, None, 0, total_queries, completed)

    def save_query(
        self, stage: str, query_id: str, row: dict[str, Any], *, total_queries: int,
        error: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = self.load_stage(stage) or {"stage": stage, "rows": {}, "errors": {}}
        payload["rows"][query_id] = row
        if error:
            payload["errors"][query_id] = error
        else:
            payload["errors"].pop(query_id, None)
        atomic_write_json(self.stage_path(stage), payload)
        self.save_progress(stage, query_id, len(payload["rows"]), total_queries, None)
        return payload

    def save_progress(
        self, stage: str | None, current_query: str | None, completed_queries: int,
        total_queries: int, completed_stages: set[str] | None,
    ) -> None:
        previous = read_json(self.progress_path)
        started_at = datetime.fromisoformat(read_json(self.manifest_path)["started_at"])
        atomic_write_json(self.progress_path, {
            **previous,
            "current_stage": stage,
            "completed_stages": sorted(completed_stages if completed_stages is not None else set(previous.get("completed_stages", []))),
            "current_query": current_query,
            "completed_queries": completed_queries,
            "total_queries": total_queries,
            "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            "last_checkpoint": utc_now(),
        })

    def finalize(self, status: str) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unknown evaluation run status: {status}")
        manifest = read_json(self.manifest_path)
        manifest.update({"status": status, "updated_at": utc_now(), "finished_at": utc_now()})
        atomic_write_json(self.manifest_path, manifest)
