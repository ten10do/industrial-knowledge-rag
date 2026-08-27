"""Read-only index integrity validation.

Validates that the configured index is present, loadable, and consistent with
the runtime configuration BEFORE the application queries it. Never mutates and
NEVER rebuilds: an incompatible index must surface as an explicit readiness
failure / operator action, not silent repair.

Strictness model:
 - hard facts (exists / openable / chunk count>0 / embedding dimension) FAIL;
 - an OPTIONAL sidecar ``index_manifest.json`` (schema documented in
   OPERATIONS_RUNBOOK.md) is enforced strictly when present;
 - full private/production indexes without a manifest (e.g. frozen research
   indexes created before manifests existed) validate on hard facts only and
   are reported as warnings, not failures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "index_manifest.json"
MANIFEST_VERSION = 1
FULL_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
FULL_EMBEDDING_DIM = 384


@dataclass
class IndexIntegrityReport:
    ok: bool
    index_kind: str  # "full" | "light"
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": name, "ok": bool(passed), "detail": detail})
        if not passed:
            self.ok = False

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "index_kind": self.index_kind,
            "checks": list(self.checks),
            "warnings": list(self.warnings),
        }


def _safe_count_detail(exc: Exception) -> str:
    return f"index_unreadable:{type(exc).__name__}"


def validate_full_index(persist_dir: Path) -> IndexIntegrityReport:
    report = IndexIntegrityReport(ok=True, index_kind="full")
    persist_dir = Path(persist_dir)

    if not persist_dir.is_dir():
        report.add("index_exists", False, f"missing_directory:{persist_dir.name}")
        return report
    report.add("index_exists", True, "present")

    if not (persist_dir / "chroma.sqlite3").is_file():
        report.add("chroma_store_present", False, "chroma.sqlite3 missing")
        return report
    report.add("chroma_store_present", True, "present")

    try:
        from langchain_chroma import Chroma

        db = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=None,  # read-only count path; no model load
        )
        collection = db._collection
        count = collection.count()
    except Exception as exc:  # unreadable store must fail readiness loudly
        report.add("index_openable", False, _safe_count_detail(exc))
        return report
    report.add("index_openable", True, "opened_read_only")

    if count <= 0:
        report.add("chunk_count", False, "chunk_count=0")
    else:
        report.add("chunk_count", True, "non_empty")

    dimension = None
    try:
        sample = collection.get(limit=1, include=["embeddings"])
        vectors = sample.get("embeddings") or []
        if vectors:
            dimension = len(vectors[0])
    except Exception as exc:  # non-fatal: some stores restrict embedding reads
        report.warn(f"embedding_dimension_unknown:{type(exc).__name__}")
    if dimension is not None:
        if dimension == FULL_EMBEDDING_DIM:
            report.add("embedding_dimension", True, f"dim={dimension}")
        else:
            report.add(
                "embedding_dimension",
                False,
                f"dim={dimension},expected={FULL_EMBEDDING_DIM}",
            )

    manifest_path = persist_dir / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.add("manifest_valid", False, f"manifest_parse_error:{type(exc).__name__}")
            return report
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            report.add(
                "manifest_valid",
                False,
                f"manifest_version={manifest.get('manifest_version')!r}",
            )
            return report
        report.add("manifest_valid", True, f"version={MANIFEST_VERSION}")
        required_fields = {
            "index_version",
            "embedding_model",
            "embedding_dimension",
            "metadata_schema_version",
            "chunk_count",
        }
        missing_fields = sorted(required_fields - set(manifest))
        if missing_fields:
            report.add("manifest_schema", False, "required_fields_missing")
            return report
        report.add("manifest_schema", True, "complete")
        expected_model = manifest.get("embedding_model")
        if expected_model not in (None, FULL_EMBEDDING_MODEL):
            report.add(
                "embedding_model_identity",
                False,
                "model mismatch with runtime embedding configuration",
            )
        expected_dimension = manifest.get("embedding_dimension")
        if expected_dimension != FULL_EMBEDDING_DIM:
            report.add(
                "manifest_embedding_dimension",
                False,
                "embedding dimension mismatch with runtime configuration",
            )
        expected_count = manifest.get("chunk_count")
        if isinstance(expected_count, int) and expected_count != count:
            report.add(
                "manifest_chunk_count",
                False,
                "manifest count does not match index",
            )
    else:
        report.warn("no_manifest:facts_only_validation")
    return report


def validate_light_index(index_path: Path) -> IndexIntegrityReport:
    report = IndexIntegrityReport(ok=True, index_kind="light")
    index_path = Path(index_path)
    if not index_path.is_file():
        report.add("index_exists", False, f"missing_file:{index_path.name}")
        return report
    report.add("index_exists", True, "present")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.add("index_openable", False, f"parse_error:{type(exc).__name__}")
        return report
    report.add("index_openable", True, "parsed_json")
    if not isinstance(payload, list) or not payload:
        report.add("chunk_count", False, "document_list_empty")
        return report
    report.add("chunk_count", True, "non_empty")
    first = payload[0]
    has_shape = isinstance(first, dict) and "page_content" in first and "metadata" in first
    report.add("document_schema", has_shape, "page_content+metadata" if has_shape else "unexpected item shape")
    return report
