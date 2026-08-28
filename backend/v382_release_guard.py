"""Public V3.82 release guards for research hashes and private artifacts."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "V382_RESEARCH_FREEZE_MANIFEST.json"
RESEARCH_FREEZE_COMMIT = "3d2a1a1f32554d88f5a3d7ee35b70be72eb761ec"

PRIVATE_PATH_PARTS = (
    "backend/evaluation/benchmark_private/",
    "backend/data/",
    "backend/vector_db/",
    "backend/light_indexes/",
    "backend/public_versions/",
    "backend/runtime_state/",
    "results/",
    "vector_db/",
    "vector_db_v369/",
)
PRIVATE_SUFFIXES = {".pdf", ".sqlite", ".sqlite3", ".pkl", ".key", ".pem"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)


def _matches_frozen_file(path: Path, relative: str, expected: str) -> bool:
    current = path.read_bytes()
    if hashlib.sha256(current).hexdigest() == expected:
        return True
    frozen = subprocess.run(
        ["git", "show", f"{RESEARCH_FREEZE_COMMIT}:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if frozen.returncode != 0:
        return False
    return current.replace(b"\r\n", b"\n") == frozen.stdout.replace(
        b"\r\n", b"\n"
    )


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_research_freeze() -> dict:
    manifest = load_manifest()
    drift = []
    missing = []
    for relative, expected in manifest["file_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif not _matches_frozen_file(path, relative, expected):
            drift.append(relative)
    category_files = {
        relative
        for paths in manifest["categories"].values()
        for relative in paths
    }
    unpinned = sorted(category_files - set(manifest["file_sha256"]))
    return {
        "status": "PASS" if not (drift or missing or unpinned) else "FAIL",
        "checked_files": len(manifest["file_sha256"]),
        "drift": drift,
        "missing": missing,
        "unpinned": unpinned,
        "baseline": manifest["baseline"],
    }


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [item for item in result.stdout.decode("utf-8").split("\0") if item]


def audit_tracked_private_files() -> dict:
    private = []
    secret_hits = []
    for relative in tracked_files():
        normalized = relative.replace("\\", "/")
        lower = normalized.lower()
        suffix = Path(lower).suffix
        if lower == ".env" or suffix in PRIVATE_SUFFIXES or any(
            part in lower for part in PRIVATE_PATH_PARTS
        ):
            private.append(normalized)
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_hits.append(normalized)
    failures = sorted(set(private + secret_hits))
    return {
        "status": "PASS" if not failures else "FAIL",
        "tracked_private_files": len(failures),
        "paths": failures,
    }


def run_all_guards() -> dict:
    freeze = verify_research_freeze()
    private = audit_tracked_private_files()
    return {
        "status": "PASS" if freeze["status"] == private["status"] == "PASS" else "FAIL",
        "research_freeze": freeze,
        "private_artifacts": private,
    }


def main() -> None:
    report = run_all_guards()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
