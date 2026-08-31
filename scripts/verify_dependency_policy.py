"""Verify the repository's dependency and Chroma security boundaries."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOCKS = (
    ROOT / "backend" / "requirements.txt",
    ROOT / "backend" / "requirements-windows.txt",
    ROOT / "backend" / "requirements-full.txt",
    ROOT / "backend" / "requirements-full-windows.txt",
    ROOT / "backend" / "requirements-dev.txt",
    ROOT / "backend" / "requirements-dev-windows.txt",
    ROOT / "legacy_streamlit" / "requirements.txt",
    ROOT / "legacy_streamlit" / "requirements-windows.txt",
)


def requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "--"))
    ]


def main() -> None:
    failures: list[str] = []
    for lock in LOCKS:
        if not lock.is_file():
            failures.append(f"missing lock: {lock.relative_to(ROOT)}")
            continue
        unpinned = [line for line in requirement_lines(lock) if "==" not in line]
        if unpinned:
            failures.append(f"unpinned requirements in {lock.relative_to(ROOT)}: {unpinned}")

    production_locks = (
        ROOT / "backend" / "requirements.txt",
        ROOT / "backend" / "requirements-windows.txt",
        ROOT / "backend" / "requirements-full.txt",
        ROOT / "backend" / "requirements-full-windows.txt",
        ROOT / "backend" / "requirements-dev.txt",
        ROOT / "backend" / "requirements-dev-windows.txt",
    )
    for lock in production_locks:
        if lock.is_file() and any(
            line.lower().startswith("langchain-community==")
            for line in requirement_lines(lock)
        ):
            failures.append(f"langchain-community is forbidden in {lock.relative_to(ROOT)}")

    base_packages = {
        line.partition("==")[0].lower()
        for line in requirement_lines(ROOT / "backend" / "requirements.txt")
    }
    for package in ("chromadb", "langchain-chroma", "langchain-huggingface"):
        if package in base_packages:
            failures.append(f"full-mode package leaked into the light lock: {package}")

    full_packages = {
        line.partition("==")[0].lower()
        for line in requirement_lines(ROOT / "backend" / "requirements-full.txt")
    }
    if "chromadb" not in full_packages:
        failures.append("the full lock is missing its declared Chroma dependency")

    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "backend").rglob("*.py")
        if not path.name.startswith("test_")
    )
    forbidden_chroma_network_usage = (
        "chromadb.HttpClient",
        "chromadb.AsyncHttpClient",
        "chromadb.api.fastapi",
        "chroma run",
    )
    for marker in forbidden_chroma_network_usage:
        if marker in production_source:
            failures.append(f"networked Chroma usage is forbidden: {marker}")
    rag_source = (ROOT / "backend" / "rag_core.py").read_text(encoding="utf-8")
    if "persist_directory=" not in rag_source:
        failures.append("embedded Chroma persist_directory boundary is missing")

    if failures:
        raise SystemExit("\n".join(failures))
    print("Dependency policy: PASS")


if __name__ == "__main__":
    main()
