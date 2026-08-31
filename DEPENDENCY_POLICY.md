# Dependency Security and Reproducibility

Python 3.11.16 and Node.js 24.18.0 are the reference toolchain versions. The
container base image is pinned by its multi-platform digest, and pip is pinned
to 26.2.1 during CI and image builds. `.python-version` and `.nvmrc` expose the
same versions to compatible local version managers. GitHub Actions are pinned
to immutable commit SHAs rather than moving major-version tags.

## Lock files

The `*.in` files contain reviewed direct dependencies. The generated `*.txt`
files contain exact transitive environments. Primary lock names target Linux;
their `*-windows.txt` counterparts target Windows:

- `backend/requirements.txt`: certified Linux light runtime and container.
- `backend/requirements-full.txt`: optional Linux full RAG runtime.
- `backend/requirements-dev.txt`: Linux test, lock-generation, and audit tools.
- `legacy_streamlit/requirements.txt`: isolated Linux archived prototype.
- `backend/requirements-windows.txt`, `requirements-full-windows.txt`, and
  `requirements-dev-windows.txt`: corresponding Windows environments.
- `legacy_streamlit/requirements-windows.txt`: archived Windows prototype.

Regenerate locks from the repository root with pip-tools 7.6.1:

    python -m piptools compile --quiet --no-header --resolver=backtracking --strip-extras --output-file=backend/requirements.txt backend/requirements.in
    python -m piptools compile --quiet --no-header --resolver=backtracking --strip-extras --output-file=backend/requirements-full.txt backend/requirements-full.in
    python -m piptools compile --quiet --no-header --resolver=backtracking --strip-extras --output-file=backend/requirements-dev.txt backend/requirements-dev.in
    python -m piptools compile --quiet --no-header --resolver=backtracking --strip-extras --output-file=legacy_streamlit/requirements.txt legacy_streamlit/requirements.in

Run those commands on Linux for the primary locks. On Windows, use the same
commands with `requirements-windows.txt`, `requirements-full-windows.txt`,
`requirements-dev-windows.txt`, and `legacy_streamlit/requirements-windows.txt`
as their respective output files. CI regenerates the four Linux locks and fails
on any diff; the policy checker verifies exact pins in all eight locks. Python
environments use the lock matching their OS; the frontend uses `npm ci` with
`package-lock.json`.

## Security gates and the Chroma exception

CI runs `pip-audit` for the certified light runtime with no exceptions and
`npm audit --audit-level=high` for the complete frontend dependency tree.

ChromaDB 1.5.9 currently has four published advisories with no fixed release:
`PYSEC-2026-311`, `CVE-2026-45830`, `CVE-2026-45831`, and
`CVE-2026-45833`. The advisories concern the network server/API boundary. This
project permits a narrow exception only for optional local full mode, where
Chroma is embedded with `persist_directory`; it does not start a Chroma server,
use `chromadb.HttpClient`, or expose a Chroma port. The certified light-mode
container does not install ChromaDB.

`scripts/verify_dependency_policy.py` enforces that production locks exclude
`langchain-community`, all lock entries are exact, and the full runtime remains
inside the embedded Chroma boundary. CI ignores only the four identifiers above
when auditing the full lock. Remove the exception as soon as a fixed compatible
Chroma release exists; do not broaden it to other advisories.

The archived Streamlit proof of concept is not a production or release
dependency surface. Install it only in a separate environment as documented in
`legacy_streamlit/README.md`.
