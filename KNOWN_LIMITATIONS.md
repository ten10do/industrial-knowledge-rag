# Known Limitations — V3.83

These limitations are release facts, not hidden backlog.

## Frozen correctness boundary

- The formal aligned baseline is **54/69 correct** (78.26%), with **9 residual false answers** and **3 residual false refusals**.
- Answerable recall is 76.92%; abstention recall is 79.07%; the hard-negative slice is 10/10.
- `CORPUS_UNSUPPORTED` and out-of-domain (OOD) queries retain residual false-answer risk. A retrieved passage is not proof that every requested claim is supported.
- The V3.79 claim-support validator is not production-ready. Its experimental path remains default OFF.
- Local NLI did not provide useful signal for the current claim-support task and is not a production dependency.
- The V3.80 requirement-extraction upgrade did not improve system safety and is not wired into production.
- The V3.81 spec-existence lattice is **NO-GO**: strict precision was 0.61 against a required gate of at least 0.95.

## Structured-document boundary

- Exact table-cell ownership is unsupported.
- `TableContextBundle` is experimental and default OFF.
- Current ingestion cannot reliably recover rows, columns, headers, merged cells, or cell provenance from arbitrary PDFs.

## Deferred research

`LAYOUT_AWARE_INGESTION_REBUILD` is `DEFERRED_RESEARCH_BACKLOG` and `NOT_PART_OF_CURRENT_RELEASE`.

A future phase requires layout-aware PDF parsing; explicit rows, columns, headers, merged cells and cell provenance; structure-aware chunks; a rebuilt corpus; a new benchmark; and new safety gates. V3.83 intentionally does not implement that research.

## Operational boundary

- The native production-like lifecycle and 10-minute soak passed, but the formal Docker Compose release topology has not completed image-build or container-lifecycle validation because the local Docker daemon is unavailable.
- CI workflow configuration is valid, but this repository has no configured Git remote and the V3.83 commits were not pushed; `CI_EXECUTED_PASS` is therefore not claimed.
- No runtime-parseable Groq or DeepSeek key was available. Optional-provider degradation and safe failure passed, but a real successful generation and its `ANSWER` metric were not validated.
- V3.83 is not a claim of Kubernetes or multi-region certification.
- In-process metrics reset on process restart. External scraping/persistence is an operator responsibility.
- Memory queue/rate-limit backends are single-process. Multi-process deployments require Redis and live workers.
- Provider readiness checks validate configuration, not a billable remote completion. Real Groq/DeepSeek availability is reported only when a request is attempted.
- A legacy index without an integrity manifest is accepted only after fact-based validation and reported as degraded; new production full indexes should ship the documented manifest.
