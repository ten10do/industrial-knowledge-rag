# Known Limitations — V3.82

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

A future phase requires layout-aware PDF parsing; explicit rows, columns, headers, merged cells and cell provenance; structure-aware chunks; a rebuilt corpus; a new benchmark; and new safety gates. V3.82 intentionally does not implement that research.

## Operational boundary

- V3.82 is a development-ready production-readiness baseline, not a claim of Kubernetes, multi-region, or long-soak production certification.
- In-process metrics reset on process restart. External scraping/persistence is an operator responsibility.
- Memory queue/rate-limit backends are single-process. Multi-process deployments require Redis and live workers.
- Provider readiness checks validate configuration, not a billable remote completion. Real Groq/DeepSeek availability is reported only when a request is attempted.
- A legacy index without an integrity manifest is accepted only after fact-based validation and reported as degraded; new production full indexes should ship the documented manifest.
