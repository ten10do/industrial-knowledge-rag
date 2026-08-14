# Frozen retrieval replay

V3.12 separates expensive retrieval from Evidence/Support rule evaluation:

```text
Frozen P2 retrieval + captured Evidence input
  -> immutable retrieval-artifact-v1 JSON
  -> current Evidence rule
  -> current Support rule
  -> per-query checkpoints and aggregate metrics
```

The replay path is evaluation-only. It reconstructs `Document`,
`RetrievalCandidate`, and `RetrievalResult` objects from JSON and does not load
PDFs, BM25, Chroma, embeddings, or a CrossEncoder.

## Artifact contract

Each artifact binds its corpus manifest, annotations, query text, retrieval
configuration, and source P2 stage by SHA-256. The retrieval configuration
includes the embedding/reranker settings and hashes of the tokenizer, BM25,
fusion, ProductIdentity, scope, section, and reranker implementations. Artifact
schema version (`retrieval-artifact-v1`) is independent of the Evidence/Support
rule version (`v311.2` at initial export), so later rules can replay the same
retrieval input.

Every query contains its ground truth, query text hash, raw Evidence candidate
pool, saved query analysis, P2 final context, and retrieval decision inputs.
Candidates retain their text, complete parsed metadata, available scores/ranks,
identity/scope fields, and section provenance. Fields unavailable in the frozen
source are omitted rather than synthesized. The corpus snapshot contains only
identity metadata and identifier/parameter terms read globally by the gates.

Validation returns:

- `VALID`: complete, internally consistent, and all hashes match.
- `PARTIAL`: replay inputs or expected query rows are missing.
- `INVALID`: schema, artifact, query, snapshot, corpus, annotation, or retrieval
  configuration identity is inconsistent.

A valid artifact is immutable. Re-export under a new artifact ID and path;
existing files are never overwritten. Writes use temporary-file plus atomic
replace persistence.

## CLI

All artifact and result paths must remain under the ignored private benchmark
directory.

```powershell
python -m backend.evaluation.v312_replay_runner export `
  --corpus a `
  --artifact-id v312-frozen-a-YYYYMMDD `
  --output backend/evaluation/benchmark_private/v312_artifacts/a.json

python -m backend.evaluation.v312_replay_runner validate `
  --artifact backend/evaluation/benchmark_private/v312_artifacts/a.json

python -m backend.evaluation.v312_replay_runner inspect `
  --artifact backend/evaluation/benchmark_private/v312_artifacts/a.json

python -m backend.evaluation.v312_replay_runner replay `
  --artifact backend/evaluation/benchmark_private/v312_artifacts/a.json `
  --manifest backend/evaluation/benchmark_private/manifest.json `
  --run-id v312-replay-a

python -m backend.evaluation.v312_replay_runner compare `
  --baseline backend/evaluation/benchmark_private/v311_runtime/v3112-frozen-a/summary.json `
  --replay backend/evaluation/benchmark_private/v312_runtime/v312-replay-a/summary.json `
  --output backend/evaluation/benchmark_private/v312_runtime/compare-a.json
```

Use `--resume` with the same export artifact ID or replay run ID after an
interruption. Completed query checkpoints are hash-validated and skipped.
Combined metrics are computed from the two saved A/B replay results with
`combine_replay_results`; no Combined retrieval artifact or index is created.

## Privacy

Artifacts contain copyrighted vendor-manual excerpts and are private to the
same degree as the source PDFs. The repository ignores
`backend/evaluation/benchmark_private/`, which includes artifacts, corpus
snapshots, replay results, comparisons, checkpoints, and logs. Never commit or
publish those files.
