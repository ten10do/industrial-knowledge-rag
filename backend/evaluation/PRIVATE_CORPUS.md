# Private real-world corpus interface

Create `backend/evaluation/benchmark_private/manifest.json` locally. The whole directory is ignored by Git, so vendor PDFs, annotations, and result JSONs remain outside version control.

Use the same top-level `documents` and `queries` schema as `fixtures/industrial_challenge.json`. Every document must include `document_id`, `chunk_id`, `file`, document metadata, `source_type`, and `commit_allowed: false`. `file` is a relative path below `benchmark_private`; never use an absolute path. A document may either include a local `content` field for a pre-extracted labelled chunk, or omit `content` and point `file` to a UTF-8 `.txt` or a text-based `.pdf`. The runner reads the local file without changing the evaluation code.

The private query annotations use `relevant_chunk_ids`; when a PDF is loaded as one evaluation chunk, use the document's declared `chunk_id`. For section-level evaluation, create separate locally ignored manifest entries for the labelled chunks. Keep queries, labels, and files legal to use, and do not copy them into the committed challenge set.

Run:

```powershell
.\venv\Scripts\python.exe -m backend.evaluation.benchmark_runner --dataset private --mode hybrid
```

Until a local manifest is supplied, the runner reports `REAL_CORPUS_GATE_NOT_RUN`.
