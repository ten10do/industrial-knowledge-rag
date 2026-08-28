# Architecture

## System boundary

Industrial Knowledge RAG is a FastAPI backend plus a React/Vite frontend. The backend owns ingestion, indexes, retrieval, Evidence decisions, optional generation, knowledge-base versions, background tasks, governance, readiness and metrics. The frontend is hosted separately and calls the backend through a configured origin.

The release topology is one light-mode backend container with four persistent named volumes. Full dense retrieval is supported by code and dependency smoke tests, but the certified V3.83 container topology uses light mode.

## Request path

    Client
      -> FastAPI validation, request ID, rate limit and knowledge-base scope
      -> conversation context processing
      -> query analysis and retrieval scope
      -> BM25 plus light TF-IDF or full Chroma vector candidates
      -> RRF fusion and optional section expansion
      -> Evidence contract: ANSWER or ABSTAIN
           -> ABSTAIN: fixed refusal, zero sources, no model call
           -> ANSWER: approved candidates only
      -> optional reranker/support experiment when explicitly enabled
      -> optional Groq or DeepSeek generation
      -> answer, sources, evidence metadata and operational metrics

## Responsibility boundaries

| Layer | Owns | Must not own |
|---|---|---|
| Ingestion | document classification, chunk boundaries, provenance metadata | answerability |
| Retrieval | candidate discovery, scope, ranking and score lineage | truth or final answer |
| Evidence | deterministic ANSWER/ABSTAIN eligibility | wording or open-ended extraction |
| Support/semantic experiments | bounded relation or detail checks | silent production authority |
| Generation | wording from approved context | upgrading an ABSTAIN decision |
| Evaluation | structured correctness and regression measurement | changing runtime decisions |

The full decision contract is in [Evidence Responsibility Boundary](EVIDENCE_RESPONSIBILITY_BOUNDARY.md).

## Data lifecycle

    authorized PDF
      -> validated upload
      -> draft files and bounded task
      -> parsed pages
      -> classified industrial chunks
      -> draft index snapshot
      -> immutable published version
      -> active public knowledge-base pointer

Local runtime state lives below backend/data, backend/light_indexes, backend/public_versions, and backend/runtime_state. Full-mode Chroma data lives below backend/vector_db. These paths, private evaluation inputs, generated results, PDFs, secrets, and caches are ignored by Git.

Publishing is versioned. A new index is built as a draft, validated, then activated through the version store. Rollback changes the active version pointer rather than mutating historical content. Multi-process coordination requires the documented Redis and shared-storage configuration; memory backends are single-process defaults.

## Retrieval modes

| Mode | Lexical signal | Vector signal | Notes |
|---|---|---|---|
| light lexical | BM25 | none | keyless, dependency-light |
| light vector | none | character TF-IDF | legacy baseline, not a dense embedding |
| light hybrid | BM25 | character TF-IDF | RRF fusion; certified Compose default |
| full vector | none | HuggingFace + Chroma | fails fast if dependencies/index identity are invalid |
| full hybrid | BM25 | HuggingFace + Chroma | RRF fusion |

RRF scores rank candidates; they are not probabilities and do not by themselves authorize an answer. The frozen Evidence policy uses explicit identifiers, equipment identity, evidence-contract coverage, source scores and safety checks.

## Operational architecture

- /live reports process liveness only.
- /ready evaluates mode-aware required dependencies and index availability. Optional provider absence yields degraded HTTP 200 when required checks pass.
- /metrics exposes bounded labels and no raw query or document content.
- Production configuration is parsed once and fails fast on invalid enums, flags, URLs, paths or unsafe debug/CORS combinations.
- Management endpoints require ADMIN_TOKEN.
- Production JSON logging uses allowlisted fields and propagates a request ID.
- The Docker image runs as uid 100, excludes offline/test/private assets, and persists runtime state through named volumes.

## Frozen and experimental surfaces

The V3.84 freeze protects retrieval, Evidence, identity, score lineage, evaluation authority, generation semantics and the aligned benchmark harness. Experimental reranking, section expansion, support, table-region context and claim-support paths remain explicit flags and are OFF by default. Their presence in the repository is research evidence, not a production-readiness claim.

See [Feature Status](FEATURE_STATUS.md), [Configuration Matrix](CONFIGURATION_MATRIX.md), and [Deployment Validation](DEPLOYMENT_VALIDATION.md).
