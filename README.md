# Industrial Knowledge RAG

A safety-oriented retrieval-augmented generation system for industrial manuals, maintenance procedures, fault knowledge, SOPs, and technical specifications.

The project is intentionally evidence-first. Retrieval proposes candidate passages; a deterministic Evidence layer decides ANSWER or ABSTAIN before any optional model call. Unknown identifiers, incompatible equipment identities, and unsupported details are refused without citations. This reduces unsafe confirmation, but does not claim to eliminate hallucination.

## Release candidate status

**V3.84 release-candidate packaging. Runtime and research semantics are frozen.**

| Area | Current evidence |
|---|---|
| Correctness authority | V377 aligned private benchmark, contract-native V3.67 evaluator |
| Frozen result | 54/69 correct, 9 false answers, 3 false refusals, 78.26% accuracy; AR 76.92%, AbR 79.07% |
| Safety slice | 10/10 hard negatives handled correctly |
| Regression suite | 948 backend tests passed, 2 skipped; 34/34 frontend tests passed; frontend build passed |
| Deployment | Docker Compose image, non-root runtime, readiness, restart, persistence, recreation, shutdown, load and privacy gates passed |
| Long-run stability | 600.12-second native soak, 11,678 requests, 0 errors |
| Freeze/private boundary | 28/28 protected hashes passed; tracked private files: 0 |

These are engineering validation results, not a production-accuracy claim. The correctness benchmark uses a local, ignored three-document corpus; its documents and annotations are not published. Public fixtures validate reproducibility and behavior contracts, not private-benchmark quality.

Read [Known Limitations](KNOWN_LIMITATIONS.md) before interpreting the metrics, or continue with the [Quick Start](#quick-start).

## Why this project exists

Industrial RAG fails dangerously when “retrieved something similar” is treated as “has evidence.” This repository makes that boundary explicit:

    Industrial documents
      -> classification and structure-aware chunking
      -> BM25 + light TF-IDF or full dense retrieval
      -> scoped candidates and RRF ranking
      -> deterministic Evidence contract
           -> ABSTAIN: no model call, no citations
           -> ANSWER: approved context only
      -> optional Groq or DeepSeek generation
      -> cited response

The system also contains versioned knowledge-base publishing, background jobs, conversation context management, operational metrics, privacy-safe logging, mode-aware readiness, and a React/Vite administration and query interface.

## What is implemented

- Text-based PDF ingestion with conservative industrial document classification, structured chunking, stable document/chunk identifiers, and source-page provenance.
- Light retrieval using BM25 plus a legacy character TF-IDF signal, and full retrieval using HuggingFace embeddings plus Chroma.
- Query scoping for industrial identifiers and equipment identities, unified retrieval candidates, RRF fusion, and traceable score lineage.
- Evidence-based ANSWER/ABSTAIN decisions independent of answer wording.
- Optional model generation, reranking, and experimental support/table paths behind explicit flags; experimental paths are OFF by default.
- Draft knowledge-base build, immutable public-version publishing, history, rollback, bounded uploads, task lifecycle, and local/Redis operational modes.
- FastAPI health, liveness, readiness and Prometheus-style metrics endpoints.
- Docker Compose backend topology with persistent named volumes and a separately hosted frontend.

See [Feature Status](FEATURE_STATUS.md) for the exact implemented, experimental, unsupported, and externally pending boundary.

## Quick start

Requirements: Python 3.11+, Node.js 20+ for local development, or Docker Desktop/Engine with Compose.

Backend:

    python -m venv venv
    .\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    Copy-Item .env.example .env
    .\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

Frontend, in another terminal:

    cd frontend
    npm ci
    npm run dev

The frontend defaults to http://localhost:5173. The backend exposes:

- http://127.0.0.1:8000/live — process liveness only
- http://127.0.0.1:8000/ready — required dependency/index readiness plus optional degradation
- http://127.0.0.1:8000/metrics — low-cardinality operational metrics
- http://127.0.0.1:8000/docs — OpenAPI UI

Management operations require ADMIN_TOKEN. A generation provider is optional: without GROQ_API_KEY or DEEPSEEK_API_KEY, readiness is degraded but the keyless retrieval, Evidence, refusal, and operational paths remain available.

## Docker Compose

    Copy-Item .env.example .env
    docker compose config --quiet
    docker compose build
    docker compose up -d
    Invoke-RestMethod http://127.0.0.1:8000/live
    Invoke-RestMethod http://127.0.0.1:8000/ready -Headers @{'X-Knowledge-Base-ID'='kb-public-shared-00000001'}

Stop containers while retaining named volumes:

    docker compose down

Do not add -v unless volume deletion is explicitly intended.

## Reproducible public demo

The public demo is keyless and uses only synthetic text created by this repository:

    .\venv\Scripts\python.exe scripts\v384_public_demo.py

It demonstrates liveness/readiness/metrics, a supported answerability decision, a safe API refusal, an unknown-model hard negative, request-ID propagation, and index immutability. It does not fabricate a provider-generated answer. See [Demo Scenarios](DEMO_SCENARIOS.md).

## Verification

    .\venv\Scripts\python.exe -m backend.v382_release_guard
    .\venv\Scripts\python.exe -m pytest backend -v
    .\venv\Scripts\python.exe scripts\v384_public_demo.py
    .\venv\Scripts\python.exe -m backend.evaluation.run
    cd frontend
    npm ci
    npm run test
    npm run build

The public CI workflow runs backend tests, the public operational smoke, offline public evaluation, a full-dependency import smoke, frontend tests/build, Compose validation, and an image build. Remote CI has not been executed in this local-only release-candidate phase and is not claimed as passed.

## Evaluation boundary

Evaluation is separated into three layers:

1. Public synthetic fixtures test parser, retrieval, API, refusal, and operational contracts.
2. Public challenge fixtures provide deterministic stress tests and failure analysis; they are not production-accuracy evidence.
3. The V377 aligned benchmark uses a legally obtained local corpus and private annotations under ignored paths. Only aggregate metrics, hashes, evaluator contracts, and reproducibility metadata are published.

The V3.67 contract-native evaluator is the correctness authority. It compares structured decisions and required/forbidden claims rather than accepting plausible-looking answer text. The frozen V377 replay digest is dcd548906ef2b1f233282283b8684f8c82066dc5eea9f0179ff45a4034a5aa1d.

See [Evaluation](EVALUATION.md), [Research Decisions](RESEARCH_DECISIONS.md), and [Known Limitations](KNOWN_LIMITATIONS.md).

## Data and secret boundary

No private PDF, private benchmark payload/result, vector database, model cache, .env file, token, or API key belongs in Git. Runtime data and private evaluation directories are ignored. Use only documents you are authorized to process and publish.

The container image excludes tests, offline evaluation assets, local results, PDFs, indexes, logs, and developer-specific paths. The validated image runs as a non-root application user.

## Documentation map

- [Architecture](ARCHITECTURE.md)
- [Evaluation](EVALUATION.md)
- [Research Decisions](RESEARCH_DECISIONS.md)
- [Feature Status](FEATURE_STATUS.md)
- [Deployment](DEPLOYMENT.md)
- [Deployment Validation](DEPLOYMENT_VALIDATION.md)
- [Operations Runbook](OPERATIONS_RUNBOOK.md)
- [Configuration Matrix](CONFIGURATION_MATRIX.md)
- [Demo Scenarios](DEMO_SCENARIOS.md)
- [Project Evidence](PROJECT_EVIDENCE.md)
- [Interview Stories](INTERVIEW_STORIES.md)
- [Release Checklist](RELEASE_CHECKLIST.md)
- [Known Limitations](KNOWN_LIMITATIONS.md)
- [Historical Research Index](docs/research/README.md)

## If you only have 5 minutes

1. Read [Architecture](ARCHITECTURE.md) for the system and safety boundary.
2. Read [Evaluation](EVALUATION.md) for the 54/69 baseline and benchmark validity.
3. Read [Deployment Validation](DEPLOYMENT_VALIDATION.md) for the executed RC evidence.
4. Read [Known Limitations](KNOWN_LIMITATIONS.md) before interpreting any metric.
5. Run the command in [Demo Scenarios](DEMO_SCENARIOS.md).

## Honest limitations

- The frozen private benchmark still has 9 false answers and 3 false refusals.
- Cell-level table ownership is unsupported. Flat text cannot safely recover merged-cell and column semantics from many industrial manuals.
- Claim-support validation, requirement-extraction upgrades, local NLI gating, reranking, and table-context logic are not production-ready and remain OFF by default.
- A real optional provider answer path and remote CI execution are external-pending, not silently treated as passed.
- The validated topology is a single backend service with persistent volumes, not Kubernetes, high availability, or a multi-region production platform.

The future independent research direction is layout-aware industrial document ingestion. It is explicitly outside the V3.84 release-candidate freeze.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

Third-party dependencies and external data or model weights remain subject to their respective licenses.
