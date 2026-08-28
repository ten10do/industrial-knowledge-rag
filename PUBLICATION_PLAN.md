# Public Repository Publication Plan

## Current decision state

**PUBLIC_REPOSITORY_PUBLISHED; REMOTE_CI_EXECUTED_PASS; TAG_RELEASE_PENDING_OWNER_DECISION.**

The MIT-licensed repository is public at https://github.com/ten10do/industrial-knowledge-rag. Local `main` tracks `origin/main`. GitHub Actions run `33156355678` passed for exact SHA `5df4ef42efc0d9e9922087a763aa65ba92dd8c4a`. No tag or GitHub Release exists.

## Proposed repository metadata

| Field | Draft |
|---|---|
| Repository name | industrial-knowledge-rag |
| Description | Deployment-validated industrial knowledge RAG with hybrid retrieval, model-aware evidence checks, safe abstention, structured evaluation, and Docker/FastAPI operations. |
| Default branch | main |
| Visibility | Public |
| Frozen RC | V3.84 deployment-validated development RC |
| Runtime/package source | b4b75a1086a05e887a8da39d757f2fb079010508 plus V3.85 publication-only documentation |

Suggested topics:

- rag
- industrial-ai
- fastapi
- retrieval
- hybrid-search
- llm
- observability
- docker
- evaluation
- safe-ai

## Owner decisions required before an external release

1. Choose the release identifier: rc-v3.84 or v0.1.0-rc1.
2. Separately authorize any Git tag and GitHub Release.

## License audit context

The project is licensed under MIT. This covers repository code and documentation only to the extent held by the copyright owner. Installed direct Python and frontend packages declare permissive MIT, BSD, Apache-2.0, ISC, 0BSD, Zlib or CC0-family metadata. This is an engineering inventory, not legal advice or a complete transitive-license opinion.

- pdfplumber 0.11.10 declares MIT and is research-only, not in the product requirements.
- PyMuPDF 1.28.2 declares AGPL-3.0/commercial dual licensing. It is not declared in product requirements or included in the Docker image; one historical validation script can optionally import it.
- The experimental cross-encoder/nli-deberta-v3-xsmall model card declares Apache-2.0. No model weight or cache is tracked or shipped.
- Apache-licensed dependencies may carry notice obligations when redistributed; the owner should include this in the final license/legal review.

## Required CI on the exact publication HEAD

The existing read-only workflow must complete:

1. Backend syntax, V3.82 release guard, backend public tests, public smoke and offline evaluation.
2. Full-mode dependency installation and import smoke.
3. Frontend npm ci, tests and production build.
4. Docker Compose configuration validation and non-root image build.

CI success is tied to the exact pushed SHA. Run `33156355678` established `REMOTE_CI_EXECUTED_PASS` for `5df4ef42efc0d9e9922087a763aa65ba92dd8c4a`; later public commits require their own exact-SHA run.

Official GitHub actions currently use major-version refs: actions/checkout@v4, actions/setup-python@v5 and actions/setup-node@v4. They are official and current-looking, but not immutable commit-SHA pins. SHA pinning is an optional supply-chain hardening decision, not represented as already complete.

## Publication state and remaining sequence

1. Public repository creation: complete.
2. `origin/main` publication: complete.
3. Exact-SHA remote CI: complete for the recorded public baseline.
4. Validate every later public HEAD with its own CI run.
5. Ask for separate tag/release authorization.
6. If authorized, create the chosen tag and publish the prepared release notes.

## Release draft

Proposed title: **V3.84 RC — Deployment-Validated Industrial Knowledge RAG**

Draft notes: [V3.84 RC Release Notes](RELEASE_NOTES_V3.84_RC.md)

No semantic version or tag is selected. rc-v3.84 preserves the internal phase name; v0.1.0-rc1 is the SemVer-style alternative. Final choice is OWNER DECISION.

## Current external boundaries

- No tag created.
- No GitHub Release created.
- No history rewrite performed.
- No provider secret uploaded.
