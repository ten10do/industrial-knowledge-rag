# Public Repository Publication Plan

## Current decision state

**PUBLIC_REPOSITORY_PRECHECK_READY; AWAITING OWNER EXTERNAL-WRITE AUTHORIZATION.**

The owner selected the MIT License. The root LICENSE file is present with Copyright (c) 2026 ten10do. No GitHub repository, remote, push, tag or release has been created.

## Proposed repository metadata

| Field | Draft |
|---|---|
| Repository name | industrial-knowledge-rag |
| Description | Deployment-validated industrial knowledge RAG with hybrid retrieval, model-aware evidence checks, safe abstention, structured evaluation, and Docker/FastAPI operations. |
| Default branch | main |
| Visibility | Public after explicit external-write authorization and exact-SHA CI validation |
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

## Owner decisions required before external writes

1. Authorize repository creation/remote setup/push in a later turn.
2. Confirm the GitHub account or organization and public visibility.
3. After exact-SHA CI passes, choose the release identifier: rc-v3.84 or v0.1.0-rc1.
4. Separately authorize any Git tag and GitHub Release.

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

CI success must be tied to the exact pushed SHA. CI_CONFIGURED is not CI PASS.

Official GitHub actions currently use major-version refs: actions/checkout@v4, actions/setup-python@v5 and actions/setup-node@v4. They are official and current-looking, but not immutable commit-SHA pins. SHA pinning is an optional supply-chain hardening decision, not represented as already complete.

## Publication sequence after future authorization

1. Create or confirm the GitHub repository.
2. Add the authorized remote and push main only.
3. Observe CI for the exact pushed SHA.
4. Fix only publication/CI defects if necessary; do not reopen RAG research.
5. Record exact-SHA CI evidence.
6. Ask for separate tag/release authorization.
7. If authorized, create the chosen tag and publish the prepared release notes.

## Release draft

Proposed title: **V3.84 RC — Deployment-Validated Industrial Knowledge RAG**

Draft notes: [V3.84 RC Release Notes](RELEASE_NOTES_V3.84_RC.md)

No semantic version or tag is selected. rc-v3.84 preserves the internal phase name; v0.1.0-rc1 is the SemVer-style alternative. Final choice is OWNER DECISION.

## Explicit non-actions

- No remote created.
- No repository created.
- No push performed.
- No tag created.
- No GitHub Release created.
- No history rewrite performed.
- No provider secret uploaded.
