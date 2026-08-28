# Feature Status

Status meanings:

- READY — implemented and supported by current regression/operational evidence.
- EXPERIMENTAL — implemented behind an explicit flag, OFF by default, not production-ready.
- PARTIAL — a bounded capability exists but the intended quality or generalization gate was not met.
- UNSUPPORTED — deliberately not claimed.
- EXTERNAL PENDING — requires infrastructure or credentials outside the local RC phase.

| Capability | Status | Boundary |
|---|---|---|
| Text PDF ingestion and industrial metadata | READY | Scanned/OCR-only PDFs are not generally supported |
| Stable chunks, page provenance and index snapshots | READY | Quality depends on extracted text |
| BM25 light retrieval | READY | Keyless and offline |
| Light TF-IDF retrieval | READY | Legacy vector-like signal; not a dense embedding |
| Full HuggingFace/Chroma retrieval | READY in code | Requires full dependencies/model/index identity; not the certified Compose default |
| Hybrid RRF candidate ranking | READY | Ranking signal, not answerability probability |
| Industrial identifier and equipment identity scoping | READY | Corpus coverage still limits recall |
| Evidence ANSWER/ABSTAIN contract | READY/FROZEN | 9 FA and 3 FR remain on V377 |
| Safe refusal without model call | READY | Returns no citations on Evidence abstention |
| Groq/DeepSeek generation | IMPLEMENTED | Optional provider capability not validated in the certified keyless container |
| Source citations | READY when answering | Citations do not prove answer correctness |
| Knowledge-base draft/build/publish/history/rollback | READY | Multi-process use requires shared backends/configuration |
| Background task center | READY | Memory default is single-process; Redis mode available |
| Conversation context management | READY | Query rewrite does not override Evidence authority |
| Liveness/readiness/metrics/logging/request IDs | READY | Metrics are process-local unless externally scraped/aggregated |
| Docker Compose backend topology | READY | Single backend service, separately hosted frontend |
| Frontend UI | READY | 34/34 tests and production build passed in V3.83 evidence |
| Cross-encoder reranking | EXPERIMENTAL | RERANK_ENABLED=false by default |
| Section expansion | EXPERIMENTAL | SECTION_EXPANSION_ENABLED=false by default |
| Rule-based support gate | EXPERIMENTAL/PARTIAL | SUPPORT_GATE_ENABLED=false; not production authority |
| Claim-support experiment | PARTIAL | Low precision; no safe aggregation policy |
| Local NLI gate | UNSUPPORTED for production | Evaluated recall 0 on the relevant aligned cohort |
| Requirement extraction upgrade | PARTIAL | Improved extraction syntax but worsened support safety |
| Table-region context bundle | EXPERIMENTAL/PARTIAL | Same-region context is not ownership proof |
| Cell-level table ownership | UNSUPPORTED | Authority NONE; flat-text inference forbidden |
| OCR and layout-aware ingestion | UNSUPPORTED | Candidate future independent phase |
| Provider-backed successful answer validation | EXTERNAL PENDING | Requires authorized credential/network/model |
| Remote CI execution | EXTERNAL PENDING | Workflow configured; no remote/push action in V3.84 |
| Kubernetes/HA/multi-region operations | UNSUPPORTED | Not part of the validated topology |

For quantified limits see [Known Limitations](KNOWN_LIMITATIONS.md). For experiment rationale see [Research Decisions](RESEARCH_DECISIONS.md).
