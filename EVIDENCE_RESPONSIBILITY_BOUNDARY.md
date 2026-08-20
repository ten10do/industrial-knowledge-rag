# Evidence / Answer Responsibility Boundary

Formal boundary between the evidence layer and the downstream answer/generation
layer. This document is the source of truth for what each stage may (and may not)
decide.

> V3.29 conclusion: `EVIDENCE_SHOULD_NOT_OWN_ANSWER_EXTRACTION`.

## Layer responsibilities

| Stage | Owns | Does NOT own |
|---|---|---|
| **Retrieval** | Find candidate evidence (BM25 / dense / RRF / identity / section / budget / rerank). | Answer eligibility, answer wording. |
| **Evidence** | Whether the query is *answerable* given approved evidence (`ANSWER` / `ABSTAIN`). | Final answer object extraction, normalization, wording. |
| **Semantic Judge** | Relation-level entailment / contradiction / neutral for polar verification. | Answer synthesis; open-slot fill. |
| **Grounding** | Best-effort, traceable answer-bearing spans / objects (debugging, citations, answer hints). | The evidence decision. |
| **Support** | Final detail sufficiency / safety on approved evidence. | Answer representation. |
| **Answer / Generation** | Compose the final answer from **evidence-approved** candidates only. | Re-adjudicating eligibility; using rejected/unapproved context. |

## Decision authority

- **Evidence Contract + Selective Semantic Judge: YES** — the *only* parties that
  may change `ANSWER` / `ABSTAIN`.
- **Grounding: NO** (`GROUNDING_DECISION_AUTHORITY = "NONE"`).
- **Normalization: NO**.
- **Generation: NO at the evidence stage** (`GENERATION_USED = "NO"`).

Grounding metadata is bidirectionally inert:

1. grounding NONE → does not change ANSWER.
2. normalization FAIL → does not change ANSWER.
3. grounding AMBIGUOUS → does not automatically ABSTAIN.
4. grounding success → does not upgrade ABSTAIN to ANSWER.

## Non-gating flags

- `EXTRACTION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"`
- `NORMALIZATION_SUCCESS_REQUIRED_FOR_ANSWER = "NO"`
- `GROUNDING_DECISION_AUTHORITY = "NONE"`
- `GROUNDING_ENRICHMENT_DEFAULT = "OFF"`
- `GENERATION_USED = "NO"`, `LLM_ANSWER_USED = "NO"`

## EvidenceDecision V2 contract

Required (authoritative):

- `decision`: `ANSWER` | `ABSTAIN`
- `reason`

Optional (non-authoritative, never change `decision`):

- `query_type` (intent)
- `supporting_candidate_ids`
- `citations` / provenance
- `relation_judge_result`
- `grounded_objects` / `grounded_spans`
- `normalization`
- `metadata` / diagnostics

## Downstream AnswerContext

The only context the answer layer may consume. It carries **approved** evidence
only (`approved_candidate_ids` + citations + grounded objects empty on `ABSTAIN`).
The answer layer must never draw on a rejected candidate, unapproved context, or
external knowledge. This is the foundation of the future generation contract.

## Status

- Evidence rule baseline: `evidence-v323.1-candidate` (frozen; unchanged).
- Semantic judge: V3.25 local NLI (frozen; thresholds ent=0.5 / contra=0.5 /
  floor=0.33, UNKNOWN = rule fallback).
- Support: `support-v316.1` (frozen; byte-unchanged).
- Retrieval: unchanged.
- V3.27 modes: `VERIFIER_ONLY` = reference; `EXTRACT_ONLY` /
  `EXTRACT_THEN_VERIFY` = experimental, not recommended.
- V3.28 grounded span: `OPTIONAL_EVIDENCE_PAYLOAD` (experimental enrichment, never
  a gate).
- V3.29 boundary candidate: `evidence-v329-boundary-candidate`
  (`EXPERIMENTAL_CANDIDATE`).