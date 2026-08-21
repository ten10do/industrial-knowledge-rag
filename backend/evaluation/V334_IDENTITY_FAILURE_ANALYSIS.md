# V3.34 Identity Failure Analysis

## Scope

This analysis is the pre-implementation record for
`identity-aware-evidence-v334-candidate`. It uses the aggregate V3.33 report,
the public Evidence contract, and the public product-identity implementation.
It does not inspect D/E/H/J plaintext or V3.33 K-CHECK query text, and it does
not modify or replay the V3.33 baseline.

## Problem confirmation

V3.33 was runtime-valid but only partially generalized. On the independent
K-CHECK split, accuracy was `0.6042`, answerable recall was `0.7500`, abstention
recall was `0.4583`, false-answer rate was `0.5417`, and false-refusal rate was
`0.2500`. Read-only attribution assigned 27 failures to the base rule and only
five to missing retrieval evidence. The primary V3.34 target is therefore the
Evidence reasoning boundary, not retrieval.

The existing implementation has useful document-level safeguards:

- query identities are normalized against document metadata;
- a more specific query is not normally compatible with a broader
  `ProductIdentity` when both levels are represented correctly;
- evidence requirements and support validation already reject many model,
  parameter, protocol, and value mismatches.

The remaining gap is claim scope. `CandidateClaim.identity` is constructed from
document metadata. It does not extract the identity asserted by the candidate
sentence. A series manual can legitimately list member models in metadata or
aliases. When a query mentions one member, the resolver can select that series
document as an explicit identity. The subsequent claim check sees the same
document metadata on every chunk, including chunks that only make a family- or
series-wide statement. The implementation can therefore observe:

```text
query member alias -> matching series document metadata -> compatible claim
```

when the semantically relevant path should be:

```text
specific query identity -> identity stated by candidate claim -> compatibility
```

This is how family evidence can be treated as model evidence even though the
existing directionality rule appears correct at the metadata level. The defect
is not that all family evidence is unsafe. It is that document identity and
claim identity are currently the same object.

## Failure taxonomy

### 1. Family-to-model leakage

A query names a model while the supporting sentence only names its product
family. A family-level statement cannot establish a property for a particular
model. Example: `CX family supports protocol P` does not by itself support
`CX5140 supports protocol P`.

### 2. Series-to-option leakage

A query names an option or option code while the candidate only describes the
base series. Options are conditional branches, not automatically inherited
properties of every series member.

### 3. Module-to-controller leakage

A query asks about a module, but the candidate describes its controller or
host. Containment and interoperability do not make module and controller
identities interchangeable.

### 4. Firmware/version leakage

A query is constrained to one firmware or document version while the candidate
states another. Same product identity does not override an explicit version
mismatch.

### 5. Accessory/extension leakage

A query names an accessory, extension, adapter, or add-on while the evidence
only covers the base product, or covers a different accessory. Accessory scope
is a sibling or child branch that requires explicit support.

### 6. Parameter-scope leakage

A query names one parameter or register while the candidate states a value or
behavior for another parameter, or only for a product-global setting. Topic and
product matches do not establish the local parameter association.

## Identity model

V3.34 adds a lightweight, deterministic claim-boundary model. It does not use
an LLM and does not change retrieval, ranking, parsing, NLI, open sufficiency,
or support thresholds.

```text
ProductIdentity
  manufacturer
  family
  series
  model
  module
  option
  firmware
  protocol
  parameter
  scope_level: GLOBAL | FAMILY | SERIES | MODEL | MODULE | OPTION
```

The query identity is extracted from the query plus existing candidate/corpus
metadata. The evidence identity is extracted from each candidate's claim text,
using document metadata only as a vocabulary and as a fallback for an
unambiguously single-model document. Explicit claim text has precedence over
document-level scope.

Compatibility is directional:

| Query identity | Evidence identity | Boundary result |
| --- | --- | --- |
| exact model | same exact model | `COMPATIBLE` |
| family/series | matching descendant model | `COMPATIBLE` |
| exact model | family or series only | `INCOMPATIBLE` |
| module | owning controller only | `INCOMPATIBLE` |
| option/accessory | base series only or different option | `INCOMPATIBLE` |
| firmware X | explicit firmware Y | `INCOMPATIBLE` |
| parameter X | explicit parameter Y | `INCOMPATIBLE` |
| insufficient reliable identity information | insufficient information | `UNKNOWN` |

The governing invariant is: a more specific query cannot be supported by
broader evidence. `UNKNOWN` is deliberately non-authoritative and preserves the
existing V3.32 mixed-evidence behavior.

## Candidate boundary

The candidate is additive and versioned as
`identity-aware-evidence-v334-candidate`:

```text
Query
  -> deterministic query identity extraction
  -> deterministic candidate-claim identity extraction
  -> directional compatibility
       COMPATIBLE   -> existing mixed Evidence contract
       INCOMPATIBLE -> ABSTAIN at identity boundary
       UNKNOWN      -> existing mixed Evidence contract
```

`EvidenceDecisionV2` is not changed. The candidate wrapper records its boundary
decision and delegates unchanged cases to the existing V3.32 entry point. The
identity boundary may only downgrade an otherwise answerable path; it cannot
upgrade an existing abstention.

## Fixed evaluation discipline

Evaluation will use a new document-disjoint DEV set, not V3.33 K-CHECK. It will
contain at least 50 balanced positive/negative hard near-miss cases covering 20
family/model, 10 module/controller, five firmware/version, five
option/accessory, and 10 parameter-scope cases. Baseline is the unchanged V3.33
mixed candidate; treatment is the V3.34 wrapper. Acceptance requires a lower
false-answer rate and no more than a five-percentage-point increase in
false-refusal rate. At most three candidate experiments are permitted.
