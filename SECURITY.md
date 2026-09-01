# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch only. Historical
research reports, archived Streamlit code, old release-candidate evidence, and
unreleased local forks are not maintained security branches.

| Version | Supported |
|---|---|
| Current `main` | Yes |
| Historical snapshots and archived prototype | No |

## Reporting a vulnerability

Do not open a public issue, discussion, or pull request for a suspected
vulnerability. Use GitHub's private vulnerability reporting form:

https://github.com/ten10do/industrial-knowledge-rag/security/advisories/new

Include the affected component and commit, reproduction steps or proof of
concept, expected impact, and any suggested mitigation. Remove API keys,
private industrial documents, customer data, model credentials, and other
secrets from the report.

Reports will be acknowledged and triaged through the private advisory. Timing
for remediation and disclosure depends on severity, reproducibility, and the
availability of an upstream fix. Please allow coordinated remediation before
public disclosure.

## Scope

Good-faith testing against code and synthetic fixtures in this repository is in
scope. Testing third-party providers, GitHub, hosted systems, private data, or
infrastructure not explicitly owned by this repository is out of scope. Do not
perform denial-of-service testing, social engineering, destructive testing, or
access data that is not yours.

Dependency findings should identify whether they affect the certified light
runtime, the optional embedded full runtime, frontend development tooling, or
the archived prototype. The documented embedded-only Chroma exception is not a
network-service authorization; exposing Chroma over a network remains outside
the supported security boundary.
