"""Keyless, synthetic V3.84 portfolio demo with no private-corpus dependency."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.update({
    "RUNTIME_ENV": "test",
    "APP_DEBUG": "false",
    "RAG_MODE": "light",
    "RETRIEVAL_MODE": "hybrid",
    "TASK_QUEUE_BACKEND": "memory",
    "RATE_LIMIT_BACKEND": "memory",
    "GROQ_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "RERANK_ENABLED": "false",
    "SECTION_EXPANSION_ENABLED": "false",
    "SUPPORT_GATE_ENABLED": "false",
    "TABLE_REGION_CONTEXT_ENABLED": "false",
    "CLAIM_SUPPORT_EXPERIMENT_ENABLED": "false",
})

from fastapi.testclient import TestClient  # noqa: E402
from backend import main  # noqa: E402
from backend.observability import REQUEST_ID_HEADER  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


KB_ID = "kb-v384-public-demo-00000001"
SUPPORTED_QUERY = "What is the maximum operating pressure of the AX-100 cooling pump?"
WRONG_MODEL_QUERY = "What is the maximum operating pressure of the AX-200 cooling pump?"
UNKNOWN_FAULT_QUERY = "What does fault code F999 mean?"


def _payload() -> list[dict]:
    return [
        {
            "page_content": (
                "Synthetic public manual. The AX-100 cooling pump has a maximum "
                "operating pressure of 8 bar."
            ),
            "metadata": {
                "source": "synthetic-public-manual.pdf",
                "file_name": "synthetic-public-manual.pdf",
                "document_id": "synthetic-public-document",
                "chunk_id": "synthetic-public-pressure",
                "document_type": "technical_spec",
                "equipment_type": "cooling pump",
                "equipment_model": "AX-100",
                "manufacturer": "Example Industrial",
                "knowledge_type": "parameter",
                "page": 0,
            },
        },
        {
            "page_content": (
                "Synthetic public manual. AX-100 alarm F101 means inlet pressure "
                "is too low. Inspect the inlet valve before restart."
            ),
            "metadata": {
                "source": "synthetic-public-manual.pdf",
                "file_name": "synthetic-public-manual.pdf",
                "document_id": "synthetic-public-document",
                "chunk_id": "synthetic-public-f101",
                "document_type": "fault_code",
                "equipment_type": "cooling pump",
                "equipment_model": "AX-100",
                "manufacturer": "Example Industrial",
                "knowledge_type": "fault",
                "error_code": "F101",
                "page": 1,
            },
        },
    ]


def run_demo() -> dict:
    index_path = Path(main.get_index_storage_path(KB_ID))
    if index_path.exists():
        raise RuntimeError("Synthetic demo index already exists; refusing to overwrite it.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(_payload(), ensure_ascii=True), encoding="utf-8")
    digest_before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    headers = {
        "X-Knowledge-Base-ID": KB_ID,
        REQUEST_ID_HEADER: "req-v384-public-demo-0001",
    }

    try:
        with TestClient(main.app) as client:
            live = client.get("/live", headers=headers)
            ready = client.get("/ready", headers=headers)

            supported_result = main.retrieve_docs(
                SUPPORTED_QUERY,
                knowledge_base_id=KB_ID,
                retrieval_mode="hybrid",
            )
            supported_evidence = main.analyze_evidence(
                SUPPORTED_QUERY,
                supported_result,
                "hybrid",
            )

            wrong_model_result = main.retrieve_docs(
                WRONG_MODEL_QUERY,
                knowledge_base_id=KB_ID,
                retrieval_mode="hybrid",
            )
            wrong_model_evidence = main.analyze_evidence(
                WRONG_MODEL_QUERY,
                wrong_model_result,
                "hybrid",
            )

            refusal = client.post(
                "/ask",
                headers=headers,
                json={"question": UNKNOWN_FAULT_QUERY},
            )
            metrics = client.get("/metrics", headers=headers)

        digest_after = hashlib.sha256(index_path.read_bytes()).hexdigest()
        refusal_body = refusal.json()
        checks = {
            "liveness_200": live.status_code == 200 and live.json().get("status") == "ok",
            "readiness_required_checks_pass": (
                ready.status_code == 200 and ready.json().get("ready") is True
            ),
            "supported_answerability_decision": (
                supported_evidence.decision == "ANSWER"
                and bool(supported_result.candidates)
            ),
            "wrong_model_hard_negative_abstains": (
                wrong_model_evidence.decision == "ABSTAIN"
                and wrong_model_evidence.reason == "MODEL_MISMATCH"
            ),
            "unknown_fault_api_refuses": (
                refusal.status_code == 200
                and refusal_body.get("is_refused") is True
                and refusal_body.get("sources") == []
                and refusal_body.get("evidence", {}).get("decision") == "ABSTAIN"
            ),
            "metrics_200": (
                metrics.status_code == 200
                and "http_requests_total" in metrics.text
                and "rag_abstains_total" in metrics.text
            ),
            "request_id_round_trip": (
                live.headers.get(REQUEST_ID_HEADER) == headers[REQUEST_ID_HEADER]
            ),
            "index_unchanged": digest_before == digest_after,
            "experimental_defaults_off": not any((
                main.reranker.requested,
                main.support_gate_enabled(),
            )),
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "scenarios": {
                "supported_answerability": {
                    "decision": supported_evidence.decision,
                    "reason": supported_evidence.reason,
                    "candidate_count": len(supported_result.candidates),
                    "provider_generation_executed": False,
                },
                "wrong_model_hard_negative": {
                    "decision": wrong_model_evidence.decision,
                    "reason": wrong_model_evidence.reason,
                },
                "unknown_fault_api": {
                    "http_status": refusal.status_code,
                    "decision": refusal_body.get("evidence", {}).get("decision"),
                    "is_refused": refusal_body.get("is_refused"),
                    "source_count": len(refusal_body.get("sources", [])),
                },
                "readiness": ready.json().get("status"),
            },
        }
    finally:
        index_path.unlink(missing_ok=True)


if __name__ == "__main__":
    report = run_demo()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)
