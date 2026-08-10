from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.retrieval.candidates import RetrievalCandidate, RetrievalResult
from backend.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    load_reranker_config,
)


def _candidate(chunk_id: str, content: str, rank: int, **metadata):
    document = SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": metadata.pop("document_id", f"doc-{chunk_id}"),
            "source": metadata.pop("source", "manual.pdf"),
            "page": metadata.pop("page", 1),
            "section": metadata.pop("section", "Faults"),
            **metadata,
        },
    )
    return RetrievalCandidate(document=document, retrieval_source="hybrid", final_rank=rank)


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        del show_progress_bar
        return [float(passage.count("relevant")) for _, passage in pairs]


def _reranker(factory=lambda *_: FakeCrossEncoder()):
    return CrossEncoderReranker(
        RerankerConfig(enabled=True, candidate_k=10, top_k=5),
        model_factory=factory,
    )


def test_empty_and_single_candidate_interfaces():
    reranker = _reranker()
    empty = reranker.rerank("query", RetrievalResult([]), top_k=3)
    assert empty.reranker_effective is True
    assert empty.result.candidates == []

    candidate = _candidate("one", "relevant", 1)
    single = reranker.rerank("query", RetrievalResult([candidate]), top_k=3)
    assert [item.chunk_id for item in single.result.candidates] == ["one"]
    assert candidate.pre_rerank_rank == 1
    assert candidate.rerank_rank == 1


def test_multiple_candidates_move_relevant_up_and_obey_top_k_deterministically():
    candidates = [
        _candidate("first", "unrelated", 1),
        _candidate("target", "relevant relevant", 2),
        _candidate("third", "relevant", 3),
    ]
    result = RetrievalResult(candidates, retrieval_mode="hybrid")
    first = _reranker().rerank("query", result, top_k=2)
    second_candidates = [
        _candidate("first", "unrelated", 1),
        _candidate("target", "relevant relevant", 2),
        _candidate("third", "relevant", 3),
    ]
    second = _reranker().rerank("query", RetrievalResult(second_candidates), top_k=2)
    assert [item.chunk_id for item in first.result.candidates] == ["target", "third"]
    assert [item.chunk_id for item in second.result.candidates] == ["target", "third"]


def test_reranking_preserves_document_and_retrieval_metadata():
    candidate = _candidate(
        "F0002", "relevant", 2, document_id="fault-doc", source="g120.pdf",
        page=4, section="Fault diagnosis", equipment_model="G120", error_code="F0002",
    )
    output = _reranker().rerank("F0002", RetrievalResult([candidate]), top_k=1).result.candidates[0]
    assert output.metadata == candidate.metadata
    assert output.lexical_rank == candidate.lexical_rank
    assert output.final_rank == 2


@pytest.mark.parametrize(
    ("query", "matching", "distractor"),
    [
        ("F0002 是什么故障", "G120 F0002 relevant", "G120 F0001"),
        ("P1080 的用途", "G120 P1080 relevant", "G120 rated voltage"),
        ("S7-1200 MW20", "S7-1200 MW20 relevant", "S7-1500 MW20"),
    ],
)
def test_exact_industrial_identifiers_do_not_regress(query, matching, distractor):
    result = RetrievalResult([
        _candidate("matching", matching, 1),
        _candidate("distractor", distractor, 2),
    ])
    output = _reranker().rerank(query, result, top_k=2)
    assert output.result.candidates[0].chunk_id == "matching"


def test_model_failure_retains_original_order_and_reports_fallback():
    def unavailable(*_):
        raise OSError("model cache missing")

    result = RetrievalResult([
        _candidate("first", "one", 1),
        _candidate("second", "two", 2),
    ])
    output = _reranker(unavailable).rerank("query", result, top_k=2)
    assert [item.chunk_id for item in output.result.candidates] == ["first", "second"]
    assert output.reranker_requested is True
    assert output.reranker_effective is False
    assert "model cache missing" in output.reranker_fallback_reason


def test_invalid_environment_configuration_is_safe_and_observable():
    with patch.dict(
        "os.environ",
        {"RERANK_ENABLED": "true", "RERANK_CANDIDATE_K": "0"},
    ):
        config, error = load_reranker_config()
    assert config.enabled is False
    assert "Invalid reranker configuration" in error
