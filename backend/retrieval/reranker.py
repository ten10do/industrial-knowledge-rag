"""Optional local Cross-Encoder reranking over unified retrieval candidates."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from threading import Lock
from typing import Callable

from .candidates import RetrievalResult


DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANK_CANDIDATE_K = 5
DEFAULT_RERANK_TOP_K = 3
SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model_name: str = DEFAULT_RERANK_MODEL
    candidate_k: int = DEFAULT_RERANK_CANDIDATE_K
    top_k: int = DEFAULT_RERANK_TOP_K
    device: str = "cpu"

    def __post_init__(self):
        if not self.model_name.strip():
            raise ValueError("RERANK_MODEL must not be empty.")
        if self.candidate_k <= 0 or self.top_k <= 0:
            raise ValueError("Reranker K values must be positive.")
        if self.top_k > self.candidate_k:
            raise ValueError("RERANK_TOP_K must not exceed RERANK_CANDIDATE_K.")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError("RERANK_DEVICE must be cpu, cuda, or mps.")


@dataclass
class RerankOutcome:
    result: RetrievalResult
    reranker_requested: bool
    reranker_effective: bool
    reranker_fallback_reason: str = ""
    model: str = ""
    candidate_count: int = 0
    output_count: int = 0

    def status(self) -> dict:
        payload = asdict(self)
        payload.pop("result")
        return payload


def _enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("RERANK_ENABLED must be true or false.")


def load_reranker_config() -> tuple[RerankerConfig, str]:
    try:
        return RerankerConfig(
            enabled=_enabled(os.getenv("RERANK_ENABLED", "false")),
            model_name=os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL),
            candidate_k=int(os.getenv("RERANK_CANDIDATE_K", str(DEFAULT_RERANK_CANDIDATE_K))),
            top_k=int(os.getenv("RERANK_TOP_K", str(DEFAULT_RERANK_TOP_K))),
            device=os.getenv("RERANK_DEVICE", "cpu").strip().lower(),
        ), ""
    except (TypeError, ValueError) as exc:
        return RerankerConfig(), f"Invalid reranker configuration: {exc}"


def _default_model_factory(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


class CrossEncoderReranker:
    def __init__(
        self,
        config: RerankerConfig,
        *,
        model_factory: Callable | None = None,
        configuration_error: str = "",
    ):
        self.config = config
        self.configuration_error = configuration_error
        self._model_factory = model_factory or _default_model_factory
        self._model = None
        self._model_error = ""
        self._lock = Lock()

    @property
    def requested(self) -> bool:
        return self.config.enabled or bool(self.configuration_error)

    def retrieval_k(self, default_k: int) -> int:
        return self.config.candidate_k if self.config.enabled else default_k

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._model_error:
            raise RuntimeError(self._model_error)
        with self._lock:
            if self._model is None:
                try:
                    self._model = self._model_factory(
                        self.config.model_name,
                        self.config.device,
                    )
                except Exception as exc:
                    self._model_error = f"{type(exc).__name__}: {exc}"
                    raise RuntimeError(self._model_error) from exc
        return self._model

    def load(self):
        """Load the configured local model explicitly for smoke tests and benchmarks."""
        if self.configuration_error:
            raise RuntimeError(self.configuration_error)
        return self._load_model()

    def rerank(self, query: str, result: RetrievalResult, top_k: int | None = None) -> RerankOutcome:
        candidates = list(getattr(result, "candidates", []) or [])
        output_k = top_k or self.config.top_k
        if output_k <= 0:
            raise ValueError("Rerank top_k must be positive.")

        def subset(selected: list) -> RetrievalResult:
            return RetrievalResult(
                selected,
                query_analysis=getattr(result, "query_analysis", None),
                corpus_documents=getattr(result, "corpus_documents", []),
                retrieval_mode=getattr(result, "retrieval_mode", ""),
                scope_decision=getattr(result, "scope_decision", None),
                section_report=getattr(result, "section_report", None),
            )

        if not self.requested:
            return RerankOutcome(result, False, False, candidate_count=len(candidates), output_count=len(candidates))
        if self.configuration_error:
            original = subset(candidates[:output_k])
            return RerankOutcome(
                original, True, False, self.configuration_error,
                self.config.model_name, len(candidates), len(original.candidates),
            )
        if not candidates:
            return RerankOutcome(
                result, True, True, model=self.config.model_name,
                candidate_count=0, output_count=0,
            )
        for rank, candidate in enumerate(candidates, start=1):
            candidate.pre_rerank_rank = candidate.final_rank or rank
            candidate.rerank_score = None
            candidate.rerank_rank = None
        try:
            model = self._load_model()
            scores = model.predict(
                [(query, candidate.document.page_content) for candidate in candidates],
                show_progress_bar=False,
            )
            scored = list(zip(candidates, [float(score) for score in scores]))
            scored.sort(
                key=lambda item: (
                    -item[1],
                    item[0].pre_rerank_rank or 10**9,
                    item[0].chunk_id,
                )
            )
            ranked = []
            for rank, (candidate, score) in enumerate(scored, start=1):
                candidate.rerank_score = score
                candidate.rerank_rank = rank
                if rank <= output_k:
                    ranked.append(candidate)
            reranked = subset(ranked)
            return RerankOutcome(
                reranked, True, True, model=self.config.model_name,
                candidate_count=len(candidates), output_count=len(ranked),
            )
        except Exception as exc:
            original = subset(candidates[:output_k])
            return RerankOutcome(
                original, True, False, f"Reranker unavailable: {exc}",
                self.config.model_name, len(candidates), len(original.candidates),
            )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    config, error = load_reranker_config()
    return CrossEncoderReranker(config, configuration_error=error)
