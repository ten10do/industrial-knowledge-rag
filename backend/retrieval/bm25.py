"""Dependency-free BM25 scorer used by both RAG backends."""

from __future__ import annotations

import math
from collections import Counter

from .tokenizer import tokenize


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency = Counter(
            token for tokens in self.tokens for token in set(tokens)
        )
        count = len(self.tokens)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.tokens:
            return [0.0] * len(self.tokens)
        scores: list[float] = []
        for length, frequencies in zip(self.lengths, self.frequencies):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1.0)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores
