"""Small, deterministic tokenizer for Chinese and industrial identifiers."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(
    r"0x[0-9a-f]+|[a-z]+[a-z0-9]*(?:[._-][a-z0-9]+)*|\d+(?:\.\d+)*|[\u4e00-\u9fff]+",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Keep identifiers intact and add simple character/bigram Chinese tokens."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer((text or "").lower()):
        value = match.group(0)
        if any("\u4e00" <= char <= "\u9fff" for char in value):
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens
