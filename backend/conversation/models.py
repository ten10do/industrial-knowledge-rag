from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_MAX_RECENT_TURNS = 6
DEFAULT_MAX_HISTORY_TURNS = 40
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_COMPRESSION_THRESHOLD = 6000

MIN_RECENT_TURNS = 1
MAX_RECENT_TURNS = 12
MIN_HISTORY_TURNS = 1
MAX_HISTORY_TURNS = 80
MIN_CONTEXT_CHARS = 1000
MAX_CONTEXT_CHARS = 20000
MIN_COMPRESSION_THRESHOLD = 100
MAX_COMPRESSION_THRESHOLD = 16000

MAX_MESSAGE_CHARS = 4000
MAX_QUESTION_CHARS = 1000
MAX_SUMMARY_CHARS = 2000
MAX_STANDALONE_QUERY_CHARS = 1000
MAX_CONVERSATION_ID_CHARS = 64
MAX_TURN_SOURCES = 8
CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


def normalize_message_text(value: str) -> str:
    return " ".join(value.split())


class ConversationSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(min_length=1, max_length=255)
    page: int | str | None = None
    score: float | None = None


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    timestamp: datetime | None = None
    sources: list[ConversationSource] | None = Field(
        default=None,
        max_length=MAX_TURN_SOURCES,
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_type(cls, value):
        if not isinstance(value, str):
            raise ValueError("content 必须是字符串。")
        return value

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = normalize_message_text(value)
        if not normalized:
            raise ValueError("content 不能为空。")
        return normalized


class ContextOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_recent_turns: int = Field(
        default=DEFAULT_MAX_RECENT_TURNS,
        ge=MIN_RECENT_TURNS,
        le=MAX_RECENT_TURNS,
    )
    max_history_turns: int = Field(
        default=DEFAULT_MAX_HISTORY_TURNS,
        ge=MIN_HISTORY_TURNS,
        le=MAX_HISTORY_TURNS,
    )
    max_context_chars: int = Field(
        default=DEFAULT_MAX_CONTEXT_CHARS,
        ge=MIN_CONTEXT_CHARS,
        le=MAX_CONTEXT_CHARS,
    )
    compression_threshold: int = Field(
        default=DEFAULT_COMPRESSION_THRESHOLD,
        ge=MIN_COMPRESSION_THRESHOLD,
        le=MAX_COMPRESSION_THRESHOLD,
    )
    enable_query_rewrite: bool = True
    enable_context_compression: bool = True

    @model_validator(mode="after")
    def validate_related_limits(self):
        if self.max_recent_turns > self.max_history_turns:
            raise ValueError(
                "max_recent_turns 不能大于 max_history_turns。"
            )
        if self.compression_threshold > self.max_context_chars:
            raise ValueError(
                "compression_threshold 不能大于 max_context_chars。"
            )
        return self


QueryRewriteStatus = Literal[
    "not_needed",
    "disabled",
    "unchanged",
    "rewritten",
    "fallback",
    "unresolved",
]
CompressionStatus = Literal[
    "not_needed",
    "disabled",
    "compressed",
    "fallback",
]


class ConversationContext(BaseModel):
    conversation_id: str
    standalone_query: str
    history_turn_count: int = Field(ge=0)
    retained_turn_count: int = Field(ge=0)
    compressed_turn_count: int = Field(ge=0)
    was_compressed: bool
    summary_used: bool
    estimated_context_size: int = Field(ge=0)
    query_rewrite_status: QueryRewriteStatus
    compression_status: CompressionStatus
    fallback_used: bool
    context_limit_applied: bool
