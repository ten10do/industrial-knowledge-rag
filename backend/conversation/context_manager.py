from __future__ import annotations

import copy
from dataclasses import dataclass
from .budget import (
    estimate_context_chars,
    trim_to_context_budget,
    trim_turns_to_char_budget,
)
from .models import (
    MAX_STANDALONE_QUERY_CHARS,
    MAX_SUMMARY_CHARS,
    ContextOptions,
    ConversationContext,
    ConversationTurn,
    normalize_message_text,
)
from .query_rewriter import (
    QueryRewriter,
    deterministic_rewrite,
    normalize_standalone_query,
)
from .summarizer import (
    ConversationSummarizer,
    deterministic_summary,
    normalize_summary,
)


@dataclass(frozen=True)
class ContextProcessingResult:
    standalone_query: str
    summary: str
    retained_turns: list[ConversationTurn]
    metadata: ConversationContext


class ConversationContextManager:
    def __init__(
        self,
        summarizer: ConversationSummarizer,
        query_rewriter: QueryRewriter,
    ):
        self._summarizer = summarizer
        self._query_rewriter = query_rewriter

    def process(
        self,
        current_question: str,
        history: list[ConversationTurn],
        conversation_id: str,
        options: ContextOptions,
    ) -> ContextProcessingResult:
        normalized_question = normalize_message_text(current_question)
        copied_history = [
            ConversationTurn.model_validate(
                copy.deepcopy(turn.model_dump() if isinstance(turn, ConversationTurn) else turn)
            )
            for turn in list(history)
        ]
        history_turn_count = len(copied_history)
        limited_history = copied_history[-options.max_history_turns :]
        context_limit_applied = len(limited_history) < history_turn_count

        summary = ""
        deterministic_rewrite_summary = ""
        compressed_turn_count = 0
        compression_status = "not_needed"
        compression_fallback = False
        total_raw_size = estimate_context_chars(
            normalized_question,
            "",
            limited_history,
        )
        has_older_turns = len(limited_history) > options.max_recent_turns

        if has_older_turns:
            older_turns = limited_history[: -options.max_recent_turns]
            retained_turns = limited_history[-options.max_recent_turns :]
            if options.enable_context_compression:
                compressed_turn_count = len(older_turns)
                (
                    summary_turns,
                    summary_limit_applied,
                ) = trim_turns_to_char_budget(
                    older_turns,
                    options.max_context_chars,
                )
                deterministic_rewrite_summary = deterministic_summary(
                    summary_turns,
                    max_chars=MAX_SUMMARY_CHARS,
                )
                context_limit_applied = (
                    context_limit_applied or summary_limit_applied
                )
                if total_raw_size > options.compression_threshold:
                    try:
                        summary = normalize_summary(
                            self._summarizer.summarize(
                                copy.deepcopy(summary_turns),
                                MAX_SUMMARY_CHARS,
                            ),
                            MAX_SUMMARY_CHARS,
                        )
                    except Exception:
                        summary = ""
                    if summary:
                        compression_status = "compressed"
                    else:
                        summary = deterministic_rewrite_summary
                        compression_status = "fallback"
                        compression_fallback = True
                else:
                    summary = deterministic_rewrite_summary
                    compression_status = "compressed"
            else:
                compression_status = "disabled"
                context_limit_applied = True
        else:
            retained_turns = limited_history
            if not options.enable_context_compression:
                compression_status = "disabled"

        summary, retained_turns, budget_limit_applied = trim_to_context_budget(
            normalized_question,
            summary,
            retained_turns,
            options.max_context_chars,
        )
        context_limit_applied = context_limit_applied or budget_limit_applied
        query_fallback = False
        if not limited_history:
            standalone_query = normalized_question
            rewrite_status = "not_needed"
        elif not options.enable_query_rewrite:
            standalone_query = normalized_question
            rewrite_status = "disabled"
        else:
            rewrite_failed = False
            try:
                rewritten = normalize_standalone_query(
                    self._query_rewriter.rewrite(
                        normalized_question,
                        summary,
                        copy.deepcopy(retained_turns),
                        MAX_STANDALONE_QUERY_CHARS,
                    ),
                    MAX_STANDALONE_QUERY_CHARS,
                )
            except Exception:
                rewritten = ""
                rewrite_failed = True
            if not rewritten:
                rewrite_failed = True

            if rewritten:
                standalone_query = rewritten
                rewrite_status = (
                    "unchanged"
                    if rewritten == normalized_question
                    else "rewritten"
                )
            else:
                standalone_query, rewrite_status = deterministic_rewrite(
                    normalized_question,
                    retained_turns or limited_history,
                    max_chars=MAX_STANDALONE_QUERY_CHARS,
                    summary=deterministic_rewrite_summary or summary,
                )
                if rewrite_failed and rewrite_status == "unchanged":
                    rewrite_status = "fallback"
                query_fallback = True

        estimated_context_size = estimate_context_chars(
            normalized_question,
            summary,
            retained_turns,
        )
        metadata = ConversationContext(
            conversation_id=conversation_id,
            standalone_query=standalone_query,
            history_turn_count=history_turn_count,
            retained_turn_count=len(retained_turns),
            compressed_turn_count=compressed_turn_count,
            was_compressed=compressed_turn_count > 0,
            summary_used=bool(summary),
            estimated_context_size=estimated_context_size,
            query_rewrite_status=rewrite_status,
            compression_status=compression_status,
            fallback_used=compression_fallback or query_fallback,
            context_limit_applied=context_limit_applied,
        )
        return ContextProcessingResult(
            standalone_query=standalone_query,
            summary=summary,
            retained_turns=retained_turns,
            metadata=metadata,
        )
