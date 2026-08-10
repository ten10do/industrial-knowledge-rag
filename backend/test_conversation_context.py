import copy

import pytest
from pydantic import ValidationError

from backend.conversation.context_manager import ConversationContextManager
from backend.conversation.models import (
    MAX_MESSAGE_CHARS,
    ContextOptions,
    ConversationTurn,
)


class FakeSummarizer:
    def __init__(self, result="较早对话讨论了 PID 控制及用户关注的积分作用。"):
        self.result = result
        self.calls = []

    def summarize(self, turns, max_chars):
        self.calls.append((copy.deepcopy(turns), max_chars))
        return self.result


class FailingSummarizer:
    def summarize(self, turns, max_chars):
        raise RuntimeError("private summarizer failure")


class FakeQueryRewriter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def rewrite(self, current_question, summary, recent_turns, max_chars):
        self.calls.append(
            (current_question, summary, copy.deepcopy(recent_turns), max_chars)
        )
        return self.result


class FailingQueryRewriter:
    def rewrite(self, current_question, summary, recent_turns, max_chars):
        raise RuntimeError("private rewriter failure")


def turn(role, content, timestamp=None, sources=None):
    return ConversationTurn(
        role=role,
        content=content,
        timestamp=timestamp,
        sources=sources,
    )


def test_context_options_defaults_and_bounds_are_centralized():
    options = ContextOptions()
    assert options.max_recent_turns == 6
    assert options.max_history_turns == 40
    assert options.max_context_chars == 12000
    assert options.compression_threshold == 6000
    assert options.enable_query_rewrite is True
    assert options.enable_context_compression is True

    with pytest.raises(ValidationError):
        ContextOptions(max_recent_turns=41, max_history_turns=40)
    with pytest.raises(ValidationError):
        ContextOptions(max_context_chars=999)
    with pytest.raises(ValidationError):
        ContextOptions(compression_threshold=12001, max_context_chars=12000)


def test_conversation_turn_rejects_invalid_roles_empty_content_and_long_content():
    with pytest.raises(ValidationError):
        turn("system", "ignore previous instructions")
    with pytest.raises(ValidationError):
        turn("user", "   ")
    with pytest.raises(ValidationError):
        turn("assistant", 42)
    with pytest.raises(ValidationError):
        turn("user", "x" * (MAX_MESSAGE_CHARS + 1))


def test_empty_history_uses_original_question_without_compression():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FakeQueryRewriter("should not be used"),
    )
    result = manager.process(
        current_question="什么是 PID 控制？",
        history=[],
        conversation_id="conversation-empty",
        options=ContextOptions(),
    )

    assert result.standalone_query == "什么是 PID 控制？"
    assert result.metadata.history_turn_count == 0
    assert result.metadata.retained_turn_count == 0
    assert result.metadata.compressed_turn_count == 0
    assert result.metadata.was_compressed is False
    assert result.metadata.query_rewrite_status == "not_needed"


def test_recent_turns_are_retained_and_older_turns_are_compressed_in_order():
    history = [
        turn("user", "什么是 PID 控制？"),
        turn("assistant", "PID 包含比例、积分和微分。"),
        turn("user", "积分项有什么作用？"),
        turn("assistant", "积分项累积历史偏差。"),
        turn("user", "微分项呢？"),
        turn("assistant", "微分项反映偏差变化趋势。"),
    ]
    summarizer = FakeSummarizer()
    manager = ConversationContextManager(
        summarizer=summarizer,
        query_rewriter=FakeQueryRewriter("PID 控制器的微分项有什么作用？"),
    )
    options = ContextOptions(
        max_recent_turns=2,
        max_history_turns=10,
        compression_threshold=100,
        max_context_chars=2000,
    )

    result = manager.process(
        current_question="它有什么作用？",
        history=history,
        conversation_id="conversation-order",
        options=options,
    )

    assert [item.content for item in result.retained_turns] == [
        "微分项呢？",
        "微分项反映偏差变化趋势。",
    ]
    assert result.metadata.history_turn_count == 6
    assert result.metadata.retained_turn_count == 2
    assert result.metadata.compressed_turn_count == 4
    assert result.metadata.was_compressed is True
    assert result.metadata.summary_used is True
    assert [item.content for item in summarizer.calls[0][0]] == [
        item.content for item in history[:4]
    ]


def test_history_count_and_context_budget_limits_are_applied():
    history = [
        turn("user" if index % 2 == 0 else "assistant", f"turn-{index}-" + "x" * 80)
        for index in range(10)
    ]
    manager = ConversationContextManager(
        summarizer=FakeSummarizer("summary-" + "s" * 1000),
        query_rewriter=FakeQueryRewriter("独立问题"),
    )
    options = ContextOptions(
        max_recent_turns=4,
        max_history_turns=6,
        max_context_chars=1000,
        compression_threshold=500,
    )

    result = manager.process(
        current_question="当前问题",
        history=history,
        conversation_id="conversation-budget",
        options=options,
    )

    assert result.metadata.history_turn_count == 10
    assert result.metadata.context_limit_applied is True
    assert result.metadata.retained_turn_count <= 4
    assert result.metadata.estimated_context_size <= 1000
    assert result.retained_turns[-1].content.startswith("turn-9-")


def test_summarizer_input_is_bounded_before_the_llm_call():
    history = [
        turn(
            "user" if index % 2 == 0 else "assistant",
            f"{index:02d}-" + "x" * 3900,
        )
        for index in range(40)
    ]
    summarizer = FakeSummarizer("有界摘要")
    manager = ConversationContextManager(
        summarizer=summarizer,
        query_rewriter=FakeQueryRewriter("当前问题"),
    )

    result = manager.process(
        current_question="当前问题",
        history=history,
        conversation_id="conversation-summary-input-budget",
        options=ContextOptions(),
    )

    summary_turns = summarizer.calls[0][0]
    assert sum(len(item.content) for item in summary_turns) <= 12000
    assert summary_turns[-1].content.startswith("33-")
    assert result.metadata.context_limit_applied is True


def test_max_recent_turns_is_a_hard_window_even_below_compression_threshold():
    history = [
        turn(
            "user" if index % 2 == 0 else "assistant",
            f"短消息 {index}",
        )
        for index in range(10)
    ]
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FakeQueryRewriter("完整问题"),
    )

    result = manager.process(
        current_question="完整问题",
        history=history,
        conversation_id="conversation-recent-window",
        options=ContextOptions(
            max_recent_turns=6,
            compression_threshold=6000,
        ),
    )

    assert result.metadata.retained_turn_count == 6
    assert result.metadata.compressed_turn_count == 4
    assert result.metadata.was_compressed is True
    assert result.metadata.summary_used is True


def test_max_recent_turns_stays_hard_when_compression_is_disabled():
    history = [
        turn(
            "user" if index % 2 == 0 else "assistant",
            f"短消息 {index}",
        )
        for index in range(10)
    ]
    summarizer = FakeSummarizer()
    manager = ConversationContextManager(
        summarizer=summarizer,
        query_rewriter=FakeQueryRewriter("完整问题"),
    )

    result = manager.process(
        current_question="完整问题",
        history=history,
        conversation_id="conversation-disabled-hard-window",
        options=ContextOptions(
            max_recent_turns=6,
            enable_context_compression=False,
        ),
    )

    assert result.retained_turns == history[-6:]
    assert result.metadata.retained_turn_count == 6
    assert result.metadata.compressed_turn_count == 0
    assert result.metadata.was_compressed is False
    assert result.metadata.summary_used is False
    assert result.metadata.compression_status == "disabled"
    assert result.metadata.context_limit_applied is True
    assert summarizer.calls == []


def test_input_history_is_not_mutated_and_processing_is_deterministic():
    history = [
        turn("user", "  PLC   扫描周期包括哪些阶段？ \n"),
        turn("assistant", " 输入采样、程序执行、输出刷新。 "),
    ]
    original = copy.deepcopy(history)
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FakeQueryRewriter("PLC 扫描周期对输入响应有什么影响？"),
    )

    first = manager.process(
        current_question="它对输入响应有什么影响？",
        history=history,
        conversation_id="conversation-stable",
        options=ContextOptions(),
    )
    second = manager.process(
        current_question="它对输入响应有什么影响？",
        history=history,
        conversation_id="conversation-stable",
        options=ContextOptions(),
    )

    assert history == original
    assert first.standalone_query == second.standalone_query
    assert first.metadata.model_dump() == second.metadata.model_dump()
    assert first.retained_turns[0].content == "PLC 扫描周期包括哪些阶段？"


def test_summarizer_failure_uses_bounded_deterministic_fallback():
    history = [
        turn("user", "PID 控制器由哪些环节组成？"),
        turn("assistant", "包括比例、积分和微分环节。"),
        turn("user", "积分环节用于什么？"),
        turn("assistant", "用于累积偏差。"),
    ]
    manager = ConversationContextManager(
        summarizer=FailingSummarizer(),
        query_rewriter=FakeQueryRewriter("PID 控制器中的积分项有什么作用？"),
    )
    options = ContextOptions(
        max_recent_turns=2,
        max_history_turns=10,
        compression_threshold=100,
        max_context_chars=1500,
    )

    first = manager.process(
        current_question="其中积分项有什么作用？",
        history=history,
        conversation_id="conversation-summary-fallback",
        options=options,
    )
    second = manager.process(
        current_question="其中积分项有什么作用？",
        history=history,
        conversation_id="conversation-summary-fallback",
        options=options,
    )

    assert first.summary == second.summary
    assert "用户问题" in first.summary
    assert len(first.summary) <= 2000
    assert first.metadata.compression_status == "fallback"
    assert "private summarizer failure" not in first.summary


@pytest.mark.parametrize(
    ("history", "question", "expected"),
    [
        (
            [
                turn("user", "什么是 PID 控制？"),
                turn("assistant", "PID 包含比例、积分和微分环节。"),
            ],
            "其中积分项有什么作用？",
            "PID 控制器中的积分项有什么作用？",
        ),
        (
            [
                turn("user", "PLC 的扫描周期包括哪些阶段？"),
                turn("assistant", "包括输入采样、程序执行和输出刷新。"),
            ],
            "它对输入响应速度有什么影响？",
            "PLC 扫描周期对输入响应速度有什么影响？",
        ),
    ],
)
def test_query_rewriter_result_is_used_for_contextual_followups(
    history, question, expected
):
    rewriter = FakeQueryRewriter(expected)
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=rewriter,
    )

    result = manager.process(
        current_question=question,
        history=history,
        conversation_id="conversation-rewrite",
        options=ContextOptions(),
    )

    assert result.standalone_query == expected
    assert result.metadata.query_rewrite_status == "rewritten"
    assert len(rewriter.calls) == 1


def test_complete_question_is_kept_when_rewriter_returns_same_meaning():
    question = "PLC 扫描周期对输入信号响应速度有什么影响？"
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FakeQueryRewriter(question),
    )
    result = manager.process(
        current_question=question,
        history=[turn("user", "上一轮讨论了 PLC。")],
        conversation_id="conversation-complete",
        options=ContextOptions(),
    )

    assert result.standalone_query == question
    assert result.metadata.query_rewrite_status == "unchanged"


def test_rewriter_failure_uses_pid_and_plc_fallback_without_raising():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )

    pid = manager.process(
        current_question="其中积分项有什么作用？",
        history=[turn("user", "什么是 PID 控制？")],
        conversation_id="conversation-pid-fallback",
        options=ContextOptions(),
    )
    plc = manager.process(
        current_question="它对输入响应速度有什么影响？",
        history=[turn("user", "PLC 的扫描周期包括哪些阶段？")],
        conversation_id="conversation-plc-fallback",
        options=ContextOptions(),
    )

    assert pid.standalone_query == "PID 控制器中的积分项有什么作用？"
    assert plc.standalone_query == "PLC 扫描周期对输入响应速度有什么影响？"
    assert pid.metadata.query_rewrite_status == "fallback"
    assert plc.metadata.query_rewrite_status == "fallback"
    assert pid.metadata.fallback_used is True


def test_unresolved_pronoun_does_not_invent_a_topic():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    result = manager.process(
        current_question="它为什么会这样？",
        history=[turn("assistant", "这是一个没有明确主题的回答。")],
        conversation_id="conversation-unresolved",
        options=ContextOptions(),
    )

    assert result.standalone_query == "它为什么会这样？"
    assert result.metadata.query_rewrite_status == "unresolved"

    contextual_only = manager.process(
        current_question="它为什么影响稳定性？",
        history=[
            turn("user", "其中反馈环节有什么作用？"),
            turn("assistant", "反馈环节用于修正偏差。"),
        ],
        conversation_id="conversation-contextual-only",
        options=ContextOptions(),
    )
    assert contextual_only.standalone_query == "它为什么影响稳定性？"
    assert contextual_only.metadata.query_rewrite_status == "unresolved"


def test_rewriter_failure_for_complete_question_is_reported_as_fallback():
    question = "PID 控制器的积分项为什么会饱和？"
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    result = manager.process(
        current_question=question,
        history=[turn("user", "什么是 PID 控制？")],
        conversation_id="conversation-complete-fallback",
        options=ContextOptions(),
    )

    assert result.standalone_query == question
    assert result.metadata.query_rewrite_status == "fallback"
    assert result.metadata.fallback_used is True


def test_qi_pronoun_fallback_respects_word_boundaries_and_possessives():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    history = [turn("user", "什么是 PID 控制？")]

    for question in (
        "其他控制方法有哪些？",
        "其他的控制方式有哪些？",
        "其次还需要考虑哪些因素？",
        "其实这种方法有什么限制？",
        "其余参数如何设置？",
        "其间发生了哪些状态变化？",
        "其他控制方法与 PID 控制相比有哪些区别？",
        "其他阿尔法结构有哪些？",
        "目前常用的控制方法有哪些？",
        "其它方法是否适用？",
        "反馈因素尤其重要吗？",
    ):
        result = manager.process(
            current_question=question,
            history=history,
            conversation_id="conversation-non-pronoun-qi",
            options=ContextOptions(),
        )
        assert result.standalone_query == question
        assert result.metadata.query_rewrite_status == "fallback"
        assert result.metadata.fallback_used is True

    expected_rewrites = {
        "其作用是什么？": "PID 控制器的作用是什么？",
        "其影响有哪些？": "PID 控制器的影响有哪些？",
        "其原理是什么？": "PID 控制器的原理是什么？",
        "请说明其参数如何影响响应。": "请说明PID 控制器的参数如何影响响应。",
        "其对系统稳定性有什么影响？": "PID 控制器对系统稳定性有什么影响？",
    }
    for question, expected in expected_rewrites.items():
        result = manager.process(
            current_question=question,
            history=history,
            conversation_id="conversation-possessive-qi",
            options=ContextOptions(),
        )
        assert result.standalone_query == expected
        assert result.metadata.query_rewrite_status == "fallback"
        assert result.metadata.fallback_used is True


def test_fallback_recovers_explicit_subtopic_without_assistant_echo():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    history = [
        turn("user", "闭环控制是什么？"),
        turn("assistant", "闭环控制通过测量输出修正偏差。"),
        turn("user", "反馈环节有什么作用？"),
        turn("assistant", "反馈环节用于比较输出和设定值。"),
    ]
    expected = "闭环控制的反馈环节为什么影响稳定性？"

    recent = manager.process(
        current_question="它为什么影响稳定性？",
        history=history,
        conversation_id="conversation-explicit-subtopic",
        options=ContextOptions(),
    )
    compressed = manager.process(
        current_question="它为什么影响稳定性？",
        history=history,
        conversation_id="conversation-explicit-subtopic-compressed",
        options=ContextOptions(
            max_recent_turns=2,
            compression_threshold=6000,
        ),
    )

    for result in (recent, compressed):
        assert result.standalone_query == expected
        assert result.metadata.query_rewrite_status == "fallback"
        assert result.metadata.fallback_used is True


def test_fallback_keeps_parallel_multi_topic_history_unresolved():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    history = [
        turn("user", "PID 控制有什么特点？"),
        turn("assistant", "PID 控制结合比例、积分和微分作用。"),
        turn("user", "PLC 扫描周期包括哪些阶段？"),
        turn("assistant", "包括输入采样、程序执行和输出刷新。"),
        turn("user", "变频器有哪些作用？"),
        turn("assistant", "变频器可以调节电机转速。"),
    ]

    recent = manager.process(
        current_question="其作用是什么？",
        history=history,
        conversation_id="conversation-ambiguous-topics",
        options=ContextOptions(),
    )
    compressed = manager.process(
        current_question="其作用是什么？",
        history=history,
        conversation_id="conversation-ambiguous-topics-compressed",
        options=ContextOptions(
            max_recent_turns=2,
            compression_threshold=6000,
        ),
    )

    for result in (recent, compressed):
        assert result.standalone_query == "其作用是什么？"
        assert result.metadata.query_rewrite_status == "unresolved"
        assert result.metadata.fallback_used is True


def test_chained_pronoun_fallback_keeps_parent_and_detail_topic():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    result = manager.process(
        current_question="它为什么可能导致积分饱和？",
        history=[
            turn("user", "什么是 PID 控制？"),
            turn("assistant", "PID 包含比例、积分和微分环节。"),
            turn("user", "其中积分项有什么作用？"),
            turn("assistant", "积分项会累积历史偏差。"),
        ],
        conversation_id="conversation-chained-fallback",
        options=ContextOptions(),
    )

    assert result.standalone_query == (
        "PID 控制器的积分项为什么可能导致积分饱和？"
    )
    assert result.metadata.query_rewrite_status == "fallback"
    assert result.metadata.fallback_used is True


def test_generic_chained_pronoun_fallback_recovers_parent_and_recent_subject():
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=FailingQueryRewriter(),
    )
    cases = [
        (
            "什么是闭环控制？",
            "闭环控制通过反馈修正偏差。",
            "其中反馈环节有什么作用？",
            "反馈环节用于比较输出和设定值。",
            "它为什么影响稳定性？",
            "闭环控制的反馈环节为什么影响稳定性？",
        ),
        (
            "请介绍前馈控制。",
            "前馈控制根据扰动提前补偿。",
            "其中补偿环节有什么作用？",
            "补偿环节用于抵消可测扰动。",
            "它如何改善控制效果？",
            "前馈控制的补偿环节如何改善控制效果？",
        ),
        (
            "串级控制包括哪些环节？",
            "串级控制包含主回路和副回路。",
            "其中副回路有什么作用？",
            "副回路用于快速抑制内环扰动。",
            "它如何改善动态性能？",
            "串级控制的副回路如何改善动态性能？",
        ),
    ]

    for index, (
        first_question,
        first_answer,
        subject_question,
        subject_answer,
        current_question,
        expected,
    ) in enumerate(cases):
        history = [
            turn("user", first_question),
            turn("assistant", first_answer),
            turn("user", subject_question),
            turn("assistant", subject_answer),
        ]
        recent = manager.process(
            current_question=current_question,
            history=history,
            conversation_id=f"conversation-generic-chain-{index}",
            options=ContextOptions(),
        )
        compressed = manager.process(
            current_question=current_question,
            history=history,
            conversation_id=f"conversation-generic-compressed-chain-{index}",
            options=ContextOptions(
                max_recent_turns=2,
                compression_threshold=6000,
            ),
        )
        llm_compressed = manager.process(
            current_question=current_question,
            history=history,
            conversation_id=f"conversation-generic-llm-chain-{index}",
            options=ContextOptions(
                max_recent_turns=2,
                compression_threshold=100,
            ),
        )

        assert recent.standalone_query == expected
        assert compressed.standalone_query == expected
        assert llm_compressed.standalone_query == expected
        assert recent.metadata.query_rewrite_status == "fallback"
        assert compressed.metadata.query_rewrite_status == "fallback"
        assert llm_compressed.metadata.query_rewrite_status == "fallback"
        assert recent.metadata.fallback_used is True
        assert compressed.metadata.fallback_used is True
        assert llm_compressed.metadata.fallback_used is True


def test_query_rewrite_can_be_disabled_and_output_is_bounded():
    rewriter = FakeQueryRewriter("x" * 5000)
    manager = ConversationContextManager(
        summarizer=FakeSummarizer(),
        query_rewriter=rewriter,
    )
    disabled = manager.process(
        current_question="其中积分项有什么作用？",
        history=[turn("user", "什么是 PID 控制？")],
        conversation_id="conversation-disabled",
        options=ContextOptions(enable_query_rewrite=False),
    )
    bounded = manager.process(
        current_question="其中积分项有什么作用？",
        history=[turn("user", "什么是 PID 控制？")],
        conversation_id="conversation-bounded",
        options=ContextOptions(),
    )

    assert disabled.standalone_query == "其中积分项有什么作用？"
    assert disabled.metadata.query_rewrite_status == "disabled"
    assert len(bounded.standalone_query) <= 1000
