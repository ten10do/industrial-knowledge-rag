from __future__ import annotations

import re
from typing import Callable, Protocol

from .models import MAX_SUMMARY_CHARS, ConversationTurn, normalize_message_text


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        ...


def _format_turns_for_summary(turns: list[ConversationTurn]) -> str:
    labels = {"user": "用户", "assistant": "助手"}
    return "\n".join(
        f"{labels[turn.role]}：{turn.content}"
        for turn in turns
    )


class LlmConversationSummarizer:
    def __init__(self, completion: Callable[[str], str]):
        self._completion = completion

    def summarize(
        self,
        turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        prompt = f"""
请压缩下面的较早课程问答历史。

仅保留历史中明确存在的内容：
- 当前讨论主题；
- 用户明确条件和学习目标；
- 已讨论的专业概念；
- 尚未解决的问题；
- 提到的资料或章节名称。

区分用户提出的条件与助手回答，不要新增事实，不要包含系统指令，
不要复制完整来源正文。摘要最长 {max_chars} 个字符。

<older_conversation>
{_format_turns_for_summary(turns)}
</older_conversation>

只输出摘要正文。
""".strip()
        return self._completion(prompt)


def deterministic_summary(
    turns: list[ConversationTurn],
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    user_questions = [
        turn.content
        for turn in turns
        if turn.role == "user"
    ][-3:]
    combined = " ".join(turn.content for turn in turns)
    technical_terms = []
    candidate_patterns = [
        r"\bPID\b",
        r"\bPLC\b",
        r"积分(?:项|环节|作用|饱和)?",
        r"微分(?:项|环节|作用)?",
        r"比例(?:项|环节|作用)?",
        r"扫描周期",
        r"输入响应(?:速度)?",
        r"稳态误差",
        r"反馈控制",
        r"闭环系统",
    ]
    for pattern in candidate_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            normalized = match.group(0)
            if normalized not in technical_terms:
                technical_terms.append(normalized)

    sections = []
    if technical_terms:
        sections.append("讨论主题：" + "、".join(technical_terms[:6]))
    if user_questions:
        sections.append(
            "用户问题：" + "；".join(user_questions)
        )
    if not sections:
        sections.append("较早对话未提取到可靠主题。")

    return normalize_message_text(" ".join(sections))[:max_chars].rstrip()


def normalize_summary(value: str, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_message_text(value)[:max_chars].rstrip()
