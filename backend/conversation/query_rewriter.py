from __future__ import annotations

import re
from typing import Callable, Protocol

from .models import (
    MAX_STANDALONE_QUERY_CHARS,
    ConversationTurn,
    normalize_message_text,
)


QI_PRONOUN_PATTERN = (
    r"(?<![尤与极何任听如])"
    r"其"
    r"(?![中他它次实余间后前一二三四五六七八九十])"
)
CONTEXT_REFERENCE_PATTERN = (
    rf"(?:其中|(?<!其)它|{QI_PRONOUN_PATTERN}|"
    r"该(?:概念|方法|过程|环节)?|上述|这个)"
)
QI_DIRECT_SUFFIX_PATTERN = (
    r"^(?:对|在|与|由|会|将|可|能|是否|能否|如何|为何|为什么)"
)
QUESTION_SUBJECT_PATTERN = (
    r"^(?:其中|该|这个)?(.+?)(?="
    r"有什么|有何|有哪些|是什么|为什么|为何|如何|怎么|会不会|"
    r"是否|能否|可以|能够|包括哪些|包含哪些|由哪些|对|的作用)"
)
REFERENCE_SUBJECT_PREFIXES = ("其中", "该", "这个")
GENERIC_SUBJECT_SUFFIX_PATTERN = (
    r"(?:环节|项目|项|模块|部分|阶段|过程|方法|参数|回路)$"
)
STRUCTURAL_SUBTOPIC_SUFFIX_PATTERN = (
    r"(?:环节|项|模块|部分|阶段|参数|回路)$"
)


class QueryRewriter(Protocol):
    def rewrite(
        self,
        current_question: str,
        summary: str,
        recent_turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        ...


def _format_recent_turns(turns: list[ConversationTurn]) -> str:
    labels = {"user": "用户", "assistant": "助手"}
    return "\n".join(
        f"{labels[turn.role]}：{turn.content}"
        for turn in turns
    )


class LlmQueryRewriter:
    def __init__(self, completion: Callable[[str], str]):
        self._completion = completion

    def rewrite(
        self,
        current_question: str,
        summary: str,
        recent_turns: list[ConversationTurn],
        max_chars: int,
    ) -> str:
        prompt = f"""
请把当前课程追问改写为可独立检索的问题。

规则：
- 只使用历史中明确出现的信息；
- 消解“其中”“它”“该概念”等指代；
- 不改变用户原意，不新增主题；
- 已经完整的问题保持原意；
- 无法可靠消解时原样返回；
- 最长 {max_chars} 个字符。

<conversation_summary>
{summary or "无"}
</conversation_summary>

<recent_conversation>
{_format_recent_turns(recent_turns) or "无"}
</recent_conversation>

<current_question>
{current_question}
</current_question>

只输出改写后的问题。
""".strip()
        return self._completion(prompt)


def normalize_standalone_query(value: str, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = normalize_message_text(value)
    normalized = re.sub(
        r"^(?:独立问题|改写结果|standalone[_ ]query)\s*[:：]\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.strip("\"'")[:max_chars].rstrip()


def _normalize_topic_label(value: str) -> str:
    topic = normalize_message_text(value).strip("？?。 ")
    topic = re.sub(
        r"PLC\s*的\s*扫描周期",
        "PLC 扫描周期",
        topic,
        flags=re.I,
    )
    if re.fullmatch(r"PID\s*控制", topic, flags=re.I):
        topic = "PID 控制器"
    return topic[:120].strip()


def _extract_topic(text: str) -> str:
    text = text.strip("？?。 ")
    patterns = [
        r"^什么是(.+)$",
        r"^(.+?)是什么$",
        r"^(.+?)(?:包括|包含|由)哪些(?:阶段|环节|部分|内容).*$",
        r"^请(?:介绍|说明|解释)(.+)$",
    ]
    topic = ""
    has_summary_markers = re.search(
        r"(?:用户问题|讨论主题|当前讨论主题)\s*[:：]",
        text,
    )
    if not has_summary_markers:
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                topic = match.group(1)
                break

    if not topic:
        summarized_questions = re.search(
            r"(?:用户问题|问题)\s*[:：]\s*(.+)",
            text,
        )
        summarized_topic = re.search(
            r"(?:讨论主题|当前讨论主题)\s*[:：]\s*"
            r"(.+?)(?=\s+用户问题\s*[:：]|[。；\n]|$)",
            text,
        )
        if summarized_questions:
            for question in reversed(
                re.split(r"[；\n]", summarized_questions.group(1))
            ):
                for pattern in patterns:
                    match = re.match(
                        pattern,
                        question.strip("？?。 "),
                    )
                    if match:
                        topic = match.group(1)
                        break
                if topic:
                    break
        if not topic and summarized_topic:
            topic = summarized_topic.group(1)

    if not topic:
        pid_match = re.search(r"PID\s*控制(?:器)?", text, re.IGNORECASE)
        plc_match = re.search(r"PLC\s*的?扫描周期", text, re.IGNORECASE)
        if pid_match:
            topic = pid_match.group(0)
        elif plc_match:
            topic = plc_match.group(0)

    return _normalize_topic_label(topic)


def _extract_followup_subject(text: str) -> str:
    normalized = normalize_message_text(text).strip("？?。 ")
    match = re.match(QUESTION_SUBJECT_PATTERN, normalized)
    if not match:
        return ""
    subject = _normalize_topic_label(match.group(1)).strip("的 ")
    if not subject or re.search(CONTEXT_REFERENCE_PATTERN, subject):
        return ""
    return subject


def _has_reference_subject_prefix(text: str) -> bool:
    normalized = normalize_message_text(text)
    return normalized.startswith(REFERENCE_SUBJECT_PREFIXES)


def _is_subject_anchored(subject: str, assistant_text: str) -> bool:
    normalized_answer = normalize_message_text(assistant_text)
    if subject in normalized_answer:
        return True
    base_subject = re.sub(GENERIC_SUBJECT_SUFFIX_PATTERN, "", subject)
    return len(base_subject) >= 2 and base_subject in normalized_answer


def _is_structural_subtopic(subject: str) -> bool:
    return bool(re.search(STRUCTURAL_SUBTOPIC_SUFFIX_PATTERN, subject))


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _extract_summary_topics(summary: str) -> list[str]:
    if not summary:
        return []

    topics: list[str] = []
    summarized_questions = re.search(
        r"(?:用户问题|问题)\s*[:：]\s*(.+)",
        summary,
    )
    if summarized_questions:
        for question in re.split(r"[；\n]", summarized_questions.group(1)):
            normalized = question.strip("？?。 ")
            if _has_reference_subject_prefix(normalized):
                continue
            _append_unique(
                topics,
                _extract_topic(normalized)
                or _extract_followup_subject(normalized),
            )

    if not topics:
        _append_unique(topics, _extract_topic(summary))
    return topics


def _extract_recent_user_topic(
    turns: list[ConversationTurn],
    summary: str = "",
) -> str:
    summary_topics = _extract_summary_topics(summary)
    root_topics = list(summary_topics)
    current_root = root_topics[0] if len(root_topics) == 1 else ""
    detail = ""
    recent_root_seen = False

    for index, turn in enumerate(turns):
        if turn.role != "user":
            continue

        text = turn.content
        explicit_topic = _extract_topic(text)
        subject = _extract_followup_subject(text)
        previous_assistant = (
            turns[index - 1].content
            if index > 0 and turns[index - 1].role == "assistant"
            else ""
        )

        if explicit_topic:
            _append_unique(root_topics, explicit_topic)
            current_root = explicit_topic
            detail = ""
            recent_root_seen = True
            continue

        if not subject:
            continue

        has_reference_prefix = _has_reference_subject_prefix(text)
        anchored_to_root = (
            len(root_topics) == 1
            and (
                has_reference_prefix
                or _is_subject_anchored(subject, previous_assistant)
                or _is_structural_subtopic(subject)
                or (
                    bool(summary_topics)
                    and not recent_root_seen
                )
            )
        )
        if anchored_to_root:
            current_root = root_topics[0]
            if subject != current_root:
                detail = subject
            continue

        if has_reference_prefix:
            continue

        _append_unique(root_topics, subject)
        current_root = subject
        detail = ""
        recent_root_seen = True

    if len(root_topics) != 1:
        return ""
    topic = root_topics[0]
    if current_root == topic and detail and detail not in topic:
        return f"{topic}的{detail}"
    return topic


def deterministic_rewrite(
    current_question: str,
    recent_turns: list[ConversationTurn],
    max_chars: int = MAX_STANDALONE_QUERY_CHARS,
    summary: str = "",
) -> tuple[str, str]:
    question = normalize_message_text(current_question)
    if not re.search(CONTEXT_REFERENCE_PATTERN, question):
        return question[:max_chars], "unchanged"

    topic = _extract_recent_user_topic(recent_turns, summary)
    if not topic:
        return question[:max_chars], "unresolved"

    qi_match = re.search(QI_PRONOUN_PATTERN, question)
    if question.startswith("其中"):
        remainder = question[len("其中") :]
        rewritten = f"{topic}中的{remainder}"
    elif question.startswith("它"):
        rewritten = topic + question[len("它") :]
    elif qi_match:
        suffix = question[qi_match.end() :]
        connector = (
            ""
            if re.match(QI_DIRECT_SUFFIX_PATTERN, suffix)
            else "的"
        )
        rewritten = (
            question[: qi_match.start()]
            + topic
            + connector
            + suffix
        )
    else:
        rewritten = re.sub(
            CONTEXT_REFERENCE_PATTERN,
            topic,
            question,
            count=1,
        )

    return normalize_standalone_query(rewritten, max_chars), "fallback"
