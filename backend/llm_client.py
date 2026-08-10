import os
import re
from pathlib import Path

from dotenv import load_dotenv

if __package__:
    from .model_governance import (
        create_model_governor,
        current_model_scope,
    )
else:
    from model_governance import (
        create_model_governor,
        current_model_scope,
    )


GROQ_MODEL = "llama-3.1-8b-instant"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MODEL_MAX_RETRIES = 1
DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 2048

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
model_governor = create_model_governor()


def get_model_client_options() -> dict:
    try:
        timeout = float(
            os.getenv(
                "MODEL_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS),
            )
        )
    except ValueError:
        timeout = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    try:
        max_retries = int(
            os.getenv(
                "MODEL_MAX_RETRIES",
                str(DEFAULT_MODEL_MAX_RETRIES),
            )
        )
    except ValueError:
        max_retries = DEFAULT_MODEL_MAX_RETRIES
    return {
        "timeout": timeout if timeout > 0 else DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
        "max_retries": max_retries if max_retries >= 0 else DEFAULT_MODEL_MAX_RETRIES,
    }


def get_model_max_output_tokens() -> int:
    try:
        value = int(
            os.getenv(
                "MODEL_MAX_OUTPUT_TOKENS",
                str(DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
            )
        )
    except ValueError:
        value = DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    return value if value > 0 else DEFAULT_MODEL_MAX_OUTPUT_TOKENS


def estimate_token_reservation(messages: list[dict]) -> int:
    input_chars = sum(
        len(str(message.get("content", "")))
        for message in messages
    )
    return input_chars + get_model_max_output_tokens()


def response_total_tokens(response) -> int | None:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        return None
    try:
        return max(0, int(total_tokens))
    except (TypeError, ValueError):
        return None


def create_governed_completion(
    provider: str,
    messages: list[dict],
    *,
    temperature: float,
):
    if provider == "Groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("没有找到 GROQ_API_KEY，请检查运行环境。")
        client = Groq(api_key=api_key, **get_model_client_options())
        model = GROQ_MODEL
    elif provider == "DeepSeek":
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查运行环境。")
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            **get_model_client_options(),
        )
        model = DEEPSEEK_MODEL
    else:
        raise ValueError("不支持的大模型服务，请选择 Groq 或 DeepSeek。")

    scope, state = current_model_scope()
    reservation, decision = model_governor.begin(
        scope,
        estimate_token_reservation(messages),
    )
    if decision is not None:
        state["quota"] = {
            "limit": decision.limit,
            "remaining": decision.remaining,
            "reset_after": decision.reset_after,
        }
    succeeded = False
    actual_tokens = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=get_model_max_output_tokens(),
        )
        actual_tokens = response_total_tokens(response)
        if actual_tokens is not None:
            state["used_tokens"] += actual_tokens
        succeeded = True
        return response.choices[0].message.content
    finally:
        model_governor.finish(
            reservation,
            actual_tokens=actual_tokens,
            succeeded=succeeded,
        )
        if (
            succeeded
            and decision is not None
            and actual_tokens is not None
            and actual_tokens < reservation.reserved_tokens
        ):
            state["quota"]["remaining"] = min(
                decision.limit,
                state["quota"]["remaining"]
                + reservation.reserved_tokens
                - actual_tokens,
            )


def build_prompt(
    question: str,
    docs,
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    context_parts = []
    for i, (doc, score) in enumerate(docs):
        metadata = getattr(doc, "metadata", {}) or {}
        source = Path(str(metadata.get("source", "未知来源"))).name
        page = metadata.get("page")
        page_label = f"，第 {page + 1} 页" if isinstance(page, int) else ""
        context_parts.append(
            f"[S{i + 1}] 来源：{source}{page_label}\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    conversation_parts = []
    if conversation_summary:
        conversation_parts.append(
            "较早对话摘要（只用于理解当前问题，不是知识来源）：\n"
            + conversation_summary
        )
    if conversation_history:
        labels = {"user": "用户", "assistant": "助手"}
        history_text = "\n".join(
            f"{labels.get(item.get('role'), '消息')}：{item.get('content', '')}"
            for item in conversation_history
        )
        conversation_parts.append(
            "最近对话（只用于理解指代，不是知识来源）：\n"
            + history_text
        )
    conversation_context = (
        "\n\n".join(conversation_parts)
        if conversation_parts
        else "无"
    )

    return f"""
你是一个工业知识智能助手。

请严格根据下面的参考资料回答用户问题。
如果参考资料中没有相关信息，请明确回答：“知识库证据不足，无法确认”。
不得编造设备参数、故障码或操作步骤；区分资料事实与基于资料的推测。

对话上下文：
{conversation_context}

参考资料：
{context}

当前用户问题：
{question}

回答要求：
1. 用中文回答
2. 解释要适合自动化专业本科生理解
3. 不要编造参考资料中没有的信息
4. 如果涉及自动控制、PLC、传感器、电机等内容，请尽量结合自动化专业背景说明
5. 回答尽量条理清晰
6. 历史助手回答不能替代本轮参考资料
7. 对资料中的事实使用 [S1]、[S2] 形式标注依据，只能引用上面实际存在的编号
"""


def build_messages(
    question: str,
    docs,
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    return [
        {
            "role": "system",
            "content": "你是严谨的工业知识助手，只能依据给定资料回答；不得编造设备参数、故障码或操作步骤，并应区分资料事实与推测。"
        },
        {
            "role": "user",
            "content": build_prompt(
                question,
                docs,
                conversation_summary=conversation_summary,
                conversation_history=conversation_history,
            )
        }
    ]


def ensure_valid_citations(answer: str, docs) -> str:
    normalized = (answer or "").strip()
    valid_ids = {str(index) for index in range(1, len(docs) + 1)}
    valid_citations = {
        citation
        for citation in CITATION_PATTERN.findall(normalized)
        if citation in valid_ids
    }
    normalized = CITATION_PATTERN.sub(
        lambda match: match.group(0) if match.group(1) in valid_ids else "",
        normalized,
    ).strip()
    if valid_ids and not valid_citations:
        fallback_ids = " ".join(
            f"[S{index}]" for index in range(1, min(len(docs), 3) + 1)
        )
        normalized = f"{normalized}\n\n参考依据：{fallback_ids}".strip()
    return normalized


def generate_with_groq(
    question: str,
    docs,
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    return create_governed_completion(
        "Groq",
        build_messages(
            question,
            docs,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        ),
        temperature=0.2,
    )


def generate_with_deepseek(
    question: str,
    docs,
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    return create_governed_completion(
        "DeepSeek",
        build_messages(
            question,
            docs,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        ),
        temperature=0.2,
    )


def generate_llm_answer(
    question: str,
    docs,
    provider: str = "Groq",
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    if provider == "Groq":
        answer = generate_with_groq(
            question,
            docs,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        )
        return ensure_valid_citations(answer, docs)

    if provider == "DeepSeek":
        answer = generate_with_deepseek(
            question,
            docs,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        )
        return ensure_valid_citations(answer, docs)

    raise ValueError("不支持的大模型服务，请选择 Groq 或 DeepSeek。")


def generate_context_text(prompt: str, provider: str = "Groq"):
    messages = [
        {
            "role": "system",
            "content": (
                "你只负责压缩课程对话或改写检索问题。"
                "不得新增历史中不存在的事实，只输出任务要求的文本。"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return create_governed_completion(
        provider,
        messages,
        temperature=0.0,
    )
