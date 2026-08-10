import os

from dotenv import load_dotenv


GROQ_MODEL = "llama-3.1-8b-instant"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

load_dotenv()


def build_prompt(question: str, docs):
    context = "\n\n".join(
        [
            f"参考片段 {i + 1}：\n{doc.page_content}"
            for i, (doc, score) in enumerate(docs)
        ]
    )

    return f"""
你是一个工业知识智能助手。

请严格根据下面的参考资料回答用户问题。
如果参考资料中没有相关信息，请回答：“知识库证据不足，无法确认”。

参考资料：
{context}

用户问题：
{question}

回答要求：
1. 用中文回答
2. 使用清晰、准确的工程语言
3. 不要编造参考资料中没有的信息
4. 不得编造设备参数、故障码或操作步骤；区分资料事实与推测
5. 尽量给出可追溯的参考片段编号
6. 回答尽量条理清晰
"""


def build_messages(question: str, docs):
    return [
        {
            "role": "system",
            "content": "你是严谨的工业知识助手，只能根据给定资料回答；不得编造设备参数、故障码或操作步骤，并应区分资料事实与推测。"
        },
        {
            "role": "user",
            "content": build_prompt(question, docs)
        }
    ]


def generate_with_groq(question: str, docs):
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("没有找到 GROQ_API_KEY，请检查 .env 文件。")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=build_messages(question, docs),
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_with_deepseek(question: str, docs):
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=build_messages(question, docs),
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_llm_answer(question: str, docs, provider: str = "Groq"):
    if provider == "Groq":
        return generate_with_groq(question, docs)

    if provider == "DeepSeek":
        return generate_with_deepseek(question, docs)

    raise ValueError("不支持的大模型服务，请选择 Groq 或 DeepSeek。")
