import math
import os
from types import SimpleNamespace


DEFAULT_BATCH_CHARS = 24000
DEFAULT_MAX_BATCHES = 8


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def batch_documents(scored_docs):
    if not scored_docs:
        return []

    max_batch_chars = _positive_int_env(
        "LEARNING_BATCH_CHARS",
        DEFAULT_BATCH_CHARS,
    )
    max_batches = _positive_int_env(
        "LEARNING_MAX_BATCHES",
        DEFAULT_MAX_BATCHES,
    )
    total_chars = sum(len(doc.page_content) for doc, _ in scored_docs)
    if total_chars > max_batch_chars * max_batches:
        raise ValueError(
            "工业知识资料过多，无法在当前知识辅助上下文预算内完整处理。"
        )

    batch_count = max(1, math.ceil(total_chars / max_batch_chars))
    target_chars = max(1, math.ceil(total_chars / batch_count))
    batches = []
    current = []
    current_chars = 0

    for item in scored_docs:
        item_chars = len(item[0].page_content)
        if current and current_chars + item_chars > target_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


def generate_hierarchical_learning_content(
    task_prompt: str,
    scored_docs,
    provider: str,
    generator,
):
    batches = batch_documents(scored_docs)
    if not batches:
        raise ValueError("知识库中没有可用于学习辅助的资料。")
    if len(batches) == 1:
        return generator(task_prompt, batches[0], provider=provider)

    partials = []
    for index, batch in enumerate(batches, start=1):
        partial_prompt = (
            f"{task_prompt}\n\n"
            f"当前是全部工业知识资料的第 {index}/{len(batches)} 批。"
            "请生成忠于本批资料的结构化中间提要，保留关键概念、条件和结论，"
            "不要声称这是全部资料的最终结果。"
        )
        partial = generator(partial_prompt, batch, provider=provider)
        partials.append(
            (
                SimpleNamespace(
                    page_content=partial,
                    metadata={"source": f"工业资料批次 {index}", "page": index - 1},
                ),
                0.0,
            )
        )

    final_prompt = (
        f"{task_prompt}\n\n"
        "下面的参考片段是覆盖全部工业知识资料后生成的分批提要。"
        "请去重、合并跨主题关联，并生成最终结果。"
    )
    return generator(final_prompt, partials, provider=provider)
