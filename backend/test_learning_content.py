from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.learning_content import generate_hierarchical_learning_content
from backend.llm_client import build_prompt


def document(content: str, source: str, page: int):
    return (
        SimpleNamespace(
            page_content=content,
            metadata={"source": source, "page": page},
        ),
        0.1,
    )


def test_answer_prompt_exposes_verifiable_source_ids():
    prompt = build_prompt(
        "PLC 的扫描周期是什么？",
        [document("输入采样、程序执行、输出刷新。", "plc.pdf", 2)],
    )

    assert "[S1] 来源：plc.pdf，第 3 页" in prompt
    assert "只能引用上面实际存在的编号" in prompt


def test_hierarchical_learning_content_covers_every_batch():
    docs = [
        document("第一章反馈控制", "one.pdf", 0),
        document("第二章PID控制", "two.pdf", 0),
        document("第三章状态空间", "three.pdf", 0),
    ]
    generator = Mock(side_effect=["批次一", "批次二", "批次三", "整课总结"])

    with patch.dict(
        "os.environ",
        {
            "LEARNING_BATCH_CHARS": "8",
            "LEARNING_MAX_BATCHES": "8",
        },
    ):
        result = generate_hierarchical_learning_content(
            "生成课程总结",
            docs,
            "Groq",
            generator,
        )

    assert result == "整课总结"
    assert generator.call_count == 4
    covered_sources = [
        call.args[1][0][0].metadata["source"]
        for call in generator.call_args_list[:3]
    ]
    assert covered_sources == ["one.pdf", "two.pdf", "three.pdf"]
    final_docs = generator.call_args_list[-1].args[1]
    assert [doc.page_content for doc, _ in final_docs] == [
        "批次一",
        "批次二",
        "批次三",
    ]
