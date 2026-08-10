import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from backend import light_rag_core, llm_client
from backend.model_governance import reset_model_scope, set_model_scope


def test_answer_prompt_separates_history_from_current_retrieval_evidence():
    docs = [
        (
            SimpleNamespace(
                page_content="当前检索证据：积分作用可以消除稳态误差。"
            ),
            0.1,
        )
    ]

    prompt = llm_client.build_prompt(
        "其中积分项有什么作用？",
        docs,
        conversation_summary="较早对话讨论 PID。",
        conversation_history=[
            {"role": "assistant", "content": "历史回答只用于理解指代。"}
        ],
    )

    assert "只用于理解当前问题，不是知识来源" in prompt
    assert "只用于理解指代，不是知识来源" in prompt
    assert "当前检索证据：积分作用可以消除稳态误差。" in prompt
    assert "当前用户问题：\n其中积分项有什么作用？" in prompt
    assert "历史助手回答不能替代本轮参考资料" in prompt


@pytest.mark.parametrize(
    ("provider", "function_name"),
    [
        ("Groq", "generate_with_groq"),
        ("DeepSeek", "generate_with_deepseek"),
    ],
)
def test_both_model_providers_keep_multiturn_arguments_compatible(
    provider,
    function_name,
):
    docs = [(SimpleNamespace(page_content="检索证据"), 0.1)]
    with patch.object(
        llm_client,
        function_name,
        return_value="模型回答",
    ) as provider_call:
        answer = llm_client.generate_llm_answer(
            "当前问题",
            docs,
            provider=provider,
            conversation_summary="摘要",
            conversation_history=[
                {"role": "user", "content": "上一轮问题"}
            ],
        )

    assert answer == "模型回答\n\n参考依据：[S1]"
    provider_call.assert_called_once_with(
        "当前问题",
        docs,
        conversation_summary="摘要",
        conversation_history=[
            {"role": "user", "content": "上一轮问题"}
        ],
    )


def test_invalid_model_citations_are_removed_and_replaced_with_valid_ids():
    docs = [
        (SimpleNamespace(page_content="证据一"), 0.1),
        (SimpleNamespace(page_content="证据二"), 0.2),
    ]

    result = llm_client.ensure_valid_citations("回答 [S9]", docs)

    assert "[S9]" not in result
    assert result.endswith("参考依据：[S1] [S2]")


def test_model_client_timeout_and_retry_options_are_configurable():
    with patch.dict(
        "os.environ",
        {
            "MODEL_REQUEST_TIMEOUT_SECONDS": "12.5",
            "MODEL_MAX_RETRIES": "2",
        },
    ):
        options = llm_client.get_model_client_options()

    assert options == {"timeout": 12.5, "max_retries": 2}


def test_provider_usage_reconciles_token_quota_and_updates_scope_state():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="模型回答")
            )
        ],
        usage=SimpleNamespace(total_tokens=12),
    )
    create = Mock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    fake_groq = SimpleNamespace(Groq=Mock(return_value=client))
    governor = Mock()
    reservation = SimpleNamespace(reserved_tokens=300)
    decision = SimpleNamespace(
        limit=1000,
        remaining=700,
        reset_after=60,
    )
    governor.begin.return_value = (reservation, decision)
    scope_token, state = set_model_scope("client:kb")

    try:
        with patch.dict(sys.modules, {"groq": fake_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
                with patch.object(
                    llm_client,
                    "model_governor",
                    governor,
                ):
                    result = llm_client.create_governed_completion(
                        "Groq",
                        [{"role": "user", "content": "问题"}],
                        temperature=0.2,
                    )
    finally:
        reset_model_scope(scope_token)

    assert result == "模型回答"
    assert state["used_tokens"] == 12
    assert state["quota"]["remaining"] == 988
    governor.finish.assert_called_once_with(
        reservation,
        actual_tokens=12,
        succeeded=True,
    )


def test_light_mode_forwards_bounded_context_to_the_shared_llm_client():
    docs = [(SimpleNamespace(page_content="检索证据"), 0.1)]
    with patch.object(
        light_rag_core,
        "generate_llm_answer",
        return_value="回答",
    ) as generate:
        result = light_rag_core.generate_answer(
            "当前问题",
            docs,
            provider="Groq",
            conversation_summary="摘要",
            conversation_history=[
                {"role": "assistant", "content": "最近回答"}
            ],
        )

    assert result == "回答"
    generate.assert_called_once_with(
        "当前问题",
        docs,
        provider="Groq",
        conversation_summary="摘要",
        conversation_history=[
            {"role": "assistant", "content": "最近回答"}
        ],
    )
