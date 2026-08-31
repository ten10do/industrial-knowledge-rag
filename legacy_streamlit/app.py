import os

import streamlit as st

from rag_core import (
    build_knowledge_base,
    retrieve_docs,
    has_relevant_docs,
    generate_answer,
    generate_learning_content,
    clear_knowledge_base,
    REFUSAL_MESSAGE,
    EMPTY_KNOWLEDGE_BASE_MESSAGE
)


DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"

os.makedirs(DATA_DIR, exist_ok=True)


st.set_page_config(
    page_title="Industrial Knowledge RAG",
    page_icon="📚",
    layout="wide"
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 760;
        color: #172033;
        margin-bottom: 0.35rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5f6878;
        margin-bottom: 1.4rem;
    }
    .section-card {
        border: 1px solid #e6eaf0;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .answer-card {
        border: 1px solid #dfe6ef;
        border-left: 4px solid #4f6f9f;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        background: #f8fafc;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .muted-text {
        color: #687385;
        font-size: 0.92rem;
    }
    div[data-testid="stButton"] button {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_page_number(page):
    if isinstance(page, int):
        return page + 1
    return page


def show_reference_sources(docs):
    if not docs:
        st.warning("没有检索到相关内容。")
        return

    with st.expander("参考来源", expanded=False):
        for i, (doc, score) in enumerate(docs, start=1):
            page = get_page_number(doc.metadata.get("page", "未知页码"))
            source = doc.metadata.get("source", "未知来源")
            source_name = os.path.basename(source)

            st.markdown(
                f"**参考片段 {i}**  \n"
                f"来源文件：`{source_name}`  \n"
                f"页码：`{page}`  \n"
                f"距离分数：`{score:.4f}`（越小越相关）"
            )
            st.text_area(
                "参考内容",
                value=doc.page_content,
                height=120,
                key=f"reference_content_{i}",
                disabled=True
            )
            if i != len(docs):
                st.divider()


def show_learning_result(title, content):
    with st.expander(title, expanded=True):
        st.write(content)


st.markdown(
    '<div class="main-title">Industrial Knowledge RAG｜工业知识智能检索与问答平台</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">支持多 PDF 工业知识资料上传、RAG 问答、来源追溯、模型切换与知识辅助功能。</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("系统设置")

    selected_model = st.selectbox(
        "大模型服务",
        options=["Groq", "DeepSeek"]
    )

    st.divider()

    st.subheader("知识库管理")

    uploaded_files = st.file_uploader(
        "上传课程 PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    file_paths = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_paths.append(file_path)

        st.success(f"已选择 {len(file_paths)} 个 PDF")

    if st.button("构建 / 重新构建知识库", use_container_width=True):
        if not file_paths:
            st.warning("请先上传 PDF 文件。")
        else:
            try:
                with st.spinner("正在解析 PDF 并构建知识库..."):
                    page_count, chunk_count = build_knowledge_base(file_paths)

                st.success(f"知识库构建完成：{page_count} 页，{chunk_count} 个文本块。")

            except Exception as e:
                st.error("知识库构建失败。")
                st.code(str(e))
                st.warning("建议使用文字版 PDF，不要使用扫描版或图片版教材。")

    if st.button("清空知识库", use_container_width=True):
        clear_knowledge_base()
        st.success("知识库已清空，请重新上传工业知识资料。")

    st.divider()

    st.subheader("当前状态")
    pdf_count = len([name for name in os.listdir(DATA_DIR) if name.lower().endswith(".pdf")])
    if os.path.exists(VECTOR_DB_DIR):
        st.success("知识库：已构建")
    else:
        st.info("知识库：未构建")
    st.caption(f"已保存 PDF：{pdf_count} 个")

    st.divider()

    st.subheader("使用步骤")
    st.markdown(
        """
        1. 选择 Groq 或 DeepSeek。
        2. 上传一份或多份课程 PDF。
        3. 构建知识库。
        4. 在主页面提问或使用知识辅助功能。
        """
    )


st.header("1. 智能问答")
st.markdown(
    '<div class="muted-text">输入与工业知识资料相关的问题，系统会先检索知识库，再基于参考片段生成回答。</div>',
    unsafe_allow_html=True
)

with st.form("qa_form"):
    question = st.text_input(
        "问题",
        placeholder="例如：水塔水位控制系统的主要任务是什么？"
    )

    top_k = st.slider(
        "返回参考片段数量",
        min_value=1,
        max_value=8,
        value=4
    )

    ask_button = st.form_submit_button("生成回答")

if ask_button:
    if not question:
        st.warning("请先输入问题。")
    else:
        try:
            with st.spinner("正在从知识库中检索相关内容..."):
                docs = retrieve_docs(question, k=top_k)

            if has_relevant_docs(docs):
                with st.spinner(f"正在调用 {selected_model} 大模型生成回答..."):
                    answer = generate_answer(question, docs, provider=selected_model)

                st.subheader("AI 回答")
                st.info(answer)
            else:
                st.warning(REFUSAL_MESSAGE)

            show_reference_sources(docs)

        except Exception as e:
            st.error("生成回答失败。")
            st.write("错误原因：")
            st.code(str(e))
            st.warning("请确认已构建知识库、所选模型 API Key 已写入 .env，且网络正常。")


st.divider()

st.header("2. 知识辅助")
st.markdown(
    '<div class="muted-text">基于当前知识库中的代表性片段生成总结、知识点和复习题。</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

learning_tasks = [
    (col1, "生成文档摘要", "summary"),
    (col2, "提取关键知识", "knowledge_points"),
    (col3, "生成核对问题", "review_questions"),
]

for column, button_label, task_type in learning_tasks:
    with column:
        if st.button(button_label, use_container_width=True):
            try:
                with st.spinner(f"正在调用 {selected_model} 生成知识辅助内容..."):
                    result = generate_learning_content(task_type, provider=selected_model)

                show_learning_result(button_label, result)

            except ValueError as e:
                if str(e) == EMPTY_KNOWLEDGE_BASE_MESSAGE:
                    st.warning(EMPTY_KNOWLEDGE_BASE_MESSAGE)
                else:
                    st.error("知识辅助内容生成失败。")
                    st.code(str(e))

            except Exception as e:
                st.error("知识辅助内容生成失败。")
                st.write("错误原因：")
                st.code(str(e))
                st.warning("请确认已构建知识库、所选模型 API Key 已写入 .env，且网络正常。")
