import os
import shutil
from types import SimpleNamespace

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_client import generate_llm_answer


PERSIST_DIR = "vector_db"
DATA_DIR = "data"
REFUSAL_MESSAGE = "知识库证据不足，无法根据已上传的工业知识资料回答该问题。"
EMPTY_KNOWLEDGE_BASE_MESSAGE = "请先上传 PDF 并构建知识库。"

# Chroma 的 similarity_search_with_score 在当前配置下返回的是原始距离值，
# 不是 0~1 的相似度；距离越小表示越相关。当前项目使用未显式归一化的
# HuggingFace Embeddings，结合现有知识库抽样，相关问题通常低于 20，
# 明显无关问题会高于该值，因此用 20.0 作为拒答阈值。
MAX_RELEVANT_DISTANCE = 20.0

load_dotenv()


def load_pdf(file_path: str):
    """
    读取 PDF 文件，并过滤空白页面。
    V3.73: attach document-level identity metadata from filename.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    source_name = os.path.basename(file_path)
    for page_number, doc in enumerate(documents):
        doc.metadata["source"] = source_name
        doc.metadata.setdefault("page", page_number)

    # V3.73: enrich with document-level identity (query-agnostic).
    try:
        from backend.retrieval.document_identity_v373 import (
            resolve_document_identity,
        )
        identity = resolve_document_identity(file_path)
        for doc in documents:
            for key in ("manufacturer", "product_family", "product_series",
                        "equipment_model", "identity_source"):
                if identity.get(key):
                    doc.metadata.setdefault(key, identity[key])
    except ImportError:
        pass  # Resolver not available; keep existing behaviour.

    return documents


def split_documents(documents):
    """
    把 PDF 文档切分成多个小文本块。
    """
    if not documents:
        raise ValueError(
            "PDF 没有读取到有效文字内容。请使用文字版 PDF，不要使用扫描版 PDF。"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "PDF 已读取，但切分后没有得到有效文本块。请换一个文字版 PDF 测试。"
        )

    return chunks


def get_embedding_model():
    """
    加载本地 Embedding 模型。
    第一次运行可能会下载模型。
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embeddings


def build_vector_db(chunks):
    """
    将文本块转换成向量，并保存到 Chroma 向量数据库。
    """
    if not chunks:
        raise ValueError("chunks 为空，无法建立向量数据库。")

    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vector_db


def load_vector_db():
    """
    加载已经建立好的 Chroma 向量数据库。
    """
    if not os.path.exists(PERSIST_DIR):
        raise ValueError("还没有建立知识库，请先上传 PDF 并点击“建立知识库”。")

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vector_db


def build_knowledge_base(pdf_path: str):
    """
    从一个或多个 PDF 文件一键建立知识库。

    流程：
    1. 逐个读取 PDF
    2. 分别切分文本
    3. 清空旧向量库，避免新旧文档混在一起
    4. 本地 Embedding 向量化
    5. 统一存入同一个 Chroma 向量库
    """
    pdf_paths = [pdf_path] if isinstance(pdf_path, str) else list(pdf_path)

    if not pdf_paths:
        raise ValueError("请先上传 PDF 文件。")

    all_documents = []
    all_chunks = []

    for path in pdf_paths:
        documents = load_pdf(path)

        if not documents:
            raise ValueError(
                f"没有从 {os.path.basename(path)} 中读取到有效文字。请使用文字版 PDF，不要使用扫描版 PDF。"
            )

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(f"{os.path.basename(path)} 文本切分失败，没有生成有效文本块。")

        all_documents.extend(documents)
        all_chunks.extend(chunks)

    if not all_documents:
        raise ValueError(
            "没有从 PDF 中读取到有效文字。请使用文字版 PDF，不要使用扫描版 PDF。"
        )

    if not all_chunks:
        raise ValueError("文本切分失败，没有生成有效文本块。")

    clear_vector_db()
    build_vector_db(all_chunks)

    return len(all_documents), len(all_chunks)


def retrieve_docs(question: str, k: int = 4):
    """
    根据用户问题，从向量数据库中检索最相关的文本块，并返回距离分数。
    """
    if not question or not question.strip():
        raise ValueError("问题不能为空。")

    vector_db = load_vector_db()

    docs = vector_db.similarity_search_with_score(
        question,
        k=k
    )

    return docs


def has_relevant_docs(scored_docs):
    """
    判断最相关片段是否达到相关性阈值。

    scored_docs 的元素为 (Document, score)。Chroma 返回的 score 是距离值，
    分数越小越相关，所以只有最小距离不超过 MAX_RELEVANT_DISTANCE 时才回答。
    """
    if not scored_docs:
        return False

    best_score = scored_docs[0][1]
    return best_score <= MAX_RELEVANT_DISTANCE


def get_representative_docs(k: int = 8):
    """
    从当前知识库中取少量代表性 chunk，供文档摘要、关键知识提取和核对问题生成使用。

    这里直接读取 Chroma 中靠前的若干条文本，避免一次性把整个知识库塞入 prompt。
    返回结构保持为 (doc, score)，便于复用 llm_client 中已有的上下文拼接逻辑。
    """
    vector_db = load_vector_db()
    result = vector_db.get(
        include=["documents", "metadatas"],
        limit=k
    )

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    representative_docs = []

    for index, content in enumerate(documents):
        if not content or not content.strip():
            continue

        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        doc = SimpleNamespace(
            page_content=content,
            metadata=metadata
        )
        representative_docs.append((doc, 0.0))

    return representative_docs


def get_learning_task_prompt(task_type: str):
    prompts = {
        "summary": """
请基于当前工业知识资料生成文档摘要，必须包括：
1）资料主要内容
2）核心章节或主题
3）重点概念
4）适用范围或注意事项（仅当资料中有明确依据时）

要求：只依据参考资料总结，不要脱离资料自由发挥。
""",
        "knowledge_points": """
请基于当前工业知识资料提取关键知识点。

要求：
1. 每个知识点后面给出简短解释
2. 尽量按资料主题分类
3. 只依据参考资料提取；不要编造设备参数、故障码或操作步骤
""",
        "review_questions": """
请基于当前工业知识资料生成核对问题。

要求：
1. 生成 5 个问题，每个问题都要给出参考答案
2. 不要生成与资料无关的题目
3. 只依据参考资料出题，不要脱离资料自由发挥
"""
    }

    if task_type not in prompts:
        raise ValueError("不支持的学习辅助功能。")

    return prompts[task_type]


def generate_learning_content(task_type: str, provider: str = "Groq"):
    docs = get_representative_docs()

    if not docs:
        raise ValueError(EMPTY_KNOWLEDGE_BASE_MESSAGE)

    return generate_llm_answer(
        get_learning_task_prompt(task_type),
        docs,
        provider=provider
    )


def generate_answer(question: str, docs, provider: str = "Groq"):
    """
    调用所选大模型 API，根据检索到的参考片段生成中文回答。
    """
    if not question or not question.strip():
        raise ValueError("问题不能为空。")

    if not docs:
        raise ValueError("没有检索到参考资料，无法生成回答。")

    return generate_llm_answer(question, docs, provider=provider)


def clear_vector_db():
    """
    清空本地向量库。
    """
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)


def clear_data_dir():
    """
    清空本地上传的 PDF 文件。
    """
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)


def clear_knowledge_base():
    """
    清空本地知识库和已上传的 PDF 文件。
    """
    clear_vector_db()
    clear_data_dir()
