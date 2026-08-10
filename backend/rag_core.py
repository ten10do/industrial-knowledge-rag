import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__:
    from .ingestion import PageText, ingest_pages
    from .learning_content import generate_hierarchical_learning_content
    from .llm_client import generate_llm_answer
else:
    from ingestion import PageText, ingest_pages
    from learning_content import generate_hierarchical_learning_content
    from llm_client import generate_llm_answer


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "vector_db"
DATA_DIR = BASE_DIR / "data"

REFUSAL_MESSAGE = "知识库证据不足，无法根据已上传的工业知识资料回答该问题。"
EMPTY_KNOWLEDGE_BASE_MESSAGE = "请先上传 PDF 并构建知识库。"

# Chroma 在当前 Embedding 配置下返回原始距离值，数值越小表示越相关。
MAX_RELEVANT_DISTANCE = 20.0
DEFAULT_MAX_KNOWLEDGE_BASE_CHUNKS = 240


def load_pdf(file_path: str | os.PathLike):
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    source_name = Path(file_path).name
    for page_number, doc in enumerate(documents):
        doc.metadata["source"] = source_name
        doc.metadata.setdefault("page", page_number)
        doc.metadata["_file_path"] = str(file_path)

    return documents


def split_documents(documents):
    if not documents:
        raise ValueError(
            "PDF 没有读取到有效文字内容。请使用文字版 PDF，不要使用扫描版 PDF。"
        )

    first_metadata = getattr(documents[0], "metadata", {}) or {}
    file_path = first_metadata.get("_file_path") or first_metadata.get(
        "source",
        "document.pdf",
    )
    pages = [
        PageText(
            int((getattr(document, "metadata", {}) or {}).get("page", index)),
            document.page_content,
        )
        for index, document in enumerate(documents)
        if document.page_content and document.page_content.strip()
    ]

    def recursive_fallback(text: str, chunk_size: int, overlap: int):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return splitter.split_text(text)

    industrial_chunks = ingest_pages(
        file_path,
        pages,
        fallback_splitter=recursive_fallback,
    )
    try:
        from langchain_core.documents import Document
    except ImportError:
        document_type = type(documents[0])
        chunks = [
            document_type(
                page_content=chunk.page_content,
                metadata=chunk.metadata,
            )
            for chunk in industrial_chunks
        ]
    else:
        chunks = [
            Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata,
            )
            for chunk in industrial_chunks
        ]

    if not chunks:
        raise ValueError("PDF 切分后没有得到有效文本块。")

    return chunks


def get_data_dir(knowledge_base_id: str = "default") -> Path:
    return DATA_DIR if knowledge_base_id == "default" else DATA_DIR / knowledge_base_id


def get_persist_dir(knowledge_base_id: str = "default") -> Path:
    return (
        PERSIST_DIR
        if knowledge_base_id == "default"
        else PERSIST_DIR / knowledge_base_id
    )


def get_index_storage_path(knowledge_base_id: str = "default") -> Path:
    return get_persist_dir(knowledge_base_id)


def reload_knowledge_base(knowledge_base_id: str = "default") -> bool:
    if not is_knowledge_base_ready(knowledge_base_id):
        return False
    try:
        vector_db = load_vector_db(knowledge_base_id)
        vector_db.get(limit=1)
        del vector_db
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def build_vector_db(chunks, persist_dir: Path | None = None):
    if not chunks:
        raise ValueError("chunks 为空，无法建立向量数据库。")

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=str(persist_dir or PERSIST_DIR)
    )


def _max_knowledge_base_chunks() -> int:
    try:
        value = int(
            os.getenv(
                "MAX_KNOWLEDGE_BASE_CHUNKS",
                str(DEFAULT_MAX_KNOWLEDGE_BASE_CHUNKS),
            )
        )
    except ValueError:
        return DEFAULT_MAX_KNOWLEDGE_BASE_CHUNKS
    return value if value > 0 else DEFAULT_MAX_KNOWLEDGE_BASE_CHUNKS


def load_vector_db(knowledge_base_id: str = "default"):
    persist_dir = get_persist_dir(knowledge_base_id)
    if not persist_dir.exists():
        raise ValueError(EMPTY_KNOWLEDGE_BASE_MESSAGE)

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embedding_model()
    )


def is_knowledge_base_ready(knowledge_base_id: str = "default"):
    persist_dir = get_persist_dir(knowledge_base_id)
    return persist_dir.exists() and any(persist_dir.iterdir())


def _serialize_chunks(chunks) -> list[dict]:
    return [
        {
            "page_content": chunk.page_content,
            "metadata": dict(chunk.metadata),
        }
        for chunk in chunks
    ]


def _deserialize_chunks(payload):
    from langchain_core.documents import Document

    return [
        Document(
            page_content=item["page_content"],
            metadata=dict(item["metadata"]),
        )
        for item in payload
        if item.get("page_content")
    ]


def build_knowledge_base_incremental(
    pdf_paths,
    knowledge_base_id: str = "default",
    *,
    previous_index_path=None,
    previous_cache: dict | None = None,
):
    if isinstance(pdf_paths, (str, os.PathLike)):
        pdf_paths = [pdf_paths]
    else:
        pdf_paths = list(pdf_paths)

    if not pdf_paths:
        raise ValueError("请先上传 PDF 文件。")

    all_chunks = []
    files_cache = {}
    previous_files = (
        previous_cache.get("files", {})
        if previous_cache
        and previous_cache.get("schema_version") == 2
        and previous_cache.get("rag_mode") == "full"
        else {}
    )
    reused_file_count = 0
    changed_files = set()

    for value in pdf_paths:
        path = Path(value)
        digest_source = (
            path.read_bytes()
            if path.exists()
            else str(path).encode("utf-8")
        )
        digest = hashlib.sha256(digest_source).hexdigest()
        cached = previous_files.get(path.name, {})
        if (
            cached.get("sha256") == digest
            and cached.get("documents")
            and cached.get("ids")
        ):
            chunks = _deserialize_chunks(cached["documents"])
            ids = list(cached["ids"])
            reused_file_count += 1
        else:
            documents = load_pdf(path)
            if not documents:
                raise ValueError(
                    f"没有从 {path.name} 中读取到有效文字。"
                    "请使用文字版 PDF。"
                )
            chunks = split_documents(documents)
            ids = [
                str(
                    chunk.metadata.get("chunk_id")
                    or hashlib.sha256(
                        f"{path.name}:{digest}:{index}".encode("utf-8")
                    ).hexdigest()
                )
                for index, chunk in enumerate(chunks)
            ]
            changed_files.add(path.name)
        all_chunks.extend(chunks)
        files_cache[path.name] = {
            "sha256": digest,
            "ids": ids,
            "documents": _serialize_chunks(chunks),
        }

    if len(all_chunks) > _max_knowledge_base_chunks():
        raise ValueError(
            "工业知识资料切分后的文本块数量超过限制 "
            f"{_max_knowledge_base_chunks()}。"
        )

    persist_dir = get_persist_dir(knowledge_base_id)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = persist_dir.with_name(
        f".{persist_dir.name}.staging-{uuid4().hex}"
    )
    backup_dir = persist_dir.with_name(
        f".{persist_dir.name}.backup-{uuid4().hex}"
    )
    try:
        can_update = (
            previous_index_path
            and Path(previous_index_path).exists()
            and previous_files
        )
        if can_update:
            shutil.copytree(previous_index_path, staging_dir)
            vector_db = Chroma(
                persist_directory=str(staging_dir),
                embedding_function=get_embedding_model(),
            )
            removed_or_changed = (
                set(previous_files) - set(files_cache)
            ) | changed_files
            removed_ids = [
                item_id
                for filename in removed_or_changed
                for item_id in previous_files.get(filename, {}).get(
                    "ids",
                    [],
                )
            ]
            if removed_ids:
                vector_db.delete(ids=removed_ids)
            changed_chunks = []
            changed_ids = []
            for filename in changed_files:
                changed_chunks.extend(
                    _deserialize_chunks(
                        files_cache[filename]["documents"]
                    )
                )
                changed_ids.extend(files_cache[filename]["ids"])
            if changed_chunks:
                vector_db.add_documents(
                    documents=changed_chunks,
                    ids=changed_ids,
                )
        else:
            all_ids = [
                item_id
                for filename in files_cache
                for item_id in files_cache[filename]["ids"]
            ]
            vector_db = Chroma.from_documents(
                documents=all_chunks,
                ids=all_ids,
                embedding=get_embedding_model(),
                persist_directory=str(staging_dir),
            )
        del vector_db
        if persist_dir.exists():
            os.replace(persist_dir, backup_dir)
        os.replace(staging_dir, persist_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if persist_dir.exists() and backup_dir.exists():
            shutil.rmtree(persist_dir)
        if backup_dir.exists():
            os.replace(backup_dir, persist_dir)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    page_count = len(
        {
            (
                chunk.metadata.get("source"),
                page_number,
            )
            for chunk in all_chunks
            for page_number in range(
                int(chunk.metadata.get("page_start", chunk.metadata.get("page", 0))),
                int(chunk.metadata.get("page_end", chunk.metadata.get("page", 0))) + 1,
            )
        }
    )
    cache = {
        "schema_version": 2,
        "rag_mode": "full",
        "files": files_cache,
    }
    stats = {
        "reused_file_count": reused_file_count,
        "parsed_file_count": len(pdf_paths) - reused_file_count,
        "updated_vector_file_count": len(changed_files),
        "removed_vector_file_count": len(
            set(previous_files) - set(files_cache)
        ),
    }
    return page_count, len(all_chunks), cache, stats


def build_knowledge_base(pdf_paths, knowledge_base_id: str = "default"):
    page_count, chunk_count, _, _ = build_knowledge_base_incremental(
        pdf_paths,
        knowledge_base_id,
    )
    return page_count, chunk_count


def retrieve_docs(question: str, k: int = 4, knowledge_base_id: str = "default"):
    if not question or not question.strip():
        raise ValueError("问题不能为空。")

    vector_db = load_vector_db(knowledge_base_id)
    return vector_db.similarity_search_with_score(question, k=k)


def has_relevant_docs(scored_docs):
    if not scored_docs:
        return False

    return bool(filter_relevant_docs(scored_docs))


def get_relevance_threshold() -> float:
    try:
        return float(
            os.getenv(
                "FULL_MAX_RELEVANT_DISTANCE",
                str(MAX_RELEVANT_DISTANCE),
            )
        )
    except ValueError:
        return MAX_RELEVANT_DISTANCE


def filter_relevant_docs(scored_docs):
    threshold = get_relevance_threshold()
    return [
        (document, score)
        for document, score in scored_docs
        if score <= threshold
    ]


def get_all_docs(knowledge_base_id: str = "default"):
    vector_db = load_vector_db(knowledge_base_id)
    result = vector_db.get(
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    representative_docs = []

    for index, content in enumerate(documents):
        if not content or not content.strip():
            continue

        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        doc = SimpleNamespace(page_content=content, metadata=metadata)
        representative_docs.append((doc, 0.0))

    return representative_docs


def get_representative_docs(k: int = 8, knowledge_base_id: str = "default"):
    return get_all_docs(knowledge_base_id)[:k]


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
要求：每个知识点给出简短解释，并尽量按资料主题分类；不要编造设备参数、故障码或操作步骤。
""",
        "review_questions": """
请基于当前工业知识资料生成 5 个文档核对问题，并给出参考答案。
每个问题和答案都必须可由资料支撑；不要生成与资料无关的内容。
"""
    }

    if task_type not in prompts:
        raise ValueError("不支持的学习辅助功能。")

    return prompts[task_type]


def generate_learning_content(
    task_type: str,
    provider: str = "Groq",
    knowledge_base_id: str = "default",
):
    docs = get_all_docs(knowledge_base_id)
    if not docs:
        raise ValueError(EMPTY_KNOWLEDGE_BASE_MESSAGE)

    return generate_hierarchical_learning_content(
        get_learning_task_prompt(task_type),
        docs,
        provider,
        generate_llm_answer,
    )


def generate_answer(
    question: str,
    docs,
    provider: str = "Groq",
    conversation_summary: str | None = None,
    conversation_history: list[dict] | None = None,
):
    if not question or not question.strip():
        raise ValueError("问题不能为空。")

    if not docs:
        raise ValueError("没有检索到参考资料，无法生成回答。")

    if conversation_summary or conversation_history:
        return generate_llm_answer(
            question,
            docs,
            provider=provider,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        )
    return generate_llm_answer(question, docs, provider=provider)


def clear_vector_db(knowledge_base_id: str = "default"):
    persist_dir = get_persist_dir(knowledge_base_id)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def clear_data_dir(knowledge_base_id: str = "default"):
    data_dir = get_data_dir(knowledge_base_id)
    if data_dir.exists():
        shutil.rmtree(data_dir)


def clear_knowledge_base(knowledge_base_id: str = "default"):
    clear_vector_db(knowledge_base_id)
    clear_data_dir(knowledge_base_id)
