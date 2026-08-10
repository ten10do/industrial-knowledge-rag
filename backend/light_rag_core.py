import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

if __package__:
    from .ingestion import PageText, ingest_pages
    from .learning_content import generate_hierarchical_learning_content
    from .llm_client import generate_llm_answer
    from .retrieval import BM25Index, RetrievalCandidate, RetrievalResult, analyze_query, analyze_retrieval_evidence, filter_documents, rrf_fuse
    from .retrieval.filters import has_exact_metadata_match
else:
    from ingestion import PageText, ingest_pages
    from learning_content import generate_hierarchical_learning_content
    from llm_client import generate_llm_answer
    from retrieval import BM25Index, RetrievalCandidate, RetrievalResult, analyze_query, analyze_retrieval_evidence, filter_documents, rrf_fuse
    from retrieval.filters import has_exact_metadata_match


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REFUSAL_MESSAGE = "知识库证据不足，无法根据已上传的工业知识资料回答该问题。"
EMPTY_KNOWLEDGE_BASE_MESSAGE = "请先上传 PDF 并构建知识库。"

# Light 模式将余弦相似度转换为距离，数值越小表示越相关。
MAX_RELEVANT_DISTANCE = 0.81
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_MAX_KNOWLEDGE_BASE_CHUNKS = 240
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_LEXICAL_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_HYBRID_TOP_K = 5
DEFAULT_RRF_K = 60


@dataclass
class LightDocument:
    page_content: str
    metadata: dict


@dataclass
class LightKnowledgeBase:
    documents: list
    vectorizer: object
    tfidf_matrix: object
    bm25: BM25Index


_knowledge_bases: dict[str, LightKnowledgeBase] = {}


def _normalized_knowledge_base_id(knowledge_base_id: str) -> str:
    return knowledge_base_id or "default"


def get_data_dir(knowledge_base_id: str = "default") -> Path:
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    return DATA_DIR if knowledge_base_id == "default" else DATA_DIR / knowledge_base_id


def _index_path(knowledge_base_id: str) -> Path:
    index_dir = DATA_DIR.parent / "light_indexes"
    return index_dir / f"{knowledge_base_id}.json"


def get_index_storage_path(knowledge_base_id: str = "default") -> Path:
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    return _index_path(knowledge_base_id)


def reload_knowledge_base(knowledge_base_id: str = "default") -> bool:
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    _knowledge_bases.pop(knowledge_base_id, None)
    return load_knowledge_base(knowledge_base_id)


def _fit_index(documents) -> LightKnowledgeBase:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=15000,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(
        [document.page_content for document in documents]
    )
    return LightKnowledgeBase(
        documents,
        vectorizer,
        tfidf_matrix,
        BM25Index([document.page_content for document in documents]),
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


def _persist_index(knowledge_base_id: str, index: LightKnowledgeBase) -> None:
    path = _index_path(knowledge_base_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in index.documents
    ]
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_knowledge_base(knowledge_base_id: str = "default") -> bool:
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    if knowledge_base_id in _knowledge_bases:
        return True

    path = _index_path(knowledge_base_id)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            LightDocument(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in payload
            if item.get("page_content")
        ]
        if not documents:
            return False
        _knowledge_bases[knowledge_base_id] = _fit_index(documents)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _documents_for_pdf(pdf_path: Path) -> list[LightDocument]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(PageText(page_number, page_text))
    return [
        LightDocument(chunk.page_content, chunk.metadata)
        for chunk in ingest_pages(
            pdf_path,
            pages,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
    ]


def _serialize_documents(documents) -> list[dict]:
    return [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]


def _deserialize_documents(payload) -> list[LightDocument]:
    return [
        LightDocument(
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
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    else:
        pdf_paths = list(pdf_paths)

    if not pdf_paths:
        raise ValueError("请先上传 PDF 文件。")

    del previous_index_path
    previous_files = (
        previous_cache.get("files", {})
        if previous_cache
        and previous_cache.get("schema_version") == 2
        and previous_cache.get("rag_mode") == "light"
        else {}
    )
    documents = []
    files_cache = {}
    reused_file_count = 0

    for value in pdf_paths:
        pdf_path = Path(value)
        digest_source = (
            pdf_path.read_bytes()
            if pdf_path.exists()
            else str(pdf_path).encode("utf-8")
        )
        digest = hashlib.sha256(digest_source).hexdigest()
        cached = previous_files.get(pdf_path.name, {})
        if (
            cached.get("sha256") == digest
            and cached.get("documents")
        ):
            file_documents = _deserialize_documents(
                cached["documents"]
            )
            reused_file_count += 1
        else:
            file_documents = _documents_for_pdf(pdf_path)
        documents.extend(file_documents)
        files_cache[pdf_path.name] = {
            "sha256": digest,
            "documents": _serialize_documents(file_documents),
        }

    if not documents:
        raise ValueError(
            "PDF 没有读取到有效文字内容。请使用文字版 PDF，不要使用扫描版 PDF。"
        )
    if len(documents) > _max_knowledge_base_chunks():
        raise ValueError(
            "工业知识资料切分后的文本块数量超过限制 "
            f"{_max_knowledge_base_chunks()}。"
        )

    index = _fit_index(documents)
    _persist_index(knowledge_base_id, index)
    _knowledge_bases[knowledge_base_id] = index
    page_count = len(
        {
            (
                document.metadata.get("source"),
                page_number,
            )
            for document in documents
            for page_number in range(
                int(document.metadata.get("page_start", document.metadata.get("page", 0))),
                int(document.metadata.get("page_end", document.metadata.get("page", 0))) + 1,
            )
        }
    )
    cache = {
        "schema_version": 2,
        "rag_mode": "light",
        "files": files_cache,
    }
    stats = {
        "reused_file_count": reused_file_count,
        "parsed_file_count": len(pdf_paths) - reused_file_count,
    }
    return page_count, len(documents), cache, stats


def build_knowledge_base(pdf_paths, knowledge_base_id: str = "default"):
    page_count, chunk_count, _, _ = build_knowledge_base_incremental(
        pdf_paths,
        knowledge_base_id,
    )
    return page_count, chunk_count


def is_knowledge_base_ready(knowledge_base_id: str = "default"):
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    if not load_knowledge_base(knowledge_base_id):
        return False
    index = _knowledge_bases[knowledge_base_id]
    return (
        bool(index.documents)
        and index.vectorizer is not None
        and index.tfidf_matrix is not None
        and index.bm25 is not None
    )


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def get_retrieval_mode(mode: str | None = None) -> str:
    selected = (mode or os.getenv("RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE)).lower()
    aliases = {"bm25": "lexical", "tfidf": "vector"}
    selected = aliases.get(selected, selected)
    if selected not in {"lexical", "vector", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE must be lexical, vector, or hybrid.")
    return selected


def _lexical_candidates(index, question: str, documents: list, analysis, top_k: int):
    scores = index.bm25.score(question)
    ranked = sorted(
        range(len(documents)),
        key=lambda item: (-scores[item], str(documents[item].metadata.get("chunk_id", ""))),
    )[: min(top_k, len(documents))]
    return [
        RetrievalCandidate(
            document=documents[item],
            retrieval_source="lexical",
            lexical_rank=rank,
            lexical_score=float(scores[item]),
            evidence_score=0.0 if scores[item] > 0 else 1.0,
            exact_metadata_match=has_exact_metadata_match(documents[item], analysis),
        )
        for rank, item in enumerate(ranked, start=1)
        if scores[item] > 0
    ]


def _tfidf_candidates(index, question: str, documents: list, analysis, top_k: int):
    from sklearn.metrics.pairwise import cosine_similarity
    query_vector = index.vectorizer.transform([question.strip()])
    similarities = cosine_similarity(query_vector, index.tfidf_matrix).ravel()
    ranked = similarities.argsort()[::-1][: min(top_k, len(documents))]
    return [
        RetrievalCandidate(
            document=documents[item],
            retrieval_source="vector",
            vector_rank=rank,
            vector_score=float(similarities[item]),
            evidence_score=float(1.0 - similarities[item]),
            exact_metadata_match=has_exact_metadata_match(documents[item], analysis),
        )
        for rank, item in enumerate(ranked, start=1)
    ]


def retrieve_docs(
    question: str,
    k: int = 4,
    knowledge_base_id: str = "default",
    retrieval_mode: str | None = None,
):
    if not question or not question.strip():
        raise ValueError("问题不能为空。")
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    if not is_knowledge_base_ready(knowledge_base_id):
        raise ValueError(EMPTY_KNOWLEDGE_BASE_MESSAGE)

    index = _knowledge_bases[knowledge_base_id]
    analysis = analyze_query(question, index.documents)
    mode = get_retrieval_mode(retrieval_mode)
    documents, filter_applied = filter_documents(index.documents, analysis)
    if analysis.error_code and not filter_applied:
        return RetrievalResult(
            [], query_analysis=analysis, corpus_documents=index.documents,
            retrieval_mode=mode,
        )
    if documents is not index.documents:
        filtered_index = _fit_index(documents)
    else:
        filtered_index = index
    lexical = _lexical_candidates(
        filtered_index,
        question,
        documents,
        analysis,
        _positive_int("LEXICAL_TOP_K", DEFAULT_LEXICAL_TOP_K),
    )
    vector = _tfidf_candidates(
        filtered_index,
        question,
        documents,
        analysis,
        _positive_int("VECTOR_TOP_K", DEFAULT_VECTOR_TOP_K),
    )
    if mode == "lexical":
        candidates = lexical[:k]
    elif mode == "vector":
        candidates = vector[:k]
    else:
        candidates = rrf_fuse(
            lexical,
            vector,
            rrf_k=_positive_int("RRF_K", DEFAULT_RRF_K),
            top_k=min(k, _positive_int("HYBRID_TOP_K", DEFAULT_HYBRID_TOP_K)),
        )
    for rank, candidate in enumerate(candidates, start=1):
        candidate.final_rank = rank
    return RetrievalResult(
        candidates, query_analysis=analysis, corpus_documents=index.documents,
        retrieval_mode=mode,
    )


def analyze_evidence(question: str, result, retrieval_mode: str | None = None, *, policy=None):
    return analyze_retrieval_evidence(
        question,
        result,
        getattr(result, "corpus_documents", []),
        get_retrieval_mode(retrieval_mode or getattr(result, "retrieval_mode", None)),
        policy=policy,
    )


def has_relevant_docs(scored_docs):
    if not scored_docs:
        return False
    return bool(filter_relevant_docs(scored_docs))


def get_relevance_threshold() -> float:
    try:
        return float(
            os.getenv(
                "LIGHT_MAX_RELEVANT_DISTANCE",
                str(MAX_RELEVANT_DISTANCE),
            )
        )
    except ValueError:
        return MAX_RELEVANT_DISTANCE


def filter_relevant_docs(scored_docs):
    candidates = getattr(scored_docs, "candidates", None)
    if candidates is not None:
        relevant = [
            candidate
            for candidate in candidates
            if candidate.exact_metadata_match
            or (
                candidate.vector_score is not None
                and candidate.evidence_score <= get_relevance_threshold()
            )
            or (
                candidate.vector_score is None
                and candidate.lexical_score is not None
                and candidate.lexical_score > 0
            )
        ]
        return RetrievalResult(relevant)
    threshold = get_relevance_threshold()
    return [
        (document, score)
        for document, score in scored_docs
        if score <= threshold
    ]


def get_all_docs(knowledge_base_id: str = "default"):
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    if not is_knowledge_base_ready(knowledge_base_id):
        raise ValueError(EMPTY_KNOWLEDGE_BASE_MESSAGE)
    return [
        (document, 0.0)
        for document in _knowledge_bases[knowledge_base_id].documents
    ]


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
""",
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


def clear_knowledge_base(knowledge_base_id: str = "default"):
    knowledge_base_id = _normalized_knowledge_base_id(knowledge_base_id)
    _knowledge_bases.pop(knowledge_base_id, None)

    index_path = _index_path(knowledge_base_id)
    if index_path.exists():
        index_path.unlink()

    data_dir = get_data_dir(knowledge_base_id)
    if data_dir.exists():
        shutil.rmtree(data_dir)
