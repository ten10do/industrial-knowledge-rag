import unittest
import importlib.util
import sys
import types
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("groq", types.SimpleNamespace(Groq=object))
sys.modules.setdefault(
    "langchain_community.document_loaders",
    types.SimpleNamespace(PyPDFLoader=object),
)
sys.modules.setdefault(
    "langchain_text_splitters",
    types.SimpleNamespace(RecursiveCharacterTextSplitter=object),
)
sys.modules.setdefault("langchain_chroma", types.SimpleNamespace(Chroma=object))
sys.modules.setdefault(
    "langchain_community.embeddings",
    types.SimpleNamespace(HuggingFaceEmbeddings=object),
)

_MODULE_NAME = "streamlit_rag_core"
_MODULE_PATH = Path(__file__).with_name("rag_core.py")
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
rag_core = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = rag_core
assert _SPEC.loader is not None
_SPEC.loader.exec_module(rag_core)


class FakeVectorDb:
    def __init__(self):
        self.called_with = None
        self.get_called_with = None

    def similarity_search_with_score(self, question, k):
        self.called_with = (question, k)
        return [("doc-a", 0.42)]

    def get(self, include, limit):
        self.get_called_with = (include, limit)
        return {
            "documents": ["chunk-a", "chunk-b"],
            "metadatas": [
                {"source": "a.pdf", "page": 0},
                {"source": "b.pdf", "page": 1},
            ],
        }


class RetrieveDocsTests(unittest.TestCase):
    def test_retrieve_docs_returns_documents_with_scores(self):
        fake_db = FakeVectorDb()

        with patch("streamlit_rag_core.load_vector_db", return_value=fake_db):
            results = rag_core.retrieve_docs("什么是闭环控制？", k=3)

        self.assertEqual(fake_db.called_with, ("什么是闭环控制？", 3))
        self.assertEqual(results, [("doc-a", 0.42)])

    def test_is_relevant_rejects_empty_or_high_distance_results(self):
        self.assertFalse(rag_core.has_relevant_docs([]))
        self.assertFalse(
            rag_core.has_relevant_docs([("doc-a", rag_core.MAX_RELEVANT_DISTANCE + 0.01)])
        )

    def test_is_relevant_accepts_distance_under_threshold(self):
        self.assertTrue(
            rag_core.has_relevant_docs([("doc-a", rag_core.MAX_RELEVANT_DISTANCE - 0.01)])
        )


class BuildKnowledgeBaseTests(unittest.TestCase):
    def test_build_knowledge_base_combines_multiple_pdfs_and_rebuilds_vector_db(self):
        docs_by_path = {
            "data/a.pdf": [SimpleNamespace(metadata={})],
            "data/b.pdf": [SimpleNamespace(metadata={}), SimpleNamespace(metadata={})],
        }
        chunks_by_doc_count = {
            1: [SimpleNamespace(metadata={"source": "a.pdf", "page": 0})],
            2: [
                SimpleNamespace(metadata={"source": "b.pdf", "page": 0}),
                SimpleNamespace(metadata={"source": "b.pdf", "page": 1}),
            ],
        }

        with patch("streamlit_rag_core.load_pdf", side_effect=lambda path: docs_by_path[path]):
            with patch(
                "streamlit_rag_core.split_documents",
                side_effect=lambda docs: chunks_by_doc_count[len(docs)],
            ):
                with patch("streamlit_rag_core.clear_vector_db") as clear_vector_db:
                    with patch("streamlit_rag_core.build_vector_db") as build_vector_db:
                        page_count, chunk_count = rag_core.build_knowledge_base(
                            ["data/a.pdf", "data/b.pdf"]
                        )

        clear_vector_db.assert_called_once()
        build_vector_db.assert_called_once()
        self.assertEqual(page_count, 3)
        self.assertEqual(chunk_count, 3)
        self.assertEqual(len(build_vector_db.call_args.args[0]), 3)

    def test_load_pdf_keeps_source_as_filename_and_page_metadata(self):
        loaded_docs = [
            SimpleNamespace(page_content="第一页内容", metadata={"source": "old", "page": 0}),
            SimpleNamespace(page_content="第二页内容", metadata={"page": 1}),
        ]

        class FakeLoader:
            def __init__(self, file_path):
                self.file_path = file_path

            def load(self):
                return loaded_docs

        with patch("streamlit_rag_core.PyPDFLoader", FakeLoader):
            docs = rag_core.load_pdf("data/course-a.pdf")

        self.assertEqual([doc.metadata["source"] for doc in docs], ["course-a.pdf", "course-a.pdf"])
        self.assertEqual([doc.metadata["page"] for doc in docs], [0, 1])

    def test_clear_knowledge_base_removes_vector_db_and_data_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            vector_dir = temp_path / "vector_db"
            data_dir = temp_path / "data"
            vector_dir.mkdir()
            data_dir.mkdir()
            (vector_dir / "index.bin").write_text("old vector", encoding="utf-8")
            (data_dir / "old.pdf").write_text("old pdf", encoding="utf-8")

            with patch("streamlit_rag_core.PERSIST_DIR", str(vector_dir)):
                with patch("streamlit_rag_core.DATA_DIR", str(data_dir)):
                    rag_core.clear_knowledge_base()

            self.assertFalse(vector_dir.exists())
            self.assertTrue(data_dir.exists())
            self.assertEqual(list(data_dir.iterdir()), [])


class GenerateAnswerTests(unittest.TestCase):
    def test_generate_answer_delegates_to_selected_llm_provider(self):
        docs = [(SimpleNamespace(page_content="参考内容"), 0.1)]

        with patch("streamlit_rag_core.generate_llm_answer", return_value="模型回答") as generate_llm_answer:
            answer = rag_core.generate_answer("测试问题", docs, provider="DeepSeek")

        self.assertEqual(answer, "模型回答")
        generate_llm_answer.assert_called_once_with(
            "测试问题",
            docs,
            provider="DeepSeek",
        )


class LearningAssistantTests(unittest.TestCase):
    def test_get_representative_docs_loads_limited_chunks_from_vector_db(self):
        fake_db = FakeVectorDb()

        with patch("streamlit_rag_core.load_vector_db", return_value=fake_db):
            docs = rag_core.get_representative_docs(k=2)

        self.assertEqual(fake_db.get_called_with, (["documents", "metadatas"], 2))
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0][0].page_content, "chunk-a")
        self.assertEqual(docs[0][0].metadata["source"], "a.pdf")
        self.assertEqual(docs[0][1], 0.0)

    def test_generate_learning_content_uses_selected_provider_and_task_prompt(self):
        representative_docs = [(SimpleNamespace(page_content="工业知识资料", metadata={}), 0.0)]

        with patch("streamlit_rag_core.get_representative_docs", return_value=representative_docs):
            with patch("streamlit_rag_core.generate_llm_answer", return_value="学习辅助结果") as generate_llm_answer:
                result = rag_core.generate_learning_content("summary", provider="DeepSeek")

        self.assertEqual(result, "学习辅助结果")
        question, docs = generate_llm_answer.call_args.args
        self.assertIn("资料主要内容", question)
        self.assertIn("不要脱离资料", question)
        self.assertEqual(docs, representative_docs)
        self.assertEqual(generate_llm_answer.call_args.kwargs["provider"], "DeepSeek")

    def test_generate_learning_content_requires_existing_knowledge_base(self):
        with patch("streamlit_rag_core.get_representative_docs", return_value=[]):
            with self.assertRaisesRegex(ValueError, "请先上传 PDF 并构建知识库。"):
                rag_core.generate_learning_content("summary", provider="Groq")


if __name__ == "__main__":
    unittest.main()
