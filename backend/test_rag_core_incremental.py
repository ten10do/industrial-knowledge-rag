import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


sys.modules.setdefault(
    "langchain_chroma",
    types.SimpleNamespace(Chroma=object),
)
sys.modules.setdefault(
    "langchain_community.document_loaders",
    types.SimpleNamespace(PyPDFLoader=object),
)
sys.modules.setdefault(
    "langchain_community.embeddings",
    types.SimpleNamespace(HuggingFaceEmbeddings=object),
)
sys.modules.setdefault(
    "langchain_text_splitters",
    types.SimpleNamespace(RecursiveCharacterTextSplitter=object),
)
sys.modules.setdefault(
    "langchain_core.documents",
    types.SimpleNamespace(Document=FakeDocument),
)

from backend import rag_core


class FakeChroma:
    events = []

    def __init__(self, persist_directory, embedding_function=None):
        self.persist_directory = Path(persist_directory)
        state_path = self.persist_directory / "state.json"
        self.ids = json.loads(state_path.read_text()) if state_path.exists() else []

    @classmethod
    def from_documents(
        cls,
        documents,
        ids,
        embedding,
        persist_directory,
    ):
        instance = cls(persist_directory)
        instance.ids = list(ids)
        instance._save()
        cls.events.append(("create", list(ids)))
        return instance

    def _save(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        (self.persist_directory / "state.json").write_text(
            json.dumps(self.ids)
        )

    def delete(self, ids):
        self.ids = [item for item in self.ids if item not in set(ids)]
        self._save()
        self.events.append(("delete", list(ids)))

    def add_documents(self, documents, ids):
        self.ids.extend(ids)
        self._save()
        self.events.append(("add", list(ids)))


def test_full_incremental_build_updates_only_changed_vectors(
    tmp_path,
    monkeypatch,
):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a-v1")
    pdf_b.write_bytes(b"b-v1")

    def load_pdf(path):
        value = Path(path)
        return [
            SimpleNamespace(
                page_content=value.read_bytes().decode(),
                metadata={"source": value.name, "page": 0},
            )
        ]

    FakeChroma.events = []
    monkeypatch.setattr(rag_core, "PERSIST_DIR", tmp_path / "vector")
    monkeypatch.setattr(rag_core, "Chroma", FakeChroma)
    monkeypatch.setattr(rag_core, "load_pdf", load_pdf)
    monkeypatch.setattr(rag_core, "split_documents", lambda documents: documents)
    monkeypatch.setattr(rag_core, "get_embedding_model", lambda: object())

    _, _, first_cache, _ = rag_core.build_knowledge_base_incremental(
        [pdf_a, pdf_b],
        "full-incremental-test",
    )
    index_path = rag_core.get_index_storage_path("full-incremental-test")
    old_a_id = first_cache["files"]["a.pdf"]["ids"][0]

    pdf_a.write_bytes(b"a-v2")
    _, _, second_cache, second_stats = (
        rag_core.build_knowledge_base_incremental(
            [pdf_a, pdf_b],
            "full-incremental-test",
            previous_index_path=index_path,
            previous_cache=first_cache,
        )
    )

    assert second_stats == {
        "reused_file_count": 1,
        "parsed_file_count": 1,
        "updated_vector_file_count": 1,
        "removed_vector_file_count": 0,
    }
    assert ("delete", [old_a_id]) in FakeChroma.events
    assert (
        "add",
        second_cache["files"]["a.pdf"]["ids"],
    ) in FakeChroma.events

    _, _, third_cache, third_stats = (
        rag_core.build_knowledge_base_incremental(
            [pdf_a],
            "full-incremental-test",
            previous_index_path=index_path,
            previous_cache=second_cache,
        )
    )
    persisted_ids = json.loads((index_path / "state.json").read_text())

    assert third_stats["reused_file_count"] == 1
    assert third_stats["removed_vector_file_count"] == 1
    assert persisted_ids == third_cache["files"]["a.pdf"]["ids"]
