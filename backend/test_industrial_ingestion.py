import json
from pathlib import Path

import pytest

from backend.ingestion import PageText, ingest_pages
from backend.ingestion.classifier import classify_document


FIXTURE_PATH = (
    Path(__file__).parent
    / "ingestion"
    / "fixtures"
    / "industrial_documents.json"
)


@pytest.fixture(scope="module")
def fixture_documents():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["documents"]


def ingest_fixture(document):
    pages = [
        PageText(page=index, text=text)
        for index, text in enumerate(document["pages"])
    ]
    return ingest_pages(document["file_name"], pages)


def test_classifier_covers_controlled_types(fixture_documents):
    by_name = {item["file_name"]: item for item in fixture_documents}
    assert classify_document(
        "plc_manual_sample.pdf",
        "\n".join(by_name["plc_manual_sample.pdf"]["pages"]),
    ) == "manual"
    assert classify_document(
        "fault_code_sample.pdf",
        "\n".join(by_name["fault_code_sample.pdf"]["pages"]),
    ) == "fault_code"
    assert classify_document(
        "sop_sample.pdf",
        "\n".join(by_name["sop_sample.pdf"]["pages"]),
    ) == "sop"
    assert classify_document("notes.pdf", "普通项目会议记录") == "general"


def test_metadata_and_ids_are_stable_and_flat(fixture_documents):
    chunks_a = ingest_fixture(fixture_documents[0])
    chunks_b = ingest_fixture(fixture_documents[0])
    assert [chunk.metadata["chunk_id"] for chunk in chunks_a] == [
        chunk.metadata["chunk_id"] for chunk in chunks_b
    ]
    assert chunks_a[0].metadata["document_id"] == chunks_b[0].metadata["document_id"]
    metadata = chunks_a[0].metadata
    assert metadata["document_type"] == "manual"
    assert metadata["manufacturer"] == "Synthetic Controls"
    assert metadata["equipment_type"] == "PLC"
    assert metadata["equipment_model"] == "IK-PLC-100"
    assert metadata["document_version"] == "V1.0"
    assert metadata["publish_date"] == "2026-08-10"
    assert metadata["page"] == metadata["page_start"]
    assert metadata["page_end"] >= metadata["page_start"]
    assert metadata["source"] == "plc_manual_sample.pdf"
    assert all(isinstance(value, (str, int)) for value in metadata.values())


def test_fault_codes_remain_atomic_with_exact_metadata(fixture_documents):
    chunks = ingest_fixture(fixture_documents[1])
    faults = {
        chunk.metadata["error_code"]: chunk
        for chunk in chunks
        if chunk.metadata["error_code"]
    }
    assert set(faults) == {"F0001", "F0002", "A0503"}
    assert "直流母线过压" in faults["F0002"].page_content
    assert "延长减速时间" in faults["F0002"].page_content
    assert faults["F0002"].metadata["knowledge_type"] == "fault"
    assert faults["A0503"].metadata["page"] == 1


def test_sop_steps_stay_together(fixture_documents):
    chunks = ingest_fixture(fixture_documents[2])
    procedure = next(
        chunk for chunk in chunks if chunk.metadata["section"] == "操作步骤"
    )
    assert all(f"{number}." in procedure.page_content for number in range(1, 5))
    assert procedure.metadata["section"] == "操作步骤"


def test_parameter_and_maintenance_knowledge_types(fixture_documents):
    manual_chunks = ingest_fixture(fixture_documents[0])
    assert any(
        chunk.metadata["knowledge_type"] == "parameter"
        and "额定输入电压：24 V" in chunk.page_content
        for chunk in manual_chunks
    )
    maintenance_chunks = ingest_fixture(fixture_documents[3])
    assert any(
        chunk.metadata["knowledge_type"] == "maintenance"
        for chunk in maintenance_chunks
    )


def test_long_general_section_uses_fallback_split_and_keeps_page_metadata():
    text = "普通说明\n" + "工业设备状态记录。" * 300
    chunks = ingest_pages(
        "general.pdf",
        [PageText(page=4, text=text)],
        chunk_size=200,
        overlap=30,
    )
    assert len(chunks) > 1
    assert all(chunk.metadata["document_type"] == "general" for chunk in chunks)
    assert all(chunk.metadata["page"] == 4 for chunk in chunks)
    assert all(len(chunk.page_content) <= 220 for chunk in chunks)


def test_identical_parts_from_a_parser_are_deduplicated_by_stable_chunk_id():
    chunks = ingest_pages(
        "repeated_table.pdf",
        [PageText(page=3, text="Technical specifications\nRepeated table footnote")],
        fallback_splitter=lambda _text, _size, _overlap: ["Repeated table footnote"] * 2,
    )
    assert len(chunks) == 1
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)


def test_pdf_footnotes_do_not_replace_english_chapter_section_metadata():
    chunks = ingest_pages(
        "manual.pdf",
        [PageText(page=8, text="Chapter 4 Installation\n(1) Available at firmware revision 31.\nGround the controller before wiring.")],
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "Chapter 4 Installation"
    assert "Available at firmware revision 31" in chunks[0].page_content


def test_empty_text_pdf_fails_cleanly():
    with pytest.raises(ValueError, match="没有读取到有效文字"):
        ingest_pages("empty.pdf", [PageText(page=0, text="   ")])
