import importlib
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.modules.setdefault("dotenv", SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class LightRagCoreTests(unittest.TestCase):
    def test_incremental_build_reuses_unchanged_pdf_chunks(self):
        light_rag_core = importlib.import_module("backend.light_rag_core")
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            first_pdf = root / "first.pdf"
            second_pdf = root / "second.pdf"
            first_pdf.write_bytes(b"first-version")
            second_pdf.write_bytes(b"second-version")
            calls = []

            def fake_reader(path):
                calls.append(Path(path).name)
                text = {
                    "first.pdf": "PID 积分项用于消除稳态误差。",
                    "second.pdf": "PLC 按扫描周期执行程序。",
                }[Path(path).name]
                return SimpleNamespace(pages=[FakePage(text)])

            with patch.object(light_rag_core, "DATA_DIR", data_dir):
                with patch.object(
                    light_rag_core,
                    "PdfReader",
                    side_effect=fake_reader,
                ):
                    _, _, cache, first_stats = (
                        light_rag_core.build_knowledge_base_incremental(
                            [first_pdf],
                            "kb-incremental-test-0001",
                            previous_cache={},
                        )
                    )
                    _, chunks, _, second_stats = (
                        light_rag_core.build_knowledge_base_incremental(
                            [first_pdf, second_pdf],
                            "kb-incremental-test-0001",
                            previous_cache=cache,
                        )
                    )
                    incremental_results = (
                        light_rag_core.retrieve_docs(
                            "PLC 扫描周期",
                            k=2,
                            knowledge_base_id=(
                                "kb-incremental-test-0001"
                            ),
                        )
                    )
                    light_rag_core.build_knowledge_base_incremental(
                        [first_pdf, second_pdf],
                        "kb-incremental-full-0001",
                        previous_cache={},
                    )
                    full_results = light_rag_core.retrieve_docs(
                        "PLC 扫描周期",
                        k=2,
                        knowledge_base_id="kb-incremental-full-0001",
                    )

            self.assertEqual(first_stats["parsed_file_count"], 1)
            self.assertEqual(second_stats["reused_file_count"], 1)
            self.assertEqual(second_stats["parsed_file_count"], 1)
            self.assertEqual(chunks, 2)
            self.assertEqual(
                [
                    item[0].metadata["source"]
                    for item in incremental_results
                ],
                [item[0].metadata["source"] for item in full_results],
            )
            self.assertEqual(
                [round(item[1], 6) for item in incremental_results],
                [round(item[1], 6) for item in full_results],
            )
            self.assertEqual(
                calls,
                [
                    "first.pdf",
                    "second.pdf",
                    "first.pdf",
                    "second.pdf",
                ],
            )

    def test_relevance_threshold_is_configurable_and_filters_each_document(self):
        light_rag_core = importlib.import_module("backend.light_rag_core")
        scored_docs = [
            (SimpleNamespace(page_content="相关", metadata={}), 0.10),
            (SimpleNamespace(page_content="较弱", metadata={}), 0.30),
        ]

        with patch.dict(
            os.environ,
            {"LIGHT_MAX_RELEVANT_DISTANCE": "0.20"},
        ):
            filtered = light_rag_core.filter_relevant_docs(scored_docs)

        self.assertEqual(filtered, scored_docs[:1])

    def test_scoped_indexes_are_isolated_and_reload_from_disk(self):
        light_rag_core = importlib.import_module("backend.light_rag_core")
        pdf_pages = {
            "plc.pdf": ["PLC 按照输入采样、程序执行、输出刷新循环扫描。"],
            "motor.pdf": ["异步电动机通过变频器改变供电频率实现调速。"],
        }

        def fake_reader(path):
            pages = [FakePage(text) for text in pdf_pages[Path(path).name]]
            return SimpleNamespace(pages=pages)

        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            with patch.object(light_rag_core, "DATA_DIR", data_dir):
                with patch.object(light_rag_core, "PdfReader", side_effect=fake_reader):
                    light_rag_core.build_knowledge_base(
                        [Path("plc.pdf")],
                        knowledge_base_id="kb-isolated-plc-0001",
                    )
                    light_rag_core.build_knowledge_base(
                        [Path("motor.pdf")],
                        knowledge_base_id="kb-isolated-motor-01",
                    )

                plc_result = light_rag_core.retrieve_docs(
                    "PLC 扫描周期",
                    knowledge_base_id="kb-isolated-plc-0001",
                )
                motor_result = light_rag_core.retrieve_docs(
                    "变频器调速",
                    knowledge_base_id="kb-isolated-motor-01",
                )
                self.assertEqual(plc_result[0][0].metadata["source"], "plc.pdf")
                self.assertEqual(motor_result[0][0].metadata["source"], "motor.pdf")

                light_rag_core._knowledge_bases.clear()
                self.assertTrue(
                    light_rag_core.is_knowledge_base_ready(
                        "kb-isolated-plc-0001"
                    )
                )
                restored = light_rag_core.retrieve_docs(
                    "程序执行",
                    knowledge_base_id="kb-isolated-plc-0001",
                )
                self.assertEqual(restored[0][0].metadata["source"], "plc.pdf")

                light_rag_core.clear_knowledge_base("kb-isolated-plc-0001")
                self.assertFalse(
                    light_rag_core.is_knowledge_base_ready(
                        "kb-isolated-plc-0001"
                    )
                )
                self.assertTrue(
                    light_rag_core.is_knowledge_base_ready(
                        "kb-isolated-motor-01"
                    )
                )

    def test_failed_light_rebuild_keeps_previous_index(self):
        light_rag_core = importlib.import_module("backend.light_rag_core")

        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            with patch.object(light_rag_core, "DATA_DIR", data_dir):
                with patch.object(
                    light_rag_core,
                    "PdfReader",
                    return_value=SimpleNamespace(
                        pages=[FakePage("PID 积分项用于消除稳态误差。")]
                    ),
                ):
                    light_rag_core.build_knowledge_base(
                        [Path("old.pdf")],
                        knowledge_base_id="kb-rollback-light-0001",
                    )

                with patch.object(
                    light_rag_core,
                    "PdfReader",
                    side_effect=ValueError("broken pdf"),
                ):
                    with self.assertRaisesRegex(ValueError, "broken pdf"):
                        light_rag_core.build_knowledge_base(
                            [Path("new.pdf")],
                            knowledge_base_id="kb-rollback-light-0001",
                        )

                result = light_rag_core.retrieve_docs(
                    "积分项",
                    knowledge_base_id="kb-rollback-light-0001",
                )
                self.assertEqual(result[0][0].metadata["source"], "old.pdf")

    def test_multi_pdf_retrieval_metadata_refusal_and_reset(self):
        module_spec = importlib.util.find_spec("backend.light_rag_core")
        self.assertIsNotNone(module_spec)
        if module_spec is None:
            return

        light_rag_core = importlib.import_module("backend.light_rag_core")
        pdf_pages = {
            "feedback.pdf": [
                "反馈控制通过检测输出、形成偏差并调整控制作用来减小误差。",
                "负反馈能够提高抗干扰能力，但设计不当可能导致振荡。",
            ],
            "stability.pdf": [
                "稳定性是控制系统的基本要求，闭环极点应位于左半平面。",
            ],
        }

        def fake_reader(path):
            pages = [FakePage(text) for text in pdf_pages[Path(path).name]]
            return SimpleNamespace(pages=pages)

        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            with patch.object(light_rag_core, "DATA_DIR", data_dir):
                with patch.object(light_rag_core, "PdfReader", side_effect=fake_reader):
                    page_count, chunk_count = light_rag_core.build_knowledge_base(
                        [Path("feedback.pdf"), Path("stability.pdf")]
                    )

                self.assertEqual(page_count, 3)
                self.assertEqual(chunk_count, 3)
                self.assertTrue(light_rag_core.is_knowledge_base_ready())

                results = light_rag_core.retrieve_docs("反馈控制为什么需要稳定性分析", k=3)
                self.assertEqual(len(results), 3)
                self.assertLess(results[0][1], 1.0)
                self.assertTrue(
                    light_rag_core.has_relevant_docs(results),
                    msg=f"top distance: {results[0][1]}",
                )
                self.assertEqual(
                    {result[0].metadata["source"] for result in results},
                    {"feedback.pdf", "stability.pdf"},
                )
                self.assertTrue(
                    all("page" in result[0].metadata for result in results)
                )

                unrelated = light_rag_core.retrieve_docs("量子化学分子轨道", k=1)
                self.assertFalse(light_rag_core.has_relevant_docs(unrelated))

                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "feedback.pdf").write_bytes(b"temporary")
                light_rag_core.clear_knowledge_base()
                self.assertFalse(light_rag_core.is_knowledge_base_ready())
                self.assertFalse(data_dir.exists())


if __name__ == "__main__":
    unittest.main()
