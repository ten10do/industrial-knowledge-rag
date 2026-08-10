# Industrial Knowledge RAG

工业知识智能检索与问答平台。

## 当前阶段

`V2 Hybrid Retrieval`

这是一个面向工业设备技术手册、维护文档、故障知识、SOP 和工业控制资料的 RAG 系统。V1 加入结构感知的工业文档摄取、工业知识 chunk 和受控 metadata；V2 在其上加入规则型 Query Analysis、metadata 预过滤、BM25、统一候选结果与 RRF 融合。Reranker、Agent、知识图谱、OCR 和 LLM Query Rewrite 仍未实现。

## 当前已实现能力

- React + Vite 前端与 FastAPI 后端。
- 管理员上传多个文字型 PDF，构建草稿知识库并发布为只读版本。
- PDF 解析、文本分块、增量索引快照与来源页码追溯。
- 基于文件名、标题和前五页文本的无 LLM 文档分类：`manual`、`fault_code`、`sop`、`maintenance`、`technical_spec`、`general`。
- 保守识别章节、故障码块、SOP 步骤、参数块和维护内容；普通文档继续使用 fallback chunking。
- 稳定的 `document_id` / `chunk_id`，以及可供过滤、引用和评测的扁平工业 metadata。
- `light` 模式：默认使用 BM25 + 字符级 TF-IDF 的 RRF；TF-IDF 仅是 legacy baseline，不将其误称为 embedding vector。
- `full` 模式：BM25 + HuggingFace Embeddings + Chroma；向量依赖不可用时，服务回退到 light/BM25，而不会伪装为 Full Hybrid。
- 基于相关性阈值的拒答；答案和来源片段返回 `[S1]` 等引用标记。
- Groq 或 DeepSeek 生成回答；文档摘要、关键知识、核对问题等辅助输出。
- 多轮对话上下文压缩、管理 Token、任务队列、版本历史与回滚能力。

## 架构

```text
Industrial PDF documents
  -> PyPDF / PyPDFLoader
  -> Industrial Document Classifier
  -> Section-aware Parser
  -> Industrial-aware Chunker
  -> Rich flat metadata
  -> RecursiveCharacterTextSplitter fallback (full mode)
  -> Query Analysis + metadata filter
  -> BM25 lexical retrieval + (light legacy TF-IDF | full Chroma vector)
  -> RRF fusion -> unified retrieval candidates
  -> evidence-based relevance/refusal (not an RRF threshold)
  -> Groq / DeepSeek LLM
  -> Answer + source file, page, score, citations
```

## 目录与数据边界

```text
backend/
  ingestion/        # V1 分类、metadata schema、结构解析与 chunking
  data/             # 运行时上传的原始资料（已忽略）
  light_indexes/    # light 模式运行时索引（已忽略）
  vector_db/        # Chroma 运行时索引（已忽略）
  public_versions/  # 本地发布版本（已忽略）
  runtime_state/    # 任务运行状态（已忽略）
frontend/           # React + Vite UI
```

## V1 Industrial Document RAG

V1 实际写入每个 chunk 的 metadata 字段如下：

```text
document_id, source, file_name, document_type,
manufacturer, equipment_type, equipment_model,
title, section, subsection,
page, page_start, page_end,
language, document_version, publish_date,
knowledge_type, error_code, chunk_id, chunk_index
```

所有写入 TF-IDF JSON 或 Chroma 的值均为字符串或整数，不包含嵌套对象。可选字段以空字符串表示。内部页码从 0 开始，API 引用页码从 1 开始。

结构识别是保守启发式，不声称能从所有 PDF 恢复视觉标题层级。故障码和短 SOP/参数单元优先保持完整；超长单元递归切分。完全普通的 PDF 分类为 `general` 并继续正常建库。

这些目录是运行时数据边界，不包含正式工业知识库。请在后续 ingest 时使用经授权、脱敏的工业资料；不要把 API Key、上传文档、向量索引或临时文件提交到 Git。

## V2 Hybrid Retrieval

V2 的检索链路为 `Query -> rule analysis -> metadata filter -> BM25 / vector -> RRF -> unified candidates -> existing answer flow`。候选对象保留 `chunk_id`、`document_id`、正文与原始 metadata，并单独携带 `retrieval_source`、lexical/vector rank 与 score、`fusion_score` 和 `final_rank`，不会污染工业文档的 metadata。

规则分析优先识别错误/报警码和已入库的设备型号；错误码存在精确 metadata 命中时严格过滤，设备型号也仅在有候选时严格过滤，避免分类 metadata 不完整造成空结果。未知错误码（如 `F9999`）不会近似命中其他故障码。拒答判断使用 exact metadata、BM25 evidence 或原向量距离；RRF 仅负责排序。

Implemented:

- Dependency-free BM25 tokenizer/scorer, preserving `F0002`、`S7-1200`、`0x8001`、`MW20` and mixed Chinese/English tokens.
- Light real benchmark: legacy TF-IDF, BM25, and BM25 + TF-IDF RRF.
- Full-code integration: BM25 reads Chroma documents without an embedding function; vector retrieval continues to use the original embedding/Chroma path.

Verified:

- Unit coverage for tokenizer, BM25, query analysis/filter fallback, RRF deduplication, unified candidates, API validation, and unknown identifier refusal.
- The synthetic V2 benchmark measures Hit@1/3, MRR, exact identifier/fault/model metrics, model confusion, OOD refusal, and median retrieval latency.

## V2.5 Full Vector Validation

Verified with the existing `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model, `HuggingFaceEmbeddings`, and temporary Chroma collections. The real smoke test passes V1 industrial chunks through `ingest_pages`, embeddings, Chroma persistence, a vector query, and citation metadata recovery. The model/cache and temporary databases remain outside Git.

`backend/evaluation/full_vector_benchmark.py` compares the same documents, chunks, cases, and labels across legacy TF-IDF, BM25, Dense Vector (HuggingFace + Chroma), and BM25 + Dense Vector + RRF. It also reports per-query ranks, vector debug metadata/distances, warm retrieval latency, semantic-query Hit@1, and Hybrid Win Rate.

The benchmark is deliberately small and synthetic. It validates the retrieval wiring and exposes failure modes; it does not claim production accuracy. The `RUN_FULL_VECTOR_SMOKE=1` test opt-in performs the real model/Chroma smoke test, while normal test runs stay offline and fast.

Environment notes:

- `backend/requirements-full.txt` is the authoritative dependency declaration and now installs cleanly with `pip check`.
- The configured model is cached by HuggingFace outside the repository. Windows may warn when its cache cannot use symlinks; this is a storage-efficiency warning, not a retrieval fallback.
- Full dependency failure remains safe: startup records requested/effective RAG and retrieval modes and falls back to light/BM25 instead of claiming a dense hybrid run.

## 本地运行

要求：Python 3.11+、Node.js 20+。

```powershell
# 后端
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

另开终端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

默认前端地址为 `http://localhost:5173`，后端健康检查为 `http://127.0.0.1:8000/health`。配置 `ADMIN_TOKEN` 后才能上传、构建、发布或回滚知识库。默认 `RAG_MODE=light`；`full` 模式需要 `backend/requirements-full.txt` 的额外依赖和可下载的 embedding 模型。

## 配置

复制 `.env.example` 为 `.env` 后按需填写：

- `GROQ_API_KEY` 或 `DEEPSEEK_API_KEY`：真实生成服务。
- `ADMIN_TOKEN`：管理接口保护令牌。
- `RAG_MODE=light|full`：选择检索实现。
- `RETRIEVAL_MODE=lexical|vector|hybrid`：默认检索策略；默认 `hybrid` 保持 V0/V1 拒答基线，light 中 `vector` 表示 legacy TF-IDF，不是 embedding vector。
- `LEXICAL_TOP_K`、`VECTOR_TOP_K`、`HYBRID_TOP_K`、`RRF_K`：候选规模与 RRF 参数，默认分别为 `10/10/5/60`。
- `FRONTEND_ORIGIN`：允许访问 API 的前端源。
- `PUBLIC_VERSION_*`、`REDIS_URL`、`TASK_QUEUE_*`：版本存储与生产任务队列配置。

`.env` 已被 Git 忽略；不要把真实密钥写入 `.env.example`。

## 验证

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m backend.evaluation.retrieval_benchmark
.\venv\Scripts\python.exe -m backend.evaluation.full_vector_benchmark
# Optional real Full smoke gate (downloads/loads the configured embedding model)
$env:RUN_FULL_VECTOR_SMOKE="1"; .\venv\Scripts\python.exe -m pytest backend\test_full_vector_smoke.py
cd frontend
npm.cmd test
npm.cmd run build
```

离线测试覆盖 PDF 边界、工业分类、结构化分块、metadata 稳定性、light 检索、FastAPI 接口、版本存储、任务队列和前端组件。`backend/ingestion/fixtures/` 与 `backend/evaluation/fixtures/industrial_dataset.json` 是仓库自建的小型合成回归资料，不含厂商手册原文，也不代表生产质量 benchmark。

工业 fixture 真实计算 Hit Rate@1、Hit Rate@3、MRR、metadata 完整率、章节命中率、页码命中率和故障码精确命中率。真实 LLM / embedding 端到端调用需要有效密钥、网络访问和相应模型资源，不能用离线测试替代。

## Roadmap

```text
V0 Base RAG
  -> V1 Industrial Document RAG (implemented)
  -> V2 Hybrid Retrieval (implemented and Full Vector validated)
  -> V3 Reranker
  -> V4 RAG Evaluation
  -> V5 Observable Industrial RAG
```
