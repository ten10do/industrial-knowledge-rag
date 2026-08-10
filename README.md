# Industrial Knowledge RAG

工业知识智能检索与问答平台。

## 当前阶段

`V0 Base RAG`

这是一个面向工业设备技术手册、维护文档、故障知识、SOP 和工业控制资料的基础 RAG 系统。当前版本保留原有可运行的检索与问答主链路；尚未实现 Hybrid Search、Reranker、Agent、知识图谱、OCR 或 RAG 评测平台等后续能力。

## 当前已实现能力

- React + Vite 前端与 FastAPI 后端。
- 管理员上传多个文字型 PDF，构建草稿知识库并发布为只读版本。
- PDF 解析、文本分块、增量索引快照与来源页码追溯。
- `light` 模式：TF-IDF 检索，适合低资源环境。
- `full` 模式：HuggingFace Embeddings + Chroma 向量检索。
- 基于相关性阈值的拒答；答案和来源片段返回 `[S1]` 等引用标记。
- Groq 或 DeepSeek 生成回答；文档摘要、关键知识、核对问题等辅助输出。
- 多轮对话上下文压缩、管理 Token、任务队列、版本历史与回滚能力。

## 架构

```text
Industrial PDF documents
  -> PyPDF / PyPDFLoader
  -> RecursiveCharacterTextSplitter
  -> light: TF-IDF  |  full: HuggingFace Embeddings
  -> light index    |  full: Chroma vector store
  -> relevance-filtered retrieval
  -> Groq / DeepSeek LLM
  -> Answer + source file, page, score, citations
```

## 目录与数据边界

```text
backend/
  data/             # 运行时上传的原始资料（已忽略）
  light_indexes/    # light 模式运行时索引（已忽略）
  vector_db/        # Chroma 运行时索引（已忽略）
  public_versions/  # 本地发布版本（已忽略）
  runtime_state/    # 任务运行状态（已忽略）
frontend/           # React + Vite UI
```

这些目录是运行时数据边界，不包含正式工业知识库。请在后续 ingest 时使用经授权、脱敏的工业资料；不要把 API Key、上传文档、向量索引或临时文件提交到 Git。

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
- `FRONTEND_ORIGIN`：允许访问 API 的前端源。
- `PUBLIC_VERSION_*`、`REDIS_URL`、`TASK_QUEUE_*`：版本存储与生产任务队列配置。

`.env` 已被 Git 忽略；不要把真实密钥写入 `.env.example`。

## 验证

```powershell
.\venv\Scripts\python.exe -m pytest
cd frontend
npm.cmd test
npm.cmd run build
```

离线测试覆盖 PDF 边界、分块、light 检索、FastAPI 接口、版本存储、任务队列和前端组件。真实 LLM / embedding 端到端调用需要有效密钥、网络访问和相应模型资源，不能用测试替代。

## Roadmap（未来计划，未在 V0 实现）

```text
V0 Base RAG
  -> V1 Industrial Document RAG
  -> V2 Hybrid Retrieval
  -> V3 Reranker
  -> V4 RAG Evaluation
  -> V5 Observable Industrial RAG
```
