import importlib
import logging
import os
import re
import shutil
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

if __package__:
    from .conversation.context_manager import ConversationContextManager
    from .conversation.models import (
        CONVERSATION_ID_PATTERN,
        MAX_CONVERSATION_ID_CHARS,
        MAX_HISTORY_TURNS,
        MAX_QUESTION_CHARS,
        ContextOptions,
        ConversationContext,
        ConversationTurn,
        normalize_message_text,
    )
    from .conversation.query_rewriter import LlmQueryRewriter
    from .conversation.summarizer import LlmConversationSummarizer
    from .model_governance import (
        ModelGovernanceError,
        reset_model_scope,
        set_model_scope,
    )
    from .index_snapshot import (
        build_fingerprint,
        create_index_snapshot,
        extract_index_snapshot,
        snapshot_is_compatible,
    )
    from .security import create_rate_limiter, require_admin_token
    from .task_queue import create_job_id, create_task_queue
    from .version_sync import PublicVersionSynchronizer
    from .version_store import (
        create_pdf_bundle,
        create_version_store,
        extract_pdf_bundle,
        sha256_hex,
        validate_job_id,
        validate_version_id,
    )
else:
    from conversation.context_manager import ConversationContextManager
    from conversation.models import (
        CONVERSATION_ID_PATTERN,
        MAX_CONVERSATION_ID_CHARS,
        MAX_HISTORY_TURNS,
        MAX_QUESTION_CHARS,
        ContextOptions,
        ConversationContext,
        ConversationTurn,
        normalize_message_text,
    )
    from conversation.query_rewriter import LlmQueryRewriter
    from conversation.summarizer import LlmConversationSummarizer
    from model_governance import (
        ModelGovernanceError,
        reset_model_scope,
        set_model_scope,
    )
    from index_snapshot import (
        build_fingerprint,
        create_index_snapshot,
        extract_index_snapshot,
        snapshot_is_compatible,
    )
    from security import create_rate_limiter, require_admin_token
    from task_queue import create_job_id, create_task_queue
    from version_sync import PublicVersionSynchronizer
    from version_store import (
        create_pdf_bundle,
        create_version_store,
        extract_pdf_bundle,
        sha256_hex,
        validate_job_id,
        validate_version_id,
    )

RAG_MODE = os.getenv("RAG_MODE", "light").strip().lower()
if RAG_MODE not in {"full", "light"}:
    raise RuntimeError("RAG_MODE 只支持 full 或 light。")

REQUESTED_RAG_MODE = RAG_MODE
REQUESTED_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
RAG_BACKEND_NAME = "rag_core" if RAG_MODE == "full" else "light_rag_core"
RAG_MODE_FALLBACK_REASON = ""


def _load_rag_backend(name: str):
    if __package__:
        return importlib.import_module(f".{name}", package=__package__)
    return importlib.import_module(name)


try:
    rag_backend = _load_rag_backend(RAG_BACKEND_NAME)
except ModuleNotFoundError as exc:
    if RAG_MODE != "full":
        raise
    RAG_MODE = "light"
    RAG_BACKEND_NAME = "light_rag_core"
    RAG_MODE_FALLBACK_REASON = f"Full dependencies unavailable: {exc.name}"
    os.environ["RETRIEVAL_MODE"] = "lexical"
    rag_backend = _load_rag_backend(RAG_BACKEND_NAME)

EFFECTIVE_RAG_MODE = RAG_MODE
EFFECTIVE_RETRIEVAL_MODE = os.getenv(
    "RETRIEVAL_MODE",
    "hybrid",
).strip().lower()

if __package__:
    llm_module = importlib.import_module(".llm_client", package=__package__)
else:
    llm_module = importlib.import_module("llm_client")

DATA_DIR = rag_backend.DATA_DIR
REFUSAL_MESSAGE = rag_backend.REFUSAL_MESSAGE
build_knowledge_base = rag_backend.build_knowledge_base
build_knowledge_base_incremental = getattr(
    rag_backend,
    "build_knowledge_base_incremental",
    None,
)
clear_knowledge_base = rag_backend.clear_knowledge_base
generate_answer = rag_backend.generate_answer
generate_learning_content = rag_backend.generate_learning_content
has_relevant_docs = rag_backend.has_relevant_docs
is_knowledge_base_ready = rag_backend.is_knowledge_base_ready
retrieve_docs = rag_backend.retrieve_docs
filter_relevant_docs = rag_backend.filter_relevant_docs
get_data_dir = rag_backend.get_data_dir
get_index_storage_path = rag_backend.get_index_storage_path
reload_knowledge_base = rag_backend.reload_knowledge_base


ModelProvider = Literal["Groq", "DeepSeek"]
knowledge_base_locks = tuple(Lock() for _ in range(64))
logger = logging.getLogger(__name__)
KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"^kb-[A-Za-z0-9_-]{16,64}$")
UPLOAD_CHUNK_BYTES = 1024 * 1024


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


MAX_UPLOAD_FILES = positive_int_env("MAX_UPLOAD_FILES", 10)
MAX_UPLOAD_FILE_BYTES = positive_int_env(
    "MAX_UPLOAD_FILE_BYTES",
    15 * 1024 * 1024,
)
MAX_UPLOAD_TOTAL_BYTES = positive_int_env(
    "MAX_UPLOAD_TOTAL_BYTES",
    40 * 1024 * 1024,
)
MAX_PDF_PAGES = positive_int_env("MAX_PDF_PAGES", 200)
MAX_INDEX_SNAPSHOT_BYTES = positive_int_env(
    "MAX_INDEX_SNAPSHOT_BYTES",
    512 * 1024 * 1024,
)
TASK_INPUT_RETENTION_SECONDS = positive_int_env(
    "TASK_INPUT_RETENTION_SECONDS",
    86400,
)
TASK_STALLED_SECONDS = positive_int_env("TASK_STALLED_SECONDS", 600)
PUBLIC_VERSION_SYNC_INTERVAL_SECONDS = positive_int_env(
    "PUBLIC_VERSION_SYNC_INTERVAL_SECONDS",
    5,
)
PUBLIC_VERSION_EVENT_CHANNEL = os.getenv(
    "PUBLIC_VERSION_EVENT_CHANNEL",
    "industrial-knowledge-rag:public-version-changed",
).strip() or "industrial-knowledge-rag:public-version-changed"
RATE_LIMITS = {
    "health": (positive_int_env("HEALTH_RATE_LIMIT", 120), 60),
    "ask": (positive_int_env("ASK_RATE_LIMIT", 30), 60),
    "study": (positive_int_env("STUDY_RATE_LIMIT", 10), 3600),
    "upload": (positive_int_env("UPLOAD_RATE_LIMIT", 5), 3600),
    "reset": (positive_int_env("RESET_RATE_LIMIT", 10), 3600),
    "publish": (positive_int_env("PUBLISH_RATE_LIMIT", 5), 3600),
    "versions": (positive_int_env("VERSION_LIST_RATE_LIMIT", 30), 3600),
    "rollback": (positive_int_env("ROLLBACK_RATE_LIMIT", 5), 3600),
    "jobs": (positive_int_env("JOB_STATUS_RATE_LIMIT", 120), 60),
    "job_retry": (positive_int_env("JOB_RETRY_RATE_LIMIT", 10), 3600),
}
MANAGEMENT_RATE_LIMIT_BUCKETS = {
    "upload",
    "reset",
    "publish",
    "versions",
    "rollback",
    "jobs",
    "job_retry",
}
rate_limiter = create_rate_limiter()
PUBLIC_KNOWLEDGE_BASE_ID = os.getenv(
    "PUBLIC_KNOWLEDGE_BASE_ID",
    "kb-public-shared-00000001",
).strip()
if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(PUBLIC_KNOWLEDGE_BASE_ID):
    raise RuntimeError("PUBLIC_KNOWLEDGE_BASE_ID 格式无效。")
version_store = create_version_store(Path(__file__).resolve().parent)


def run_knowledge_task(task_type: str, payload: dict, report):
    if __package__:
        from .knowledge_tasks import execute_job
    else:
        from knowledge_tasks import execute_job
    return execute_job(task_type, payload, report)


task_queue = create_task_queue(run_knowledge_task)
ACTIVE_VERSION_MARKER_PATH = (
    DATA_DIR.parent / "runtime_state" / "public_active_version.txt"
)
LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_allowed_origins():
    origins = list(LOCAL_FRONTEND_ORIGINS)
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
    if frontend_origin and frontend_origin not in origins:
        origins.append(frontend_origin)
    return origins


@asynccontextmanager
async def lifespan(_app: FastAPI):
    public_version_synchronizer.start()
    try:
        yield
    finally:
        public_version_synchronizer.stop()


app = FastAPI(
    title="Industrial Knowledge RAG API",
    version="1.1.0",
    description="面向工业技术文档的 RAG 检索、问答与知识辅助后端。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Retry-After",
        "RateLimit-Limit",
        "RateLimit-Remaining",
        "RateLimit-Reset",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-RateLimit-Policy",
        "X-Model-Token-Limit",
        "X-Model-Token-Remaining",
        "X-Model-Token-Reset",
        "X-Model-Tokens-Used",
    ],
)


@app.middleware("http")
async def add_governance_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in getattr(
        request.state,
        "governance_headers",
        {},
    ).items():
        response.headers[name] = str(value)
    return response


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    model_provider: ModelProvider = "Groq"
    top_k: int = Field(default=4, ge=1, le=8)
    retrieval_mode: Literal["lexical", "vector", "hybrid"] | None = None
    conversation_id: str | None = Field(
        default=None,
        max_length=MAX_CONVERSATION_ID_CHARS,
        pattern=CONVERSATION_ID_PATTERN,
    )
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
    )
    context_options: ContextOptions = Field(default_factory=ContextOptions)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = normalize_message_text(value)
        if not normalized:
            raise ValueError("问题不能为空。")
        return normalized


class StudyRequest(BaseModel):
    model_provider: ModelProvider = "Groq"


class SourceItem(BaseModel):
    citation_id: str
    source: str
    page: int | str
    score: float
    content: str
    document_id: str = ""
    document_type: str = ""
    manufacturer: str = ""
    equipment_type: str = ""
    equipment_model: str = ""
    section: str = ""
    subsection: str = ""
    page_start: int | None = None
    page_end: int | None = None
    knowledge_type: str = ""
    error_code: str = ""
    chunk_id: str = ""
    retrieval_source: str = ""
    lexical_rank: int | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    vector_score: float | None = None
    fusion_score: float | None = None
    final_rank: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    is_refused: bool
    conversation_context: ConversationContext | None = None


class UploadResponse(BaseModel):
    page_count: int
    chunk_count: int
    files: list[str]


class VersionActivationResponse(UploadResponse):
    version_id: str
    created_at: str


class KnowledgeBaseVersion(BaseModel):
    version_id: str
    created_at: str
    page_count: int
    chunk_count: int
    files: list[str]
    active: bool
    index_snapshot_ready: bool = False


class VersionListResponse(BaseModel):
    versions: list[KnowledgeBaseVersion]


class JobSubmissionResponse(BaseModel):
    job_id: str
    task_type: str
    status: str
    progress: int
    message: str
    trace_id: str


class JobStatusResponse(JobSubmissionResponse):
    stage: str = ""
    failed_stage: str = ""
    error: str = ""
    result: dict | None = None
    attempt: int = 1
    retry_of: str = ""
    created_at: str
    updated_at: str
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float | None = None
    is_stalled: bool = False


class TaskCenterResponse(BaseModel):
    jobs: list[JobStatusResponse]
    metrics: dict
    worker: dict


class StudyResponse(BaseModel):
    content: str


def require_knowledge_base_id(
    value: Annotated[
        str,
        Header(alias="X-Knowledge-Base-ID", min_length=19, max_length=67),
    ],
) -> str:
    if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=422,
            detail="X-Knowledge-Base-ID 格式无效。",
        )
    return value


def require_management_token(
    value: Annotated[
        str | None,
        Header(alias="X-Admin-Token"),
    ] = None,
) -> None:
    require_admin_token(value)


def require_idempotency_key(
    value: Annotated[
        str | None,
        Header(alias="Idempotency-Key", max_length=128),
    ] = None,
) -> str:
    key = (value or "").strip()
    if not key:
        return uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", key):
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key 格式无效。",
        )
    return key


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def set_governance_headers(request: Request, values: dict) -> None:
    current = getattr(request.state, "governance_headers", {})
    request.state.governance_headers = {**current, **values}


def limit_headers(decision) -> dict:
    return {
        "RateLimit-Limit": decision.limit,
        "RateLimit-Remaining": decision.remaining,
        "RateLimit-Reset": decision.reset_after,
        "X-RateLimit-Limit": decision.limit,
        "X-RateLimit-Remaining": decision.remaining,
        "X-RateLimit-Reset": decision.reset_after,
    }


def enforce_rate_limit(
    request: Request,
    knowledge_base_id: str,
    bucket: str,
) -> None:
    limit, window_seconds = RATE_LIMITS[bucket]
    client_host = request.client.host if request.client else "unknown"
    try:
        decision = rate_limiter.consume_result(
            bucket,
            f"{client_host}:{knowledge_base_id}",
            limit,
            window_seconds,
        )
    except Exception as exc:
        management = bucket in MANAGEMENT_RATE_LIMIT_BUCKETS
        fail_open = bool_env(
            (
                "RATE_LIMIT_MANAGEMENT_FAIL_OPEN"
                if management
                else "RATE_LIMIT_PUBLIC_FAIL_OPEN"
            ),
            not management,
        )
        logger.exception("rate_limit_backend_failed")
        if fail_open:
            set_governance_headers(
                request,
                {"X-RateLimit-Policy": "degraded-open"},
            )
            return
        raise HTTPException(
            status_code=503,
            detail="限流服务暂时不可用。",
            headers={"Retry-After": "5"},
        ) from exc

    headers = limit_headers(decision)
    set_governance_headers(request, headers)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试。",
            headers={
                **{name: str(value) for name, value in headers.items()},
                "Retry-After": str(decision.reset_after),
            },
        )


def model_scope_key(request: Request, knowledge_base_id: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    return f"{client_host}:{knowledge_base_id}"


def set_model_quota_headers(request: Request, state: dict) -> None:
    quota = state.get("quota")
    if not quota:
        return
    set_governance_headers(
        request,
        {
            "X-Model-Token-Limit": quota["limit"],
            "X-Model-Token-Remaining": quota["remaining"],
            "X-Model-Token-Reset": quota["reset_after"],
            "X-Model-Tokens-Used": state.get("used_tokens", 0),
        },
    )


def model_governance_http_error(exc: ModelGovernanceError) -> HTTPException:
    headers = {"Retry-After": str(exc.retry_after)}
    if exc.limit is not None and exc.remaining is not None:
        headers.update(
            {
                "X-Model-Token-Limit": str(exc.limit),
                "X-Model-Token-Remaining": str(exc.remaining),
                "X-Model-Token-Reset": str(
                    exc.quota_reset_after or exc.retry_after
                ),
            }
        )
    return HTTPException(
        status_code=429,
        detail=str(exc),
        headers=headers,
    )


def get_knowledge_base_lock(knowledge_base_id: str) -> Lock:
    return knowledge_base_locks[hash(knowledge_base_id) % len(knowledge_base_locks)]


@contextmanager
def lock_knowledge_bases(*knowledge_base_ids: str):
    unique_locks = {
        id(get_knowledge_base_lock(knowledge_base_id)): get_knowledge_base_lock(
            knowledge_base_id
        )
        for knowledge_base_id in knowledge_base_ids
    }
    locks = sorted(unique_locks.values(), key=id)
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def require_draft_knowledge_base(knowledge_base_id: str) -> None:
    if knowledge_base_id == PUBLIC_KNOWLEDGE_BASE_ID:
        raise HTTPException(
            status_code=409,
            detail="公共知识库为只读，请先构建草稿库再发布。",
        )


def get_knowledge_base_status(knowledge_base_id: str):
    ready = is_knowledge_base_ready(knowledge_base_id)
    data_dir = get_data_dir(knowledge_base_id)
    pdf_count = (
        sum(1 for path in data_dir.iterdir() if path.suffix.lower() == ".pdf")
        if data_dir.exists()
        else 0
    )
    return ready, pdf_count


def sanitize_pdf_filename(filename: str | None):
    safe_name = Path(filename or "").name
    if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError("只支持上传 PDF 文件。")
    return safe_name


def save_validated_uploads(upload_files: list[UploadFile], staging_dir: Path):
    filenames = [sanitize_pdf_filename(upload.filename) for upload in upload_files]
    if len(set(filenames)) != len(filenames):
        raise ValueError("同一次上传中不能包含重名 PDF。")

    saved_paths = []
    total_bytes = 0
    for upload, filename in zip(upload_files, filenames):
        target_path = staging_dir / filename
        file_bytes = 0
        upload.file.seek(0)
        with target_path.open("wb") as target:
            while True:
                chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                file_bytes += len(chunk)
                total_bytes += len(chunk)
                if file_bytes > MAX_UPLOAD_FILE_BYTES:
                    raise ValueError(
                        f"{filename} 超过单文件大小限制 "
                        f"{MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB。"
                    )
                if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                    raise ValueError(
                        "上传文件总大小超过限制 "
                        f"{MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB。"
                    )
                target.write(chunk)

        with target_path.open("rb") as saved_file:
            if saved_file.read(5) != b"%PDF-":
                raise ValueError(f"{filename} 不是有效的 PDF 文件。")
        saved_paths.append(target_path)

    page_count = 0
    for path in saved_paths:
        try:
            page_count += len(PdfReader(str(path)).pages)
        except Exception as exc:
            raise ValueError(f"{path.name} 无法解析为有效 PDF。") from exc
        if page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF 总页数不能超过 {MAX_PDF_PAGES} 页。")
    return saved_paths, filenames


def prepare_draft_task_input(
    upload_files: list[UploadFile],
    knowledge_base_id: str,
    job_id: str,
) -> dict:
    if not upload_files:
        raise ValueError("请至少上传一个 PDF 文件。")
    if len(upload_files) > MAX_UPLOAD_FILES:
        raise ValueError(f"一次最多上传 {MAX_UPLOAD_FILES} 个 PDF 文件。")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = DATA_DIR / f".task-input-{job_id}"
    staging_dir.mkdir(parents=True)
    try:
        saved_paths, filenames = save_validated_uploads(
            upload_files,
            staging_dir,
        )
        bundle = create_pdf_bundle(saved_paths)
        created_at = datetime.now(timezone.utc)
        manifest = {
            "job_id": job_id,
            "knowledge_base_id": knowledge_base_id,
            "files": filenames,
            "sha256": sha256_hex(bundle),
            "size_bytes": len(bundle),
            "created_at": created_at.isoformat(),
            "expires_at": (
                created_at + timedelta(seconds=TASK_INPUT_RETENTION_SECONDS)
            ).isoformat(),
        }
        try:
            version_store.cleanup_expired_task_inputs(
                created_at.isoformat()
            )
        except Exception:
            logger.warning("expired_task_input_cleanup_failed")
        version_store.save_task_input(job_id, manifest, bundle)
        return manifest
    finally:
        remove_storage_path(staging_dir)


def job_submission_response(record: dict) -> JobSubmissionResponse:
    return JobSubmissionResponse(
        job_id=record["job_id"],
        task_type=record["task_type"],
        status=record["status"],
        progress=record["progress"],
        message=record["message"],
        trace_id=record.get("trace_id", ""),
    )


def log_task_submission(record: dict, created: bool) -> None:
    logger.info(
        "knowledge_task_submitted",
        extra={
            "job_id": record["job_id"],
            "task_type": record["task_type"],
            "scope": record.get("scope", ""),
            "trace_id": record.get("trace_id", ""),
            "attempt": record.get("attempt", 1),
            "created": created,
        },
    )


def is_job_stalled(record: dict, worker_health: dict) -> bool:
    if record.get("status") != "running":
        return False
    if not worker_health.get("healthy"):
        return True
    updated_at = record.get("updated_at", "")
    if not updated_at:
        return False
    try:
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)
        ).total_seconds()
    except ValueError:
        return False
    return age > TASK_STALLED_SECONDS


def job_status_response(
    record: dict,
    worker_health: dict | None = None,
) -> JobStatusResponse:
    worker_health = worker_health or {"healthy": True}
    return JobStatusResponse(
        job_id=record["job_id"],
        task_type=record["task_type"],
        status=record["status"],
        progress=record["progress"],
        message=record["message"],
        trace_id=record.get("trace_id", ""),
        stage=record.get("stage", ""),
        failed_stage=record.get("failed_stage", ""),
        error=record.get("error", ""),
        result=record.get("result"),
        attempt=record.get("attempt", 1),
        retry_of=record.get("retry_of", ""),
        created_at=record.get("created_at", ""),
        updated_at=record.get("updated_at", ""),
        started_at=record.get("started_at", ""),
        finished_at=record.get("finished_at", ""),
        duration_seconds=record.get("duration_seconds"),
        is_stalled=is_job_stalled(record, worker_health),
    )


def rebuild_knowledge_base(
    upload_files: list[UploadFile],
    knowledge_base_id: str,
):
    if not upload_files:
        raise ValueError("请至少上传一个 PDF 文件。")
    if len(upload_files) > MAX_UPLOAD_FILES:
        raise ValueError(f"一次最多上传 {MAX_UPLOAD_FILES} 个 PDF 文件。")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = DATA_DIR / f".{knowledge_base_id}.staging-{uuid4().hex}"
    backup_dir = DATA_DIR / f".{knowledge_base_id}.backup-{uuid4().hex}"
    data_dir = get_data_dir(knowledge_base_id)
    staging_dir.mkdir(parents=True)
    backup_created = False

    try:
        saved_paths, filenames = save_validated_uploads(
            upload_files,
            staging_dir,
        )
        if data_dir.exists():
            os.replace(data_dir, backup_dir)
            backup_created = True
        os.replace(staging_dir, data_dir)
        final_paths = [data_dir / path.name for path in saved_paths]
        page_count, chunk_count = build_knowledge_base(
            final_paths,
            knowledge_base_id=knowledge_base_id,
        )
        if backup_created:
            try:
                shutil.rmtree(backup_dir)
            except OSError:
                logger.warning(
                    "knowledge_base_backup_cleanup_failed",
                    extra={"knowledge_base_id": knowledge_base_id},
                )
    except Exception:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if backup_created and backup_dir.exists():
            os.replace(backup_dir, data_dir)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return page_count, chunk_count, filenames


def create_version_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"v-{timestamp}-{uuid4().hex[:8]}"


def remove_storage_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def public_pdf_filenames(source_dir: Path) -> list[str]:
    if not source_dir.exists():
        return []
    return sorted(
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def write_active_version_marker(version_id: str) -> None:
    validate_version_id(version_id)
    ACTIVE_VERSION_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    staging_path = ACTIVE_VERSION_MARKER_PATH.with_name(
        f".{ACTIVE_VERSION_MARKER_PATH.name}-{uuid4().hex}"
    )
    staging_path.write_text(version_id, encoding="utf-8")
    os.replace(staging_path, ACTIVE_VERSION_MARKER_PATH)


def read_active_version_marker() -> str | None:
    if not ACTIVE_VERSION_MARKER_PATH.exists():
        return None
    try:
        return validate_version_id(
            ACTIVE_VERSION_MARKER_PATH.read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError):
        return None


def replace_knowledge_base_from_directory(
    source_dir: Path,
    knowledge_base_id: str,
    after_build=None,
    previous_build_cache: dict | None = None,
):
    filenames = public_pdf_filenames(source_dir)
    if not filenames:
        raise ValueError("知识库版本为空，无法激活。")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = get_data_dir(knowledge_base_id)
    index_path = get_index_storage_path(knowledge_base_id)
    staging_dir = DATA_DIR / f".{knowledge_base_id}.staging-{uuid4().hex}"
    data_backup = DATA_DIR / f".{knowledge_base_id}.backup-{uuid4().hex}"
    index_backup = index_path.with_name(
        f".{index_path.name}.backup-{uuid4().hex}"
    )
    staging_dir.mkdir(parents=True)
    data_backup_created = False
    index_backup_created = False

    try:
        for filename in filenames:
            shutil.copy2(source_dir / filename, staging_dir / filename)
        if target_dir.exists():
            os.replace(target_dir, data_backup)
            data_backup_created = True
        if index_path.exists():
            os.replace(index_path, index_backup)
            index_backup_created = True

        os.replace(staging_dir, target_dir)
        final_paths = [target_dir / filename for filename in filenames]
        build_metadata = {"cache": None, "stats": {}}
        if (
            build_knowledge_base_incremental
            and previous_build_cache is not None
        ):
            (
                page_count,
                chunk_count,
                build_metadata["cache"],
                build_metadata["stats"],
            ) = build_knowledge_base_incremental(
                final_paths,
                knowledge_base_id=knowledge_base_id,
                previous_index_path=(
                    index_backup if index_backup_created else None
                ),
                previous_cache=previous_build_cache,
            )
        else:
            page_count, chunk_count = build_knowledge_base(
                final_paths,
                knowledge_base_id=knowledge_base_id,
            )
        callback_result = (
            after_build(
                page_count,
                chunk_count,
                filenames,
                final_paths,
                build_metadata,
            )
            if after_build
            else None
        )
    except Exception:
        remove_storage_path(target_dir)
        remove_storage_path(index_path)
        if data_backup_created and data_backup.exists():
            os.replace(data_backup, target_dir)
        if index_backup_created and index_backup.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(index_backup, index_path)
        reload_knowledge_base(knowledge_base_id)
        remove_storage_path(staging_dir)
        raise

    for backup in (data_backup, index_backup):
        try:
            remove_storage_path(backup)
        except OSError:
            logger.warning(
                "public_knowledge_base_backup_cleanup_failed",
                extra={"path": str(backup)},
            )
    return page_count, chunk_count, filenames, callback_result


def replace_public_knowledge_base(source_dir: Path, after_build=None):
    return replace_knowledge_base_from_directory(
        source_dir,
        PUBLIC_KNOWLEDGE_BASE_ID,
        after_build=after_build,
    )


def replace_knowledge_base_from_snapshot(
    source_dir: Path,
    knowledge_base_id: str,
    snapshot_metadata: dict,
    snapshot_bundle: bytes,
    *,
    after_activate=None,
):
    filenames = public_pdf_filenames(source_dir)
    if not filenames:
        raise ValueError("知识库版本为空，无法激活。")

    target_dir = get_data_dir(knowledge_base_id)
    index_path = get_index_storage_path(knowledge_base_id)
    staging_dir = DATA_DIR / f".{knowledge_base_id}.staging-{uuid4().hex}"
    staged_index = index_path.with_name(
        f".{index_path.name}.snapshot-{uuid4().hex}"
    )
    data_backup = DATA_DIR / f".{knowledge_base_id}.backup-{uuid4().hex}"
    index_backup = index_path.with_name(
        f".{index_path.name}.backup-{uuid4().hex}"
    )
    staging_dir.mkdir(parents=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    data_backup_created = False
    index_backup_created = False

    try:
        for filename in filenames:
            shutil.copy2(source_dir / filename, staging_dir / filename)
        extract_index_snapshot(
            snapshot_bundle,
            snapshot_metadata,
            staged_index,
            max_total_bytes=MAX_INDEX_SNAPSHOT_BYTES,
        )
        if target_dir.exists():
            os.replace(target_dir, data_backup)
            data_backup_created = True
        if index_path.exists():
            os.replace(index_path, index_backup)
            index_backup_created = True
        os.replace(staging_dir, target_dir)
        os.replace(staged_index, index_path)
        if not reload_knowledge_base(knowledge_base_id):
            raise ValueError("索引快照加载失败。")
        callback_result = after_activate() if after_activate else None
    except Exception:
        remove_storage_path(target_dir)
        remove_storage_path(index_path)
        if data_backup_created and data_backup.exists():
            os.replace(data_backup, target_dir)
        if index_backup_created and index_backup.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(index_backup, index_path)
        reload_knowledge_base(knowledge_base_id)
        remove_storage_path(staging_dir)
        remove_storage_path(staged_index)
        raise

    remove_storage_path(data_backup)
    remove_storage_path(index_backup)
    return filenames, callback_result


def publish_source_directory(
    draft_dir: Path,
    draft_knowledge_base_id: str,
    *,
    prebuilt_snapshot: tuple[dict, bytes] | None = None,
    build_stats: tuple[int, int] | None = None,
):
    filenames = public_pdf_filenames(draft_dir)
    if not filenames:
        raise ValueError("草稿知识库为空，请先上传并构建草稿。")

    version_id = create_version_id()
    created_at = datetime.now(timezone.utc).isoformat()
    bundle = create_pdf_bundle(
        [draft_dir / filename for filename in filenames]
    )

    def persist_version(
        page_count,
        chunk_count,
        built_files,
        _paths,
        _build_metadata,
        snapshot=None,
    ):
        snapshot_metadata, snapshot_bundle = (
            snapshot
            if snapshot
            else create_index_snapshot(
                get_index_storage_path(PUBLIC_KNOWLEDGE_BASE_ID),
                RAG_MODE,
            )
        )
        manifest = {
            "version_id": version_id,
            "created_at": created_at,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "files": built_files,
            "sha256": sha256_hex(bundle),
            "size_bytes": len(bundle),
            "source_draft_id": draft_knowledge_base_id,
            "index_snapshot": snapshot_metadata,
            "index_snapshot_reused": snapshot is not None,
        }
        version_store.save_version(manifest, bundle)
        if hasattr(version_store, "save_version_snapshot"):
            version_store.save_version_snapshot(
                version_id,
                snapshot_bundle,
            )
        version_store.set_active_version(version_id)
        return manifest

    if (
        prebuilt_snapshot
        and build_stats
        and snapshot_is_compatible(prebuilt_snapshot[0], RAG_MODE)
    ):
        page_count, chunk_count = build_stats

        def activate_prebuilt():
            return persist_version(
                page_count,
                chunk_count,
                filenames,
                [],
                {},
                snapshot=prebuilt_snapshot,
            )

        filenames, manifest = replace_knowledge_base_from_snapshot(
            draft_dir,
            PUBLIC_KNOWLEDGE_BASE_ID,
            prebuilt_snapshot[0],
            prebuilt_snapshot[1],
            after_activate=activate_prebuilt,
        )
    else:
        (
            page_count,
            chunk_count,
            filenames,
            manifest,
        ) = replace_public_knowledge_base(
            draft_dir,
            after_build=persist_version,
        )
    try:
        write_active_version_marker(version_id)
    except OSError:
        logger.warning("public_active_version_marker_write_failed")
    return page_count, chunk_count, filenames, manifest


def publish_knowledge_base(draft_knowledge_base_id: str):
    require_draft_knowledge_base(draft_knowledge_base_id)
    return publish_source_directory(
        get_data_dir(draft_knowledge_base_id),
        draft_knowledge_base_id,
    )


def activate_stored_public_version(
    version_id: str,
    *,
    update_remote_pointer: bool = True,
):
    version_id = validate_version_id(version_id)
    manifest, bundle = version_store.load_version(version_id)
    if sha256_hex(bundle) != manifest.get("sha256"):
        raise ValueError("知识库版本完整性校验失败。")
    if len(bundle) != manifest.get("size_bytes"):
        raise ValueError("知识库版本大小校验失败。")

    restore_dir = DATA_DIR / f".version.restore-{uuid4().hex}"

    def activate_pointer(_pages=None, _chunks=None, _files=None, _paths=None, _metadata=None):
        if update_remote_pointer:
            version_store.set_active_version(version_id)

    try:
        extract_pdf_bundle(
            bundle,
            list(manifest.get("files", [])),
            restore_dir,
            max_total_bytes=MAX_UPLOAD_TOTAL_BYTES,
        )
        snapshot_metadata = manifest.get("index_snapshot") or {}
        snapshot_loaded = False
        if snapshot_is_compatible(snapshot_metadata, RAG_MODE):
            try:
                snapshot_bundle = version_store.load_version_snapshot(
                    version_id
                )
                filenames, _ = replace_knowledge_base_from_snapshot(
                    restore_dir,
                    PUBLIC_KNOWLEDGE_BASE_ID,
                    snapshot_metadata,
                    snapshot_bundle,
                    after_activate=activate_pointer,
                )
                page_count = int(manifest["page_count"])
                chunk_count = int(manifest["chunk_count"])
                snapshot_loaded = True
                logger.info(
                    "public_knowledge_base_snapshot_activated",
                    extra={"version_id": version_id},
                )
            except Exception:
                logger.exception(
                    "public_knowledge_base_snapshot_fallback"
                )
        if not snapshot_loaded:
            (
                page_count,
                chunk_count,
                filenames,
                _,
            ) = replace_public_knowledge_base(
                restore_dir,
                after_build=activate_pointer,
            )
    finally:
        remove_storage_path(restore_dir)

    try:
        write_active_version_marker(version_id)
    except OSError:
        logger.warning("public_active_version_marker_write_failed")
    return page_count, chunk_count, filenames, manifest


def restore_active_public_version() -> bool:
    version_id = version_store.get_active_version_id()
    if not version_id:
        return False
    if (
        read_active_version_marker() == version_id
        and is_knowledge_base_ready(PUBLIC_KNOWLEDGE_BASE_ID)
        and public_pdf_filenames(
            get_data_dir(PUBLIC_KNOWLEDGE_BASE_ID)
        )
    ):
        return False
    with get_knowledge_base_lock(PUBLIC_KNOWLEDGE_BASE_ID):
        activate_stored_public_version(
            version_id,
            update_remote_pointer=False,
        )
    logger.info(
        "public_knowledge_base_restored",
        extra={"version_id": version_id},
    )
    return True


def activate_remote_public_version(version_id: str) -> None:
    with task_queue.lock(f"public:{PUBLIC_KNOWLEDGE_BASE_ID}"):
        with get_knowledge_base_lock(PUBLIC_KNOWLEDGE_BASE_ID):
            activate_stored_public_version(
                version_id,
                update_remote_pointer=False,
            )


def public_knowledge_base_is_ready() -> bool:
    return (
        is_knowledge_base_ready(PUBLIC_KNOWLEDGE_BASE_ID)
        and bool(public_pdf_filenames(get_data_dir(PUBLIC_KNOWLEDGE_BASE_ID)))
    )


public_version_synchronizer = PublicVersionSynchronizer(
    get_active_version=lambda: version_store.get_active_version_id(),
    activate_version=activate_remote_public_version,
    is_loaded_ready=public_knowledge_base_is_ready,
    event_source=task_queue,
    event_channel=PUBLIC_VERSION_EVENT_CHANNEL,
    interval_seconds=PUBLIC_VERSION_SYNC_INTERVAL_SECONDS,
    loaded_version_id=read_active_version_marker(),
)


def ensure_public_version_current(knowledge_base_id: str) -> None:
    if knowledge_base_id == PUBLIC_KNOWLEDGE_BASE_ID:
        public_version_synchronizer.ensure_current()


def publish_public_version_event(version_id: str) -> None:
    public_version_synchronizer.mark_loaded(version_id)
    try:
        task_queue.publish_event(
            PUBLIC_VERSION_EVENT_CHANNEL,
            version_id,
        )
    except Exception:
        logger.exception("public_knowledge_base_version_event_publish_failed")


def serialize_sources(docs):
    sources = []
    candidates = getattr(docs, "candidates", [])

    for index, (doc, score) in enumerate(docs, start=1):
        candidate = candidates[index - 1] if index <= len(candidates) else None
        metadata = getattr(doc, "metadata", {}) or {}
        source = Path(str(metadata.get("source", "未知来源"))).name
        page = metadata.get("page", "未知页码")
        if isinstance(page, int):
            page += 1
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if isinstance(page_start, int):
            page_start += 1
        else:
            page_start = None
        if isinstance(page_end, int):
            page_end += 1
        else:
            page_end = None

        sources.append(
            SourceItem(
                citation_id=f"S{index}",
                source=source,
                page=page,
                score=float(score),
                content=doc.page_content,
                document_id=str(metadata.get("document_id", "")),
                document_type=str(metadata.get("document_type", "")),
                manufacturer=str(metadata.get("manufacturer", "")),
                equipment_type=str(metadata.get("equipment_type", "")),
                equipment_model=str(metadata.get("equipment_model", "")),
                section=str(metadata.get("section", "")),
                subsection=str(metadata.get("subsection", "")),
                page_start=page_start,
                page_end=page_end,
                knowledge_type=str(metadata.get("knowledge_type", "")),
                error_code=str(metadata.get("error_code", "")),
                chunk_id=str(metadata.get("chunk_id", "")),
                retrieval_source=(candidate.retrieval_source if candidate else ""),
                lexical_rank=(candidate.lexical_rank if candidate else None),
                vector_rank=(candidate.vector_rank if candidate else None),
                lexical_score=(candidate.lexical_score if candidate else None),
                vector_score=(candidate.vector_score if candidate else None),
                fusion_score=(candidate.fusion_score if candidate else None),
                final_rank=(candidate.final_rank if candidate else None),
            )
        )

    return sources


def create_context_manager(provider: ModelProvider):
    completion = lambda prompt: llm_module.generate_context_text(
        prompt,
        provider=provider,
    )
    return ConversationContextManager(
        summarizer=LlmConversationSummarizer(completion),
        query_rewriter=LlmQueryRewriter(completion),
    )


def generate_conversation_id():
    return f"conversation-{uuid4()}"


def serialize_prompt_history(turns: list[ConversationTurn]):
    return [
        {
            "role": turn.role,
            "content": turn.content,
        }
        for turn in turns
    ]


def run_study_task(
    task_type: str,
    request: StudyRequest,
    knowledge_base_id: str,
    http_request: Request,
):
    scope_token, quota_state = set_model_scope(
        model_scope_key(http_request, knowledge_base_id)
    )
    try:
        ensure_public_version_current(knowledge_base_id)
        with get_knowledge_base_lock(knowledge_base_id):
            content = generate_learning_content(
                task_type,
                provider=request.model_provider,
                knowledge_base_id=knowledge_base_id,
            )
        return StudyResponse(content=content)
    except ModelGovernanceError as exc:
        raise model_governance_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="学习辅助内容生成失败。") from exc
    finally:
        set_model_quota_headers(http_request, quota_state)
        reset_model_scope(scope_token)


@app.get("/health")
def health(
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
):
    enforce_rate_limit(request, knowledge_base_id, "health")
    ensure_public_version_current(knowledge_base_id)
    ready, pdf_count = get_knowledge_base_status(knowledge_base_id)
    response = {
        "status": "ok",
        "knowledge_base_ready": ready,
        "pdf_count": pdf_count,
    }
    if RAG_MODE_FALLBACK_REASON:
        response["retrieval_fallback"] = {
            "requested_rag_mode": REQUESTED_RAG_MODE,
            "effective_rag_mode": EFFECTIVE_RAG_MODE,
            "requested_retrieval_mode": REQUESTED_RETRIEVAL_MODE,
            "effective_retrieval_mode": EFFECTIVE_RETRIEVAL_MODE,
            "reason": RAG_MODE_FALLBACK_REASON,
        }
    if knowledge_base_id == PUBLIC_KNOWLEDGE_BASE_ID:
        sync_status = public_version_synchronizer.status()
        response["version_sync"] = sync_status
        governance = {
            "rate_limit": rate_limiter.health(),
            "model_quota": llm_module.model_governor.health(),
        }
        response["governance"] = governance
        if (
            sync_status["status"] == "degraded"
            or not governance["rate_limit"]["healthy"]
            or not governance["model_quota"]["healthy"]
        ):
            response["status"] = "degraded"
    return response


@app.post(
    "/upload",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def upload(
    request: Request,
    files: list[UploadFile] = File(...),
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
    idempotency_key: str = Depends(require_idempotency_key),
):
    require_draft_knowledge_base(knowledge_base_id)
    enforce_rate_limit(request, knowledge_base_id, "upload")
    job_id = create_job_id()
    try:
        prepare_draft_task_input(files, knowledge_base_id, job_id)
        try:
            record, created = task_queue.submit(
                "build_draft",
                {
                    "knowledge_base_id": knowledge_base_id,
                    "input_job_id": job_id,
                },
                scope=knowledge_base_id,
                idempotency_key=idempotency_key,
                job_id=job_id,
            )
        except Exception:
            version_store.delete_task_input(job_id)
            raise
        if not created:
            version_store.delete_task_input(job_id)
        log_task_submission(record, created)
        return job_submission_response(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="知识库构建失败。") from exc


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_defaults=True,
)
def ask(
    request: AskRequest,
    http_request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
):
    enforce_rate_limit(http_request, knowledge_base_id, "ask")
    ensure_public_version_current(knowledge_base_id)
    scope_token, quota_state = set_model_scope(
        model_scope_key(http_request, knowledge_base_id)
    )
    try:
        conversation_id = request.conversation_id or generate_conversation_id()
        context_result = create_context_manager(
            request.model_provider
        ).process(
            current_question=request.question,
            history=request.history,
            conversation_id=conversation_id,
            options=request.context_options,
        )
        context_metadata = context_result.metadata
        logger.info(
            "conversation_context_processed",
            extra={
                "conversation_id": conversation_id,
                "history_turn_count": context_metadata.history_turn_count,
                "retained_turn_count": context_metadata.retained_turn_count,
                "compressed_turn_count": context_metadata.compressed_turn_count,
                "was_compressed": context_metadata.was_compressed,
                "query_rewrite_status": context_metadata.query_rewrite_status,
                "standalone_query_length": len(
                    context_result.standalone_query
                ),
            },
        )
        with get_knowledge_base_lock(knowledge_base_id):
            retrieval_arguments = {
                "k": request.top_k,
                "knowledge_base_id": knowledge_base_id,
            }
            if request.retrieval_mode:
                retrieval_arguments["retrieval_mode"] = request.retrieval_mode
            docs = retrieve_docs(context_result.standalone_query, **retrieval_arguments)

        docs = filter_relevant_docs(docs)
        sources = serialize_sources(docs)
        if not has_relevant_docs(docs):
            return AskResponse(
                answer=REFUSAL_MESSAGE,
                sources=sources,
                is_refused=True,
                conversation_context=context_metadata,
            )

        if request.history:
            answer = generate_answer(
                request.question,
                docs,
                provider=request.model_provider,
                conversation_summary=context_result.summary or None,
                conversation_history=serialize_prompt_history(
                    context_result.retained_turns
                ),
            )
        else:
            answer = generate_answer(
                request.question,
                docs,
                provider=request.model_provider,
            )
        return AskResponse(
            answer=answer,
            sources=sources,
            is_refused=False,
            conversation_context=context_metadata,
        )
    except ModelGovernanceError as exc:
        raise model_governance_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="问答生成失败。") from exc
    finally:
        set_model_quota_headers(http_request, quota_state)
        reset_model_scope(scope_token)


@app.post("/study/summary", response_model=StudyResponse)
def study_summary(
    request: StudyRequest,
    http_request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
):
    enforce_rate_limit(http_request, knowledge_base_id, "study")
    return run_study_task(
        "summary",
        request,
        knowledge_base_id,
        http_request,
    )


@app.post("/study/knowledge-points", response_model=StudyResponse)
def study_knowledge_points(
    request: StudyRequest,
    http_request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
):
    enforce_rate_limit(http_request, knowledge_base_id, "study")
    return run_study_task(
        "knowledge_points",
        request,
        knowledge_base_id,
        http_request,
    )


@app.post("/study/quiz", response_model=StudyResponse)
def study_quiz(
    request: StudyRequest,
    http_request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
):
    enforce_rate_limit(http_request, knowledge_base_id, "study")
    return run_study_task(
        "review_questions",
        request,
        knowledge_base_id,
        http_request,
    )


@app.post("/reset")
def reset(
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
):
    require_draft_knowledge_base(knowledge_base_id)
    enforce_rate_limit(request, knowledge_base_id, "reset")
    try:
        with task_queue.lock(f"draft:{knowledge_base_id}"):
            with get_knowledge_base_lock(knowledge_base_id):
                version_store.delete_draft(knowledge_base_id)
                clear_knowledge_base(knowledge_base_id)
        return {"message": "知识库已清空。"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="知识库清空失败。") from exc


@app.post(
    "/publish",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def publish(
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
    idempotency_key: str = Depends(require_idempotency_key),
):
    require_draft_knowledge_base(knowledge_base_id)
    enforce_rate_limit(request, knowledge_base_id, "publish")
    try:
        record, created = task_queue.submit(
            "publish",
            {"knowledge_base_id": knowledge_base_id},
            scope=knowledge_base_id,
            idempotency_key=idempotency_key,
        )
        log_task_submission(record, created)
        return job_submission_response(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="公共知识库发布失败。") from exc


@app.get("/versions", response_model=VersionListResponse)
def versions(
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
):
    enforce_rate_limit(request, knowledge_base_id, "versions")
    try:
        active_version_id = version_store.get_active_version_id()
        items = [
            KnowledgeBaseVersion(
                version_id=item["version_id"],
                created_at=item["created_at"],
                page_count=item["page_count"],
                chunk_count=item["chunk_count"],
                files=item["files"],
                active=item["version_id"] == active_version_id,
                index_snapshot_ready=bool(item.get("index_snapshot")),
            )
            for item in version_store.list_versions()
        ]
        return VersionListResponse(versions=items)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="知识库版本历史读取失败。",
        ) from exc


@app.post(
    "/versions/{version_id}/rollback",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def rollback_version(
    version_id: str,
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
    idempotency_key: str = Depends(require_idempotency_key),
):
    enforce_rate_limit(request, knowledge_base_id, "rollback")
    try:
        validate_version_id(version_id)
        record, created = task_queue.submit(
            "rollback",
            {"version_id": version_id},
            scope=knowledge_base_id,
            idempotency_key=idempotency_key,
        )
        log_task_submission(record, created)
        return job_submission_response(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="公共知识库回滚失败。",
        ) from exc


@app.get("/jobs", response_model=TaskCenterResponse)
def jobs(
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
    limit: int = 50,
):
    enforce_rate_limit(request, knowledge_base_id, "jobs")
    try:
        limit = max(1, min(100, limit))
        worker_health = task_queue.health()
        records = task_queue.list(
            scope=knowledge_base_id,
            limit=limit,
        )
        return TaskCenterResponse(
            jobs=[
                job_status_response(record, worker_health)
                for record in records
            ],
            metrics=task_queue.metrics(scope=knowledge_base_id),
            worker=worker_health,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="任务中心读取失败。",
        ) from exc


@app.post(
    "/jobs/{job_id}/retry",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def retry_job(
    job_id: str,
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
    idempotency_key: str = Depends(require_idempotency_key),
):
    enforce_rate_limit(request, knowledge_base_id, "job_retry")
    try:
        validate_job_id(job_id)
        original = task_queue.get(job_id)
        if original.get("scope") != knowledge_base_id:
            raise ValueError("任务不属于当前草稿知识库。")
        record, created = task_queue.retry(
            job_id,
            idempotency_key=idempotency_key,
        )
        logger.info(
            "knowledge_task_retried",
            extra={
                "job_id": record["job_id"],
                "retry_of": job_id,
                "trace_id": record.get("trace_id", ""),
                "created": created,
            },
        )
        return job_submission_response(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="任务重试失败。") from exc


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(
    job_id: str,
    request: Request,
    knowledge_base_id: str = Depends(require_knowledge_base_id),
    _: None = Depends(require_management_token),
):
    enforce_rate_limit(request, knowledge_base_id, "jobs")
    try:
        validate_job_id(job_id)
        record = task_queue.get(job_id)
        if record.get("scope") != knowledge_base_id:
            raise ValueError("任务不属于当前草稿知识库。")
        return job_status_response(record, task_queue.health())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="任务状态读取失败。") from exc
