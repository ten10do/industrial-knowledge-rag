from pathlib import Path
from uuid import uuid4


def _main_module():
    try:
        from . import main
    except ImportError:
        import main
    return main


def _rq_reporter():
    try:
        from rq import get_current_job
    except ImportError:
        return lambda _progress, _message, _stage=None: None

    job = get_current_job()
    if job is None:
        return lambda _progress, _message, _stage=None: None

    def report(
        progress: int,
        message: str,
        stage: str | None = None,
    ) -> None:
        from .task_queue import utc_now

        values = {
            "progress": max(0, min(99, int(progress))),
            "message": message,
            "updated_at": utc_now(),
        }
        if stage:
            values["stage"] = stage
        job.meta.update(values)
        job.save_meta()

    return report


def _record_rq_error(error: str) -> None:
    try:
        from rq import get_current_job
        from .task_queue import utc_now
    except ImportError:
        return
    job = get_current_job()
    if job is None:
        return
    job.meta.update(
        {
            "error": error,
            "message": "任务执行失败。",
            "failed_stage": job.meta.get("stage", ""),
            "updated_at": utc_now(),
        }
    )
    job.save_meta()


def _verified_artifact(main, manifest: dict, bundle: bytes) -> None:
    if main.sha256_hex(bundle) != manifest.get("sha256"):
        raise ValueError("任务资料完整性校验失败。")
    if len(bundle) != manifest.get("size_bytes"):
        raise ValueError("任务资料大小校验失败。")


def _build_draft(payload: dict, report):
    main = _main_module()
    job_id = payload["input_job_id"]
    knowledge_base_id = payload["knowledge_base_id"]
    restore_dir = main.DATA_DIR / f".draft-task-{uuid4().hex}"
    completed = False
    expired = False
    try:
        previous_build_cache = main.version_store.load_draft_build_cache(
            knowledge_base_id
        )
    except (AttributeError, ValueError):
        previous_build_cache = {}

    try:
        report(10, "正在读取上传资料。", "loading_input")
        manifest, bundle = main.version_store.load_task_input(job_id)
        _verified_artifact(main, manifest, bundle)
        expires_at = manifest.get("expires_at", "")
        if (
            expires_at
            and main.datetime.fromisoformat(expires_at)
            <= main.datetime.now(main.timezone.utc)
        ):
            expired = True
            raise ValueError("失败任务输入已过期，请重新上传 PDF。")
        main.extract_pdf_bundle(
            bundle,
            list(manifest.get("files", [])),
            restore_dir,
            max_total_bytes=main.MAX_UPLOAD_TOTAL_BYTES,
        )
        report(30, "PDF 校验完成，正在构建草稿索引。", "indexing")

        def persist_draft(
            page_count,
            chunk_count,
            files,
            _paths,
            build_metadata,
        ):
            submitted_at = manifest.get("created_at", "")
            try:
                current_manifest, _ = main.version_store.load_draft(
                    knowledge_base_id
                )
            except ValueError:
                current_manifest = None
            if (
                current_manifest
                and current_manifest.get("submitted_at", "") > submitted_at
            ):
                raise ValueError("该构建任务已被更新的上传任务取代。")
            snapshot_metadata, snapshot_bundle = (
                main.create_index_snapshot(
                    main.get_index_storage_path(knowledge_base_id),
                    main.RAG_MODE,
                )
            )
            draft_manifest = {
                "knowledge_base_id": knowledge_base_id,
                "files": files,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "sha256": main.sha256_hex(bundle),
                "size_bytes": len(bundle),
                "ready": True,
                "submitted_at": submitted_at,
                "built_at": main.datetime.now(
                    main.timezone.utc
                ).isoformat(),
                "index_snapshot": snapshot_metadata,
                "incremental_build": build_metadata.get("stats", {}),
            }
            report(85, "索引构建完成，正在保存草稿。", "persisting_draft")
            main.version_store.save_draft(
                knowledge_base_id,
                draft_manifest,
                bundle,
            )
            main.version_store.save_draft_snapshot(
                knowledge_base_id,
                snapshot_bundle,
            )
            main.version_store.save_draft_build_cache(
                knowledge_base_id,
                build_metadata.get("cache") or {},
            )
            return draft_manifest

        with main.task_queue.lock(f"draft:{knowledge_base_id}"):
            with main.get_knowledge_base_lock(knowledge_base_id):
                if previous_build_cache:
                    index_path = main.get_index_storage_path(
                        knowledge_base_id
                    )
                    try:
                        previous_manifest, _ = (
                            main.version_store.load_draft(
                                knowledge_base_id
                            )
                        )
                        snapshot_metadata = previous_manifest[
                            "index_snapshot"
                        ]
                        if not main.snapshot_is_compatible(
                            snapshot_metadata,
                            main.RAG_MODE,
                        ):
                            raise ValueError("草稿索引快照与当前配置不兼容。")
                        snapshot_bundle = (
                            main.version_store.load_draft_snapshot(
                                knowledge_base_id
                            )
                        )
                        main.remove_storage_path(index_path)
                        index_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        main.extract_index_snapshot(
                            snapshot_bundle,
                            snapshot_metadata,
                            index_path,
                            max_total_bytes=(
                                main.MAX_INDEX_SNAPSHOT_BYTES
                            ),
                        )
                        if not main.reload_knowledge_base(
                            knowledge_base_id
                        ):
                            raise ValueError("草稿索引快照加载失败。")
                    except Exception:
                        main.remove_storage_path(index_path)
                        main.logger.exception(
                            "draft_index_snapshot_restore_failed"
                        )
                page_count, chunk_count, files, draft_manifest = (
                    main.replace_knowledge_base_from_directory(
                        restore_dir,
                        knowledge_base_id,
                        after_build=persist_draft,
                        previous_build_cache=previous_build_cache,
                    )
                )
        report(95, "草稿库已保存。", "activating_draft")
        completed = True
        return {
            "page_count": page_count,
            "chunk_count": chunk_count,
            "files": files,
            "incremental_build": draft_manifest.get(
                "incremental_build",
                {},
            ),
        }
    finally:
        try:
            main.remove_storage_path(restore_dir)
        except OSError:
            main.logger.warning("draft_task_restore_cleanup_failed")
        if completed or expired:
            try:
                main.version_store.delete_task_input(job_id)
            except Exception:
                main.logger.warning("draft_task_input_cleanup_failed")


def _publish(payload: dict, report):
    main = _main_module()
    knowledge_base_id = payload["knowledge_base_id"]
    restore_dir = main.DATA_DIR / f".publish-task-{uuid4().hex}"

    try:
        with main.task_queue.lock(f"draft:{knowledge_base_id}"):
            report(10, "正在读取已构建草稿。", "loading_draft")
            manifest, bundle = main.version_store.load_draft(
                knowledge_base_id
            )
            if not manifest.get("ready"):
                raise ValueError("草稿尚未构建完成，不能发布。")
            _verified_artifact(main, manifest, bundle)
            main.extract_pdf_bundle(
                bundle,
                list(manifest.get("files", [])),
                restore_dir,
                max_total_bytes=main.MAX_UPLOAD_TOTAL_BYTES,
            )
            try:
                prebuilt_snapshot = (
                    manifest["index_snapshot"],
                    main.version_store.load_draft_snapshot(
                        knowledge_base_id
                    ),
                )
            except (KeyError, AttributeError, ValueError):
                prebuilt_snapshot = None
            report(30, "正在激活公共知识库索引。", "indexing")
            with main.task_queue.lock(
                f"public:{main.PUBLIC_KNOWLEDGE_BASE_ID}"
            ):
                with main.lock_knowledge_bases(
                    knowledge_base_id,
                    main.PUBLIC_KNOWLEDGE_BASE_ID,
                ):
                    page_count, chunk_count, files, version = (
                        main.publish_source_directory(
                            restore_dir,
                            knowledge_base_id,
                            prebuilt_snapshot=prebuilt_snapshot,
                            build_stats=(
                                int(manifest["page_count"]),
                                int(manifest["chunk_count"]),
                            ),
                        )
                    )
        main.publish_public_version_event(version["version_id"])
        report(95, "公共版本已激活。", "activating_version")
        return {
            "page_count": page_count,
            "chunk_count": chunk_count,
            "files": files,
            "version_id": version["version_id"],
            "created_at": version["created_at"],
            "index_snapshot_reused": bool(
                version.get("index_snapshot_reused")
            ),
        }
    finally:
        try:
            main.remove_storage_path(restore_dir)
        except OSError:
            main.logger.warning("publish_task_restore_cleanup_failed")


def _rollback(payload: dict, report):
    main = _main_module()
    version_id = payload["version_id"]
    report(15, "正在读取历史版本。", "loading_version")
    with main.task_queue.lock(
        f"public:{main.PUBLIC_KNOWLEDGE_BASE_ID}"
    ):
        with main.get_knowledge_base_lock(main.PUBLIC_KNOWLEDGE_BASE_ID):
            report(35, "正在重建历史版本索引。", "indexing")
            page_count, chunk_count, files, version = (
                main.activate_stored_public_version(version_id)
            )
    main.publish_public_version_event(version["version_id"])
    report(95, "历史版本已激活。", "activating_version")
    return {
        "page_count": page_count,
        "chunk_count": chunk_count,
        "files": files,
        "version_id": version["version_id"],
        "created_at": version["created_at"],
    }


TASK_HANDLERS = {
    "build_draft": _build_draft,
    "publish": _publish,
    "rollback": _rollback,
}


def execute_job(task_type: str, payload: dict, report=None):
    reporter = report or _rq_reporter()
    handler = TASK_HANDLERS.get(task_type)
    if handler is None:
        raise ValueError(f"不支持的任务类型：{task_type}")
    try:
        return handler(payload, reporter)
    except Exception as exc:
        _record_rq_error(str(exc) or exc.__class__.__name__)
        raise
