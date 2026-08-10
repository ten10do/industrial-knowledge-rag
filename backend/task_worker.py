import os


def main():
    try:
        from redis import Redis
        from rq import Queue, Worker
    except ImportError as exc:
        raise RuntimeError("任务 Worker 需要安装 redis 和 rq。") from exc

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL 不能为空。")
    queue_name = os.getenv("TASK_QUEUE_NAME", "knowledge").strip()
    connection = Redis.from_url(redis_url)
    worker = Worker(
        [Queue(queue_name, connection=connection)],
        connection=connection,
    )
    worker.work()


if __name__ == "__main__":
    main()
