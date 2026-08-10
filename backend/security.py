import hashlib
import math
import os
import secrets
import time
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after: int


class FixedWindowRateLimiter:
    backend_name = "memory"

    def __init__(self):
        self._lock = Lock()
        self._windows = {}

    def consume_result(
        self,
        bucket: str,
        key: str,
        limit: int,
        window_seconds: int,
        now: float | None = None,
        amount: int = 1,
    ) -> LimitDecision:
        current_time = time.monotonic() if now is None else now
        window_id = int(current_time // window_seconds)
        state_key = (bucket, key)

        with self._lock:
            stored_window, count = self._windows.get(
                state_key,
                (window_id, 0),
            )
            if stored_window != window_id:
                stored_window, count = window_id, 0
            allowed = count + amount <= limit
            if allowed:
                count += amount
                self._windows[state_key] = (stored_window, count)
            if len(self._windows) > 4096:
                self._windows = {
                    item_key: value
                    for item_key, value in self._windows.items()
                    if value[0] >= window_id - 1
                }

        window_end = (window_id + 1) * window_seconds
        return LimitDecision(
            allowed=allowed,
            limit=limit,
            remaining=max(0, limit - count),
            reset_after=max(1, math.ceil(window_end - current_time)),
        )

    def consume(
        self,
        bucket: str,
        key: str,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> int | None:
        decision = self.consume_result(
            bucket,
            key,
            limit,
            window_seconds,
            now=now,
        )
        return None if decision.allowed else decision.reset_after

    def refund(
        self,
        bucket: str,
        key: str,
        window_seconds: int,
        amount: int,
        now: float | None = None,
    ) -> None:
        current_time = time.monotonic() if now is None else now
        window_id = int(current_time // window_seconds)
        state_key = (bucket, key)
        with self._lock:
            stored_window, count = self._windows.get(
                state_key,
                (window_id, 0),
            )
            if stored_window == window_id:
                self._windows[state_key] = (
                    stored_window,
                    max(0, count - amount),
                )

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()

    def health(self) -> dict:
        return {
            "backend": self.backend_name,
            "healthy": True,
            "last_error": "",
        }


class RedisFixedWindowRateLimiter:
    backend_name = "redis"
    CONSUME_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl = redis.call('TTL', KEYS[1])
if current + amount > limit then
  if ttl < 1 then ttl = tonumber(ARGV[1]) end
  return {0, current, ttl}
end
current = redis.call('INCRBY', KEYS[1], amount)
if current == amount then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
ttl = redis.call('TTL', KEYS[1])
return {1, current, ttl}
"""
    REFUND_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
local updated = math.max(0, current - amount)
redis.call('SET', KEYS[1], updated, 'KEEPTTL')
return updated
"""

    def __init__(self, redis_url: str, redis_client=None):
        if not redis_url and redis_client is None:
            raise RuntimeError("REDIS_URL 不能为空。")
        if redis_client is None:
            try:
                from redis import Redis
            except ImportError as exc:
                raise RuntimeError("Redis 限流需要安装 redis。") from exc
            redis_client = Redis.from_url(redis_url)
        self.redis = redis_client
        self._healthy = True
        self._last_error = ""

    @staticmethod
    def _window_key(
        bucket: str,
        key: str,
        window_seconds: int,
        now: float,
    ) -> str:
        digest = hashlib.sha256(f"{bucket}:{key}".encode("utf-8")).hexdigest()
        window_id = int(now // window_seconds)
        return f"industrial-knowledge-rag:ratelimit:{digest}:{window_id}"

    def consume_result(
        self,
        bucket: str,
        key: str,
        limit: int,
        window_seconds: int,
        now: float | None = None,
        amount: int = 1,
    ) -> LimitDecision:
        current_time = time.time() if now is None else now
        redis_key = self._window_key(
            bucket,
            key,
            window_seconds,
            current_time,
        )
        window_id = int(current_time // window_seconds)
        reset_after = max(
            1,
            math.ceil(
                (window_id + 1) * window_seconds - current_time
            ),
        )
        try:
            allowed, count, ttl = self.redis.eval(
                self.CONSUME_SCRIPT,
                1,
                redis_key,
                reset_after,
                amount,
                limit,
            )
            self._healthy = True
            self._last_error = ""
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc) or exc.__class__.__name__
            raise
        return LimitDecision(
            allowed=bool(allowed),
            limit=limit,
            remaining=max(0, limit - int(count)),
            reset_after=max(1, int(ttl)),
        )

    def consume(
        self,
        bucket: str,
        key: str,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> int | None:
        decision = self.consume_result(
            bucket,
            key,
            limit,
            window_seconds,
            now=now,
        )
        return None if decision.allowed else decision.reset_after

    def refund(
        self,
        bucket: str,
        key: str,
        window_seconds: int,
        amount: int,
        now: float | None = None,
    ) -> None:
        current_time = time.time() if now is None else now
        redis_key = self._window_key(
            bucket,
            key,
            window_seconds,
            current_time,
        )
        try:
            self.redis.eval(
                self.REFUND_SCRIPT,
                1,
                redis_key,
                amount,
            )
            self._healthy = True
            self._last_error = ""
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc) or exc.__class__.__name__
            raise

    def clear(self) -> None:
        keys = list(self.redis.scan_iter("industrial-knowledge-rag:ratelimit:*"))
        if keys:
            self.redis.delete(*keys)

    def health(self) -> dict:
        try:
            self.redis.ping()
            self._healthy = True
            self._last_error = ""
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc) or exc.__class__.__name__
        return {
            "backend": self.backend_name,
            "healthy": self._healthy,
            "last_error": self._last_error,
        }


def create_rate_limiter():
    backend = os.getenv("RATE_LIMIT_BACKEND", "memory").strip().lower()
    if backend == "memory":
        return FixedWindowRateLimiter()
    if backend == "redis":
        return RedisFixedWindowRateLimiter(
            os.getenv("REDIS_URL", "").strip()
        )
    raise RuntimeError("RATE_LIMIT_BACKEND 只支持 memory 或 redis。")


def require_admin_token(provided_token: str | None) -> None:
    configured_token = os.getenv("ADMIN_TOKEN", "").strip()
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail="服务端尚未配置管理 Token。",
        )
    if not provided_token or not secrets.compare_digest(
        provided_token,
        configured_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="管理 Token 无效。",
            headers={"WWW-Authenticate": "X-Admin-Token"},
        )
