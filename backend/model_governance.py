import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock

if __package__:
    from .security import (
        FixedWindowRateLimiter,
        RedisFixedWindowRateLimiter,
    )
else:
    from security import FixedWindowRateLimiter, RedisFixedWindowRateLimiter


SECONDS_PER_DAY = 86400


class ModelGovernanceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retry_after: int,
        limit: int | None = None,
        remaining: int | None = None,
        quota_reset_after: int | None = None,
    ):
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))
        self.limit = limit
        self.remaining = remaining
        self.quota_reset_after = quota_reset_after


class ModelQuotaExceeded(ModelGovernanceError):
    pass


class ModelConcurrencyExceeded(ModelGovernanceError):
    pass


class MemoryConcurrencyLimiter:
    backend_name = "memory"

    def __init__(self):
        self.lock = Lock()
        self.counts = {}

    def acquire(self, key: str, limit: int, _ttl: int) -> bool:
        with self.lock:
            current = self.counts.get(key, 0)
            if current >= limit:
                return False
            self.counts[key] = current + 1
            return True

    def release(self, key: str) -> None:
        with self.lock:
            current = self.counts.get(key, 0)
            if current <= 1:
                self.counts.pop(key, None)
            else:
                self.counts[key] = current - 1

    def health(self) -> dict:
        return {"backend": self.backend_name, "healthy": True}


class RedisConcurrencyLimiter:
    backend_name = "redis"
    ACQUIRE_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current >= tonumber(ARGV[1]) then return 0 end
current = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""
    RELEASE_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current <= 1 then
  redis.call('DEL', KEYS[1])
  return 0
end
return redis.call('DECR', KEYS[1])
"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self._healthy = True
        self._last_error = ""

    @staticmethod
    def _key(key: str) -> str:
        import hashlib

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"industrial-knowledge-rag:model:concurrency:{digest}"

    def acquire(self, key: str, limit: int, ttl: int) -> bool:
        try:
            acquired = self.redis.eval(
                self.ACQUIRE_SCRIPT,
                1,
                self._key(key),
                limit,
                ttl,
            )
            self._healthy = True
            self._last_error = ""
            return bool(acquired)
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc) or exc.__class__.__name__
            raise

    def release(self, key: str) -> None:
        try:
            self.redis.eval(
                self.RELEASE_SCRIPT,
                1,
                self._key(key),
            )
            self._healthy = True
            self._last_error = ""
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc) or exc.__class__.__name__

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


@dataclass
class ModelReservation:
    scope: str
    reserved_tokens: int
    quota_reserved: bool
    slot_acquired: bool


class ModelGovernor:
    def __init__(
        self,
        *,
        quota_limiter,
        concurrency_limiter,
        daily_token_limit: int,
        concurrency_limit: int,
        slot_ttl_seconds: int,
        fail_open: bool,
    ):
        self.quota_limiter = quota_limiter
        self.concurrency_limiter = concurrency_limiter
        self.daily_token_limit = daily_token_limit
        self.concurrency_limit = concurrency_limit
        self.slot_ttl_seconds = slot_ttl_seconds
        self.fail_open = fail_open
        self._degraded = False
        self._last_error = ""

    def begin(self, scope: str, estimated_tokens: int) -> tuple:
        estimated_tokens = max(1, int(estimated_tokens))
        quota_reserved = False
        slot_acquired = False
        decision = None
        try:
            decision = self.quota_limiter.consume_result(
                "model_daily_tokens",
                scope,
                self.daily_token_limit,
                SECONDS_PER_DAY,
                now=time.time(),
                amount=estimated_tokens,
            )
            if not decision.allowed:
                raise ModelQuotaExceeded(
                    "今日模型 Token 配额已用完。",
                    retry_after=decision.reset_after,
                    limit=decision.limit,
                    remaining=decision.remaining,
                    quota_reset_after=decision.reset_after,
                )
            quota_reserved = True
            slot_acquired = self.concurrency_limiter.acquire(
                scope,
                self.concurrency_limit,
                self.slot_ttl_seconds,
            )
            if not slot_acquired:
                self.quota_limiter.refund(
                    "model_daily_tokens",
                    scope,
                    SECONDS_PER_DAY,
                    estimated_tokens,
                    now=time.time(),
                )
                quota_reserved = False
                raise ModelConcurrencyExceeded(
                    "当前模型调用过多，请稍后重试。",
                    retry_after=1,
                    limit=decision.limit,
                    remaining=min(
                        decision.limit,
                        decision.remaining + estimated_tokens,
                    ),
                    quota_reset_after=decision.reset_after,
                )
            self._degraded = False
            self._last_error = ""
        except ModelGovernanceError:
            raise
        except Exception as exc:
            self._degraded = True
            self._last_error = str(exc) or exc.__class__.__name__
            if not self.fail_open:
                raise ModelGovernanceError(
                    "模型配额服务暂时不可用。",
                    retry_after=5,
                ) from exc

        reservation = ModelReservation(
            scope=scope,
            reserved_tokens=estimated_tokens,
            quota_reserved=quota_reserved,
            slot_acquired=slot_acquired,
        )
        return reservation, decision

    def finish(
        self,
        reservation: ModelReservation,
        *,
        actual_tokens: int | None,
        succeeded: bool,
    ) -> None:
        try:
            if (
                succeeded
                and reservation.quota_reserved
                and actual_tokens is not None
                and actual_tokens < reservation.reserved_tokens
            ):
                self.quota_limiter.refund(
                    "model_daily_tokens",
                    reservation.scope,
                    SECONDS_PER_DAY,
                    reservation.reserved_tokens - max(0, actual_tokens),
                    now=time.time(),
                )
        finally:
            if reservation.slot_acquired:
                self.concurrency_limiter.release(reservation.scope)

    def health(self) -> dict:
        quota_health = self.quota_limiter.health()
        concurrency_health = self.concurrency_limiter.health()
        return {
            "backend": quota_health["backend"],
            "healthy": (
                not self._degraded
                and quota_health.get("healthy", True)
                and concurrency_health.get("healthy", True)
            ),
            "fail_open": self.fail_open,
            "daily_token_limit": self.daily_token_limit,
            "concurrency_limit": self.concurrency_limit,
            "last_error": (
                self._last_error
                or quota_health.get("last_error", "")
                or concurrency_health.get("last_error", "")
            ),
        }


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_model_governor() -> ModelGovernor:
    backend = os.getenv("RATE_LIMIT_BACKEND", "memory").strip().lower()
    if backend == "redis":
        quota_limiter = RedisFixedWindowRateLimiter(
            os.getenv("REDIS_URL", "").strip()
        )
        concurrency_limiter = RedisConcurrencyLimiter(
            quota_limiter.redis
        )
    elif backend == "memory":
        quota_limiter = FixedWindowRateLimiter()
        concurrency_limiter = MemoryConcurrencyLimiter()
    else:
        raise RuntimeError("RATE_LIMIT_BACKEND 只支持 memory 或 redis。")
    return ModelGovernor(
        quota_limiter=quota_limiter,
        concurrency_limiter=concurrency_limiter,
        daily_token_limit=positive_int_env(
            "MODEL_DAILY_TOKEN_LIMIT",
            200000,
        ),
        concurrency_limit=positive_int_env(
            "MODEL_MAX_CONCURRENT_PER_USER",
            2,
        ),
        slot_ttl_seconds=positive_int_env(
            "MODEL_CONCURRENCY_SLOT_TTL_SECONDS",
            180,
        ),
        fail_open=bool_env("RATE_LIMIT_PUBLIC_FAIL_OPEN", True),
    )


model_scope: ContextVar[tuple[str, dict] | None] = ContextVar(
    "model_scope",
    default=None,
)


def set_model_scope(scope: str):
    state = {"used_tokens": 0, "quota": None}
    return model_scope.set((scope, state)), state


def reset_model_scope(token) -> None:
    model_scope.reset(token)


def current_model_scope() -> tuple[str, dict]:
    value = model_scope.get()
    if value is None:
        return "unscoped", {"used_tokens": 0, "quota": None}
    return value
