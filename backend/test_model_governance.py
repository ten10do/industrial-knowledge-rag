from unittest.mock import Mock

import pytest

from backend.model_governance import (
    MemoryConcurrencyLimiter,
    ModelConcurrencyExceeded,
    ModelGovernanceError,
    ModelGovernor,
    RedisConcurrencyLimiter,
)
from backend.security import FixedWindowRateLimiter


def create_governor(
    *,
    daily_limit=100,
    concurrency_limit=1,
    fail_open=False,
    quota_limiter=None,
):
    return ModelGovernor(
        quota_limiter=quota_limiter or FixedWindowRateLimiter(),
        concurrency_limiter=MemoryConcurrencyLimiter(),
        daily_token_limit=daily_limit,
        concurrency_limit=concurrency_limit,
        slot_ttl_seconds=120,
        fail_open=fail_open,
    )


def test_token_reservation_is_reconciled_to_provider_usage():
    governor = create_governor()
    first, _ = governor.begin("user", 80)
    governor.finish(first, actual_tokens=20, succeeded=True)

    second, decision = governor.begin("user", 80)

    assert decision.allowed
    assert decision.remaining == 0
    governor.finish(second, actual_tokens=80, succeeded=True)


def test_concurrency_slot_is_atomic_and_released():
    governor = create_governor()
    first, _ = governor.begin("user", 10)

    with pytest.raises(ModelConcurrencyExceeded):
        governor.begin("user", 10)

    governor.finish(first, actual_tokens=10, succeeded=True)
    second, _ = governor.begin("user", 10)
    governor.finish(second, actual_tokens=10, succeeded=True)


def test_two_redis_concurrency_limiters_share_slots():
    class SharedRedis:
        def __init__(self):
            self.values = {}

        def eval(self, script, _count, key, *args):
            current = self.values.get(key, 0)
            if script == RedisConcurrencyLimiter.ACQUIRE_SCRIPT:
                limit = int(args[0])
                if current >= limit:
                    return 0
                self.values[key] = current + 1
                return 1
            if script == RedisConcurrencyLimiter.RELEASE_SCRIPT:
                if current <= 1:
                    self.values.pop(key, None)
                    return 0
                self.values[key] = current - 1
                return self.values[key]
            raise AssertionError("unexpected script")

    redis = SharedRedis()
    first = RedisConcurrencyLimiter(redis)
    second = RedisConcurrencyLimiter(redis)

    assert first.acquire("user", 1, 120)
    assert not second.acquire("user", 1, 120)
    first.release("user")
    assert second.acquire("user", 1, 120)


def test_governance_backend_failure_obeys_fail_open_policy():
    broken = Mock()
    broken.consume_result.side_effect = RuntimeError("redis unavailable")
    broken.health.return_value = {
        "backend": "redis",
        "healthy": False,
        "last_error": "redis unavailable",
    }

    open_governor = create_governor(
        fail_open=True,
        quota_limiter=broken,
    )
    reservation, decision = open_governor.begin("user", 10)
    assert decision is None
    assert not reservation.quota_reserved
    assert not open_governor.health()["healthy"]

    closed_governor = create_governor(
        fail_open=False,
        quota_limiter=broken,
    )
    with pytest.raises(ModelGovernanceError):
        closed_governor.begin("user", 10)
