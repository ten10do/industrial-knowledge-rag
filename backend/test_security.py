from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.security import (
    FixedWindowRateLimiter,
    RedisFixedWindowRateLimiter,
    require_admin_token,
)


class SharedFakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def eval(self, script, _key_count, key, *args):
        if script == RedisFixedWindowRateLimiter.CONSUME_SCRIPT:
            ttl, amount, limit = map(int, args)
            current = self.values.get(key, 0)
            if current + amount > limit:
                return [0, current, self.ttls.get(key, ttl)]
            current += amount
            self.values[key] = current
            self.ttls.setdefault(key, ttl)
            return [1, current, self.ttls[key]]
        if script == RedisFixedWindowRateLimiter.REFUND_SCRIPT:
            amount = int(args[0])
            self.values[key] = max(
                0,
                self.values.get(key, 0) - amount,
            )
            return self.values[key]
        raise AssertionError("unexpected script")

    def scan_iter(self, pattern):
        prefix = pattern.removesuffix("*")
        return [key for key in self.values if key.startswith(prefix)]

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.ttls.pop(key, None)


def test_fixed_window_rate_limiter_returns_retry_after():
    limiter = FixedWindowRateLimiter()

    assert limiter.consume("ask", "client", 2, 60, now=10) is None
    assert limiter.consume("ask", "client", 2, 60, now=11) is None
    assert limiter.consume("ask", "client", 2, 60, now=12) == 48
    assert limiter.consume("ask", "client", 2, 60, now=61) is None


def test_two_redis_limiter_instances_share_the_same_atomic_budget():
    redis = SharedFakeRedis()
    first = RedisFixedWindowRateLimiter("", redis_client=redis)
    second = RedisFixedWindowRateLimiter("", redis_client=redis)

    assert first.consume("ask", "client", 2, 60, now=10) is None
    assert second.consume("ask", "client", 2, 60, now=11) is None

    decision = first.consume_result("ask", "client", 2, 60, now=12)
    assert not decision.allowed
    assert decision.remaining == 0
    assert decision.reset_after == 50


def test_admin_token_fails_closed_and_uses_constant_time_comparison():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(HTTPException) as missing_config:
            require_admin_token("anything")
    assert missing_config.value.status_code == 503

    with patch.dict("os.environ", {"ADMIN_TOKEN": "server-secret"}, clear=True):
        with pytest.raises(HTTPException) as invalid:
            require_admin_token("wrong")
        require_admin_token("server-secret")
    assert invalid.value.status_code == 401
