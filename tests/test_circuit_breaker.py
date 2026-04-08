from __future__ import annotations

import unittest

from app.agent.invoke import invoke_with_timeout
from app.resilience.circuit_breaker import CircuitBreaker


class CircuitBreakerTests(unittest.TestCase):
    def test_opens_after_threshold_and_recovers(self) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        breaker = CircuitBreaker(
            name="t",
            failure_threshold=2,
            cooldown_sec=10,
            half_open_success_threshold=1,
            half_open_max_calls=1,
            clock=clock,
        )

        # 2 failures -> OPEN
        token1 = breaker.begin_call()
        breaker.record_failure(token1)
        token2 = breaker.begin_call()
        breaker.record_failure(token2)
        self.assertEqual(breaker.state, "open")

        # Still in cooldown -> reject
        with self.assertRaises(Exception):
            breaker.begin_call()

        # After cooldown -> HALF_OPEN and success -> CLOSED
        now += 11
        token3 = breaker.begin_call()
        self.assertEqual(breaker.state, "half_open")
        breaker.record_success(token3)
        self.assertEqual(breaker.state, "closed")

    def test_invoke_with_timeout_shortcircuits_when_open(self) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        breaker = CircuitBreaker(
            name="t",
            failure_threshold=1,
            cooldown_sec=10,
            clock=clock,
        )

        token = breaker.begin_call()
        breaker.record_failure(token)
        self.assertEqual(breaker.state, "open")

        called = {"n": 0}

        def _fn(arg: int) -> int:
            called["n"] += 1
            return arg + 1

        res = invoke_with_timeout(_fn, 1, timeout_sec=1, breaker=breaker)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.error_type, "circuit_open")
        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()

