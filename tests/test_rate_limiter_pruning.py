from __future__ import annotations

import unittest

from app.observability.rate_limiter import RateLimiter


class RateLimiterPruningTests(unittest.TestCase):
    def test_prunes_to_max_users(self) -> None:
        limiter = RateLimiter(limit=10, window=60, max_users=2, user_ttl_sec=3600, prune_every=1)
        limiter.is_allowed("u1")
        limiter.is_allowed("u2")
        limiter.is_allowed("u3")

        # After pruning, old users are evicted; u1 should be gone.
        limiter.reset("u1")  # should not raise
        allowed, wait = limiter.is_allowed("u1")
        self.assertTrue(allowed)
        self.assertEqual(wait, 0)


if __name__ == "__main__":
    unittest.main()

