from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.observability.token_usage import TokenBudgetManager


class TokenBudgetManagerPruningTests(unittest.TestCase):
    def _make_settings(self, **overrides):
        base = {
            "max_total_token_budget": 10_000,
            "max_user_token_budget": 1_000,
            "token_budget_warning_threshold": 0.8,
            "token_budget_max_users": 2,
            "token_budget_user_ttl_sec": 3600,
            "token_budget_prune_every": 1,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_prunes_to_max_users_lru(self) -> None:
        mgr = TokenBudgetManager(self._make_settings())
        mgr.update_usage("u1", 1, 0)
        mgr.update_usage("u2", 1, 0)
        # touch u1 so u2 becomes LRU
        mgr.has_budget("u1", estimated_needed=0)
        mgr.update_usage("u3", 1, 0)

        # u2 should be evicted (max_users=2)
        self.assertTrue(mgr.has_budget("u1", estimated_needed=0))
        self.assertTrue(mgr.has_budget("u3", estimated_needed=0))
        self.assertTrue(mgr.get_user_remaining("u2") == 1000)  # treated as new user


if __name__ == "__main__":
    unittest.main()

