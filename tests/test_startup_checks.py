from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.startup_checks import check_env_vars


class StartupChecksTests(unittest.TestCase):
    def test_accepts_openrouter_key_without_openai_key(self) -> None:
        env = {
            "TELEGRAM_TOKEN": "123:ABC",
            "OPENROUTER_API_KEY": "or-test",
        }
        with patch.dict(os.environ, env, clear=True):
            check_env_vars(for_web=False)

    def test_raises_runtime_error_when_llm_key_missing(self) -> None:
        env = {
            "TELEGRAM_TOKEN": "123:ABC",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as exc:
                check_env_vars(for_web=False)
        self.assertIn("OPENAI_API_KEY or OPENROUTER_API_KEY", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
