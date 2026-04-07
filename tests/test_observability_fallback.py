from __future__ import annotations

import unittest
from unittest.mock import patch

import app.observability.langfuse_client as langfuse_client


class ObservabilityFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        langfuse_client._client_init_attempted = False
        langfuse_client._langfuse_client = None
        langfuse_client._callback_handler_class = None

    @patch("app.observability.langfuse_client._is_enabled", return_value=False)
    def test_langfuse_client_is_none_when_disabled(self, _mock_enabled: object) -> None:
        self.assertIsNone(langfuse_client.get_langfuse_client())

    @patch("app.observability.langfuse_client._is_enabled", return_value=False)
    def test_callback_handler_is_none_when_disabled(self, _mock_enabled: object) -> None:
        self.assertIsNone(langfuse_client.get_langchain_callback_handler())


if __name__ == "__main__":
    unittest.main()
