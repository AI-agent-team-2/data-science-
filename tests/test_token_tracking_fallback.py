from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID

from app.graph import TokenTrackingCallbackHandler


class TokenTrackingFallbackTests(unittest.TestCase):
    def test_extracts_usage_from_message_response_metadata(self) -> None:
        handler = TokenTrackingCallbackHandler()

        message = MagicMock()
        message.response_metadata = {"token_usage": {"prompt_tokens": 3, "completion_tokens": 7}}
        chunk = MagicMock()
        chunk.message = message
        chunk.generation_info = None
        response = MagicMock()
        response.llm_output = {}
        response.generations = [[chunk]]

        with patch("app.graph.token_manager.update_usage") as mock_update:
            handler.on_llm_end(response, metadata={"user_id": "u1"})
            mock_update.assert_called_once_with("u1", 3, 7)

    def test_fallback_estimate_uses_prompt_from_on_chat_model_start(self) -> None:
        handler = TokenTrackingCallbackHandler()
        run_id = UUID("00000000-0000-0000-0000-000000000001")

        msg = MagicMock()
        msg.content = "abcd" * 10
        handler.on_chat_model_start(serialized={}, messages=[[msg]], run_id=run_id)

        message = MagicMock()
        message.response_metadata = {}
        message.content = "abcd" * 10
        chunk = MagicMock()
        chunk.message = message
        chunk.generation_info = None
        response = MagicMock()
        response.llm_output = {}
        response.generations = [[chunk]]

        with patch("app.graph.token_manager.update_usage") as mock_update:
            handler.on_llm_end(response, metadata={"user_id": "u1"}, run_id=run_id)
            args = mock_update.call_args[0]
            self.assertEqual(args[0], "u1")
            self.assertGreater(args[1], 0)  # prompt estimate
            self.assertGreater(args[2], 0)  # completion estimate


if __name__ == "__main__":
    unittest.main()

