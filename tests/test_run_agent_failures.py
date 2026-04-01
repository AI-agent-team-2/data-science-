from __future__ import annotations

import unittest
from unittest.mock import patch

from app.run_agent import InvocationResult, _run_agent_pipeline


class RunAgentFailureTests(unittest.TestCase):
    @patch("app.run_agent._invoke_with_timeout")
    @patch("app.run_agent.build_context")
    @patch("app.run_agent.load_messages", return_value=[])
    def test_model_failure_returns_explicit_internal_error(
        self,
        _mock_history,
        mock_build_context,
        mock_invoke,
    ) -> None:
        mock_build_context.return_value = type(
            "Ctx",
            (),
            {
                "context_text": "[RAG] text=ok",
                "web_urls": [],
                "used_web": False,
                "used_source": "rag",
                "terminal_response": "",
                "failed_sources": [],
                "attempted_sources": ["rag"],
                "source_status_map": {"rag": "used"},
                "fallback_reason": "primary_source_succeeded",
            },
        )()
        mock_invoke.return_value = InvocationResult(
            status="failed",
            error_type="timeout",
            error_message="timeout>45s",
        )

        answer = _run_agent_pipeline(
            {
                "user_text": "Какой срок службы у трубы PE-Xa EVOH ONDO?",
                "session_id": "u1",
                "hashed_user": "u_hash",
                "source_order": ["rag"],
            }
        )

        self.assertIn("временно недоступны", answer)


if __name__ == "__main__":
    unittest.main()
