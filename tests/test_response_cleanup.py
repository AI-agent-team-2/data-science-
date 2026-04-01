from __future__ import annotations

import unittest

from app.context import _clean_web_text, ensure_sources_block, parse_tool_payload
from app.run_agent import _prepare_user_answer


class ResponseCleanupTests(unittest.TestCase):
    class _FakeToolMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    def test_clean_web_text_removes_truncated_markers(self) -> None:
        value = "Цветовая палитра смещается...[truncated]"
        self.assertEqual(_clean_web_text(value), "Цветовая палитра смещается")

    def test_prepare_user_answer_keeps_urls_unchanged(self) -> None:
        value = "Источники:\n- https://example.com/product/abc123XYZ987"
        self.assertIn("https://example.com/product/abc123XYZ987", _prepare_user_answer(value))

    def test_prepare_user_answer_removes_truncated_marker(self) -> None:
        value = "Текст ответа ...(truncated]"
        self.assertNotIn("truncated", _prepare_user_answer(value).lower())

    def test_ensure_sources_block_overrides_broken_model_block(self) -> None:
        answer = "Ответ.\n\nИсточники:\n- https://bad/link/[secret]"
        fixed = ensure_sources_block(answer, ["https://valid.example/path"])
        self.assertIn("https://valid.example/path", fixed)
        self.assertNotIn("[secret]", fixed)

    def test_parse_tool_payload_accepts_dict(self) -> None:
        payload = {"results": [{"name": "x"}]}
        self.assertEqual(parse_tool_payload(payload), payload)

    def test_parse_tool_payload_accepts_object_with_content(self) -> None:
        raw = self._FakeToolMessage('{"results":[{"name":"ok"}]}')
        parsed = parse_tool_payload(raw)
        self.assertEqual(parsed.get("results", [{}])[0].get("name"), "ok")


if __name__ == "__main__":
    unittest.main()
