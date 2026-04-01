from __future__ import annotations

import unittest

from app.context import _clean_web_text
from app.run_agent import _prepare_user_answer


class ResponseCleanupTests(unittest.TestCase):
    def test_clean_web_text_removes_truncated_markers(self) -> None:
        value = "Цветовая палитра смещается...[truncated]"
        self.assertEqual(_clean_web_text(value), "Цветовая палитра смещается")

    def test_prepare_user_answer_keeps_urls_unchanged(self) -> None:
        value = "Источники:\n- https://example.com/product/abc123XYZ987"
        self.assertIn("https://example.com/product/abc123XYZ987", _prepare_user_answer(value))

    def test_prepare_user_answer_removes_truncated_marker(self) -> None:
        value = "Текст ответа ...(truncated]"
        self.assertNotIn("truncated", _prepare_user_answer(value).lower())


if __name__ == "__main__":
    unittest.main()
