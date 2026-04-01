from __future__ import annotations

import unittest

from app.context import build_context


class ContextRoutingTests(unittest.TestCase):
    def test_lookup_sku_not_found_stops_fallback_chain(self) -> None:
        calls: list[str] = []

        def invoke_tool(func, payload, op_name, config=None):
            calls.append(op_name)
            if op_name == "tool_lookup":
                return {
                    "query": payload["query"],
                    "count": 0,
                    "results": [],
                    "mode": "sku_not_found",
                    "note": "Точный артикул не найден в базе товаров.",
                }
            raise AssertionError(f"Unexpected fallback call: {op_name}")

        result = build_context(
            query="Что за товар OSPNC220?",
            source_order=["lookup", "rag", "web"],
            invoke_tool=invoke_tool,
        )

        self.assertEqual(calls, ["tool_lookup"])
        self.assertEqual(result.used_source, "lookup")
        self.assertEqual(result.terminal_response, "Точный артикул не найден в базе товаров.")
        self.assertEqual(result.context_text, "")

    def test_first_useful_source_short_circuits_pipeline(self) -> None:
        calls: list[str] = []

        def invoke_tool(func, payload, op_name, config=None):
            calls.append(op_name)
            if op_name == "tool_rag":
                return {
                    "query": payload["query"],
                    "count": 1,
                    "results": [
                        {
                            "text": "Срок службы трубы составляет до 50 лет.",
                            "score": 0.9,
                            "metadata": {"source": "kb.txt", "section": "SPEC", "doc_id": "doc1", "product": "PE-Xa"},
                        }
                    ],
                }
            raise AssertionError(f"Unexpected fallback call: {op_name}")

        result = build_context(
            query="Какой срок службы у трубы PE-Xa EVOH ONDO?",
            source_order=["rag", "lookup", "web"],
            invoke_tool=invoke_tool,
        )

        self.assertEqual(calls, ["tool_rag"])
        self.assertEqual(result.used_source, "rag")
        self.assertTrue(result.context_text)


if __name__ == "__main__":
    unittest.main()
