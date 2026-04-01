from __future__ import annotations

import unittest

from app.routing import is_domain_query, resolve_source_order, should_prefer_lookup, should_prefer_web


class RoutingSkuTests(unittest.TestCase):
    def test_sku_query_prefers_lookup(self) -> None:
        query = "Что за товар OGBKP 001?"
        self.assertTrue(should_prefer_lookup(query))
        self.assertEqual(resolve_source_order(query), ["lookup", "rag", "web"])

    def test_web_priority_not_triggered_by_sku(self) -> None:
        query = "Какая сейчас цена OSPNC220?"
        self.assertFalse(should_prefer_web(query))
        self.assertEqual(resolve_source_order(query), ["lookup", "rag", "web"])

    def test_sku_is_domain_signal_even_with_noise(self) -> None:
        self.assertTrue(is_domain_query("что это за OXSF-1616"))
        self.assertTrue(is_domain_query("артикул opxa16221"))


if __name__ == "__main__":
    unittest.main()
