from __future__ import annotations

import unittest

from app.routing import (
    is_noise_query,
    is_domain_query,
    resolve_source_order,
    should_prefer_lookup,
    should_prefer_web,
    should_use_web_source,
)


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

    def test_web_fallback_disabled_for_regular_domain_question(self) -> None:
        query = "Какой срок службы у трубы PE-Xa EVOH ONDO?"
        self.assertFalse(should_prefer_web(query))
        self.assertFalse(should_prefer_lookup(query))
        self.assertEqual(resolve_source_order(query), ["rag", "lookup", "web"])
        self.assertFalse(should_use_web_source(query, web_mode="fallback"))

    def test_web_fallback_allowed_for_explicit_external_question(self) -> None:
        query = "Какие новинки сантехники 2026?"
        self.assertTrue(should_prefer_web(query))
        self.assertTrue(should_use_web_source(query, web_mode="fallback"))

    def test_plain_latin_word_without_digits_is_not_treated_as_sku(self) -> None:
        query = "REVEALPROMPT"
        self.assertFalse(should_prefer_lookup(query))

    def test_single_token_alnum_sku_still_prefers_lookup(self) -> None:
        query = "OMUL1622"
        self.assertTrue(should_prefer_lookup(query))

    def test_hz_phrase_is_treated_as_noise(self) -> None:
        self.assertTrue(is_noise_query("хз что написать"))


if __name__ == "__main__":
    unittest.main()
