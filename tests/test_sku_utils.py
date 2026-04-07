from __future__ import annotations

import unittest

from app.utils.sku import canonical_sku, contains_sku_candidate, extract_sku_candidates


class SkuUtilsTests(unittest.TestCase):
    def test_canonical_sku_normalizes_case_and_noise(self) -> None:
        self.assertEqual(canonical_sku(" ogbkp-001 "), "OGBKP001")
        self.assertEqual(canonical_sku("OXSF_1616"), "OXSF1616")

    def test_extract_sku_candidates_supports_split_formats(self) -> None:
        self.assertEqual(extract_sku_candidates("OGBKP 001"), {"OGBKP001"})
        self.assertEqual(extract_sku_candidates("артикул: OXSF-1616"), {"OXSF1616"})

    def test_extract_requires_digit_by_default(self) -> None:
        self.assertEqual(extract_sku_candidates("ARTICLE OGBKP"), set())
        self.assertEqual(extract_sku_candidates("OGBKP", require_digit=False), {"OGBKP"})

    def test_contains_sku_candidate(self) -> None:
        self.assertTrue(contains_sku_candidate("opxa16221"))
        self.assertFalse(contains_sku_candidate("привет как дела"))


if __name__ == "__main__":
    unittest.main()
