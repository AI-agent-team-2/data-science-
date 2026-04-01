from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.ingest import ProductRecord, build_sku_index
from app.rag.sku_index import load_sku_index, save_sku_index


class SkuIndexTests(unittest.TestCase):
    def test_build_sku_index_groups_products_by_normalized_sku(self) -> None:
        records = [
            ProductRecord(
                product_id="product:a",
                lookup_text="A",
                metadata={"source": "a.txt", "articles": "OGBKP-001, OXSF-1616", "product": "A"},
            ),
            ProductRecord(
                product_id="product:b",
                lookup_text="B",
                metadata={"source": "b.txt", "articles": "OGBKP 001", "product": "B"},
            ),
        ]

        index = build_sku_index(records)

        self.assertIn("OGBKP001", index)
        self.assertEqual(len(index["OGBKP001"]), 2)
        self.assertIn("OXSF1616", index)

    def test_save_and_load_sku_index_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.rag.sku_index.sku_index_path", return_value=Path(tmpdir) / "sku_index.json"):
                payload = {"OGBKP001": [{"source": "a.txt", "product": "A"}]}
                save_sku_index(payload)
                loaded = load_sku_index()
                self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
