from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.rag.health import get_index_health, require_ready_index


class RagHealthTests(unittest.TestCase):
    @patch("app.rag.health.chromadb.PersistentClient")
    def test_get_index_health_returns_counts(self, mock_client_cls: MagicMock) -> None:
        client = MagicMock()
        chunk_collection = MagicMock()
        chunk_collection.count.return_value = 10
        product_collection = MagicMock()
        product_collection.count.return_value = 3

        def _get_collection(name: str):
            return chunk_collection if name == "sanitary_goods" else product_collection

        client.get_collection.side_effect = _get_collection
        mock_client_cls.return_value = client

        health = get_index_health()

        self.assertEqual(health.chunk_count, 10)
        self.assertEqual(health.product_count, 3)
        self.assertTrue(health.is_ready)

    @patch("app.rag.health.get_index_health")
    def test_require_ready_index_raises_when_counts_are_empty(self, mock_health: MagicMock) -> None:
        mock_health.return_value = MagicMock(
            is_ready=False,
            chunk_collection="sanitary_goods",
            chunk_count=0,
            product_collection="sanitary_goods_products",
            product_count=0,
        )
        with self.assertRaises(RuntimeError):
            require_ready_index()


if __name__ == "__main__":
    unittest.main()
