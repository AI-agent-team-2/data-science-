from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import app.bootstrap as bootstrap


class BootstrapTests(unittest.TestCase):
    @patch.dict("os.environ", {"AUTO_INGEST_ON_START": "false"}, clear=False)
    def test_should_run_ingest_false_when_disabled(self) -> None:
        self.assertFalse(bootstrap.should_run_ingest())

    @patch.dict("os.environ", {"AUTO_INGEST_ON_START": "true"}, clear=False)
    def test_should_run_ingest_true_when_enabled(self) -> None:
        self.assertTrue(bootstrap.should_run_ingest())

    @patch("app.bootstrap._collection_count", side_effect=[12, 5])
    @patch("app.bootstrap.chromadb.PersistentClient")
    @patch("app.rag.ingest.main", return_value=0)
    @patch.dict("os.environ", {"AUTO_INGEST_ON_START": "true"}, clear=False)
    def test_ensure_index_ready_runs_ingest_and_verifies_counts(
        self,
        _mock_ingest: MagicMock,
        mock_client: MagicMock,
        _mock_count: MagicMock,
    ) -> None:
        mock_client.return_value = MagicMock()
        bootstrap.ensure_index_ready()
        _mock_ingest.assert_called_once_with()

    @patch("app.rag.ingest.main", return_value=1)
    @patch.dict("os.environ", {"AUTO_INGEST_ON_START": "true"}, clear=False)
    def test_ensure_index_ready_raises_on_ingest_failure(self, _mock_ingest: MagicMock) -> None:
        with self.assertRaises(RuntimeError):
            bootstrap.ensure_index_ready()


if __name__ == "__main__":
    unittest.main()
