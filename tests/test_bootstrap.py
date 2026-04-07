from __future__ import annotations

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import app.bootstrap as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_should_run_ingest_false_when_mode_never(self) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            mock_mode.return_value = "never"
            self.assertFalse(bootstrap.should_run_ingest())

    def test_should_run_ingest_true_when_mode_always(self) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            mock_mode.return_value = "always"
            self.assertTrue(bootstrap.should_run_ingest())

    def test_should_run_ingest_true_when_index_not_ready(self) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            with patch("app.bootstrap.get_index_health") as mock_health:
                mock_mode.return_value = "if_empty"
                mock_health.return_value = MagicMock(is_ready=False)
                self.assertTrue(bootstrap.should_run_ingest())

    def test_should_run_ingest_false_when_index_ready(self) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            with patch("app.bootstrap.get_index_health") as mock_health:
                mock_mode.return_value = "if_empty"
                mock_health.return_value = MagicMock(is_ready=True)
                self.assertFalse(bootstrap.should_run_ingest())

    @patch("app.bootstrap.get_index_health")
    @patch("app.rag.ingest.main", return_value=0)
    def test_ensure_index_ready_runs_ingest_and_verifies_counts(
        self,
        _mock_ingest: MagicMock,
        mock_health: MagicMock,
    ) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            mock_mode.return_value = "always"
            mock_health.return_value = MagicMock(
                is_ready=True,
                chunk_collection="sanitary_goods",
                chunk_count=12,
                product_collection="sanitary_goods_products",
                product_count=5,
            )
            bootstrap.ensure_index_ready()
            _mock_ingest.assert_called_once_with()

    @patch("app.rag.ingest.main", return_value=1)
    def test_ensure_index_ready_raises_on_ingest_failure(self, _mock_ingest: MagicMock) -> None:
        with patch.object(type(bootstrap.settings), "resolved_startup_index_mode", new_callable=PropertyMock) as mock_mode:
            mock_mode.return_value = "always"
            with self.assertRaises(RuntimeError):
                bootstrap.ensure_index_ready()

    @patch("app.bootstrap.os.execv")
    @patch("app.bootstrap.ensure_index_ready")
    def test_main_ensures_index_and_execs_bot(
        self,
        mock_ensure_index_ready: MagicMock,
        mock_execv: MagicMock,
    ) -> None:
        bootstrap.main()

        mock_ensure_index_ready.assert_called_once_with()
        mock_execv.assert_called_once_with(
            bootstrap.sys.executable,
            [bootstrap.sys.executable, "-m", "app.bot.telegram_bot"],
        )


if __name__ == "__main__":
    unittest.main()
