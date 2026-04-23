from __future__ import annotations

import os
import sys
import unittest
from importlib import import_module
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _fresh_import(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return import_module(module_name)


class HistoryStoreOrderTests(unittest.TestCase):
    def test_history_is_loaded_in_stable_chronological_order(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "history.db")
            os.environ["HISTORY_DB_PATH"] = db_path

            _fresh_import("app.config")
            store = _fresh_import("app.history_store")

            store.save_turn("s1", "u1", "a1")
            store.save_turn("s1", "u2", "a2")
            store.save_turn("s1", "u3", "a3")

            messages = store.load_messages("s1")
            self.assertEqual(
                messages,
                [
                    ("human", "u1"),
                    ("ai", "a1"),
                    ("human", "u2"),
                    ("ai", "a2"),
                    ("human", "u3"),
                    ("ai", "a3"),
                ],
            )

            turns = store.load_turns("s1")
            self.assertEqual(len(turns), 3)
            self.assertTrue(turns[0]["timestamp"])
            self.assertEqual([t["user"] for t in turns], ["u1", "u2", "u3"])

    def test_cleanup_is_not_called_on_every_save(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "history.db")
            os.environ["HISTORY_DB_PATH"] = db_path

            _fresh_import("app.config")
            store = _fresh_import("app.history_store")

            with patch.object(store, "_cleanup_old") as mock_cleanup:
                for index in range(49):
                    store.save_turn("s2", f"u{index}", f"a{index}")
                self.assertEqual(mock_cleanup.call_count, 0)

                store.save_turn("s2", "u50", "a50")
                self.assertEqual(mock_cleanup.call_count, 1)


if __name__ == "__main__":
    unittest.main()
