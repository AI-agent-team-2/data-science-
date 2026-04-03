from __future__ import annotations

import unittest

from app.dialog_rules import resolve_followup_reference


class RunAgentFollowupTests(unittest.TestCase):
    def test_followup_query_adds_two_recent_skus_from_history(self) -> None:
        history = [
            ("human", "Что за товар OSPNC220?"),
            ("ai", "Это сервопривод."),
            ("human", "А есть вариант OSPNO220?"),
            ("ai", "Да, есть."),
        ]

        rewritten = resolve_followup_reference("Какой из них нормально закрытый?", history)

        self.assertIn("OSPNO220", rewritten)
        self.assertIn("OSPNC220", rewritten)
        self.assertIn("сравни SKU", rewritten)

    def test_followup_query_infers_nc_no_pair_when_only_one_sku_in_history(self) -> None:
        history = [
            ("human", "Что за товар OSPNC220?"),
            ("ai", "Это сервопривод."),
        ]

        rewritten = resolve_followup_reference("А чем он отличается от второго варианта?", history)

        self.assertIn("OSPNC220", rewritten)
        self.assertIn("OSPNO220", rewritten)


if __name__ == "__main__":
    unittest.main()
