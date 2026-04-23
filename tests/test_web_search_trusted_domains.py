from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.tools.web_search import filter_by_trusted_domains


class WebSearchTrustedDomainsTests(unittest.TestCase):
    def test_filter_uses_hostname_not_substring(self) -> None:
        items = [
            {"url": "https://trusted.example/product/1", "title": "ok", "snippet": "ok"},
            {"url": "https://trusted.example.evil.com/phish", "title": "bad", "snippet": "bad"},
        ]
        mock_settings = SimpleNamespace(
            web_trusted_domains_enabled=True,
            web_trusted_domains=["trusted.example"],
            web_min_sources=2,
        )
        with patch("app.tools.web_search.settings", mock_settings):
            filtered = filter_by_trusted_domains(items)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["url"], "https://trusted.example/product/1")

    def test_filter_does_not_fallback_to_untrusted_results_when_too_few(self) -> None:
        items = [
            {"url": "https://trusted.example/a", "title": "ok", "snippet": "ok"},
            {"url": "https://evil.example/b", "title": "bad", "snippet": "bad"},
        ]
        mock_settings = SimpleNamespace(
            web_trusted_domains_enabled=True,
            web_trusted_domains=["trusted.example"],
            web_min_sources=2,
        )
        with patch("app.tools.web_search.settings", mock_settings):
            filtered = filter_by_trusted_domains(items)

        self.assertEqual(filtered, [items[0]])


if __name__ == "__main__":
    unittest.main()
