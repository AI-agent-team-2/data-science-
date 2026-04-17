from __future__ import annotations

import os
import sys
import unittest
from importlib import import_module
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


def _fresh_import(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return import_module(module_name)


def _load_api_app():
    os.environ["WEB_API_KEY"] = "test-key"
    os.environ["WEB_ALLOWED_ORIGINS"] = "http://allowed.example"

    _fresh_import("app.config")
    api = _fresh_import("web.api")
    return api.app


class WebApiSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")
        self.client = TestClient(_load_api_app())

    def test_health_is_public(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    def test_chat_requires_api_key(self) -> None:
        resp = self.client.post("/api/chat", json={"message": "hi", "session_id": "s1"})
        self.assertEqual(resp.status_code, 401)

        resp = self.client.post(
            "/api/chat",
            json={"message": "hi", "session_id": "s1"},
            headers={"X-API-Key": "wrong"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_chat_accepts_valid_api_key(self) -> None:
        with patch("web.api.run_agent", return_value="OK"):
            resp = self.client.post(
                "/api/chat",
                json={"message": "hi", "session_id": "s1"},
                headers={"X-API-Key": "test-key"},
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload.get("reply"), "OK")
        self.assertEqual(payload.get("session_id"), "s1")

    def test_history_requires_api_key(self) -> None:
        resp = self.client.get("/api/history", params={"session_id": "s1"})
        self.assertEqual(resp.status_code, 401)

    def test_cors_allows_only_configured_origins(self) -> None:
        # Allowed origin
        resp = self.client.options(
            "/api/chat",
            headers={
                "Origin": "http://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key",
            },
        )
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://allowed.example")

        # Disallowed origin
        resp = self.client.options(
            "/api/chat",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key",
            },
        )
        self.assertNotEqual(resp.headers.get("access-control-allow-origin"), "http://evil.example")


if __name__ == "__main__":
    unittest.main()
