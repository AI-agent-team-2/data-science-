from __future__ import annotations

import os
import unittest
import sys
import importlib
from unittest.mock import MagicMock, patch


class VisionInvokeConfigTests(unittest.TestCase):
    def test_recognize_photo_passes_config_to_model_invoke(self) -> None:
        try:
            import telebot  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("telebot is not installed")

        os.environ["TELEGRAM_TOKEN"] = "123:ABC"
        os.environ["OPENAI_API_KEY"] = "test-key"

        with patch("telebot.TeleBot") as _mock_bot_cls:
            if "app.bot.telegram_bot" in sys.modules:
                del sys.modules["app.bot.telegram_bot"]
            telegram_bot = importlib.import_module("app.bot.telegram_bot")

        fake_model = MagicMock()
        fake_model.invoke.return_value = MagicMock(content="ok")

        with patch.object(telegram_bot, "prepare_image_for_vision", return_value=b"x"):
            with patch.object(telegram_bot.base64, "b64encode", return_value=b"eA=="):
                with patch.object(telegram_bot, "get_model", return_value=fake_model):
                    with patch.object(telegram_bot, "get_langchain_callback_handler", return_value=None):
                        telegram_bot._recognize_photo(b"img", user_id="u1")

        _args, kwargs = fake_model.invoke.call_args
        self.assertIn("config", kwargs)
        self.assertIsInstance(kwargs["config"], dict)


if __name__ == "__main__":
    unittest.main()
