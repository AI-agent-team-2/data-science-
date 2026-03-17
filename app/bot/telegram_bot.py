from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import telebot

# Позволяет запускать файл напрямую как `python app\bot\telegram_bot.py`.
# В этом режиме добавляем корень проекта в sys.path, чтобы импорт `app.*` работал стабильно.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.run_agent import run_agent

# Базовое логирование ошибок бота (без утечки деталей пользователю).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Telegram-бота через токен из .env.
bot = telebot.TeleBot(settings.telegram_token, threaded=True)


@bot.message_handler(commands=["start"])
def start_handler(message):
    # Приветственное сообщение при первом запуске диалога.
    bot.reply_to(
        message,
        "Привет! Я ассистент по сантехническим товарам. Задай вопрос по товару, параметрам или совместимости."
    )


@bot.message_handler(func=lambda _: True)
def text_handler(message):
    # Базовый обработчик любого текстового входа.
    try:
        # Отправляем текст в агент и получаем финальный ответ модели.
        answer = run_agent(message.text, user_id=str(message.from_user.id))
        bot.reply_to(message, answer)
    except Exception as e:
        # Логируем техническую ошибку для разработчика.
        logger.exception("Failed to process Telegram message: %s", e)
        # Пользователю отдаем безопасный текст без внутренних деталей.
        bot.reply_to(
            message,
            "Не удалось обработать запрос. Попробуйте еще раз через минуту.",
        )


if __name__ == "__main__":
    # Запуск бесконечного long-polling цикла Telegram API.
    # Увеличиваем таймаут ожидания новых сообщений (timeout) и интервал между запросами (long_polling_timeout).
    bot.infinity_polling()
