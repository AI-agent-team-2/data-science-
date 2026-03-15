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

# Настройка прокси, если она задана в .env.
if settings.proxy_url:
    # Устанавливаем прокси для библиотеки telebot (через apihelper)
    telebot.apihelper.proxy = {'https': settings.proxy_url, 'http': settings.proxy_url}
    # Также устанавливаем переменные окружения для requests, на случай если apihelper не подхватит
    os.environ['HTTPS_PROXY'] = settings.proxy_url
    os.environ['HTTP_PROXY'] = settings.proxy_url
    logger.info("Using proxy for Telegram: %s", settings.proxy_url)

# Увеличиваем глобальный таймаут сессии для всех запросов к API Telegram.
telebot.apihelper.SESSION_TIME_OUT = 90

# Инициализация Telegram-бота через токен из .env.
bot = telebot.TeleBot(settings.telegram_token, threaded=True)

# Проверяем соединение перед запуском
try:
    logger.info("Checking Telegram connection...")
    user = bot.get_me()
    logger.info("Bot connected successfully: @%s", user.username)
except Exception as e:
    logger.error("Failed to connect to Telegram: %s", e)
    logger.warning("If you are in Russia, please ensure PROXY_URL is set in .env")

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
        # Показываем статус "печать", пока агент думает.
        bot.send_chat_action(message.chat.id, 'typing')
        
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
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
