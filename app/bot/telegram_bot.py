from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import telebot
from telebot.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

# Позволяет запускать файл напрямую: `python app\bot\telegram_bot.py`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.history_store import clear_history
from app.observability import hash_user_id
from app.rag.health import get_index_health
from app.rag.retriever import ChromaRetriever
from app.run_agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_VERSION: Final[str] = "2.0.0"
MAX_STATUS_ERROR_LEN: Final[int] = 200
INLINE_KEYBOARD_ROW_WIDTH: Final[int] = 2

WELCOME_TEXT: Final[str] = (
    "Привет! Я ассистент по сантехническим товарам. "
    "Задай вопрос по товару, параметрам или совместимости.\n\n"
    "Используй /help для списка команд."
)

HELP_TEXT: Final[str] = """
🤖 *Ассистент по сантехнике*

*Команды:*
/start - Начать диалог
/help - Показать эту справку
/clear - Очистить историю диалога
/status - Статус бота и версия
/id - Показать информацию о чате

*Как задавать вопросы:*
- По артикулу: `O12345`
- По модели: `смеситель Hansgrohe`
- По цене: `сколько стоит унитаз`
- По новинкам: `новинки 2026`

*Источники информации:*
- База товаров (по артикулам)
- База знаний (характеристики, совместимость)
- Интернет (цены, отзывы, новинки)
"""

GENERIC_PROCESS_ERROR_TEXT: Final[str] = (
    "❌ Не удалось обработать запрос. Попробуйте еще раз через минуту."
)

# Список поддерживаемых команд.
KNOWN_COMMANDS: Final[list[str]] = ["start", "help", "clear", "status", "id"]

bot = telebot.TeleBot(settings.telegram_token, threaded=True)


def _build_help_keyboard() -> InlineKeyboardMarkup:
    """Создает инлайн-клавиатуру для раздела справки."""
    markup = InlineKeyboardMarkup(row_width=INLINE_KEYBOARD_ROW_WIDTH)
    markup.add(
        InlineKeyboardButton("🔍 Поиск в интернете", callback_data="search_web"),
        InlineKeyboardButton("📚 Поиск в базе", callback_data="search_rag"),
        InlineKeyboardButton("🗑 Очистить историю", callback_data="clear_history"),
    )
    return markup


def _safe_error_text(exc: Exception, max_len: int = MAX_STATUS_ERROR_LEN) -> str:
    """Возвращает укороченный текст ошибки для безопасного вывода в Telegram."""
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _format_status_text() -> str:
    """Формирует текст статуса бота с проверкой доступности RAG-хранилища."""
    rag_status: str
    try:
        health = get_index_health()
        if health.is_ready:
            rag_status = f"✅ ({health.chunk_count} чанков, {health.product_count} товаров)"
        else:
            rag_status = (
                f"⚠️ ({health.chunk_collection}={health.chunk_count}, "
                f"{health.product_collection}={health.product_count})"
            )
    except Exception as exc:
        logger.exception("Не удалось инициализировать RAG-ретривер")
        rag_status = f"❌ ({_safe_error_text(exc)})"

    return (
        "\n📊 *Статус бота*\n\n"
        f"*Версия:* {BOT_VERSION}\n"
        f"*LLM:* {settings.resolved_model_name}\n"
        f"*RAG:* {rag_status}\n"
        f"*Веб-поиск:* {'✅' if settings.enable_web_search else '❌'}\n"
        f"*История:* {'✅' if settings.history_db_path else '❌'}\n\n"
        f"*Команды:* {', '.join('/' + cmd for cmd in KNOWN_COMMANDS)}\n"
    )


def _format_id_text(message: Message) -> str:
    """Формирует информационный блок с Chat ID, User ID и Session ID."""
    user_id = message.from_user.id
    chat_id = message.chat.id
    session_id = str(user_id)

    return (
        "\n🆔 *Идентификаторы*\n\n"
        f"*Chat ID:* `{chat_id}`\n"
        f"*User ID:* `{user_id}`\n"
        f"*Session ID:* `{session_id}`\n\n"
        "Эти данные могут понадобиться для отладки.\n"
    )


def _clear_user_history(user_id: int) -> None:
    """Очищает историю диалога пользователя по его идентификатору."""
    clear_history(session_id=str(user_id))


def _handle_unknown_command(message: Message) -> None:
    """Отправляет ответ о неизвестной slash-команде."""
    text = str(message.text or "").strip()
    command = text.split()[0].lower() if text else "/unknown"
    bot.reply_to(
        message,
        f"❌ Неизвестная команда `{command}`.\n"
        "Используйте /help для списка доступных команд.",
        parse_mode="Markdown",
    )


def _send_search_hint(call: CallbackQuery, mode: str) -> None:
    """Отправляет подсказку для веб-поиска или поиска по базе знаний."""
    if mode == "web":
        answer_text = "🔍 Напиши свой запрос для поиска в интернете"
        message_text = "Введи запрос, и я найду информацию в интернете:"
    else:
        answer_text = "📚 Напиши свой запрос для поиска в базе знаний"
        message_text = "Введи запрос, и я найду информацию в базе:"

    bot.answer_callback_query(call.id, answer_text)
    bot.send_message(call.message.chat.id, message_text)


def _handle_text_message(message: Message) -> None:
    """Обрабатывает пользовательский текст и возвращает ответ от агента."""
    text = str(message.text or "").strip()
    if not text:
        bot.reply_to(message, "Пожалуйста, отправьте текстовый запрос.")
        return

    if text.startswith("/"):
        _handle_unknown_command(message)
        return

    session_user_id = str(message.from_user.id)
    logger.debug("Обработка сообщения Telegram для сессии=%s", hash_user_id(session_user_id))
    answer = run_agent(text, user_id=session_user_id)
    bot.reply_to(message, answer)


# ========== Обработчики команд ==========

@bot.message_handler(commands=["start"])
def start_handler(message: Message) -> None:
    """Отправляет приветственное сообщение и краткую инструкцию."""
    bot.reply_to(message, WELCOME_TEXT)


@bot.message_handler(commands=["help"])
def help_handler(message: Message) -> None:
    """Показывает справку и инлайн-кнопки действий."""
    bot.reply_to(message, HELP_TEXT, parse_mode="Markdown", reply_markup=_build_help_keyboard())


@bot.message_handler(commands=["clear"])
def clear_handler(message: Message) -> None:
    """Очищает историю диалога текущего пользователя."""
    _clear_user_history(message.from_user.id)
    bot.reply_to(message, "✅ История диалога очищена!")


@bot.message_handler(commands=["status"])
def status_handler(message: Message) -> None:
    """Показывает текущий статус бота и подключенных подсистем."""
    bot.reply_to(message, _format_status_text(), parse_mode="Markdown")


@bot.message_handler(commands=["id"])
def id_handler(message: Message) -> None:
    """Показывает технические идентификаторы чата и пользователя."""
    bot.reply_to(message, _format_id_text(message), parse_mode="Markdown")


# ========== Обработчик неизвестных slash-команд ==========

@bot.message_handler(func=lambda message: bool(message.text and message.text.startswith('/')))
def unknown_command_handler(message: Message) -> None:
    """Перехватывает неизвестные slash-команды и возвращает подсказку."""
    _handle_unknown_command(message)


# ========== Инлайн-кнопки ==========

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call: CallbackQuery) -> None:
    """Обрабатывает действия из инлайн-клавиатуры справки."""
    if call.data == "search_web":
        _send_search_hint(call, mode="web")
        return

    if call.data == "search_rag":
        _send_search_hint(call, mode="rag")
        return

    if call.data == "clear_history":
        _clear_user_history(call.from_user.id)
        bot.answer_callback_query(call.id, "✅ История очищена")
        bot.edit_message_text(
            "✅ История диалога очищена!",
            call.message.chat.id,
            call.message.message_id,
        )
        return

    bot.answer_callback_query(call.id, "Действие не поддерживается")


# ========== Основной обработчик текста ==========

@bot.message_handler(func=lambda message: True)
def text_handler(message: Message) -> None:
    """Обрабатывает текстовые сообщения, которые не являются командами."""
    try:
        _handle_text_message(message)
    except Exception:
        logger.exception("Не удалось обработать сообщение Telegram")
        bot.reply_to(message, GENERIC_PROCESS_ERROR_TEXT)


if __name__ == "__main__":
    logger.info("Запуск SAN Bot v%s", BOT_VERSION)
    logger.info("Доступные команды: %s", KNOWN_COMMANDS)
    bot.infinity_polling()
