from __future__ import annotations

import base64
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
from app.observability.rate_limiter import rate_limiter
from app.rag.health import get_index_health
from app.run_agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_VERSION: Final[str] = "2.0.0"
MAX_STATUS_ERROR_LEN: Final[int] = 200
INLINE_KEYBOARD_ROW_WIDTH: Final[int] = 2

WELCOME_TEXT: Final[str] = (
    "Привет! Я ассистент по сантехническим товарам. "
    "Задай вопрос по товару, параметрам или совместимости.\n\n"
    "📸 Также можешь отправить фото детали — я распознаю товар и найду аналоги.\n\n"
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
- 📸 **Отправьте фото детали** — я распознаю товар и найду похожие в каталоге

*Источники информации:*
- База товаров (по артикулам)
- База знаний (характеристики, совместимость)
- Интернет (цены, отзывы, новинки)
"""

GENERIC_PROCESS_ERROR_TEXT: Final[str] = (
    "❌ Не удалось обработать запрос. Попробуйте еще раз через минуту."
)

# Тексты для фото-обработки
PHOTO_RECEIVED_TEXT: Final[str] = "📸 Получил фото. Распознаю товар..."
PHOTO_ERROR_TEXT: Final[str] = (
    "❌ Не удалось распознать товар на фото. Попробуй:\n"
    "- сфотографировать деталь крупнее\n"
    "- сделать фото при хорошем освещении\n"
    "- добавить текстовое описание к фото"
)

# Vision LLM промпт
VISION_SYSTEM_PROMPT: Final[str] = """Ты эксперт по сантехнике. 
Определи, что за товар на фото. 
Отвечай строго в формате:
- ТИП: [что это]
- БРЕНД: [если виден на фото]
- ХАРАКТЕРИСТИКИ: [резьба, диаметр, материал, назначение]
- АРТИКУЛЫ: [если видны на фото]
- ОПИСАНИЕ: [краткое описание в 1-2 предложениях]"""

VISION_USER_PROMPT: Final[str] = "Что это за сантехническое изделие? Определи тип, бренд, характеристики."

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
        if hasattr(health, 'is_ready') and health.is_ready:
            rag_status = f"✅ ({health.chunk_count} чанков, {health.product_count} товаров)"
        else:
            rag_status = f"⚠️ (чанки: {getattr(health, 'chunk_count', 0)}, товары: {getattr(health, 'product_count', 0)})"
    except Exception as exc:
        logger.exception("Не удалось инициализировать RAG-ретривер")
        rag_status = f"❌ ({_safe_error_text(exc)})"

    return (
        "\n📊 *Статус бота*\n\n"
        f"*Версия:* {BOT_VERSION}\n"
        f"*LLM:* {settings.resolved_model_name}\n"
        f"*RAG:* {rag_status}\n"
        f"*Веб-поиск:* {'✅' if settings.enable_web_search else '❌'}\n"
        f"*История:* {'✅' if settings.history_db_path else '❌'}\n"
        f"*Распознавание фото:* ✅ (Vision LLM)\n"
        f"*Rate limiting:* ✅ ({settings.rate_limit_requests} запросов/{settings.rate_limit_window_sec} сек)\n\n"
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
    rate_limiter.reset(str(user_id))


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


def _recognize_photo(image_bytes: bytes) -> str:
    """Отправляет фото в Vision LLM и возвращает распознанное описание."""
    from app.graph import model
    from langchain_core.messages import HumanMessage, SystemMessage

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    response = model.invoke([
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": VISION_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ])

    return response.content.strip()


def _find_similar_products(description: str, limit: int = 3) -> list[dict]:
    """Ищет товары в каталоге по описанию."""
    from app.tools.product_lookup import product_lookup

    result = product_lookup.invoke({"query": description, "limit": limit})
    data = result if isinstance(result, dict) else {}
    return data.get("results", [])


def _format_photo_response(description: str, products: list[dict]) -> str:
    """Форматирует ответ пользователю с результатами распознавания."""
    lines = [f"Распознано:\n{description}\n"]

    if products:
        lines.append("Похожие товары в каталоге:")
        for item in products[:3]:
            name = item.get('name', '')
            brand = item.get('brand', '')
            sku_list = item.get('sku_list', [])
            category = item.get('category', '')

            line = f"\n- {name}"
            if brand:
                line += f" ({brand})"
            if sku_list:
                skus = sku_list[:3]
                line += f"\n  Артикулы: {', '.join(skus)}"
            if category:
                line += f"\n  Категория: {category}"
            lines.append(line)
    else:
        lines.append("❌ Не нашёл похожих товаров в каталоге.")
        lines.append("Попробуй сфотографировать деталь крупнее, с разных ракурсов, или добавь текстовое описание.")

    return "\n".join(lines)


def _handle_text_message(message: Message) -> None:
    """Обрабатывает пользовательский текст и возвращает ответ от агента."""
    text = str(message.text or "").strip()
    if not text:
        bot.reply_to(message, "Пожалуйста, отправьте текстовый запрос.")
        return

    if text.startswith("/"):
        _handle_unknown_command(message)
        return

    user_id = str(message.from_user.id)

    # Rate limiting
    allowed, wait_seconds = rate_limiter.is_allowed(user_id)
    if not allowed:
        bot.reply_to(message, f"⏳ Слишком много запросов. Подождите {wait_seconds} секунд.")
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


# ========== Обработчик фото ==========

@bot.message_handler(content_types=['photo'])
def photo_handler(message: Message) -> None:
    """Обрабатывает фотографии: распознаёт товар через Vision LLM и ищет аналоги."""
    user_id = str(message.from_user.id)

    # Rate limiting
    allowed, wait_seconds = rate_limiter.is_allowed(user_id)
    if not allowed:
        bot.reply_to(message, f"⏳ Слишком много запросов. Подождите {wait_seconds} секунд.")
        return

    try:
        bot.reply_to(message, PHOTO_RECEIVED_TEXT)

        # Получить самое большое фото
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        downloaded = bot.download_file(file_info.file_path)

        # Распознать через Vision LLM
        description = _recognize_photo(downloaded)

        # Найти похожие товары
        products = _find_similar_products(description)

        # Сформировать и отправить ответ
        response_text = _format_photo_response(description, products)
        bot.reply_to(message, response_text)

    except Exception:
        logger.exception("Ошибка при обработке фото")
        bot.reply_to(message, PHOTO_ERROR_TEXT)


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
    logger.info("Распознавание фото: включено (Vision LLM)")
    logger.info("Rate limiting: %s запросов / %s сек", settings.rate_limit_requests, settings.rate_limit_window_sec)
    bot.infinity_polling()
