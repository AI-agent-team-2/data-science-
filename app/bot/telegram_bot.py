from __future__ import annotations

import logging
import sys
from pathlib import Path

import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

# Позволяет запускать файл напрямую как `python app\bot\telegram_bot.py`
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.run_agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(settings.telegram_token, threaded=True)

# Список всех поддерживаемых команд
KNOWN_COMMANDS = ["start", "help", "clear", "status", "id"]


# ========== Обработчики команд ==========

@bot.message_handler(commands=["start"])
def start_handler(message):
    bot.reply_to(
        message,
        "Привет! Я ассистент по сантехническим товарам. Задай вопрос по товару, параметрам или совместимости.\n\n"
        "Используй /help для списка команд."
    )


@bot.message_handler(commands=["help"])
def help_handler(message):
    """Показать помощь"""
    help_text = """
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
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("🔍 Поиск в интернете", callback_data="search_web"),
        InlineKeyboardButton("📚 Поиск в базе", callback_data="search_rag"),
        InlineKeyboardButton("🗑 Очистить историю", callback_data="clear_history")
    )
    bot.reply_to(message, help_text, parse_mode="Markdown", reply_markup=markup)


@bot.message_handler(commands=["clear"])
def clear_handler(message):
    """Очистить историю диалога"""
    from app.history_store import clear_history
    
    user_id = str(message.from_user.id)
    clear_history(session_id=user_id)
    bot.reply_to(message, "✅ История диалога очищена!")


@bot.message_handler(commands=["status"])
def status_handler(message):
    """Показать статус бота"""
    from app.rag.retriever import ChromaRetriever
    
    try:
        retriever = ChromaRetriever()
        chunks_count = retriever.collection.count()
        rag_status = f"✅ ({chunks_count} чанков)"
    except Exception as e:
        rag_status = f"❌ ({e})"
    
    status_text = f"""
📊 *Статус бота*

*Версия:* 2.0.0
*LLM:* {settings.resolved_model_name}
*RAG:* {rag_status}
*Веб-поиск:* {'✅' if settings.enable_web_search else '❌'}
*История:* {'✅' if settings.history_db_path else '❌'}

*Команды:* {', '.join(['/' + cmd for cmd in KNOWN_COMMANDS])}
"""
    bot.reply_to(message, status_text, parse_mode="Markdown")


@bot.message_handler(commands=["id"])
def id_handler(message):
    """Показать идентификаторы"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    session_id = str(user_id)
    
    id_text = f"""
🆔 *Идентификаторы*

*Chat ID:* `{chat_id}`
*User ID:* `{user_id}`
*Session ID:* `{session_id}`

Эти данные могут понадобиться для отладки.
"""
    bot.reply_to(message, id_text, parse_mode="Markdown")


# ========== Обработчик неизвестных slash-команд ==========

@bot.message_handler(func=lambda message: message.text and message.text.startswith('/'))
def unknown_command_handler(message):
    """Перехват неизвестных slash-команд"""
    command = message.text.split()[0].lower()
    bot.reply_to(
        message,
        f"❌ Неизвестная команда `{command}`.\n"
        f"Используйте /help для списка доступных команд.",
        parse_mode="Markdown"
    )


# ========== Инлайн-кнопки ==========

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call: CallbackQuery):
    """Обработчик инлайн-кнопок"""
    if call.data == "search_web":
        bot.answer_callback_query(call.id, "🔍 Напиши свой запрос для поиска в интернете")
        bot.send_message(call.message.chat.id, "Введи запрос, и я найду информацию в интернете:")
    
    elif call.data == "search_rag":
        bot.answer_callback_query(call.id, "📚 Напиши свой запрос для поиска в базе знаний")
        bot.send_message(call.message.chat.id, "Введи запрос, и я найду информацию в базе:")
    
    elif call.data == "clear_history":
        from app.history_store import clear_history
        user_id = str(call.from_user.id)
        clear_history(session_id=user_id)
        bot.answer_callback_query(call.id, "✅ История очищена")
        bot.edit_message_text(
            "✅ История диалога очищена!",
            call.message.chat.id,
            call.message.message_id
        )


# ========== Основной обработчик текста ==========

@bot.message_handler(func=lambda message: True)
def text_handler(message):
    """Обработчик текстовых сообщений (не команд)"""
    try:
        # Безопасность: убеждаемся, что это не команда (на всякий случай)
        if message.text.startswith('/'):
            unknown_command_handler(message)
            return
        
        answer = run_agent(message.text, user_id=str(message.from_user.id))
        bot.reply_to(message, answer)
        
    except Exception as e:
        logger.exception("Failed to process Telegram message: %s", e)
        bot.reply_to(
            message,
            "❌ Не удалось обработать запрос. Попробуйте еще раз через минуту.",
        )


if __name__ == "__main__":
    logger.info("Starting SAN Bot v2.0.0...")
    logger.info(f"Known commands: {KNOWN_COMMANDS}")
    bot.infinity_polling()