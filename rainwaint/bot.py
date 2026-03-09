import os

import telebot
from dotenv import load_dotenv

from rag import answer_with_rag, generate_event


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=["start"])
def handle_start(message):
    text = (
        "Добро пожаловать в гид по вселенной Earth-6160.\n\n"
        "Задавай вопросы о секторах, героях, технологиях и скрытых зонах — "
        "я буду отвечать как Гид Совета Maker'а, опираясь на внутренний путеводитель.\n\n"
        "Команды:\n"
        "/help — показать помощь\n"
        "/event <запрос> — сгенерировать сюжетное событие или сцену"
    )
    bot.reply_to(message, text)


@bot.message_handler(commands=["help"])
def handle_help(message):
    text = (
        "Я — Гид Maker'а по новой Ultimate-вселенной Earth-6160.\n\n"
        "Просто напиши вопрос по лору, и я отвечу на основе путеводителя.\n\n"
        "Дополнительные команды:\n"
        "/event <запрос> — создать событие, сцену или осложнение для игроков "
        "на основе текущего лора (например: /event стычка в подземном городе свободы)."
    )
    bot.reply_to(message, text)


@bot.message_handler(commands=["event"])
def handle_event(message):
    try:
        user_id = message.from_user.id
        # Текст после команды, если есть
        parts = message.text.split(" ", 1)
        if len(parts) < 2 or not parts[1].strip():
            bot.reply_to(
                message,
                "После команды /event добавь короткое описание запроса.\n"
                "Например: /event нападение дронов в африканском секторе.",
            )
            return

        event_request = parts[1].strip()
        print(f"[USER {user_id} /event] {event_request}")

        response_text = generate_event(event_request)
        print(f"[BOT  {user_id} /event] {response_text}")

        bot.reply_to(message, response_text)
    except Exception as e:
        bot.reply_to(message, f"Ошибка при генерации события: {str(e)}. Попробуйте позже.")


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        user_id = message.from_user.id
        print(f"[USER {user_id}] {message.text}")

        response_text = answer_with_rag(message.text)
        print(f"[BOT  {user_id}] {response_text}")

        bot.reply_to(message, response_text)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


bot.polling()