import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE = "http://localhost:11434/v1"  # LOCAL_API_URL
MODEL_NAME = "gemma3:4b"  # путь к модели

llm = ChatOpenAI(
    openai_api_key="fake_key",
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)


# Замените 'bot_token' на токен вашего бота
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

# user_id -> список пар (вопрос, ответ)
user_histories = defaultdict(list)


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        user_id = message.chat.id
        messages = []

        # Берём последние 5 пар вопрос-ответ из истории пользователя
        history = user_histories.get(user_id, [])
        last_pairs = history[-5:]

        for question, answer in last_pairs:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        # Текущее сообщение пользователя
        messages.append({"role": "user", "content": message.text})

        print(f"User {user_id}: {message.text}")
        response = llm.invoke(messages).content
        print(f"LLM: {response}")

        # Сохраняем новую пару в историю
        history.append((message.text, response))
        user_histories[user_id] = history

        # Отвечаем бота ответом LLM
        bot.reply_to(message, response)
    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


# Запуск бота
bot.polling()