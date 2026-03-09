import telebot
from telebot import apihelper
from dotenv import load_dotenv
import os
from collections import deque
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Прокси для доступа к api.telegram.org
# (если без прокси/VPN бот не подключается)
# В .env добавьте: TG_PROXY=http://127.0.0.1:ПОРТ или socks5://127.0.0.1:ПОРТ
proxy = os.getenv('TG_PROXY')
if proxy:
    apihelper.proxy = {'https': proxy, 'http': proxy}

OPENROUTER_BASE = "http://localhost:11434/v1" # "https://openrouter.ai/api/v1"
MODEL_NAME = "gemma3:4b"  # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free

# Сколько последних пар вопрос-ответ хранить в контексте
HISTORY_SIZE = 5

llm = ChatOpenAI(openai_api_key='fake_key',
                 openai_api_base=OPENROUTER_BASE,
                 model_name=MODEL_NAME,
                 )
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# История чатов: chat_id -> deque из (вопрос, ответ), не более HISTORY_SIZE пар
chat_histories: dict[int, deque[tuple[str, str]]] = {}


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        chat_id = message.chat.id
        user_text = (message.text or "").strip()
        if not user_text:
            bot.reply_to(message, "Напишите текст сообщения.")
            return

        # Получаем или создаём историю для этого чата
        if chat_id not in chat_histories:
            chat_histories[chat_id] = deque(maxlen=HISTORY_SIZE)

        history = chat_histories[chat_id]

        # Собираем контекст: последние 5 пар вопрос-ответ + текущий вопрос
        messages = []
        for q, a in history:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        messages.append(HumanMessage(content=user_text))

        print(user_text)
        response = llm.invoke(messages).content
        print(response)

        # Добавляем новую пару в историю (deque сам обрежет до HISTORY_SIZE)
        history.append((user_text, response))

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


# Запуск бота
bot.polling()
