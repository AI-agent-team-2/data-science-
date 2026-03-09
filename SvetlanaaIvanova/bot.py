import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"  # LOCAL_API_URL
MODEL_NAME = "z-ai/glm-4.5-air:free"  # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

# Замените 'bot_token' на токен вашего бота, сохранённого в .env
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

# История сообщений для каждого чата: максимум 5 последних пар вопрос–ответ (10 сообщений)
chat_histories = defaultdict(list)


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        chat_id = message.chat.id

        # Берём текущую историю чата
        history = chat_histories[chat_id]

        # Добавляем новое пользовательское сообщение
        user_msg = HumanMessage(content=message.text)
        history.append(user_msg)

        # В контекст отправляем не более 10 последних сообщений (5 пар вопрос–ответ)
        messages_for_llm = history[-10:]

        print(f"[{chat_id}] USER:", message.text)

        # Отправляем историю + новое сообщение в LLM
        response_msg = llm.invoke(messages_for_llm)
        response_text = response_msg.content

        print(f"[{chat_id}] BOT:", response_text)

        # Сохраняем ответ бота в историю
        if isinstance(response_msg, AIMessage):
            history.append(response_msg)
        else:
            history.append(AIMessage(content=response_text))

        # Оставляем только последние 10 сообщений в истории (5 Q&A)
        if len(history) > 10:
            chat_histories[chat_id] = history[-10:]

        # Отвечаем пользователю
        bot.reply_to(message, response_text)

    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


# Запуск бота
bot.polling()