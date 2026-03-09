import telebot
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
load_dotenv()

OPENROUTER_BASE = "http://localhost:11434/v1" #LOCAL_API_URL ,
MODEL_NAME = "gemma3:4b" # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free





llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 openai_api_base = OPENROUTER_BASE,
                 model_name = MODEL_NAME,
)
# Замените 'bot_token' на токен вашего бота, сохранённого в секрет колаба
BOT_TOKEN = os.getenv("BOT_TOKEN")


bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        # Отправляем сообщение в LLM
        print(message.text)
        response = llm.invoke(message.text).content
        print(response)
        # Отвечаем бота ответом LLM
        bot.reply_to(message, response)
    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")

# Запуск бота
bot.polling()