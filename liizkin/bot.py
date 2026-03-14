import telebot
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from prompt import RAG_PROMPT
from ingest import get_retriever

load_dotenv()

# OPENROUTER_BASE = "https://openrouter.ai/api/v1" #LOCAL_API_URL ,
MODEL_NAME = "llama3.1:8b" # "z-ai/glm-4.5-air:free" # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free

llm = ChatOllama(model=MODEL_NAME)
BOT_TOKEN = os.getenv('BOT_TOKEN')

#поиск для ответа на вопрос от пользователя
retriever = get_retriever()

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        text_of_message = message.text
        answer_on_questions_client = retriever.invoke(text_of_message)
        context = "\n\n".join([one_of_the_answer.page_content for one_of_the_answer in answer_on_questions_client] )
        prompt = RAG_PROMPT.format(
        context=context,
        question=text_of_message
    )
        response = llm.invoke(prompt).content

        # Отвечаем бота ответом LLM
        bot.reply_to(message, response)
    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")

# Запуск бота
bot.polling()



