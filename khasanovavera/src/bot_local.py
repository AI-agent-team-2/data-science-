import os
import re
import sys
from pathlib import Path

# Пути: скрипт в src/, корень проекта — родитель (или Python запущен из корня с путём src/bot_local.py)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_script_dir) not in sys.path and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    import telebot
    from telebot import apihelper
    from dotenv import load_dotenv
except ImportError as e:
    print("Ошибка импорта. Установите зависимости: pip install pyTelegramBotAPI python-dotenv", flush=True)
    print(f"Детали: {e}", flush=True)
    sys.exit(1)

try:
    from rag import ask
except ImportError as e:
    print("Ошибка: не найден модуль rag. Запускайте из корня проекта: python src/bot_local.py", flush=True)
    print("Или из папки src: cd src  затем  python bot_local.py", flush=True)
    print(f"Детали: {e}", flush=True)
    sys.exit(1)

try:
    from web_search import search as web_search, is_available as web_search_available
except ImportError:
    web_search_available = lambda: False
    web_search = lambda q: "Веб-поиск не подключён (модуль web_search не найден)."

try:
    from product_lookup import product_lookup
except ImportError:
    product_lookup = None

# .env ищем в корне проекта
_env_path = _project_root / ".env"
load_dotenv(_env_path)

# Прокси для доступа к api.telegram.org
# (если без прокси/VPN бот не подключается)
# В .env добавьте: TG_PROXY=http://127.0.0.1:ПОРТ или socks5://127.0.0.1:ПОРТ
proxy = os.getenv("TG_PROXY")
if proxy:
    apihelper.proxy = {"https": proxy, "http": proxy}

# Таймауты запросов к Telegram. При ReadTimeout добавьте в .env: TG_PROXY=...
apihelper.CONNECT_TIMEOUT = int(os.getenv("TG_CONNECT_TIMEOUT", "90"))
apihelper.READ_TIMEOUT = int(os.getenv("TG_READ_TIMEOUT", "120"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    print(f"Файл .env: {'найден' if _env_path.exists() else 'НЕ НАЙДЕН'}")
    print("Ожидаемая переменная в .env: BOT_TOKEN=токен_от_BotFather")
    raise SystemExit("В .env задайте BOT_TOKEN.")

bot = telebot.TeleBot(BOT_TOKEN)

# Лимит длины сообщения в Telegram
TG_MAX_MESSAGE_LENGTH = 4096

# Product Lookup: префиксы для поиска по товару/артикулу
PRODUCT_PREFIXES = ("артикул ", "товар ", "product ")

# Запрос считается похожим на артикул: одно слово, буквы/цифры/дефис/подчёркивание (например PEXOWFHF030)
PRODUCT_ARTICLE_PATTERN = re.compile(r"^[A-Za-z0-9\-\._]{2,50}$")


def _looks_like_article(text: str) -> bool:
    """Сообщение — один токен, похожий на артикул (PEXOWFHF030 и т.п.)."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return False
    parts = cleaned.split()
    return len(parts) == 1 and bool(PRODUCT_ARTICLE_PATTERN.match(parts[0]))

# Фразы в ответе RAG, при которых дополняем ответ веб-поиском (в базе не нашли)
_NO_INFO_MARKERS = (
    "в контексте нет",
    "нет информации",
    "не найден",
    "не удалось найти",
    "нет данных",
    "нет сведений",
)


def _rag_suggests_no_info(response: str) -> bool:
    """Ответ RAG указывает, что в базе нет ответа на вопрос."""
    r = response.lower().strip()
    if len(r) < 80:
        return any(m in r for m in _NO_INFO_MARKERS)
    return any(m in r for m in _NO_INFO_MARKERS)


@bot.message_handler(func=lambda message: True)
def handle_rag_message(message):
    try:
        user_text = (message.text or "").strip()
        if not user_text:
            bot.reply_to(message, "Напишите текст сообщения.")
            return

        # Product Lookup: по префиксу или по виду артикула (одно слово типа PEXOWFHF030)
        use_product_lookup = False
        query_for_product = user_text
        for prefix in PRODUCT_PREFIXES:
            if user_text.lower().startswith(prefix):
                query_for_product = user_text[len(prefix):].strip()
                use_product_lookup = bool(query_for_product)
                break
        if not use_product_lookup and product_lookup is not None:
            if _looks_like_article(user_text):
                use_product_lookup = True
                query_for_product = user_text.strip()

        if use_product_lookup and product_lookup is not None:
            print(f"[{message.chat.id}] Product Lookup: {query_for_product!r}")
            bot.send_chat_action(message.chat.id, "typing")
            status_msg = bot.reply_to(message, "Ищу товар…")
            response = product_lookup(query_for_product)
            used_product_lookup = True
        else:
            print(f"[{message.chat.id}] Вопрос: {user_text!r}")
            bot.send_chat_action(message.chat.id, "typing")
            status_msg = bot.reply_to(message, "Обрабатываю запрос…")
            response = ask(user_text)
            used_product_lookup = False
        # Веб-поиск только если спрашивали через RAG (не через product lookup) и в базе не нашли
        if (
            not used_product_lookup
            and _rag_suggests_no_info(response)
            and web_search_available()
        ):
            web_result = web_search(user_text)
            if web_result and "ошибка" not in web_result.lower()[:100]:
                response = web_result
                print(f"[{message.chat.id}] Подставлен ответ из веб-поиска")
        else:
            print(f"[{message.chat.id}] Ответ ({len(response)} символов)")

        if len(response) > TG_MAX_MESSAGE_LENGTH:
            response = response[: TG_MAX_MESSAGE_LENGTH - 50] + "\n\n[…] обрезано"

        try:
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
                text=response,
            )
        except Exception:
            bot.delete_message(message.chat.id, status_msg.message_id)
            bot.reply_to(message, response)
    except Exception as e:
        print(f"[{message.chat.id}] Ошибка: {e}", flush=True)
        try:
            bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")
        except Exception:
            pass


if __name__ == "__main__":
    print("Бот запущен (RAG). Ожидаю сообщения…", flush=True)
    if product_lookup is None:
        print("Внимание: Product Lookup недоступен (модуль product_lookup не загружен). Запросы по артикулу пойдут в общий RAG.", flush=True)
    else:
        print("Product Lookup: включён (артикулы без префикса, например PEXOWFHF030, идут в поиск по товарам).", flush=True)
    bot.polling()
