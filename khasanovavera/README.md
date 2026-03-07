# Telegram-бот с LLM

Telegram-бот, который отвечает на сообщения с помощью языковой модели (LLM). Поддерживаются два режима: облачный API (OpenRouter) и локальный (Ollama) с историей диалога.

## Возможности

- **bot.py** — облачный режим: ответы через [OpenRouter](https://openrouter.ai/) (модель по умолчанию: GLM 4.5 Air).
- **bot_local.py** — локальный режим: ответы через [Ollama](https://ollama.ai/) с учётом последних 5 пар вопрос–ответ в контексте; опционально прокси для доступа к Telegram API.

## Требования

- Python 3.x
- Токен бота от [@BotFather](https://t.me/BotFather)

Для облачного режима — API-ключ OpenRouter.  
Для локального — установленный и запущенный Ollama с нужной моделью (например, `gemma3:4b`).

## Установка

```bash
git clone <url-репозитория>
cd tg_bot
python -m venv myvenv
# Активация venv: Windows — myvenv\Scripts\activate, Linux/macOS — source myvenv/bin/activate
pip install -r requirements.txt
```

## Настройка

Создайте файл `.env` в корне проекта (он в `.gitignore`, в репозиторий не попадёт):

```env
BOT_TOKEN=ваш_токен_от_BotFather
```

**Только для bot.py (OpenRouter):**

```env
OPENROUTER_API_KEY=ваш_ключ_openrouter
```

**Только для bot_local.py (если нужен прокси к Telegram):**

```env
TG_PROXY=http://127.0.0.1:ПОРТ
# или socks5://127.0.0.1:ПОРТ
```

## Запуск

**Облачный режим (OpenRouter):**

```bash
python bot.py
```

**Локальный режим (Ollama + история):**

```bash
# Убедитесь, что Ollama запущен и модель загружена, например:
# ollama run gemma3:4b
python bot_local.py
```

## Структура проекта

| Файл            | Описание                                      |
|-----------------|-----------------------------------------------|
| `bot.py`        | Бот с облачной LLM (OpenRouter), без истории  |
| `bot_local.py`  | Бот с локальной LLM (Ollama), с историей чата |
| `requirements.txt` | Зависимости Python                         |
| `.env`          | Секреты (токен, ключи) — не коммитить         |

## Зависимости

- `telebot` — работа с Telegram Bot API
- `python-dotenv` — загрузка переменных из `.env`
- `langchain-openai` — вызов LLM (совместим с OpenRouter и Ollama)

## Лицензия

MIT (или укажите свою).
