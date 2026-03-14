# Telegram-бот с RAG и LLM

Telegram-бот с поиском по базе знаний (RAG): ответы строятся по документам из папки `knowledge_base` с помощью локальной LLM (Ollama). Поддерживаются поиск по товарам/артикулам и опциональный веб-поиск, когда в базе нет ответа.

## Возможности

- **RAG (Retrieval-Augmented Generation)** — индексация документов из `knowledge_base/` в ChromaDB, семантический поиск и ответ через Ollama (модель по умолчанию: `gemma3:4b`).
- **Поиск по товарам (Product Lookup)** — запросы по артикулу (например `PEXOWFHF030`) или по префиксу «артикул», «товар», «product» обрабатываются отдельным поиском по каталогам.
- **Веб-поиск** — если RAG не нашёл ответ в базе, бот может подставить результат поиска в интернете (DuckDuckGo; опционально Tavily при наличии API-ключа).
- **Прокси** — опциональная настройка прокси для доступа к Telegram API.

## Требования

- Python 3.x
- Токен бота от [@BotFather](https://t.me/BotFather)
- Установленный и запущенный [Ollama](https://ollama.ai/) с моделью (например `ollama run gemma3:4b`)
- Документы в папке `knowledge_base/` (`.txt` и поддерживаемые форматы из `chunk_strategy`) — при первом запросе строится индекс в `chroma_db_rag/`

## Установка

```bash
git clone <url-репозитория>
cd khasanovavera
python -m venv myvenv
# Активация venv: Windows — myvenv\Scripts\activate, Linux/macOS — source myvenv/bin/activate
pip install -r requirements.txt
```

## Настройка

Создайте файл `.env` в корне проекта (он в `.gitignore`, в репозиторий не попадёт):

```env
BOT_TOKEN=ваш_токен_от_BotFather
```

**Опционально — прокси к Telegram (если без прокси/VPN бот не подключается):**

```env
TG_PROXY=http://127.0.0.1:ПОРТ
# или socks5://127.0.0.1:ПОРТ
```

**Опционально — веб-поиск через Tavily (иначе используется только DuckDuckGo):**

```env
TAVILY_API_KEY=ваш_ключ_tavily
```

## Запуск

1. Убедитесь, что Ollama запущен и модель загружена:
   ```bash
   ollama run gemma3:4b
   ```
2. Запуск бота из корня проекта:
   ```bash
   python run_bot.py
   ```

Бот обрабатывает все текстовые сообщения: общие вопросы — через RAG по базе знаний, запросы по артикулу или с префиксом «артикул»/«товар» — через Product Lookup.

## Структура проекта

| Файл / папка       | Описание |
|--------------------|----------|
| `run_bot.py`       | Точка входа: запуск бота из корня проекта |
| `src/bot_local.py` | Логика Telegram-бота (RAG, Product Lookup, веб-поиск) |
| `src/rag.py`       | RAG: загрузка документов, чанки, ChromaDB, вызов LLM (Ollama) |
| `src/chunk_strategy.py` | Загрузка и чанкование документов из `knowledge_base` |
| `src/product_lookup.py` | Поиск по товарам/артикулам в базе знаний |
| `src/web_search.py`    | Веб-поиск (DuckDuckGo, опционально Tavily) |
| `src/check.py`     | Проверка установленных зависимостей (LangChain, PyTorch, FAISS и др.) |
| `knowledge_base/`  | Исходные документы для RAG (добавьте свои `.txt` и т.п.) |
| `chroma_db_rag/`  | Индекс ChromaDB (создаётся при первом запросе) |
| `requirements.txt`| Зависимости Python |
| `.env`             | Секреты (токен, ключи) — не коммитить |

## Зависимости (основные)

- `telebot` — работа с Telegram Bot API
- `python-dotenv` — загрузка переменных из `.env`
- `langchain-*`, `langgraph` — цепочки и вызов LLM
- `langchain-chroma` — векторное хранилище
- `sentence-transformers`, `faiss-cpu` — эмбеддинги и поиск
- `duckduckgo-search`, `tavily-python` — веб-поиск
- `pypdf`, `pdfplumber` — обработка PDF в базе знаний

## Лицензия

MIT (или укажите свою).
