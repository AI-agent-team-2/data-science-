# SAN Bot

Telegram-бот по сантехническим товарам на базе LLM + RAG (ChromaDB).

## Что умеет

- отвечает на вопросы пользователей в Telegram;
- использует 3 источника: `product_lookup`, `rag_search`, `web_search`;
- хранит историю диалога в `history.db` (SQLite);
- хранит векторный индекс в `chroma_db`.

## Быстрый старт

### 1) Установка

```bash
python -m venv myenv
# Linux/macOS:
# source myenv/bin/activate
# Windows:
# myenv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Настройка `.env`

```bash
cp .env.example .env
# Windows PowerShell:
# Copy-Item .env.example .env
```

Минимально нужны:

```env
TELEGRAM_TOKEN=
OPENAI_API_KEY=
```

Остальные параметры имеют значения по умолчанию из `app/config.py`.

### 3) Индексация базы знаний

```bash
python -m app.rag.ingest
```

База документов: `data/knowledge_base/*.txt`

### 4) Запуск бота

```bash
python app/bot/telegram_bot.py
```

или

```bash
python -m app.bot.telegram_bot
```

## Проверка качества поиска

```bash
python scripts/retrieval_eval.py --suite all --top-k 6
```

Поддерживаемые наборы:

- `rag`
- `lookup`
- `web`
- `owasp`
- `all`

## Ключевые модули

- `app/run_agent.py` — роутинг и сбор контекста;
- `app/tools/` — инструменты (`product_lookup`, `rag_search`, `web_search`);
- `app/rag/ingest.py` — индексация документов;
- `app/rag/retriever.py` — retrieval из Chroma;
- `app/bot/telegram_bot.py` — Telegram-интерфейс.

## Полезно знать

- `product_lookup` читает каталог из `data/knowledge_base/*.txt`;
- в RAG используется секционное чанкование документов;
- после обновления базы знаний нужно повторить `python -m app.rag.ingest`.

## История изменений

См. [CHANGELOG.md](CHANGELOG.md).
