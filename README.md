# SAN Bot

Telegram-бот по сантехническим товарам на базе LLM + RAG (ChromaDB).

## Возможности

- отвечает на вопросы из Telegram;
- использует `product_lookup`, `rag_search`, `web_search`;
- простой роутинг: SKU/товарный запрос -> `product_lookup`, технический/общий запрос -> `rag_search`, если не найдено -> `web_search`;
- хранит векторный индекс в `chroma_db`;
- хранит историю диалога в `history.db` (SQLite).

## Архитектура системы

```text
Telegram user
   |
   v
telegram_bot.py
   |
   v
run_agent.py (router)
   |-- product_lookup (локальный каталог из data/knowledge_base/tp)
   |-- rag_search (Chroma retriever по проиндексированным документам)
   |-- web_search (Tavily/DuckDuckGo для внешних данных)
   |
   v
LLM (app/graph.py + SYSTEM_PROMPT)
   |
   v
Ответ в Telegram
```

- История диалога хранится в `history.db` (SQLite).
- Векторный индекс хранится в `chroma_db`.
- Индексация документов выполняется через `python -m app.rag.ingest`.

## Data Pipeline

```text
data/knowledge_base/*.txt
   |
   v
preprocess_for_rag (clean_text)
   |
   v
chunk_documents (chunk_size/chunk_overlap)
   |
   v
embedding (OpenAI-compatible)
   |
   v
Chroma collection (chroma_db)
   |
   v
rag_search -> top_k chunks -> контекст для LLM
```

- Шаги индексации реализованы в `app/rag/ingest.py`.
- Поиск по векторной базе выполняет `app/rag/retriever.py`.
- Очистка текста перед индексацией выполняется в `app/rag/preprocess_text.py`.

## Требования

- Python 3.13+
- любой терминал (`bash`, `zsh`, `cmd`, PowerShell, IDE terminal)

## Установка

```bash
python -m venv myenv
# Linux/macOS:
# source myenv/bin/activate
# Windows:
# myenv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка `.env`

Скопируйте шаблон:

```bash
cp .env.example .env
# Windows PowerShell:
# Copy-Item .env.example .env
```

Минимальный пример:

```env
TELEGRAM_TOKEN=

MODEL_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=
MODEL_NAME=openai/gpt-4o-mini

EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_BATCH_SIZE=64

CHROMA_PATH=./chroma_db
COLLECTION_NAME=sanitary_goods
TOP_K=6
CHUNK_SIZE=900
CHUNK_OVERLAP=140

HISTORY_DB_PATH=./history.db
HISTORY_MAX_MESSAGES=24
HISTORY_TTL_DAYS=30

TAVILY_API_KEY=
```

## Индексация RAG

Перед первым запуском:

```bash
python -m app.rag.ingest
```

Повторно запускайте индексацию после обновления файлов в `data/knowledge_base`.

## Запуск бота

```bash
python app/bot/telegram_bot.py
```

Альтернатива:

```bash
python -m app.bot.telegram_bot
```

## Ручная проверка

1. Запустите бота и отправьте `/start`.
2. Проверьте SKU-запрос (например артикул товара): должен сработать `product_lookup`.
3. Проверьте технический вопрос по базе знаний (например про совместимость): должен сработать `rag_search`.
4. Проверьте внешний вопрос, которого нет в базе: должен сработать `web_search`.

## Мини-оценка Retrieval

```bash
python scripts/retrieval_eval.py
# или с другим k:
# python scripts/retrieval_eval.py --top-k 8
```

Скрипт печатает `hit@k` по небольшому фиксированному набору запросов.

## Структура

- `app/config.py` - настройки проекта
- `app/run_agent.py` - основная логика ответа и линейный роутинг
- `app/graph.py` - инициализация LLM-клиента
- `app/tools/` - инструменты (`product_lookup`, `rag_search`, `web_search`)
- `app/rag/` - индексация и retrieval
- `data/knowledge_base/` - исходные документы

## Изменения в версии 2.0 

### Обзор
Версия 2.0 значительно улучшает качество ответов бота, добавляет новые функции безопасности и производительности. Основные изменения направлены на повышение релевантности веб-поиска, улучшение пользовательского опыта и добавление инструментов мониторинга.

---

### Веб-поиск

| Было | Стало |
|------|-------|
| Поиск возвращал любые результаты (фильмы, новости, общий ремонт) | Добавлен фильтр `_is_sanitary_relevant`, пропускающий только результаты с ключевыми словами: *унитаз, смеситель, кран, труба, сантехника* |
| Каждый запрос отправлялся в Tavily/DuckDuckGo без кэширования | Добавлен **кэш в памяти** (100 записей, TTL 24 часа), экономящий API вызовы |
| В поиск передавался исходный запрос пользователя | Функция `enhance_search_query` автоматически улучшает запросы: «новинки 2026» → «новинки сантехники 2026 каталог» |

---

### Telegram бот

| Было | Стало |
|------|-------|
| Только команда `/start` | Добавлены команды **`/help`** (справка) и **`/clear`** (очистка истории) |
| Нет интерактивных элементов | Добавлены **инлайн-кнопки**: 🔍 Поиск в интернете, 📚 Поиск в базе, 🗑 Очистить историю |
| Нет подсказок для пользователя | `/help` показывает список команд и примеры запросов |

---

### Безопасность

| Было | Стало |
|------|-------|
| Только защита на уровне SYSTEM_PROMPT | Добавлен **предварительный фильтр запросов** с блокировкой опасных тем |
| Нет защиты от промпт-инъекций | Обнаружение паттернов: *ignore previous, ты теперь, забудь* |

**Заблокированные темы:**
- Взлом, хакинг, обход блокировок
- Фейковые/поддельные документы
- Политика, партии, протесты
- Наркотики, оружие

---

### RAG (поиск по базе знаний)

| Было | Стало |
|------|-------|
| Только векторный поиск в Chroma | Добавлен **гибридный поиск** `hybrid_search` |
| Результаты сортировались только по distance | Ключевые слова из запроса повышают релевантность чанка (бустинг) |
| — | Функция `_extract_keywords` извлекает значимые слова из запроса |

---

### История диалогов (`app/history_store.py`)

| Было | Стало |
|------|-------|
| Только сохранение и загрузка истории | Добавлен метод **`clear_history`** для очистки истории пользователя |
| — | Добавлена функция **`get_history_stats`** для получения статистики |

---

### Мониторинг (новый модуль `app/monitoring.py`)

| Было | Стало |
|------|-------|
| Нет отслеживания производительности | Добавлен декоратор **`track_time`** для замера времени выполнения функций |
| Нет метрик | Сохраняются метрики по каждому инструменту (время, количество вызовов) |
| — | Функция **`get_stats`** возвращает среднее, min, max, количество вызовов |
| — | Функция **`get_today_stats`** — статистика только за сегодня |

---

### Конфигурация (`app/config.py`)

| Было | Стало |
|------|-------|
| Базовые настройки | Добавлены новые параметры в `Settings`: |
| — | `web_cache_enabled` — включение кэша веб-поиска |
| — | `web_cache_ttl_hours` — время жизни кэша (часы) |
| — | `web_search_max_results` — максимальное количество результатов |
| — | `enable_web_search` / `enable_rag` / `enable_product_lookup` — переключение инструментов |

---

### Финальный промпт (`app/run_agent.py`)

| Было | Стало |
|------|-------|
| `"Ответь кратко и по делу..."` | Добавлена инструкция: *«Если вопрос только про сантехнику — отвечай только про сантехнику. Не добавляй информацию про ремонт, стройматериалы, мебель»* |

---

### Fallback цепочка (логика выбора источников)

| Было | Стало |
|------|-------|
| Первый источник → если нет → сразу уточняющий вопрос | Полноценная **fallback цепочка**: |
| — | **prefer_web**: web → rag → lookup → уточнение |
| — | **prefer_lookup**: lookup → rag → уточнение |
| — | **обычный запрос**: rag → lookup → web → уточнение |

---

### Новые файлы

| Файл | Назначение |
|------|------------|
| `app/monitoring.py` | Мониторинг производительности, сбор метрик |

---

### Измененные файлы

| Файл | Изменения |
|------|----------|
| `app/run_agent.py` | Добавлены: `enhance_search_query`, `_is_sanitary_relevant`, улучшенный промпт |
| `app/tools/web_search.py` | Добавлен кэш в памяти |
| `app/bot/telegram_bot.py` | Добавлены команды `/help`, `/clear`, инлайн-кнопки |
| `app/rag/retriever.py` | Добавлен метод `hybrid_search` |
| `app/history_store.py` | Добавлены `clear_history`, `get_history_stats` |
| `app/config.py` | Добавлены новые настройки |

---
