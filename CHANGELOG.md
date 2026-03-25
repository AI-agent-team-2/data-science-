# Журнал изменений

Формат файла основан на принципах [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

## [5] - 2026-03-25

### Крупные изменения
- Переход на callback/native-first observability.
- Один запрос пользователя = один `CallbackHandler` = один root trace `run_agent`.
- Инструменты (`tool_lookup`, `tool_rag`, `tool_web`) и `model_invoke` наблюдаются как нативные callback steps.

### Документация
- Обновлены `OBSERVABILITY.md` и `README.md`.

## [4] - 2026-03-25

### Крупные изменения
- Добавлена интеграция Langfuse observability.
- Введен trace `run_agent` для отслеживания обработки запроса.
- Добавлена передача metadata/контекста для привязки model вызовов к trace.

## [3] - 2026-03-24

### Крупные изменения
- Добавлен RAG-слой на ChromaDB (`rag_search`, ingestion/retrieval).
- Добавлен внешний web-поиск (`web_search`) с кешированием результатов.

## [2] - 2026-03-22

### Крупные изменения
- Добавлен поиск по каталогу товаров (`product_lookup`) с приоритизацией SKU.
- Добавлено хранение истории диалога в SQLite.

## [1] - 2026-03-21

### Крупные изменения
- Базовый релиз Telegram-бота для сантехнического домена.
- Добавлены базовые команды Telegram-бота и основной pipeline ответа.
