# Observability и Langfuse

## Зачем это нужно

Observability в проекте помогает:
- видеть полный путь обработки пользовательского запроса;
- быстро находить причины ошибок и таймаутов;
- контролировать использование модели (latency, tokens, ошибки);
- проверять, какой источник контекста сработал (`lookup`, `rag`, `web`).

## Как устроена интеграция

Интеграция состоит из двух уровней:

1. Manual tracing в `app/run_agent.py`:
   - trace `run_agent` на каждый входящий запрос;
   - orchestration spans для ключевых этапов пайплайна.
2. Callback-интеграция LangChain в `app/graph.py`:
   - `langfuse.langchain.CallbackHandler` для автоматического наблюдения вызовов модели.

Точка инициализации клиента и callback handler: `app/observability/langfuse_client.py`.

## Какие trace/span/observation создаются

### Trace

- `run_agent`
  - `session_id = hash_user_id(user_id)`
  - безопасные metadata: `model`, `provider`, `source_order`, `enable_*` флаги

### Дочерние spans

- `history_load`
- `context_build`
- `tool_lookup`
- `tool_rag`
- `tool_web`
- `model_invoke`
- `history_save`

### Что дополнительно попадает в наблюдаемость

- Инструменты:
  - `web_search`: `provider`, `cache_hit`, `result_count`
  - `rag_search`: `top_k`, `retrieved_count`, `avg_score`, `max_score`
  - `product_lookup`: `normalized_query` (sanitized), `detected_sku_count`, `result_count`, `mode`
- Ошибки и таймауты как error-event/status.
- Данные по модели (tokens/latency/usage) через callback LangChain, если провайдер возвращает usage.

## Что логируется безопасно

- очищенный текст запроса (`sanitize_text`);
- агрегированные метрики и технические статусы;
- хешированный `user_id`/`session_id`.

## Что не логируется

- raw Telegram identifiers (`chat_id`, `user_id`) в traces;
- ключи, токены, секреты;
- email и телефоны в открытом виде;
- полный текст чувствительных payload;
- полный текст RAG-чанков в production traces.

## Санитизация данных

Модуль: `app/observability/sanitize.py`

- `hash_user_id(user_id)` — стабильный безопасный идентификатор вида `u_<16hex>`
- `sanitize_text(text)` — маскирует:
  - email;
  - phone;
  - bearer/token-like строки;
  - длинные secret-like значения.
- `sanitize_payload(payload)` — рекурсивная очистка `dict/list/tuple` перед отправкой в Langfuse.

## Переменные окружения

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

## Включение и выключение

### Включить

1. Заполнить ключи и host.
2. Установить `LANGFUSE_ENABLED=true`.
3. Перезапустить сервис.

### Выключить

1. Установить `LANGFUSE_ENABLED=false`.
2. Перезапустить сервис.

При выключении observability бизнес-логика продолжает работать.

## Локальная проверка

1. Проверка импорта:
   - `python -c "from app.observability.langfuse_client import get_langfuse_client; print(get_langfuse_client() is not None)"`
2. Пробный вызов:
   - отправить сообщение боту или вызвать `run_agent` локально.
3. Проверить в UI Langfuse:
   - появился trace `run_agent`;
   - внутри есть `history_load`, `context_build`, `tool_*`, `model_invoke`, `history_save`.

## Диагностика типовых проблем

1. Trace не появляется:
   - проверьте `LANGFUSE_ENABLED`, ключи и доступность `LANGFUSE_HOST`.
2. Нет usage/tokens:
   - проверьте, что провайдер модели возвращает usage metadata.
3. Повторяющиеся ошибки callback:
   - проверьте совместимость версий `langfuse` и `langchain` из `requirements.txt`.
4. Таймауты в инструментах:
   - проверьте сетевой доступ для web-поиска и время ответа внешних API.
