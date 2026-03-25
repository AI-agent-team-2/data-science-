# Observability (Langfuse)

## Что логируется

- `trace` на каждый вызов `run_agent`:
  - `name=run_agent`
  - `session_id=hash(user_id)`
  - metadata: `model`, `provider`, `source_order`, `enable_*` flags
- дочерние `spans`:
  - `history_load`
  - `context_build`
  - `tool_lookup`
  - `tool_rag`
  - `tool_web`
  - `model_invoke`
  - `history_save`
- метрики инструментов:
  - `web_search`: `provider`, `cache_hit`, `result_count`
  - `rag_search`: `top_k`, `retrieved_count`, `avg_score`, `max_score`
  - `product_lookup`: `normalized_query` (sanitized), `detected_sku_count`, `result_count`, `mode`
- ошибки/таймауты как error event/status update.
- `model_invoke` usage/cost:
  - приоритетно через Langfuse callback handler
  - fallback (если callback недоступен): ручная отправка `generation` из `response_metadata/usage_metadata`

## Что НЕ логируется

- raw Telegram identifiers (`chat_id`, `user_id`) в traces
- секреты, токены, API keys
- email/phone в открытом виде
- полный текст чувствительного payload
- полный текст RAG-чанков в observability payload

## Sanitization

`app/observability/sanitize.py`:
- `sanitize_text(text)` маскирует `email`, `phone`, `Bearer/token-like`, `secret-like` строки
- `sanitize_payload(payload)` рекурсивно чистит dict/list/tuple
- `hash_user_id(user_id)` возвращает стабильный безопасный хэш (`u_<16hex>`)

## Env переменные

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

## Как включить

1. Установить env-переменные Langfuse.
2. Переключить `LANGFUSE_ENABLED=true`.
3. Перезапустить сервис бота.

## Как проверить локально

1. Проверка импорта:
   - `python -c "from app.observability.langfuse_client import get_langfuse_client; print('ok')"`
2. Пробный запрос в бота или `run_agent`.
3. В Langfuse UI должен появиться trace `run_agent` со span-структурой:
   - `history_load` -> `context_build` -> `tool_*` -> `model_invoke` -> `history_save`

## Как выключить

1. Поставить `LANGFUSE_ENABLED=false`.
2. Перезапустить сервис.

Бизнес-логика продолжит работать в no-op режиме observability.
