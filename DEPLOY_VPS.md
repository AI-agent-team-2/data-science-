# Deploy на VPS (безопасный rollout Langfuse)

## 1) Обновить код и зависимости

```bash
cd /opt/san_bot
git pull
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Добавить env на сервере

В `.env`:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

## 3) Health-check перед включением

```bash
python -c "from app.run_agent import run_agent; print('run_agent_import_ok')"
python -c "from app.observability.langfuse_client import get_langfuse_client; print('langfuse_import_ok')"
```

## 4) Безопасный rollout

1. Оставить `LANGFUSE_ENABLED=false`.
2. Перезапустить бота и убедиться, что он стартует штатно.
3. Переключить `LANGFUSE_ENABLED=true`.
4. Перезапустить сервис.
5. Проверить первые 10-20 запросов в Langfuse:
   - создается 1 trace `run_agent` на запрос
   - есть spans: `history_load`, `context_build`, `tool_lookup/tool_rag/tool_web`, `model_invoke`, `history_save`
   - нет raw Telegram id и секретов в payload

## 5) Rollback

1. Поставить `LANGFUSE_ENABLED=false`.
2. Перезапустить сервис.
3. При необходимости:
   - `git checkout <previous_tag_or_commit>`
   - повторный restart сервиса.
