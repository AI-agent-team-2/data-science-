# Deploy на VPS

Документ описывает безопасный rollout бота с включаемой интеграцией Langfuse.

## 1) Обновить код

```bash
cd /opt/san_bot
sudo -u botuser -H git pull
```

## 2) Обновить зависимости в venv

```bash
sudo -u botuser -H /opt/san_bot/.venv/bin/python -m pip install --upgrade pip
sudo -u botuser -H /opt/san_bot/.venv/bin/python -m pip install -r /opt/san_bot/requirements.txt
```

## 3) Проверить env для systemd

Файл окружения: `/etc/san-bot/san-bot.env`

Обязательные переменные для Langfuse:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

## 4) Health-check перед перезапуском

```bash
sudo -u botuser -H bash -lc '
set -a; source /etc/san-bot/san-bot.env; set +a
cd /opt/san_bot
/opt/san_bot/.venv/bin/python -c "import app.graph; print(\"graph_import_ok\")"
/opt/san_bot/.venv/bin/python -c "from app.run_agent import run_agent; print(\"run_agent_import_ok\")"
/opt/san_bot/.venv/bin/python -c "from app.observability.langfuse_client import get_langfuse_client; print(\"langfuse_client_ready=\", get_langfuse_client() is not None)"
'
```

## 5) Безопасный rollout

1. Оставить `LANGFUSE_ENABLED=false`.
2. Перезапустить сервис и убедиться, что бот запускается без ошибок.
3. Переключить `LANGFUSE_ENABLED=true` в `/etc/san-bot/san-bot.env`.
4. Перезапустить сервис:
   - `sudo systemctl restart san-bot.service`
5. Проверить первые 10–20 запросов:
   - в Langfuse появляется trace `run_agent`;
   - внутри есть `history_load`, `context_build`, `tool_lookup/tool_rag/tool_web`, `model_invoke`, `history_save`;
   - в payload нет raw Telegram ID и секретов.

## 6) Проверка сервиса и логов

```bash
sudo systemctl status san-bot.service --no-pager -l
sudo journalctl -u san-bot.service -n 100 --no-pager
```

## 7) Rollback

1. Установить `LANGFUSE_ENABLED=false`.
2. Перезапустить сервис:
   - `sudo systemctl restart san-bot.service`
3. Если нужно откатить код:
   - `sudo -u botuser -H git checkout <commit_or_tag>`
   - повторно перезапустить сервис.
