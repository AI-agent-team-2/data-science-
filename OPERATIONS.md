# Operations Runbook

## Быстрый статус

```bash
cd /opt/san_bot
docker compose ps
docker compose logs --tail=100 san-bot
docker compose logs --tail=100 san-bot-web
```

Поддерживаемый runtime только один: контейнер `san-bot` через `docker compose`.
Не запускайте параллельно старый `systemd`-сервис или ручной Python-процесс, иначе Telegram polling начнет возвращать `409 Conflict`.

## Health-check

```bash
docker inspect san-bot --format '{{json .State.Health}}'
docker inspect san-bot-web --format '{{json .State.Health}}'
```

Health-check отражает не только факт запуска контейнера, но и readiness retrieval-данных:
- chunk collection должна быть непустой;
- product collection должна быть непустой.

## Перезапуск

```bash
cd /opt/san_bot
docker compose restart san-bot san-bot-web
```

## Проверка данных

```bash
docker volume ls | grep san_bot
docker volume inspect san_bot_san_bot_chroma
docker volume inspect san_bot_san_bot_history
```

Проверка readiness индекса изнутри контейнера:

```bash
docker exec -i san-bot python - <<'PY'
from app.rag.health import get_index_health
health = get_index_health()
print(health)
print("ready=", health.is_ready)
PY
```

## Manual rollback

```bash
cd /opt/san_bot
export BOT_IMAGE="$(cat .previous_image_tag)"
export BOT_ENV_FILE=/etc/san-bot/san-bot.env
docker compose up -d san-bot san-bot-web
```

## Инциденты

1. Контейнер не стартует:
   - проверить env (`BOT_ENV_FILE`);
   - проверить `docker compose logs san-bot`.
2. Бот молчит в Telegram:
   - проверить `TELEGRAM_TOKEN`;
   - проверить сетевую доступность провайдера LLM.
   - убедиться, что не запущен второй poller вне Docker (`systemctl status san-bot.service`, `pgrep -af "telegram_bot.py|app.bot.telegram_bot"`).
3. Пустые ответы из базы:
   - проверить `CHROMA_PATH`;
   - проверить readiness индекса внутри контейнера;
   - при необходимости переиндексировать `python -m app.rag.ingest` в сервисном окружении;
   - проверить `STARTUP_INDEX_MODE`, если контейнер стартует на новой пустой базе.
4. Web UI/API возвращает `401` или `503`:
   - проверить, что в env задан `WEB_API_KEY`;
   - проверить, что клиент передает заголовок `X-API-Key`;
   - проверить `WEB_ALLOWED_ORIGINS` для используемого домена/IP и порта.
