# Operations Runbook

## Быстрый статус

```bash
cd /opt/san_bot
docker compose ps
docker compose logs --tail=100 san-bot
```

## Health-check

```bash
docker inspect san-bot --format '{{json .State.Health}}'
```

## Перезапуск

```bash
cd /opt/san_bot
docker compose restart san-bot
```

## Проверка данных

```bash
docker volume ls | grep san_bot
docker volume inspect san_bot_chroma
docker volume inspect san_bot_history
```

## Manual rollback

```bash
cd /opt/san_bot
export BOT_IMAGE="$(cat .previous_image_tag)"
export BOT_ENV_FILE=/etc/san-bot/san-bot.env
docker compose up -d san-bot
```

## Инциденты

1. Контейнер не стартует:
   - проверить env (`BOT_ENV_FILE`);
   - проверить `docker compose logs san-bot`.
2. Бот молчит в Telegram:
   - проверить `TELEGRAM_TOKEN`;
   - проверить сетевую доступность провайдера LLM.
3. Пустые ответы из базы:
   - проверить `CHROMA_PATH`;
   - при необходимости переиндексировать `python -m app.rag.ingest` в сервисном окружении.
