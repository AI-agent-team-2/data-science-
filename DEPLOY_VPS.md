# Deploy на VPS (Docker Compose + CI/CD)

## Контур

- CI/CD workflow: `.github/workflows/deploy.yml`
- Registry: `ghcr.io/<owner>/<repo>:<tag>`
- Runtime: `docker compose` на VPS
- Контейнеры:
  - `san-bot`
  - `san-bot-web`
- State volumes:
  - `san_bot_chroma`
  - `san_bot_history`

Важно: поддерживается только Docker-контур.
Старые сценарии через `systemd`, `.venv` и ручной `deploy.sh` считаются legacy и не должны использоваться параллельно с контейнером.

## Подготовка VPS (один раз)

1. Установить Docker и один из вариантов Compose:
   - `docker compose` plugin, или
   - standalone `docker-compose`.
2. Подготовить директорию проекта, например `/opt/san_bot`.
3. Разместить в ней `docker-compose.yml`.
4. Создать env-файл, например `/etc/san-bot/san-bot.env`.
5. (Опционально для private image) выполнить `docker login ghcr.io`.

## Минимальный env

```env
TELEGRAM_TOKEN=...
OPENAI_API_KEY=...
MODEL_PROVIDER=openrouter
MODEL_NAME=
OPENAI_BASE_URL=
CHROMA_PATH=/app/chroma_db
HISTORY_DB_PATH=/app/history/history.db
STARTUP_INDEX_MODE=if_empty
WEB_API_KEY=...
WEB_ALLOWED_ORIGINS=http://<VPS_IP>:8000
```

Опционально:

```env
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
TAVILY_API_KEY=
MODEL_MAX_RETRIES=2
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Secrets для GitHub Actions

Обязательные:

- `VPS_HOST`
- `VPS_USER`
- `VPS_SSH_KEY`
- `VPS_PORT`

Рекомендуемые:

- `VPS_APP_DIR` (например `/opt/san_bot`)
- `VPS_ENV_FILE` (например `/etc/san-bot/san-bot.env`)
- `GHCR_USERNAME`
- `GHCR_TOKEN`

## Что делает pipeline

1. Проверки кода:
   - `python -m compileall app tests`
   - `python -m unittest discover -s tests -p "test_*.py"`
   - import smoke (`app.graph`, `run_agent`)
2. Сборка и публикация Docker image:
   - `ghcr.io/<repo>:<commit_sha>`
   - `ghcr.io/<repo>:latest`
3. SSH deploy на VPS:
   - `docker compose pull san-bot san-bot-web`
   - `docker compose up -d san-bot san-bot-web`
4. Post-deploy health-check контейнера.
5. Rollback на предыдущий image tag при неуспешном health-check.

Замечание по indexing:
- при `STARTUP_INDEX_MODE=if_empty` контейнер сам выполнит ingest только на пустой базе;
- при `STARTUP_INDEX_MODE=always` ingest будет запускаться на каждом старте;
- для production по умолчанию рекомендуется `if_empty`.

## Локальные команды диагностики на VPS

```bash
cd /opt/san_bot
docker compose ps
docker compose logs --tail=100 san-bot
docker compose logs --tail=100 san-bot-web
docker inspect san-bot --format '{{json .State.Health}}'
docker inspect san-bot-web --format '{{json .State.Health}}'
```

## Rollback вручную

```bash
cd /opt/san_bot
export BOT_IMAGE="$(cat .previous_image_tag)"
export BOT_ENV_FILE=/etc/san-bot/san-bot.env
docker compose up -d san-bot san-bot-web
```

## Post-deploy checklist

1. `docker compose ps` показывает `san-bot` и `san-bot-web` в `healthy`/`running`.
2. Бот отвечает на `/start` и 1-2 доменных запроса.
3. История диалога сохраняется после перезапуска контейнера.
4. Web API отвечает с валидным `X-API-Key` и возвращает `401` при неверном ключе.
5. При `LANGFUSE_ENABLED=true` trace появляются, при `false` бот работает без деградации.
