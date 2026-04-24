# Deploy на VPS (Docker Compose + CI/CD)

## Контур

- CI/CD workflow: `.github/workflows/deploy.yml`
- Registry: `ghcr.io/<owner>/<repo>:<tag>`
- Runtime: `docker compose` на VPS
- Контейнеры:
  - `san-bot`
  - `san-bot-web`
  - `san-bot-proxy` (nginx)
- State volumes:
  - `san_bot_chroma`
  - `san_bot_history`
  - `san_bot_web_cache`

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
WEB_CACHE_ENABLED=true
WEB_CACHE_TTL_HOURS=24
MODEL_MAX_RETRIES=2
MODEL_CIRCUIT_BREAKER_ENABLED=true
MODEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
MODEL_CIRCUIT_BREAKER_COOLDOWN_SEC=30
MODEL_CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD=1
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
   - `docker compose pull san-bot san-bot-web san-bot-proxy`
   - `docker compose up -d san-bot san-bot-web san-bot-proxy`
4. Post-deploy health-check контейнера.
5. Rollback на предыдущий image tag при неуспешном health-check.
6. Post-deploy cleanup на VPS:
   - prune неиспользуемых Docker images (старые sha-теги после каждого deploy).

Замечание по indexing:
- при `STARTUP_INDEX_MODE=if_empty` контейнер сам выполнит ingest только на пустой базе;
- при `STARTUP_INDEX_MODE=always` ingest будет запускаться на каждом старте;
- для production по умолчанию рекомендуется `if_empty`.

Замечание по таймауту health-check:
- pipeline ждёт до ~10 минут, пока `san-bot` и `san-bot-web` перейдут в `healthy` (на пустой базе ingestion может занять несколько минут).

## Локальные команды диагностики на VPS

```bash
cd /opt/san_bot
docker compose ps
docker compose logs --tail=100 san-bot
docker compose logs --tail=100 san-bot-web
docker compose logs --tail=100 san-bot-proxy
docker inspect san-bot --format '{{json .State.Health}}'
docker inspect san-bot-web --format '{{json .State.Health}}'
```

## Как не забивать диск на VPS

Симптом: `git fetch` / `docker pull` падают с `No space left on device`, хотя диск кажется “достаточно большим”.
Причина: каждый deploy публикует новый image с тегом commit SHA, и на VPS постепенно копятся десятки старых образов.

Что уже сделано в проекте:

- В `docker-compose.yml` включена ротация Docker логов (`max-size`/`max-file`) для сервисов `san-bot`, `san-bot-web`, `san-bot-proxy`.
- В CI/CD deploy-скрипте добавлен auto-cleanup неиспользуемых образов после успешного deploy.

Если нужно освободить место вручную на VPS:

```bash
docker image prune -af
docker builder prune -af
df -hT
```

Если место ушло в Docker логи (обычно `/var/lib/docker/containers/*/*-json.log`):

```bash
sudo find /var/lib/docker/containers -name '*-json.log' -type f -exec sh -c ': > "$1"' _ {} \;
df -hT
```

## Rollback вручную

```bash
cd /opt/san_bot
export BOT_IMAGE="$(cat .previous_image_tag)"
export BOT_ENV_FILE=/etc/san-bot/san-bot.env
docker compose up -d san-bot san-bot-web san-bot-proxy
```

## Post-deploy checklist

1. `docker compose ps` показывает `san-bot`, `san-bot-web` и `san-bot-proxy` в `healthy`/`running`.
2. Бот отвечает на `/start` и 1-2 доменных запроса.
3. История диалога сохраняется после перезапуска контейнера.
4. Web UI доступен через `http://<VPS_IP>:8000` без ручного ввода API-ключа (ключ инжектится proxy в `/api/*`).
5. При `LANGFUSE_ENABLED=true` trace появляются, при `false` бот работает без деградации.
