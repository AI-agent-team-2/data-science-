FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PATH=/app/chroma_db \
    HISTORY_DB_PATH=/app/history/history.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY data ./data

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/chroma_db /app/history \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import app.graph; from app.run_agent import run_agent; print('healthy')" || exit 1

CMD ["python", "-m", "app.bot.telegram_bot"]
