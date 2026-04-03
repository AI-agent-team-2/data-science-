"""
FastAPI-сервер для веб-интерфейса SAN Bot.

Запуск из корня проекта:
    python -m uvicorn web.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импорт `app.*` работал
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.history_store import clear_history, init_db, load_messages
from app.run_agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic-модели запросов и ответов
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Текст сообщения")
    session_id: str = Field(..., min_length=1, max_length=100, description="ID сессии")

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)

class HistoryItem(BaseModel):
    user: str
    assistant: str
    timestamp: str | None = None

# ---------------------------------------------------------------------------
# Приложение FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAN Bot Web API",
    description="Веб-интерфейс для консультанта по сантехнике",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.resolved_web_api_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """Инициализирует локальную БД истории перед обслуживанием запросов."""
    init_db()


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    Проверяет API-ключ для web API.
    Если ключ не задан в настройках, проверка пропускается (dev-режим).
    """
    expected = settings.web_api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


@app.get("/api/history")
def get_history(
    session_id: str = Query(..., min_length=1, max_length=100),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Получить историю диалога для данной сессии.
    
    Возвращает список сообщений в формате:
    {"history": [{"user": "текст", "assistant": "текст", "timestamp": null}], "count": N}
    """
    _require_api_key(x_api_key)
    logger.info("GET /api/history  session=%s", session_id)
    try:
        # load_messages возвращает список кортежей (role, content)
        messages = load_messages(session_id=session_id)
        # Преобразуем в удобный формат
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "user": messages[i][1],
                    "assistant": messages[i + 1][1],
                    "timestamp": None  # SQLite не хранит timestamp в этой версии
                })
        return {"history": history, "count": len(history)}
    except Exception as exc:
        logger.exception("Failed to load history")
        raise HTTPException(status_code=500, detail="Ошибка загрузки истории") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Отправить сообщение боту и получить ответ.

    Эндпоинт синхронный (`def`, не `async def`), поэтому FastAPI
    автоматически выполняет его в пуле потоков — это корректно для
    блокирующего вызова `run_agent()`.
    """
    _require_api_key(x_api_key)
    logger.info("POST /api/chat  session=%s  len=%d", req.session_id, len(req.message))
    try:
        reply = run_agent(req.message, user_id=req.session_id)
    except Exception as exc:
        logger.exception("run_agent failed")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса") from exc
    return ChatResponse(reply=reply, session_id=req.session_id)


@app.post("/api/clear")
def clear(req: ClearRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Очистить историю диалога для данной сессии."""
    _require_api_key(x_api_key)
    logger.info("POST /api/clear  session=%s", req.session_id)
    clear_history(session_id=req.session_id)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Статические файлы (фронтенд) — монтируем ПОСЛЕДНИМ,
# чтобы маршруты /api/* имели приоритет.
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
