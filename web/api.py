"""
FastAPI-сервер для веб-интерфейса SAN Bot.

Запуск из корня проекта:
    python -m uvicorn web.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импорт `app.*` работал
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.run_agent import run_agent
from app.history_store import clear_history, load_turns

from app.startup_checks import check_env_vars

# Проверка переменных окружения перед запуском
check_env_vars(for_web=True)

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
    allow_origins=settings.web_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.web_api_key:
        logger.error("WEB_API_KEY is not configured; refusing to serve protected endpoints")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WEB_API_KEY is not configured",
        )

    if not x_api_key or x_api_key != settings.web_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.get("/api/health")
def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


@app.get("/api/history")
def get_history(
    session_id: str = Query(..., min_length=1, max_length=100),
    _auth: None = Depends(_require_api_key),
):
    """
    Получить историю диалога для данной сессии.
    
    Возвращает список сообщений в формате:
    {"history": [{"user": "текст", "assistant": "текст", "timestamp": "..." }], "count": N}
    """
    logger.info("GET /api/history  session=%s", session_id)
    try:
        history = load_turns(session_id=session_id)
        return {"history": history, "count": len(history)}
    except Exception as exc:
        logger.exception("Failed to load history")
        raise HTTPException(status_code=500, detail="Ошибка загрузки истории") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Отправить сообщение боту и получить ответ.

    Эндпоинт синхронный (`def`, не `async def`), поэтому FastAPI
    автоматически выполняет его в пуле потоков — это корректно для
    блокирующего вызова `run_agent()`.
    """
    logger.info("POST /api/chat  session=%s  len=%d", req.session_id, len(req.message))
    try:
        reply = run_agent(req.message, user_id=req.session_id)
    except Exception as exc:
        logger.exception("run_agent failed")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса") from exc
    return ChatResponse(reply=reply, session_id=req.session_id)


@app.post("/api/chat-stream")
async def chat_stream(req: ChatRequest):
    """
    Отправить сообщение боту и получить ответ в режиме стриминга (по частям).
    
    Ответ возвращается порциями по 3-5 символов, что создаёт эффект "печатания".
    """
    logger.info("POST /api/chat-stream  session=%s  len=%d", req.session_id, len(req.message))
    
    async def generate():
        try:
            # Получаем полный ответ от бота (синхронная функция)
            # Запускаем в потоке, чтобы не блокировать asyncio
            loop = asyncio.get_event_loop()
            reply = await loop.run_in_executor(
                None, 
                run_agent, 
                req.message, 
                req.session_id
            )
            
            # Отправляем ответ порциями (по 3-5 символов)
            chunk_size = 4
            for i in range(0, len(reply), chunk_size):
                chunk = reply[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.02)  # небольшая задержка для эффекта печати
                
        except Exception as exc:
            logger.exception("chat_stream failed")
            yield f"❌ Ошибка: {exc}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8"
    )


@app.post("/api/clear")
def clear(req: ClearRequest, _auth: None = Depends(_require_api_key)):
    """Очистить историю диалога для данной сессии."""
    logger.info("POST /api/clear  session=%s", req.session_id)
    clear_history(session_id=req.session_id)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Статические файлы (фронтенд) — монтируем ПОСЛЕДНИМ,
# чтобы маршруты /api/* имели приоритет.
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
