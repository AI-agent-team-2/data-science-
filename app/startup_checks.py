from __future__ import annotations

import os
import sys
from typing import Final

# Обязательные переменные для Telegram бота
REQUIRED_ENV_VARS: Final[dict[str, str]] = {
    "TELEGRAM_TOKEN": "Telegram bot token",
    "OPENAI_API_KEY": "OpenAI/OpenRouter API key",
}

# Опциональные для веб-интерфейса (только предупреждение, не блокируем)
WEB_OPTIONAL_ENV_VARS: Final[dict[str, str]] = {
    "WEB_API_KEY": "Web API key for authentication",
}


def check_env_vars(for_web: bool = False) -> None:
    """
    Проверяет наличие обязательных переменных окружения.
    
    Args:
        for_web: Если True, дополнительно проверяет WEB_API_KEY.
    
    Raises:
        SystemExit: Если какая-то обязательная переменная отсутствует.
    """
    # В CI окружении пропускаем проверку (тесты не должны падать из-за отсутствия переменных)
    if os.getenv("CI") == "true":
        print("⚠️ Running in CI mode, skipping env vars check")
        return
    
    missing: list[str] = []
    
    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if for_web:
        for var, description in WEB_OPTIONAL_ENV_VARS.items():
            if not os.getenv(var):
                print(f"⚠️ Optional env var not set: {var} ({description}) - some endpoints will be protected")
    
    if missing:
        print("❌ Missing required environment variables:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        sys.exit(1)
    
    print("✅ All required environment variables are set")