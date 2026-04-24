from __future__ import annotations

import os
from typing import Final

# Обязательные переменные для Telegram бота
REQUIRED_ENV_VARS: Final[dict[str, str]] = {
    "TELEGRAM_TOKEN": "Telegram bot token",
}

# LLM ключ (можно через OPENAI_API_KEY или OPENROUTER_API_KEY)
LLM_KEY_VARS: Final[list[str]] = ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]

# Опциональные для веб-интерфейса (только предупреждение, не блокируем)
WEB_OPTIONAL_ENV_VARS: Final[dict[str, str]] = {
    "WEB_API_KEY": "Web API key for authentication",
}


def check_env_vars(for_web: bool = False) -> None:
    """
    Проверяет наличие обязательных переменных окружения.
    
    Args:
        for_web: Если True, дополнительно предупреждает о WEB_API_KEY.
    """
    # В CI окружении пропускаем проверку
    if os.getenv("CI") == "true":
        print("WARN: Running in CI mode, skipping env vars check")
        return
    
    missing: list[str] = []
    
    # Проверяем Telegram токен
    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    # Проверяем LLM ключ (хотя бы один из)
    llm_key_found = any(os.getenv(var) for var in LLM_KEY_VARS)
    if not llm_key_found:
        missing.append(f"LLM API key ({' or '.join(LLM_KEY_VARS)})")
    
    if for_web:
        for var, description in WEB_OPTIONAL_ENV_VARS.items():
            if not os.getenv(var):
                print(f"WARN: Optional env var not set: {var} ({description}) - some endpoints will be protected")
    
    if missing:
        print("ERROR: Missing required environment variables:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    
    print("OK: All required environment variables are set")