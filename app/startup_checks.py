import os
import sys

# Обязательные переменные для Telegram бота
REQUIRED_ENV_VARS = {
    "TELEGRAM_TOKEN": "Telegram bot token",
    "OPENAI_API_KEY": "OpenAI/OpenRouter API key",
}

# Опциональные для веб-интерфейса
WEB_REQUIRED_ENV_VARS = {
    "WEB_API_KEY": "Web API key for authentication",
}


def check_env_vars(for_web: bool = False) -> None:
    """Проверяет наличие обязательных переменных окружения.
    
    Args:
        for_web: Если True, проверяет также WEB_API_KEY.
    
    Raises:
        SystemExit: Если какая-то обязательная переменная отсутствует.
    """
    missing = []
    
    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if for_web:
        for var, description in WEB_REQUIRED_ENV_VARS.items():
            if not os.getenv(var):
                missing.append(f"{var} ({description})")
    
    if missing:
        print("❌ Missing required environment variables:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        sys.exit(1)
    
    print("✅ All required environment variables are set")