import os

# Обязательные переменные для Telegram бота
REQUIRED_ENV_VARS = {
    "TELEGRAM_TOKEN": "Telegram bot token",
}

# Опциональные для веб-интерфейса
WEB_REQUIRED_ENV_VARS = {
    "WEB_API_KEY": "Web API key for authentication",
}


def _build_missing_vars(for_web: bool = False) -> list[str]:
    """Собирает список отсутствующих обязательных переменных окружения."""
    missing = []

    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")

    # Ключ LLM может быть задан через OPENAI_API_KEY или OPENROUTER_API_KEY.
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY (OpenAI/OpenRouter API key)")

    if for_web:
        for var, description in WEB_REQUIRED_ENV_VARS.items():
            if not os.getenv(var):
                missing.append(f"{var} ({description})")

    return missing


def check_env_vars(for_web: bool = False) -> None:
    """Проверяет наличие обязательных переменных окружения.

    Args:
        for_web: Если True, проверяет также WEB_API_KEY.

    Raises:
        RuntimeError: Если какая-то обязательная переменная отсутствует.
    """
    missing = _build_missing_vars(for_web=for_web)

    if missing:
        details = "\n".join(f"   - {item}" for item in missing)
        raise RuntimeError(
            "Missing required environment variables:\n"
            f"{details}\n"
            "Please check your .env file and ensure all required variables are set."
        )
