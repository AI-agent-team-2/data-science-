"""
Запуск Telegram-бота из корня проекта.
Использование: python run_bot.py
"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
bot_script = root / "src" / "bot_local.py"
subprocess.run([sys.executable, str(bot_script)], cwd=str(root))
