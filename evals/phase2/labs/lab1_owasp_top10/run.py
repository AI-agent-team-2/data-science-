"""
Lab 1 (Phase 2): OWASP LLM Top-10 Dynamic Red Teaming

Адаптирован для тестирования SAN Bot через run_agent.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from deepteam import red_team
from deepteam.frameworks import OWASPTop10
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

# Определяем корень репозитория динамически.
# run.py → lab1_owasp_top10 → labs → phase2 → evals → repo_root
REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.run_agent import run_agent

load_dotenv(REPO_ROOT / ".env")


# =========================================================
# НАСТРОЙКА: выберите режим работы model_callback
# =========================================================
# USE_MOCK_CALLBACK = True      # Заглушка (для проверки DeepTeam)
USE_MOCK_CALLBACK = False        # Реальный бот (run_agent)
# =========================================================


# ─── Обёртка для OpenRouter (Attacker и Judge) ────────────────────────────────

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def get_model_name(self) -> str:
        return self.model

    def load_model(self):
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


# ─── CLI-аргументы ─────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OWASP LLM Top-10 dynamic red teaming для SAN Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-model",
        default=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        help="Target модель (тестируемая)"
    )
    parser.add_argument(
        "--attacker-model",
        default=os.getenv("ATTACKER_MODEL", "openai/gpt-4o-mini"),
        help="Attacker модель (генерирует атаки)"
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "openai/gpt-4o"),
        help="Judge модель (оценивает ответы)"
    )
    parser.add_argument(
        "--attacks-per-type",
        type=int,
        default=1,
        help="Число атак на уязвимость (по умолчанию 1)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Параллельных запросов (по умолчанию 1)"
    )
    return parser


# ─── model_callback (с переключением между заглушкой и реальным ботом) ────────

def model_callback_mock(input: str) -> str:
    """Заглушка: всегда возвращает один и тот же ответ."""
    return "I cannot answer this question as it does not relate to sanitary products."


def model_callback_real(input: str) -> str:
    """Реальный вызов run_agent."""
    try:
        print(f"\n[→] Атака: {input[:150]}...")
        response = run_agent(input, user_id="redteam_test")
        
        if response is None:
            print("[!] Ответ: None")
            return "[ERROR] Empty response"
        
        print(f"[←] Ответ: {response[:200]}...")
        
        # Если бот отказался отвечать, Judge должен это оценить
        if "Не могу обработать" in response or "не могу" in response.lower():
            print("[ℹ] Бот отказался отвечать на атаку")
        
        return response
        
    except Exception as e:
        print(f"[!] Ошибка в model_callback: {e}")
        return f"[ERROR] {e}"


# Выбираем нужную версию model_callback в зависимости от настройки
if USE_MOCK_CALLBACK:
    model_callback = model_callback_mock
    print("🔧 Режим: ЗАГЛУШКА (mock callback)")
else:
    model_callback = model_callback_real
    print("🔧 Режим: РЕАЛЬНЫЙ БОТ (run_agent)")


# ─── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Ошибка: OPENROUTER_API_KEY (или OPENAI_API_KEY) не задан (например, в .env или в GitHub Secrets)."
        )

    print("=" * 60)
    print("DeepTeam: OWASP LLM Top-10 Red Teaming")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Target model:   {args.target_model}")
    print(f"Attacker model: {args.attacker_model}")
    print(f"Judge model:    {args.judge_model}")
    print(f"Attacks per type: {args.attacks_per_type}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("=" * 60)

    # ─── Attacker и Judge через OpenRouter ────────────────────────────────────
    attacker_llm = OpenRouterLLM(model=args.attacker_model, api_key=api_key)
    judge_llm = OpenRouterLLM(model=args.judge_model, api_key=api_key)

    # ─── Запуск red teaming ──────────────────────────────────────────────────
    print("\nЗапуск red teaming...\n")
    
    risk_assessment = red_team(
        model_callback=model_callback,
        framework=OWASPTop10(),
        simulator_model=attacker_llm,
        evaluation_model=judge_llm,
        attacks_per_vulnerability_type=args.attacks_per_type,
        max_concurrent=args.max_concurrent,
    )

    # ─── Результаты ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Всего тест-кейсов: {len(risk_assessment.test_cases)}")
    
    # Считаем количество успешных и неудачных
    passed = 0
    failed = 0
    errored = 0
    
    for tc in risk_assessment.test_cases:
        if hasattr(tc, 'success'):
            if tc.success:
                passed += 1
            else:
                failed += 1
        elif hasattr(tc, 'error') and tc.error:
            errored += 1
    
    print(f"Успешно (защита сработала):  {passed}")
    print(f"Провалено (атака прошла):    {failed}")
    print(f"Ошибок:                       {errored}")
    
    if hasattr(risk_assessment, 'pass_rate'):
        print(f"Pass rate: {risk_assessment.pass_rate:.1%}")

    # ─── Сохранение отчёта ────────────────────────────────────────────────────
    report_dir = PHASE2_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"owasp_{timestamp}.json"

    report_data = {
        "target_model": args.target_model,
        "attacker_model": args.attacker_model,
        "judge_model": args.judge_model,
        "attacks_per_type": args.attacks_per_type,
        "total_test_cases": len(risk_assessment.test_cases),
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "test_cases": []
    }

    for tc in risk_assessment.test_cases:
        tc_data = {}
        if hasattr(tc, 'vulnerability_type'):
            tc_data["vulnerability_type"] = str(tc.vulnerability_type)
        if hasattr(tc, 'input'):
            tc_data["input"] = tc.input
        if hasattr(tc, 'actual_output'):
            tc_data["actual_output"] = tc.actual_output
        if hasattr(tc, 'success'):
            tc_data["success"] = tc.success
        if hasattr(tc, 'error'):
            tc_data["error"] = str(tc.error)
        report_data["test_cases"].append(tc_data)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\nОтчёт сохранён: {report_path}")
    print("\nГотово!")
