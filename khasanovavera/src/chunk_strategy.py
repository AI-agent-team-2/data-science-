"""
Стратегия чанкования для knowledge_base.

Как определить нужный размер чанков
==================================

1) Контекст LLM
   - В промпт попадает несколько чанков (например retriever_k=4).
   - Сумма длин чанков не должна превышать контекст модели (например 4k–8k токенов).
   - Ориентир: 1 токен ≈ 4 символа для русского/английского.
   Пример: контекст 4096 → ~16k символов на все чанки → при k=4 разумно 300–600 символов на чанк.

2) Структура текстов в knowledge_base
   - Каталоги (tp/, cat/, qe/): блоки с заголовками (DOCUMENT, PRODUCT, # Назначение, таблицы).
   - Чанк не должен резать смысловую единицу (одна карточка товара, один подраздел).
   - Посмотри среднюю длину параграфа в своих файлах: chunk_size лучше брать не меньше
     типичного параграфа и не больше 2–3 параграфов.

3) Эмбеддинги
   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 хорошо работает с
     предложениями и короткими абзацами (128–512 токенов).
   - Слишком длинный чанк размывает вектор; слишком короткий — теряется контекст.

4) Практический подбор
   - Запусти этот скрипт на реальных файлах из knowledge_base (см. __main__).
   - Посмотри num_chunks и avg_length для разных chunk_size/overlap.
   - Overlap 10–20% от chunk_size уменьшает обрезку на границах (например 50–150 при size 500).
   - После индексации проверь качество на типичных вопросах: если ответы обрезанные или
     не находят нужный фрагмент — попробуй увеличить chunk_size или overlap.

Типичные диапазоны для каталогов/справочников:
- chunk_size: 400–800 символов
- chunk_overlap: 80–150 символов
"""
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def analyze_chunking_strategy(text: str, chunk_sizes: list[int], overlaps: list[int]):
    """Строит таблицу: для каждой пары (chunk_size, overlap) — число чанков и средняя длина."""
    rows = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                continue
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_text(text)
            rows.append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "num_chunks": len(chunks),
                "avg_length": sum(map(len, chunks)) / len(chunks) if chunks else 0,
            })
    return pd.DataFrame(rows)


def load_sample_from_knowledge_base(kb_root: Path | None = None, max_chars: int = 50_000) -> str:
    """Читает кусок текста из папки knowledge_base для теста чанкования."""
    if kb_root is None:
        kb_root = Path(__file__).resolve().parent.parent / "knowledge_base"
    if not kb_root.is_dir():
        return ""
    parts = []
    n = 0
    for path in sorted(kb_root.rglob("*.txt")):
        try:
            part = path.read_text(encoding="utf-8")
            parts.append(part)
            n += len(part)
            if n >= max_chars:
                break
        except Exception:
            continue
    return "\n\n".join(parts)[:max_chars]


def load_all_documents_from_knowledge_base(kb_root: Path | None = None) -> str:
    """Читает весь текст из всех .txt в папке knowledge_base (для RAG-индекса)."""
    if kb_root is None:
        kb_root = Path(__file__).resolve().parent.parent / "knowledge_base"
    if not kb_root.is_dir():
        return ""
    parts = []
    for path in sorted(kb_root.rglob("*.txt")):
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Пример на реальных данных из knowledge_base
    sample = load_sample_from_knowledge_base(max_chars=30_000)
    if not sample.strip():
        print("Нет .txt в knowledge_base — задайте sample_text вручную.")
        sample = "Пример текста. " * 500  # fallback

    chunk_sizes = [300, 500, 800, 1200]
    overlaps = [0, 50, 100, 150]
    df = analyze_chunking_strategy(sample, chunk_sizes, overlaps)
    print("Чанкование (фрагмент knowledge_base):")
    print(df.to_string(index=False))
