import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge_base.txt")


def load_knowledge_text() -> str:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
            ";",
            ":",
            " ",
            "",
        ],
    )
    return text_splitter.split_text(text)


def build_vector_index(chunks: List[str]) -> Tuple[TfidfVectorizer, List[str], any]:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, chunks, matrix


def retrieve_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    chunks: List[str],
    matrix,
    top_k: int = 3,
) -> List[str]:
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices if sims[i] > 0]


# --- Глобальный кэш базы знаний и индекса ---
_GLOBAL_TEXT: str | None = None
_GLOBAL_CHUNKS: List[str] | None = None
_GLOBAL_VECTORIZER: TfidfVectorizer | None = None
_GLOBAL_MATRIX = None


def _ensure_index_built() -> None:
    """
    Лениво загружает текст и строит индекс один раз.
    """
    global _GLOBAL_TEXT, _GLOBAL_CHUNKS, _GLOBAL_VECTORIZER, _GLOBAL_MATRIX

    if _GLOBAL_TEXT is not None:
        return

    text = load_knowledge_text()
    chunks = build_chunks(text)
    vectorizer, chunk_list, matrix = build_vector_index(chunks)

    _GLOBAL_TEXT = text
    _GLOBAL_CHUNKS = chunk_list
    _GLOBAL_VECTORIZER = vectorizer
    _GLOBAL_MATRIX = matrix


def create_llm() -> ChatOpenAI:
    load_dotenv()

    openrouter_base = "https://openrouter.ai/api/v1"
    model_name = "google/gemma-3n-e4b-it:free"

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    os.environ["OPENAI_API_KEY"] = openrouter_api_key or ""
    os.environ["OPENAI_BASE_URL"] = openrouter_base

    return ChatOpenAI(model=model_name)


def answer_with_rag(question: str) -> str:
    """
    Отвечает на вопрос, опираясь на базу знаний и LLM.
    """
    _ensure_index_built()

    relevant_chunks = retrieve_chunks(
        question,
        _GLOBAL_VECTORIZER,  # type: ignore[arg-type]
        _GLOBAL_CHUNKS or [],
        _GLOBAL_MATRIX,
        top_k=3,
    )

    context = "\n\n".join(relevant_chunks) if relevant_chunks else "Нет подходящих фрагментов."

    llm = create_llm()

    # Объединяем “роль гида” и инструкцию в один пользовательский промт,
    # без отдельного system-роля, чтобы не триггерить ошибку модели.
    full_prompt = (
        "Ты — официальный Гид Совета Maker'а во вселенной Earth-6160. "
        "Отвечай только на русском языке, без использования английских слов и эмодзи. "
        "Ты объясняешь устройство новой Ultimate-вселенной, её секторы, персонажей, технологии, "
        "скрытые зоны и правила выживания. Отвечай строго на основе переданного контекста "
        "из путеводителя Maker'а. Если нужной информации в контексте нет, прямо скажи об этом "
        "и не выдумывай детали.\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"
    )

    msg = llm.invoke([HumanMessage(content=full_prompt)])
    return msg.content


def generate_event(prompt: str) -> str:
    """
    Генератор сюжетных событий / сцен на основе базы знаний.
    """
    _ensure_index_built()

    # Берём релевантные куски к запросу игрока
    relevant_chunks = retrieve_chunks(
        prompt,
        _GLOBAL_VECTORIZER,  # type: ignore[arg-type]
        _GLOBAL_CHUNKS or [],
        _GLOBAL_MATRIX,
        top_k=4,
    )

    context = "\n\n".join(relevant_chunks) if relevant_chunks else "Нет подходящих фрагментов."

    llm = create_llm()

    full_prompt = (
        "Ты — сценарист и мастер, создающий события во вселенной Earth-6160 по путеводителю Maker'а. "
        "Отвечай только на русском языке, без английских слов и эмодзи. "
        "Генерируй яркие, но компактные сцены, завязки миссий, случайные встречи или осложнения "
        "для игроков. Используй только информацию из контекста и логичные выводы из неё, "
        "не вводи противоречащие лору элементы.\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ЗАПРОС ВЕДУЩЕГО ИГРЫ:\n{prompt}\n\n"
        "Сгенерируй одно конкретное событие или сцену для игроков."
    )

    msg = llm.invoke([HumanMessage(content=full_prompt)])
    return msg.content


if __name__ == "__main__":
    print("RAG-ассистент запущен. Введите вопрос (или пустую строку для выхода).")
    while True:
        q = input("Вопрос: ").strip()
        if not q:
            break
        try:
            answer = answer_with_rag(q)
            print(f"\nОтвет:\n{answer}\n")
        except Exception as e:
            print(f"Ошибка: {e}")