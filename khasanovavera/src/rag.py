"""
RAG: документы из knowledge_base → чанки → ChromaDB → поиск + ответ через LLM (Ollama).
Использование: from rag import ask, rebuild_index
"""
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from chunk_strategy import load_all_documents_from_knowledge_base

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_ROOT = PROJECT_ROOT / "knowledge_base"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db_rag"
CHUNK_TEXTS_FILE = CHROMA_PERSIST_DIR / "chunk_texts.json"
CHROMA_IDS_FILE = CHROMA_PERSIST_DIR / "chroma_ids.json"
COLLECTION_NAME = "knowledge_base"

# Параметры чанкования (по рекомендациям из chunk_strategy)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVER_K = 10  # сколько чанков отдаём в контекст
FETCH_K = 60  # сколько чанков достаём по семантике (больше — выше шанс вытащить нужный)

# Product Lookup: меньше чанков = быстрее и точнее (один товар редко в 10 чанках)
PRODUCT_RETRIEVER_K = 5
PRODUCT_FETCH_K = 20

# Ленивые синглтоны (инициализация при первом вызове ask/rebuild)
_vectorstore = None
_chain = None
_product_chain = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    return _embeddings


def _get_llm():
    try:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="gemma3:4b", temperature=0.2)
    except Exception:
        return None


def _get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _load_and_chunk() -> list[str]:
    text = load_all_documents_from_knowledge_base(KB_ROOT)
    if not text.strip():
        return []
    return _get_splitter().split_text(text)


def _get_vectorstore(force_rebuild: bool = False):
    global _vectorstore
    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    from langchain_chroma import Chroma

    # Если индекс уже сохранён на диске — открываем его и при необходимости восстанавливаем chunk_texts
    if (not force_rebuild) and CHROMA_PERSIST_DIR.is_dir():
        embeddings = _get_embeddings()
        try:
            _vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=embeddings,
            )
        except TypeError:
            _vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding=embeddings,
            )
        if not CHUNK_TEXTS_FILE.exists():
            _ensure_chunk_texts_from_collection(_vectorstore)
        return _vectorstore

    chunks = _load_and_chunk()
    if not chunks:
        raise ValueError(
            "Нет текста в knowledge_base. Добавьте .txt файлы в папку knowledge_base."
        )

    embeddings = _get_embeddings()
    # Явные id — чтобы по ним подтягивать чанки при точном совпадении запроса
    ids = [str(i) for i in range(len(chunks))]
    _vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        ids=ids,
    )
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_TEXTS_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    CHROMA_IDS_FILE.write_text(json.dumps(ids, ensure_ascii=True), encoding="utf-8")
    return _vectorstore


# Варианты слов для сопоставления запроса с текстом (опечатки, синонимы)
_QUERY_NORMALIZE = (
    ("шитого", "сшитого"),
    ("шитый", "сшитый"),
)


def _normalize_for_match(text: str) -> str:
    """Нормализация для сопоставления: нижний регистр, синонимы, пробелы, дефисы."""
    if not text:
        return ""
    t = text.lower().strip()
    for a, b in _QUERY_NORMALIZE:
        t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"-", "", t)  # PEX-OWF-HF030 и PEXOWFHF030 совпадут
    return t


def _normalize_for_product_match(text: str) -> str:
    """Жёсткая нормализация для артикулов: без пробелов и дефисов (для подстрочного поиска)."""
    t = _normalize_for_match(text)
    t = re.sub(r"\s", "", t)  # "PEX OWF HF030" -> "pexowfhf030"
    return t


def _get_docs_by_ids(vs, ids: list[str]) -> list[Document]:
    """Достаёт документы из Chroma по списку id."""
    if not ids:
        return []
    try:
        res = vs._collection.get(ids=ids)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or [{}] * len(docs)
        return [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
    except Exception:
        return []


def _get_docs_by_ids_fallback(ids: list[str]) -> list[Document]:
    """Если Chroma не вернул документы по id — собираем из chunk_texts.json по chroma_ids."""
    if not ids or not CHUNK_TEXTS_FILE.exists() or not CHROMA_IDS_FILE.exists():
        return []
    try:
        chunks = json.loads(CHUNK_TEXTS_FILE.read_text(encoding="utf-8"))
        chroma_ids = json.loads(CHROMA_IDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(chunks, list) or not isinstance(chroma_ids, list) or len(chroma_ids) < len(chunks):
        return []
    id_to_idx = {str(cid): i for i, cid in enumerate(chroma_ids)}
    out = []
    for id_ in ids:
        i = id_to_idx.get(str(id_))
        if i is not None and i < len(chunks):
            out.append(Document(page_content=str(chunks[i]), metadata={}))
    return out


def _ensure_chunk_texts_from_collection(vs) -> None:
    """Если chunk_texts.json нет — заполняем из коллекции Chroma. Сохраняем и реальные id для точного поиска."""
    if CHUNK_TEXTS_FILE.exists():
        return
    try:
        res = vs._collection.get(include=["documents"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        if not ids or not docs:
            return
        def sort_key(item):
            i, _ = item
            try:
                return (0, int(i))
            except (ValueError, TypeError):
                return (1, str(i))
        sorted_pairs = sorted(zip(ids, docs), key=sort_key)
        sorted_ids = [id_ for id_, _ in sorted_pairs]
        chunks = [doc for _, doc in sorted_pairs]
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        CHUNK_TEXTS_FILE.write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
        )
        CHROMA_IDS_FILE.write_text(
            json.dumps(sorted_ids, ensure_ascii=True), encoding="utf-8"
        )
    except Exception:
        pass


def _exact_match_chunk_ids(question: str) -> list[str]:
    """Возвращает id чанков Chroma, в которых запрос встречается как подстрока (после нормализации)."""
    if not CHUNK_TEXTS_FILE.exists():
        return []
    q = _normalize_for_match(question)
    if len(q) < 2:
        return []
    try:
        chunks = json.loads(CHUNK_TEXTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(chunks, list):
        return []
    # Сначала обычное сопоставление (с дефисами убранными)
    indices = [i for i, t in enumerate(chunks) if q in _normalize_for_match(str(t))]
    # Если не нашли и запрос похож на артикул — сопоставление без пробелов
    if not indices and re.match(r"^[\w\d\-\.\s]+$", question.strip()):
        q_strict = _normalize_for_product_match(question)
        if len(q_strict) >= 2:
            indices = [
                i for i, t in enumerate(chunks)
                if q_strict in _normalize_for_product_match(str(t))
            ]
    if not indices:
        return []
    if CHROMA_IDS_FILE.exists():
        try:
            chroma_ids = json.loads(CHROMA_IDS_FILE.read_text(encoding="utf-8"))
            if isinstance(chroma_ids, list) and len(chroma_ids) >= len(chunks):
                return [str(chroma_ids[i]) for i in indices]
        except Exception:
            pass
    return [str(i) for i in indices]


def _hybrid_retrieve(question: str, retriever_k: int, fetch_k: int):
    """Сначала чанки с точным вхождением запроса, затем семантика + переранжирование по словам."""
    vs = _get_vectorstore()
    # 1) Чанки, где запрос встречается буквально (максимальный приоритет)
    exact_ids = _exact_match_chunk_ids(question)
    exact_docs = _get_docs_by_ids(vs, exact_ids) if exact_ids else []
    seen_content = {d.page_content for d in exact_docs}

    # 2) Семантический поиск (кандидаты)
    docs = vs.similarity_search(question, k=fetch_k)
    short = " ".join(w for w in question.split() if len(w) > 2)[:80]
    if short and short != question:
        docs2 = vs.similarity_search(short, k=fetch_k // 2)
        by_content = {d.page_content: d for d in docs}
        for d in docs2:
            if d.page_content not in by_content:
                by_content[d.page_content] = d
        docs = list(by_content.values())

    # Убираем дубликаты с exact_docs
    semantic = [d for d in docs if d.page_content not in seen_content]
    q = _normalize_for_match(question)
    words = [w for w in q.split() if len(w) > 1]

    def rank(doc):
        content = doc.page_content.lower()
        content_n = _normalize_for_match(content)
        if q in content or q in content_n:
            return (0, 0)
        match_count = sum(1 for w in words if w in content or w in content_n)
        return (1, -match_count)

    semantic.sort(key=rank)
    # Сначала все с точным вхождением, потом лучшие по семантике/словам, всего retriever_k
    result = exact_docs + semantic
    return result[:retriever_k]


def _hybrid_retrieve_product(question: str, retriever_k: int, fetch_k: int):
    """
    Поиск для Product Lookup: быстрее и точнее.
    Если есть точное вхождение (артикул/название) — только эти чанки, без семантики.
    Иначе один семантический поиск и переранжирование по словам.
    """
    vs = _get_vectorstore()
    q = _normalize_for_match(question)

    exact_ids = _exact_match_chunk_ids(question)
    exact_docs = _get_docs_by_ids(vs, exact_ids) if exact_ids else []
    if exact_ids and not exact_docs:
        exact_docs = _get_docs_by_ids_fallback(exact_ids)

    if exact_docs:
        return exact_docs[:retriever_k]

    docs = vs.similarity_search(question, k=fetch_k)
    if not docs:
        return []
    words = [w for w in q.split() if len(w) > 1]

    def rank(doc):
        content = doc.page_content.lower()
        content_n = _normalize_for_match(content)
        if q in content or q in content_n:
            return (0, 0)
        match_count = sum(1 for w in words if w in content or w in content_n)
        return (1, -match_count)

    docs.sort(key=rank)
    return docs[:retriever_k]


def _build_chain(retriever_k: int = RETRIEVER_K):
    global _chain
    vs = _get_vectorstore()
    fetch_k = max(FETCH_K, retriever_k * 2)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def retrieve_and_format(question: str) -> str:
        docs = _hybrid_retrieve(question, retriever_k=retriever_k, fetch_k=fetch_k)
        return format_docs(docs)

    context_fn = RunnableLambda(lambda q: retrieve_and_format(q))
    llm = _get_llm()
    if llm is None:
        _chain = (
            {"context": context_fn, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: "Найденные фрагменты:\n\n" + x["context"])
        )
        return _chain

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты помощник. Отвечай только по приведённому контексту из базы знаний.
Если в контексте нет ответа — скажи об этом кратко. Язык ответа — как у пользователя."""),
        ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
    ])
    _chain = (
        {"context": context_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return _chain


# Промпт для поиска по товарам: только факты, без лишнего
_PRODUCT_SYSTEM = """Ты справочник по каталогу. Контекст ниже — это уже найденные фрагменты каталога.
Если в контексте есть артикул или название товара — обязательно опиши товар по ним, не пиши «Не найден».
Формат: название товара, артикул, ключевые параметры (диаметр, длина, давление и т.д.). Без вступлений.
Только если в контексте действительно нет ни артикула, ни подходящего товара — ответь: «Не найден». Язык ответа — как у пользователя."""


def _build_product_chain(retriever_k: int = PRODUCT_RETRIEVER_K):
    """Цепочка RAG для Product Lookup: меньше чанков, приоритет точному совпадению."""
    global _product_chain
    fetch_k = max(PRODUCT_FETCH_K, retriever_k * 2)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def retrieve_and_format(question: str) -> str:
        docs = _hybrid_retrieve_product(
            question, retriever_k=retriever_k, fetch_k=fetch_k
        )
        return format_docs(docs)

    context_fn = RunnableLambda(lambda q: retrieve_and_format(q))
    llm = _get_llm()
    if llm is None:
        _product_chain = (
            {"context": context_fn, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: "Найденные фрагменты (товары):\n\n" + x["context"])
        )
        return _product_chain

    prompt = ChatPromptTemplate.from_messages([
        ("system", _PRODUCT_SYSTEM),
        ("human", "Контекст:\n{context}\n\nВопрос (товар/артикул): {question}"),
    ])
    _product_chain = (
        {"context": context_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return _product_chain


def ask_product(question: str, retriever_k: int = PRODUCT_RETRIEVER_K) -> str:
    """Поиск по товарам: артикул, название, характеристики (product lookup)."""
    question = (question or "").strip()
    question = re.sub(r"\s+", " ", question)
    if not question:
        return "Напишите артикул или название товара."
    try:
        chain = _build_product_chain(retriever_k=retriever_k)
        return chain.invoke(question)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Ошибка поиска товара: {e}"


def ask(question: str, retriever_k: int = RETRIEVER_K) -> str:
    """Задать вопрос по базе знаний (RAG)."""
    question = (question or "").strip()
    if not question:
        return "Напишите вопрос."
    try:
        chain = _build_chain(retriever_k=retriever_k)
        return chain.invoke(question)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Ошибка RAG: {e}. Проверьте, что папка knowledge_base заполнена и при необходимости выполните rebuild_index()."


def product_lookup_diagnose(query: str = "PEXOWFHF030") -> str:
    """
    Диагностика: почему product lookup может не находить товар.
    Вызов: python -c "from rag import product_lookup_diagnose; print(product_lookup_diagnose('артикул'))"
    """
    lines = [
        "=== Диагностика Product Lookup ===",
        f"Запрос: {query!r}",
        f"Папка chroma_db_rag: {'есть' if CHROMA_PERSIST_DIR.is_dir() else 'НЕТ'}",
        f"Файл chunk_texts.json: {'есть' if CHUNK_TEXTS_FILE.exists() else 'НЕТ (пересоберите индекс: rebuild_index())'}",
        f"Файл chroma_ids.json: {'есть' if CHROMA_IDS_FILE.exists() else 'НЕТ'}",
    ]
    if CHUNK_TEXTS_FILE.exists():
        try:
            chunks = json.loads(CHUNK_TEXTS_FILE.read_text(encoding="utf-8"))
            n = len(chunks) if isinstance(chunks, list) else 0
            lines.append(f"Чанков в базе: {n}")
            exact_ids = _exact_match_chunk_ids(query)
            lines.append(f"Точное совпадение по запросу: найдено чанков = {len(exact_ids)}")
            if not exact_ids and n > 0:
                q = _normalize_for_match(query)
                lines.append(f"Нормализованный запрос: {q!r}")
                sample = _normalize_for_match(str(chunks[0]))[:200] if chunks else ""
                lines.append(f"Пример нормализованного чанка (начало): {sample!r}...")
        except Exception as e:
            lines.append(f"Ошибка: {e}")
    return "\n".join(lines)


def rebuild_index() -> str:
    """Переиндексировать все документы из knowledge_base (удаляет старый индекс)."""
    import shutil
    global _vectorstore, _chain, _product_chain, _embeddings
    _vectorstore = None
    _chain = None
    _product_chain = None
    _embeddings = None
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
    chunks = _load_and_chunk()
    if not chunks:
        return "Нет .txt в knowledge_base — индекс не создан."
    _get_vectorstore(force_rebuild=True)
    return f"Индекс пересобран: {len(chunks)} чанков."


if __name__ == "__main__":
    import sys
    args = [a.strip() for a in sys.argv[1:] if a.strip()]
    if args and args[0].lower() == "rebuild":
        print(rebuild_index())
    else:
        q = " ".join(args) if args else "Что есть в базе?"
        print(ask(q))
