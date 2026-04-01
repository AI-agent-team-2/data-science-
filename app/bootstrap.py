from __future__ import annotations

import logging
import os
import sys

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)
PRODUCT_COLLECTION_SUFFIX = "_products"


def _auto_ingest_enabled() -> bool:
    raw = os.getenv("AUTO_INGEST_ON_START", "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _product_collection_name() -> str:
    return f"{settings.collection_name}{PRODUCT_COLLECTION_SUFFIX}"


def _collection_count(client: chromadb.PersistentClient, collection_name: str) -> int:
    try:
        return client.get_collection(collection_name).count()
    except Exception:
        return 0


def should_run_ingest() -> bool:
    if not _auto_ingest_enabled():
        logger.info("AUTO_INGEST_ON_START disabled; skipping startup ingest")
        return False

    logger.info("AUTO_INGEST_ON_START enabled; startup ingest will run")
    return True


def ensure_index_ready() -> None:
    if not should_run_ingest():
        return

    logger.info("Running startup ingest before bot launch")
    from app.rag.ingest import main as ingest_main

    exit_code = ingest_main()
    if exit_code != 0:
        raise RuntimeError(f"Startup ingest failed with exit code {exit_code}")

    client = chromadb.PersistentClient(path=settings.chroma_path)
    main_count = _collection_count(client, settings.collection_name)
    product_count = _collection_count(client, _product_collection_name())
    if main_count == 0 or product_count == 0:
        raise RuntimeError(
            "Startup ingest completed but Chroma collections are still empty: "
            f"{settings.collection_name}={main_count}, {_product_collection_name()}={product_count}"
        )

    logger.info("Startup ingest completed successfully")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ensure_index_ready()
    os.execv(sys.executable, [sys.executable, "-m", "app.bot.telegram_bot"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
