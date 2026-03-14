"""
Product Lookup — поиск по товарам в базе знаний (артикул, название, характеристики).

Использование:
    from product_lookup import product_lookup

    product_lookup("PEXOWFHF030")
    product_lookup("редуктор давления 1/2")
"""
try:
    from rag import ask_product

    def product_lookup(query: str, retriever_k: int = 10) -> str:
        """
        Поиск товара по артикулу или названию в каталогах knowledge_base.

        Args:
            query: Артикул (например PEXOWFHF030) или описание товара.
            retriever_k: Сколько чанков подтягивать (по умолчанию 10).

        Returns:
            Текст с названием, артикулом, характеристиками или сообщение об отсутствии.
        """
        return ask_product(query, retriever_k=retriever_k)

except ImportError:

    def product_lookup(query: str, retriever_k: int = 10) -> str:
        return "Product Lookup недоступен: модуль rag не найден."


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "PEXOWFHF030"
    if q.lower() == "diagnose" or q.lower() == "диагностика":
        from rag import product_lookup_diagnose
        print(product_lookup_diagnose("PEXOWFHF030"))
    else:
        print("Запрос:", repr(q))
        print("-" * 50)
        print(product_lookup(q))  # noqa: E501
