import os
import re

PATH_DATA = "knowledge_sources"

def clean_text(text):
    #удаление спецсимволов
    text_clean = re.sub(r'[^\w\s]', '', text).strip()
    #удаление лищних пробелов
    text_clean = re.sub(r'\s+', '', text_clean).strip()
    #удаление html-тегов
    text_clean = re.sub(r'<.*?>', '', text_clean).strip
    return text_clean

def struct_data(text, filename):
    name = os.path.splitext(filename)[0]

    formatted = f"""
                        Название:
                        {name}

                        Описание:
                        {text}

                        Характеристики:
                        -

                        Применение:
                        -
                        """
    return formatted.strip()

def format_file(path):
    for file_name in os.listdir(path=path):
        if file_name.endswith(".pdf") or file_name.endswith(".md"):

            file_path = os.path.join(path, file_name)

            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                text = f.read()

            text = clean_text(text)

            formatted_text = struct_data(text, file_name)

            new_path = os.path.join(path, file_name.replace(".md", ".txt"))

            with open(new_path, "w", encoding="utf-8", errors='ignore') as f:
                f.write(formatted_text)

            print("Обработан:", file_name)


format_file(PATH_DATA)
