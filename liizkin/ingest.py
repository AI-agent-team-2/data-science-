from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from langchain_core.documents import Document
from dotenv import load_dotenv

PATH_FILES = 'knowledge_sources'

def chuck_of_text(path):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    documents = []
    for file in os.listdir(path=PATH_FILES):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": file}))
    return documents

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
def get_embeddings():
    embeddings = OpenAIEmbeddings(api_key=OPENROUTER_API_KEY, request_timeout=120)
    return embeddings
def get_retriever():
    docs = chuck_of_text(PATH_FILES)
    store = Chroma(collection_name="team2_collection", #название базы
        embedding_function=get_embeddings(), persist_directory="chroma_db")
    
    retriever=store.as_retriever(search_kwargs={'k': 3})
    return retriever