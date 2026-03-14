# проверка основных зависимостей
# Проверка основных зависимостей
import sys
print(f"Python version: {sys.version}")

try:
    import langchain
    print(f"✅ LangChain: {langchain.__version__}")
except ImportError:
    print("❌ LangChain не установлен. Выполните: pip install -r requirements.txt")

try:
    import sentence_transformers
    print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
except ImportError:
    print("❌ Sentence Transformers не установлен")

try:
    import faiss
    print(f"✅ FAISS: установлен")
except ImportError:
    print("❌ FAISS не установлен")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   GPU доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch не установлен")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy не установлен")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

print("\n🎉 Если все ✅ — можно продолжить работу!")