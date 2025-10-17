"""
embeddings.py
Local embeddings via sentence-transformers / HuggingFace
"""

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os


class LocalHFEmbedding(Embeddings):
    """
    Wraps a HuggingFace sentence-transformers model into LangChain Embeddings interface
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def get_embeddings(backend: str = "local", **kwargs) -> Embeddings:
    """
    Returns an embedding model. 
    - backend: "local" (default), more can be added
    """
    if backend == "local":
        return LocalHFEmbedding(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
