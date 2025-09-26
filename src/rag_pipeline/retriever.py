"""
retriever.py
Responsible for:
 - loading CSV/text datasets
 - basic text chunking
 - building and querying Chroma vectorstore
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class Retriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", persist_dir="chroma_db"):
        # Use persistent client (saves data locally)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Set embedding function
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model)

        # Create or get a collection
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, ids, documents, metadatas=None):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, text, n_results=5):
        return self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
