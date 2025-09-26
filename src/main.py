"""
main.py
Example script that wires everything together.

python src/main.py --csv data/processed/rag_sample_dataset_100.csv --embedding local --ollama_model llama3.2:latest
"""

import os
from utils.embeddings import get_embeddings
from rag_pipeline.retriever import Retriever
from rag_pipeline.generator import OllamaWrapper
from rag_pipeline.pipeline import RAGPipeline


def build_pipeline(data_csv: str, embedding_backend: str = "local", embedding_model: str = "all-MiniLM-L6-v2", persist_dir: str = "./chroma_db", ollama_model: str = "llama3.2:latest"):
    # 1. embeddings
    embeddings = get_embeddings(backend=embedding_backend)

    # 2. retriever
    retr = Retriever(embedding_model=embedding_model, persist_dir=persist_dir)

    # Add docs if collection empty
    # naive check: if no documents, ingest
    try:
        if len(retr.query("hello", n_results=1)["documents"][0]) == 0:
            print("No docs in vectorstore; ingesting from CSV...")
            import pandas as pd
            df = pd.read_csv(data_csv)
            ids = [str(i) for i in range(len(df))]
            docs = df["content"].tolist()
            metas = df[["id", "title"]].to_dict(orient="records")
            retr.add_documents(ids, docs, metas)
    except Exception:
        print("Exception during quick retrieve (likely empty DB). Ingesting docs.")
        import pandas as pd
        df = pd.read_csv(data_csv)
        ids = [str(i) for i in range(len(df))]
        docs = df["content"].tolist()
        metas = df[["id", "title"]].to_dict(orient="records")
        retr.add_documents(ids, docs, metas)

    # 3. generator
    gen = OllamaWrapper(model=ollama_model)

    # 4. pipeline
    pipeline = RAGPipeline(retriever=retr, generator=gen)
    return pipeline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to dataset CSV (with `content` column).")
    parser.add_argument("--embedding", default="local",
                        help="Embedding backend: local")
    parser.add_argument(
        "--ollama_model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    args = parser.parse_args()

    pipeline = build_pipeline(
        data_csv=args.csv, embedding_backend=args.embedding, ollama_model=args.ollama_model)
    print("RAG pipeline ready. Try queries now.")

    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        res = pipeline.query(q)
        print("\n--- ANSWER ---\n")
        print(res["answer"])
        print("\n--- SOURCES ---\n")
        for c in res["contexts"]:
            print(c["metadata"], "\n", c["text"][:400], "\n---\n")
