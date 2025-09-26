"""
pipeline.py
Composes Retriever + Generator into a modular RAG pipeline. Designed to be extended.
"""

from rag_pipeline.generator import build_prompt


class RAGPipeline:
    def __init__(self, retriever, generator, top_k=4):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def query(self, query_text: str, answer_max_tokens: int = 512, instructions: str = None):
        # Step 1: retrieve
        results = self.retriever.query(query_text, n_results=self.top_k)
        # Chroma returns a dict with 'documents', 'metadatas', etc.
        contexts = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results.get("documents", [[]])[0],
                                 results.get("metadatas", [[]])[0])
        ]
        contexts = [c for c in contexts if c.get("text")]
        if not contexts:
            return {"query": query_text, "answer": "No relevant context found.", "contexts": []}
        
        seen_texts = set()
        unique_contexts = []
        for c in contexts:
            text = c["text"].strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_contexts.append(c)
        contexts = unique_contexts
        print("\n[DEBUG] Retrieved contexts:")
        for c in contexts:
            print(c["metadata"], "\n", c["text"][:400], "\n---\n")

        # Step 2: build prompt
        prompt = build_prompt(query_text, contexts, instructions)
        print("\n[DEBUG] Full prompt sent to Ollama:")
        print(prompt)

        # Step 3: call generator
        answer = self.generator.generate(prompt, max_tokens=answer_max_tokens)

        return {
            "query": query_text,
            "answer": answer,
            "contexts": contexts,
            "prompt": prompt
        }
