from langchain_ollama import OllamaLLM

from src.qalf.query_complexity import QueryComplexityClassifier
from src.qalf.query_intent import QueryIntentClassifier
from src.neo4j.graph_ingestion import GraphDBManager
from src.neo4j.vector_ingestion import VectorDBManager
from src.preprocessing.document_parser import DocumentParser
from src.retriever.hybrid_retrieval import HybridRetriever
import src.utils.constants as C


def adaptive_retrieve(query, hybrid_retriever, intent_classifier, complexity_classifier, top_k=7):
    # 1. Classify
    complexity = complexity_classifier.classify(query)
    intent = intent_classifier.classify(query)
    weights = intent_classifier.get_routing_weights(intent)

    # 2. Adaptive retrieval (use vector/graph/keyword based on weights)
    # NOTE: Currently "hybrid" retrieval, TODO: make this weighted by the classifier
    results = hybrid_retriever.retrieve(query, method="hybrid", top_k=top_k)

    # 3. Optionally sort/boost by weights, total complexity, or combine accordingly
    # (Placeholder: for now, just return the results)
    return {
        "complexity": complexity,
        "intent": intent,
        "weights": weights,
        "retrieval_results": results
    }


def main():
    file_paths = [C.TEST_PDF]

    # Step 1: Parse
    parser = DocumentParser(config={"dtype": "auto", "device_map": "auto"})
    parsed_contents = parser.parse_batch(file_paths)

    # Step 2: Graph ingestion
    llm = OllamaLLM(model=C.OLLAMA_MODEL, temperature=0.0,
                    base_url=C.OLLAMA_URI)
    graph_manager = GraphDBManager(
        llm=llm,
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB)
    graph_manager.ingest_batch_parsed_content(parsed_contents)

    # Step 3: Vector embedding
    vector_manager = VectorDBManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        text_embedding_model=C.TRANSFORMER_EMBEDDING_MODEL)
    vector_manager.batch_embed_parsed_contents(parsed_contents)

    # Step 4: Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        graph_manager, config={"embedding_model": C.TRANSFORMER_EMBEDDING_MODEL})

    # Step 5: Classifiers
    complexity_classifier = QueryComplexityClassifier(nlp_model=C.SPACY_MODEL)
    intent_classifier = QueryIntentClassifier()

    # Step 6: Adaptive QALF retrieval
    while True:
        query = input("Enter a query (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        results = adaptive_retrieve(
            query, hybrid_retriever, intent_classifier, complexity_classifier)
        print("\n=== QALF Adaptive Output ===")
        print(f"Complexity: {results['complexity']}")
        print(f"Intent: {results['intent']} (weights: {results['weights']})")
        print(f"Top Results [{results['retrieval_results']['count']}]:")
        for idx, r in enumerate(results['retrieval_results']['results']):
            print(f"{idx+1}. {r.get('text') or r.get('entity') or r.get('content') or ''} ({r.get('source', r.get('method'))})")
        print("="*40)


if __name__ == "__main__":
    main()
