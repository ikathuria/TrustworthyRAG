"""
Main pipeline script for multimodal document ingestion into Neo4j.

This script orchestrates:
1. Document parsing (text, images, tables)
2. Graph ingestion with LLM-based entity/relationship extraction
3. Vector embedding generation and storage

Usage:
    python main_pipeline.py
"""

from pathlib import Path
from typing import List
from langchain_ollama import OllamaLLM

from src.preprocessing.document_parser import DocumentParser
from src.neo4j.graph_ingestion import GraphDBManager
from src.neo4j.vector_ingestion import VectorDBManager
from src.utils.base_parser import ParsedContent
import src.utils.constants as C


def load_parsed_contents(parsed_pkl_paths: List[str]) -> List[ParsedContent]:
    """Load previously parsed content from pickle files"""
    parsed_contents = []
    for pkl_path in parsed_pkl_paths:
        try:
            parsed_content = ParsedContent.from_pkl(pkl_path)
            parsed_contents.append(parsed_content)
            print(f"Loaded: {pkl_path}")
        except Exception as e:
            print(f"Failed to load {pkl_path}: {str(e)}")
    return parsed_contents


def main():
    # Configuration
    file_paths = [
        "data/raw/WickedRose_andNCPH.pdf",
        # Add more files here
    ]

    # Step 1: Parse documents (multimodal extraction)
    print("=" * 80)
    print("STEP 1: Parsing Documents")
    print("=" * 80)

    parser_config = {
        "dtype": "auto",
        "device_map": "auto"
    }
    parser = DocumentParser(config=parser_config)
    parsed_contents = parser.parse_batch(file_paths)

    print(f"\nParsed {len(parsed_contents)} documents successfully")
    for pc in parsed_contents:
        print(f"  - {pc.source_file}: "
              f"{len(pc.text.get('content', ''))} chars text, "
              f"{len(pc.images)} images, "
              f"{len(pc.tables)} tables")

    # Step 2: Initialize LLM for graph extraction
    print("\n" + "=" * 80)
    print("STEP 2: Initializing LLM")
    print("=" * 80)

    llm = OllamaLLM(
        model="llama2",  # or "mistral", "qwen2.5", etc.
        temperature=0.0,
        base_url="http://localhost:11434"  # Default Ollama URL
    )
    print("LLM initialized: Ollama (llama2)")

    # Step 3: Graph ingestion with LLM-based extraction
    print("\n" + "=" * 80)
    print("STEP 3: Graph Ingestion (LLM-based)")
    print("=" * 80)

    graph_manager = GraphDBManager(
        llm=llm,
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        allowed_nodes=[],  # Empty = domain-agnostic
        allowed_relationships=[]  # Empty = domain-agnostic
    )

    graph_stats = graph_manager.ingest_batch_parsed_content(parsed_contents)
    print(f"\nGraph Ingestion Complete:")
    print(f"  - Documents: {graph_stats['documents']}")
    print(f"  - Text Chunks: {graph_stats['text_chunks']}")
    print(f"  - Images: {graph_stats['images']}")
    print(f"  - Tables: {graph_stats['tables']}")
    print(f"  - Entities: {graph_stats['entities']}")
    print(f"  - Relationships: {graph_stats['relationships']}")

    # Step 4: Vector embedding generation and storage
    print("\n" + "=" * 80)
    print("STEP 4: Vector Embedding Generation")
    print("=" * 80)

    vector_manager = VectorDBManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        text_embedding_model="all-MiniLM-L6-v2"
    )

    embedding_stats = vector_manager.batch_embed_parsed_contents(
        parsed_contents)
    print(f"\nEmbedding Complete:")
    print(f"  - Text Embeddings: {embedding_stats['text_embeddings']}")
    print(f"  - Image Embeddings: {embedding_stats['image_embeddings']}")
    print(f"  - Table Embeddings: {embedding_stats['table_embeddings']}")
    print(f"  - Entity Embeddings: {embedding_stats['entity_embeddings']}")

    # Step 5: Test retrieval
    print("\n" + "=" * 80)
    print("STEP 5: Testing Retrieval")
    print("=" * 80)

    test_query = "What are the main cybersecurity threats?"
    print(f"\nQuery: '{test_query}'")

    results = vector_manager.similarity_search_multimodal(
        query=test_query,
        modality="all",
        k=3
    )

    print(f"\nFound {len(results)} results:")
    for idx, result in enumerate(results):
        print(f"\n  Result {idx + 1} ({result['modality']}):")
        print(f"    Content: {result['content'][:200]}...")
        print(f"    Metadata: {result['metadata']}")

    # Get graph statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)

    final_stats = graph_manager.get_statistics()
    print(f"\nGraph Database Statistics:")
    print(f"  - Total Entities: {final_stats.get('total_entities', 0)}")
    print(f"  - Total Relations: {final_stats.get('total_relations', 0)}")

    # Close connections
    graph_manager.close()
    vector_manager.close()
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
