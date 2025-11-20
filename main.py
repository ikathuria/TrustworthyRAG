"""
Complete Adaptive RAG Pipeline with QALF integration.

This script orchestrates:
1. Document parsing (text, images, tables)
2. Graph ingestion with LLM-based entity/relationship extraction
3. Vector embedding generation and storage
4. Adaptive query-based retrieval with QALF classifiers
5. Interactive query interface

Usage:
    python main_pipeline_updated.py [--mode {ingest|query|both}]
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
import argparse

from src.preprocessing.document_parser import DocumentParser
from src.neo4j.graph_ingestion import GraphDBManager
from src.neo4j.vector_ingestion import VectorDBManager
from src.retrieval.hybrid_retrieval import HybridRetriever
from src.qalf.query_complexity import QueryComplexityClassifier
from src.qalf.query_intent import QueryIntentClassifier
from src.utils.base_parser import ParsedContent
import src.utils.constants as C


def load_parsed_contents(parsed_pkl_paths: List[str]) -> List[ParsedContent]:
    """Load previously parsed content from pickle files"""
    parsed_contents = []
    for pkl_path in parsed_pkl_paths:
        try:
            parsed_content = ParsedContent.from_pkl(pkl_path)
            parsed_contents.append(parsed_content)
            print(f"✓ Loaded: {pkl_path}")
        except Exception as e:
            print(f"✗ Failed to load {pkl_path}: {str(e)}")
    return parsed_contents


def ingest_pipeline(file_paths: List[str]) -> Dict[str, Any]:
    """
    Run the ingestion pipeline: parse → graph → embeddings
    
    Args:
        file_paths: List of document paths to ingest
        
    Returns:
        Dict with ingestion statistics
    """
    stats = {
        'parsed': 0,
        'graph': {},
        'embeddings': {}
    }

    # Step 1: Parse documents
    print("\n" + "=" * 80)
    print("STEP 1: Parsing Documents")
    print("=" * 80)

    parser_config = {
        "dtype": "auto",
        "device_map": "auto"
    }
    parser = DocumentParser(config=parser_config)
    parsed_contents = parser.parse_batch(file_paths)
    stats['parsed'] = len(parsed_contents)

    print(f"\n✓ Parsed {len(parsed_contents)} documents successfully")
    for pc in parsed_contents:
        print(f"  - {pc.source_file}: "
              f"{len(pc.text.get('content', ''))} chars text, "
              f"{len(pc.images)} images, "
              f"{len(pc.tables)} tables")

    # Step 2: Initialize LLM
    print("\n" + "=" * 80)
    print("STEP 2: Initializing LLM")
    print("=" * 80)

    llm = OllamaLLM(
        model=C.OLLAMA_MODEL,
        temperature=0.0,
        base_url=C.OLLAMA_URI
    )
    print(f"✓ LLM initialized: Ollama ({C.OLLAMA_MODEL})")

    # Step 3: Graph ingestion
    print("\n" + "=" * 80)
    print("STEP 3: Graph Ingestion (LLM-based)")
    print("=" * 80)

    graph_manager = GraphDBManager(
        llm=llm,
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        allowed_nodes=[],
        allowed_relationships=[]
    )

    graph_stats = graph_manager.ingest_batch_parsed_content(parsed_contents)
    stats['graph'] = graph_stats

    print(f"\n✓ Graph Ingestion Complete:")
    print(f"  - Documents: {graph_stats['documents']}")
    print(f"  - Text Chunks: {graph_stats['text_chunks']}")
    print(f"  - Images: {graph_stats['images']}")
    print(f"  - Tables: {graph_stats['tables']}")
    print(f"  - Entities: {graph_stats['entities']}")
    print(f"  - Relationships: {graph_stats['relationships']}")

    # Step 4: Vector embeddings
    print("\n" + "=" * 80)
    print("STEP 4: Vector Embedding Generation")
    print("=" * 80)

    vector_manager = VectorDBManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        text_embedding_model=C.TRANSFORMER_EMBEDDING_MODEL
    )

    embedding_stats = vector_manager.batch_embed_parsed_contents(
        parsed_contents)
    stats['embeddings'] = embedding_stats

    print(f"\n✓ Embedding Complete:")
    print(f"  - Text Embeddings: {embedding_stats['text_embeddings']}")
    print(f"  - Image Embeddings: {embedding_stats['image_embeddings']}")
    print(f"  - Table Embeddings: {embedding_stats['table_embeddings']}")
    print(f"  - Entity Embeddings: {embedding_stats['entity_embeddings']}")

    # Cleanup
    graph_manager.close()
    vector_manager.close()

    return stats


def query_pipeline():
    """
    Run interactive query pipeline with QALF adaptive retrieval
    """
    print("\n" + "=" * 80)
    print("INITIALIZING QUERY PIPELINE")
    print("=" * 80)

    # Initialize components
    print("\n1. Loading QALF Classifiers...")
    complexity_classifier = QueryComplexityClassifier(nlp_model=C.SPACY_MODEL)
    intent_classifier = QueryIntentClassifier()
    print("✓ Classifiers loaded")

    print("\n2. Connecting to Neo4j...")
    from src.neo4j.neo4j_manager import Neo4jManager
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB
    )
    print("✓ Neo4j connected")

    print("\n3. Initializing Hybrid Retriever...")
    hybrid_retriever = HybridRetriever(
        neo4j_manager=neo4j_manager,
        embedding_model=C.TRANSFORMER_EMBEDDING_MODEL
    )
    print("✓ Hybrid retriever ready")

    print("\n" + "=" * 80)
    print("ADAPTIVE QUERY INTERFACE (QALF)")
    print("=" * 80)
    print("Commands:")
    print("  - Enter a query to search")
    print("  - 'stats' - Show database statistics")
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit the interface")
    print("=" * 80)

    # Interactive loop
    while True:
        print("\n" + "-" * 80)
        query = input("Query> ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit']:
            print("\n✓ Exiting query interface...")
            break

        if query.lower() == 'help':
            print("\nAvailable commands:")
            print("  - Enter any natural language query")
            print("  - 'stats' - Show database statistics")
            print("  - 'quit' or 'exit' - Exit")
            continue

        if query.lower() == 'stats':
            print("\nFetching database statistics...")
            stats_query = """
            MATCH (d:Document) WITH count(d) as docs
            MATCH (c:Chunk) WITH docs, count(c) as chunks
            MATCH (e:Entity) WITH docs, chunks, count(e) as entities
            MATCH ()-[r]->() WITH docs, chunks, entities, count(r) as rels
            RETURN docs, chunks, entities, rels
            """
            result = neo4j_manager.query_graph(stats_query)
            if result:
                r = result[0]
                print(f"\n📊 Database Statistics:")
                print(f"  - Documents: {r.get('docs', 0)}")
                print(f"  - Chunks: {r.get('chunks', 0)}")
                print(f"  - Entities: {r.get('entities', 0)}")
                print(f"  - Relationships: {r.get('rels', 0)}")
            continue

        # Process query with QALF
        print(f"\n🔍 Processing: '{query}'")

        # Step 1: Classify
        print("\n1️⃣ QALF Classification:")
        complexity = complexity_classifier.classify(query)
        complexity_score = complexity_classifier.total_score(query)
        intent = intent_classifier.classify(query)
        weights = intent_classifier.get_routing_weights(intent)

        print(f"   Complexity: {complexity_score} {complexity}")
        print(f"   Intent: {intent}")
        print(f"   Routing Weights: {weights}")

        # Step 2: Retrieve with adaptive weights
        print("\n2️⃣ Hybrid Retrieval:")
        results = hybrid_retriever.retrieve(
            query=query,
            method="hybrid",
            top_k=5,
            weights=weights
        )

        if not results.get("success"):
            print(
                f"   ✗ Retrieval failed: {results.get('error', 'Unknown error')}")
            continue

        print(f"   ✓ Found {results['count']} results")
        print(f"   Method: {results['method']}")
        if 'weights_used' in results:
            print(f"   Weights: {results['weights_used']}")

        # Step 3: Display results
        print("\n3️⃣ Top Results:")
        print("-" * 80)

        for idx, result in enumerate(results["results"], 1):
            print(
                f"\n[{idx}] Score: {result.get('fusion_score', result.get('score', 0.0)):.4f}")
            print(
                f"    Sources: {', '.join(result.get('retrieval_sources', [result.get('source', 'unknown')]))}")

            # Display content based on result type
            if 'entity' in result:
                print(
                    f"    Entity: {result['entity']} ({result.get('entity_type', 'N/A')})")
                if result.get('related_entities'):
                    print(
                        f"    Related: {', '.join(result['related_entities'][:3])}")

            content = result.get('content', '')
            if content:
                # Truncate long content
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"    Content: {content}")

            if result.get('modality'):
                print(f"    Modality: {result['modality']}")

            if result.get('document'):
                print(f"    Document: {result['document']}")

        print("\n" + "-" * 80)

    # Cleanup
    neo4j_manager.close()
    print("\n✓ Pipeline closed")


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Adaptive RAG Pipeline with QALF")
    parser.add_argument(
        '--mode',
        choices=['ingest', 'query', 'both'],
        default='both',
        help='Pipeline mode: ingest documents, query interface, or both'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=[C.TEST_PDF],
        help='Document files to ingest (for ingest/both modes)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ADAPTIVE MULTIMODAL RAG PIPELINE WITH QALF")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")

    try:
        if args.mode in ['ingest', 'both']:
            print(f"\nFiles to ingest: {args.files}")
            stats = ingest_pipeline(args.files)

            print("\n" + "=" * 80)
            print("INGESTION SUMMARY")
            print("=" * 80)
            print(f"✓ Parsed: {stats['parsed']} documents")
            print(
                f"✓ Graph Nodes: {stats['graph'].get('entities', 0)} entities")
            print(
                f"✓ Graph Relations: {stats['graph'].get('relationships', 0)} relationships")
            print(
                f"✓ Vector Embeddings: {sum(stats['embeddings'].values())} total")

        if args.mode in ['query', 'both']:
            if args.mode == 'both':
                input("\nPress Enter to start query interface...")
            query_pipeline()

        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
