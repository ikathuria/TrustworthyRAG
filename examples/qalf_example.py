"""
Example usage of QALF Pipeline.
Demonstrates the complete end-to-end QALF system.

This example works with a local Neo4j instance.
Make sure your Neo4j is running before executing this script.
"""

from src.neo4j.neo4j_manager import Neo4jManager
from src.retriever.qalf_pipeline import QALFPipeline
import src.utils.constants as C
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Example QALF usage with local Neo4j instance"""
    
    print("="*80)
    print("QALF Pipeline Example - Using Local Neo4j Instance")
    print("="*80)
    print(f"Connecting to Neo4j at: {C.NEO4J_URI}")
    print(f"Database: {C.NEO4J_DB}, Username: {C.NEO4J_USERNAME}")
    print("="*80 + "\n")
    
    # Initialize Neo4j connection using constants
    # Update these in src/utils/constants.py if your Neo4j uses different settings
    try:
        neo4j_manager = Neo4jManager(
            uri=C.NEO4J_URI,  # Default: "neo4j://127.0.0.1:7687"
            username=C.NEO4J_USERNAME,  # Default: "neo4j"
            password=C.NEO4J_PASSWORD,  # Default: "test1234"
            database=C.NEO4J_DB  # Default: "neo4j"
        )
        print("✅ Successfully connected to Neo4j!\n")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("\nPlease ensure:")
        print("  1. Neo4j is running locally")
        print("  2. Connection settings in src/utils/constants.py are correct")
        print("  3. Neo4j credentials are correct")
        return
    
    # Initialize QALF pipeline
    pipeline = QALFPipeline(
        neo4j_manager=neo4j_manager,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384
    )
    
    # Setup indexes (one-time operation)
    print("Setting up Neo4j indexes...")
    try:
        pipeline.setup_indexes()
    except Exception as e:
        print(f"Index setup warning: {e}")
        print("Continuing anyway (indexes may already exist)...")
    
    # Example queries
    test_queries = [
        "What is GraphRAG?",
        "Compare GraphRAG architectures from 2024 to 2025",
        "Show me the chart of attack trends",
        "Why does adversarial poisoning happen in RAG systems?",
        "What are the vulnerabilities in multimodal RAG and how can they be mitigated?"
    ]
    
    print("\n" + "="*80)
    print("QALF Pipeline - Example Queries")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        try:
            # Retrieve using QALF
            results = pipeline.qalf_retrieve(query, top_k=5)
            
            print(f"\nRetrieved {len(results)} documents:")
            for result in results:
                print(f"  {result['rank']}. {result['title']}")
                print(f"     Score: {result['score']:.4f}, "
                      f"Consensus: {result['consensus']:.2f}")
                print(f"     Intent: {result['intent']}, "
                      f"Modalities: {result['modalities']}")
                print()
        
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    # Close connection
    neo4j_manager.close()
    print("\n✅ QALF example completed!")


if __name__ == "__main__":
    main()

