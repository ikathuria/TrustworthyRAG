
"""
Example usage of the Cybersecurity Knowledge Graph Pipeline
"""
import json
from pathlib import Path

# Import the main pipeline
from retriever.main_pipeline import CybersecKnowledgeGraphPipeline, create_pipeline


def create_example_config():
    """Create an example configuration file"""
    config = {
        "parser": {
            "type": "mineru",
            "device_map": "auto",
            "torch_dtype": "auto",
            "max_new_tokens": 1024,
            "temperature": 0.1
        },
        "extractor": {
            "type": "cyner",
            "transformer_model": "xlm-roberta-large",
            "use_heuristic": True,
            "flair_model": None,
            "confidence_threshold": 0.7
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "your_password_here",
            "database": "neo4j"
        },
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "api_key": "your_openai_api_key_here"
        },
        "pipeline": {
            "batch_size": 100,
            "enable_evaluation": True,
            "output_dir": "./outputs"
        },
        "evaluation": {
            "metrics": ["precision", "recall", "f1_score"],
            "test_size": 0.2
        }
    }

    # Save configuration to file
    with open("config.json", "w") as f:
        json.dump(config, indent=2, fp=f)

    print("Created example configuration: config.json")
    return config


def basic_pipeline_example():
    """Basic pipeline usage example"""
    print("\n=== Basic Pipeline Example ===")

    # Create configuration
    config = create_example_config()

    try:
        # Initialize pipeline
        pipeline = create_pipeline("config.json")

        # Example documents to process
        document_paths = [
            "sample_threat_report.pdf",
            "malware_analysis.pdf",
            "vulnerability_disclosure.pdf"
        ]

        # Note: These files don't exist in this example
        print(f"\nWould process documents: {document_paths}")

        # Show system status
        status = pipeline.get_system_status()
        print(f"\nSystem Status:")
        print(json.dumps(status, indent=2, default=str))

        # Example of how to process documents (commented out)
        # results = pipeline.process_documents(document_paths)
        # print(f"\nProcessing Results:")
        # print(json.dumps(results, indent=2, default=str))

        # Example queries
        example_queries = [
            "Find all malware related to APT groups",
            "Show vulnerabilities affecting Windows systems",
            "List all CVEs from 2024"
        ]

        print(f"\nExample queries that could be executed:")
        for query in example_queries:
            print(f"  - {query}")
            # result = pipeline.query_knowledge_graph(query, method="hybrid")
            # print(f"    Result: {result[:100]}...")

        # Get insights about a specific entity
        print(f"\nExample entity insight query:")
        print("  - pipeline.get_entity_insights('WannaCry', 'MALWARE')")

        # Shutdown
        pipeline.shutdown()
        print("\nPipeline example completed successfully!")

    except Exception as e:
        print(f"Pipeline example failed: {e}")
        print("Note: This is expected if dependencies are not installed")


def advanced_pipeline_example():
    """Advanced pipeline usage with custom processing"""
    print("\n=== Advanced Pipeline Example ===")

    try:
        # Create pipeline with custom configuration
        config_path = "config.json"
        pipeline = CybersecKnowledgeGraphPipeline(config_path)

        # Custom entity filtering function
        def filter_high_confidence_entities(entities, threshold=0.8):
            return [e for e in entities if e.confidence >= threshold]

        # Custom relation enhancement function
        def enhance_relations_with_context(relations, context_window=200):
            enhanced_relations = []
            for rel in relations:
                # Add more context to relations
                rel.context = f"Enhanced context: {rel.context[:context_window]}"
                enhanced_relations.append(rel)
            return enhanced_relations

        print("Advanced pipeline configuration ready")
        print("Custom processing functions defined:")
        print("  - filter_high_confidence_entities()")
        print("  - enhance_relations_with_context()")

        # Example of batch processing with custom logic
        print("\nBatch processing workflow:")
        print("1. Parse documents with MinerU2.5")
        print("2. Extract entities with CyNER")
        print("3. Filter high-confidence entities")
        print("4. Extract and enhance relations")
        print("5. Ingest into Neo4j knowledge graph")
        print("6. Enable hybrid retrieval (Cypher + Vector)")

        pipeline.shutdown()

    except Exception as e:
        print(f"Advanced example setup failed: {e}")


def evaluation_example():
    """Example of system evaluation"""
    print("\n=== Evaluation Example ===")

    # Example test queries for evaluation
    test_queries = [
        {
            "query": "What malware targets Windows systems?",
            "expected_entities": ["Windows", "malware_name"],
            "expected_relations": ["TARGETS"]
        },
        {
            "query": "Which CVEs affect Apache servers?",
            "expected_entities": ["Apache", "CVE-YYYY-XXXX"],
            "expected_relations": ["AFFECTS"]
        },
        {
            "query": "Show attack patterns used by APT29",
            "expected_entities": ["APT29", "attack_pattern"],
            "expected_relations": ["USES"]
        }
    ]

    print("Example evaluation queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: {query['query']}")
        print(f"   Expected entities: {query['expected_entities']}")
        print(f"   Expected relations: {query['expected_relations']}")

    print("\nEvaluation metrics that would be calculated:")
    print("- Entity extraction precision/recall")
    print("- Relation extraction accuracy")
    print("- Query response relevance")
    print("- System performance (latency, throughput)")


def demonstrate_oop_design():
    """Demonstrate OOP design principles used"""
    print("\n=== OOP Design Principles Demonstration ===")

    print("1. ABSTRACTION:")
    print("   - BaseParser abstract class defines parsing interface")
    print("   - BaseExtractor abstract class defines extraction interface")

    print("\n2. ENCAPSULATION:")
    print("   - Each component encapsulates its specific functionality")
    print("   - Private methods (prefixed with _) hide implementation details")

    print("\n3. INHERITANCE:")
    print("   - MinerUParser inherits from BaseParser")
    print("   - CyNERExtractor inherits from BaseExtractor")

    print("\n4. POLYMORPHISM:")
    print("   - Factory functions create different parser/extractor types")
    print("   - Same interface works with different implementations")

    print("\n5. COMPOSITION:")
    print("   - Pipeline orchestrator composes multiple components")
    print("   - Neo4jManager composed with LangChain integration")

    print("\n6. DESIGN PATTERNS USED:")
    print("   - Factory Pattern: create_parser(), create_extractor()")
    print("   - Strategy Pattern: different retrieval methods")
    print("   - Template Method: process_documents() pipeline")
    print("   - Facade Pattern: Pipeline class simplifies complex operations")


if __name__ == "__main__":
    print("Cybersecurity Knowledge Graph System - Example Usage")
    print("=" * 60)

    # Run all examples
    basic_pipeline_example()
    advanced_pipeline_example()
    evaluation_example()
    demonstrate_oop_design()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nTo get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up Neo4j database")
    print("3. Configure API keys in config.json")
    print("4. Run: python example_usage.py")
