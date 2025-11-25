"""
Evaluate Adaptive RAG Pipeline with QALF.

This script orchestrates:
1. Evaluate the results using the ground truth

Usage:
    python evaluate.py [--ground_truth {ground_truth}]
"""

import json
import pandas as pd
from typing import List, Dict, Any
import argparse

from src.neo4j.neo4j_manager import Neo4jManager
from src.retriever.qalf_pipeline import QALFPipeline
import src.utils.constants as C


def load_ground_truth(ground_truth_path: str) -> Dict[str, Any]:
    """Load the ground truth from a file"""
    questions = []
    answers = []

    with open(ground_truth_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            questions.append(item['question'])
            answers.append(item['answer'])

    return questions, answers

def evaluate_pipeline(questions: List[str], answers: List[str]):
    """
    Evaluate the results using the questions and answers
    """
    df = pd.DataFrame(
        columns=["Question", "Correct Answer", "Generated Answer", "Sources"]
    )

    # Initialize Neo4j connection
    print("\n1. Connecting to Neo4j...")
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB
    )
    print("✓ Neo4j connected")
    
    qalf_pipeline = QALFPipeline(
        neo4j_manager=neo4j_manager,
        embedding_model=C.TRANSFORMER_EMBEDDING_MODEL,
        enable_generator=True,
        embedding_dim=384,
    )
    print("✓ QALF pipeline initialized")

    for question, answer in zip(questions, answers):
        full_result = qalf_pipeline.qalf_retrieve_and_generate(question, top_k=7)
        results = full_result.get("retrieval", {}).get("results", [])
        generation = full_result.get("generation")
        answer_qalf = generation.get("response", "")

        if results:
            # Display complexity and intent from first result (all results have same metadata)
            first_result = results[0]
            complexity = first_result.get('complexity', 'N/A')
            intent = first_result.get('intent', 'N/A')
            modalities = first_result.get('modalities', [])

            print(f"\n📊 Query Analysis:")
            print(f"  Complexity (4D): {complexity}")
            print(f"  Intent: {intent}")
            print(f"  Active Modalities: {', '.join(modalities)}")

            # Display generated response if available
            if generation and generation.get("success"):
                print(f"\n❓ Question: {question}")
                print(f"\n✅ Correct Answer: {answer}")

                answer_qalf = generation.get("response", "")

                print(f"\n💬 Generated Answer:")
                print(answer_qalf)
                print("-" * 80)

                # Display sources
                sources = generation.get("sources", [])
                if sources:
                    print(f"\n📚 Sources ({len(sources)} documents):")
                    for i, source in enumerate(sources, 1):
                        print(f"  {i}. {source.get('title', 'Unknown')} "
                              f"({source.get('chunks_used', 0)} chunks)")

                df = pd.concat([df, pd.DataFrame({"Question": [question], "Correct Answer": [answer], "Generated Answer": [answer_qalf], "Sources": [sources]})], ignore_index=True)

                print(
                    f"\n⏱️  Generation time: {generation.get('generation_time', 0):.2f}s")
                print("-" * 80)

            print(f"\n📄 Top Results [{len(results)}]:")
            print("-" * 80)

            for result in results:
                print(f"\n[{result['rank']}] {result['title']}")
                print(f"    Score: {result['score']:.4f} | "
                        f"Consensus: {result['consensus']:.2f}")

                # Try to get additional metadata if available
                doc_id = result.get('doc_id', '')
                if doc_id:
                    print(f"    Doc ID: {doc_id}")

            print("\n" + "-" * 80)
        else:
            print("\n⚠️  No results found.")
            print("This could mean:")
            print("  - No documents match the query")
            print("  - Indexes may need to be created")
            print("  - Documents may need to be ingested first")
        
        print("=" * 80)

    # Cleanup
    neo4j_manager.close()
    print("\n✓ QALF pipeline closed")

    print("-" * 80)
    print("\n📊 QALF Results:")
    print(df)

    df.to_csv("qalf_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Adaptive RAG Pipeline with QALF")
    parser.add_argument(
        '--ground_truth',
        type=str,
        default=C.GROUND_TRUTH_PATH,
        help='Path to the ground truth file'
    )
    args = parser.parse_args()
    questions, answers = load_ground_truth(args.ground_truth)
    evaluate_pipeline(questions, answers)
