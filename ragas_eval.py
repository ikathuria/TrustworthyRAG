import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import logging

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C
from src.retriever.rag_generator import RAGGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# RAGAS Imports (dynamic import to avoid failing if not installed)
def run_ragas_evaluation(eval_samples: List[Dict[str, Any]]):
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("Ragas not installed. Please run 'pip install ragas'")
        return

    # Convert samples to Ragas format
    data = {
        "question": [s["question"] for s in eval_samples],
        "answer": [s["answer"] for s in eval_samples],
        "contexts": [s["contexts"] for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }
    dataset = Dataset.from_dict(data)

    # Note: RAGAS defaults to OpenAI. For local Ollama usage, or custom LLMs,
    # we need to pass the LLM object but RAGAS 0.2+ has specific wrappers.
    # For this implementation, we assume environment variables or defaults are or can be set.

    logger.info(f"Running Ragas evaluation on {len(eval_samples)} samples...")
    # results = evaluate(dataset, metrics=[faithfulness, answer_relevance, context_precision, context_recall])
    # logger.info(f"Ragas Results: {results}")
    # return results

    # Placeholder for actual RAGAS call which often requires OpenAI key or heavy setup
    logger.info(
        "Ragas data prepared. Skipping actual call as it requires LLM credentials."
    )
    return data


def main(doc_bench_dir: str, num_samples: int = 5):
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    registry = SystemRegistry(neo4j_manager)
    # Initialize Generator (uses Ollama)
    generator = RAGGenerator(neo4j_manager)

    subdirs = glob.glob(os.path.join(doc_bench_dir, "*"))
    eval_samples = []

    # Simple sampling
    count = 0
    for subdir in tqdm(subdirs, desc="Collecting Ragas Samples"):
        if count >= num_samples:
            break

        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))
        if not qa_files:
            continue

        with open(qa_files[0], "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                query = item["question"]
                ground_truth = item.get("answer", "")

                # Run QALF Retrieval
                retrieved_docs = registry.run_qalf(query)

                # Generate Answer
                gen_result = generator.generate(query, retrieved_docs)
                answer = gen_result["response"]

                # Fetch Contexts
                # QALF retrieval results have doc_id, we need actual contents for Ragas
                chunks = generator.fetch_chunk_content(
                    [d["doc_id"] for d in retrieved_docs]
                )
                contexts = [c["content"] for c in chunks]

                eval_samples.append(
                    {
                        "question": query,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": ground_truth,
                    }
                )
                count += 1
                if count >= num_samples:
                    break

    # Save the prepared dataset for review
    with open("ragas_eval_data.json", "w") as f:
        json.dump(eval_samples, f, indent=2)
    logger.info("Saved RAG samples to ragas_eval_data.json")

    # run_ragas_evaluation(eval_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_bench_dir", default="data/raw/DocBench")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    main(args.doc_bench_dir, args.n)
