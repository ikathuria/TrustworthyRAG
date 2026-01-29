import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from tqdm import tqdm
import logging
import argparse

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C
from metrics import ndcg_at_k, recall_at_k, hit_rate
from src.qalf.query_complexity import QueryComplexityClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_path(path: str) -> str:
    """Normalize path to match Neo4j IDs (which depend on how they were ingested)"""
    return os.path.normpath(path)


def get_ingested_documents(neo4j_manager: Neo4jManager) -> Set[str]:
    """Get set of all document IDs currently in Neo4j"""
    query = "MATCH (d:Document) RETURN d.id as id"
    with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
        result = session.run(query)
        # Normalize paths for comparison
        documents = {normalize_path(record["id"]) for record in result}
    logger.info(f"Found {len(documents)} ingested documents in Neo4j")
    return documents


def get_retrieved_doc_ids(
    results: List[Dict[str, Any]], neo4j_manager: Neo4jManager
) -> List[str]:
    """Extract parent document IDs from retrieval results"""
    doc_ids = []

    for res in results:
        # Check standard locations for doc ID
        if "doc_id" in res:
            doc_ids.append(normalize_path(res["doc_id"]))
        elif "source" in res:
            doc_ids.append(normalize_path(res["source"]))
        elif "metadata" in res and "source" in res["metadata"]:
            doc_ids.append(normalize_path(res["metadata"]["source"]))
        # Sometimes doc_id is the source file path directly in this system

    return doc_ids


def run_evaluation(doc_bench_dir: str):
    logger.info("Initializing evaluation...")

    # Connect to Neo4j
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    # Initialize Systems
    registry = SystemRegistry(neo4j_manager)
    complexity_classifier = QueryComplexityClassifier()

    # Setup indexes to ensure all necessary indices (like chunk_fulltext) exist
    try:
        logger.info("Setting up QALF indexes...")
        registry.qalf.setup_indexes()
        logger.info("✓ QALF indexes ready")
    except Exception as e:
        logger.warning(f"⚠️  Index setup note: {e}")

    # Get available documents
    ingested_docs = get_ingested_documents(neo4j_manager)

    if not ingested_docs:
        logger.error("No documents found in Neo4j. Please run ingestion first.")
        return

    # iterate through DocBench folders
    subdirs = glob.glob(os.path.join(doc_bench_dir, "*"))

    if args.limit:
        subdirs = subdirs[: args.limit]

    all_results = []

    for subdir in tqdm(subdirs, desc="Evaluating Directories"):
        if not os.path.isdir(subdir):
            continue

        # Find PDF and QA file
        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))

        if not pdf_files or not qa_files:
            continue

        # Assume 1 PDF per dir as per structure seen
        pdf_path = normalize_path(pdf_files[0])
        qa_path = qa_files[0]

        # Check if this PDF is ingested
        # We need to robustly match the ID.
        # Ingestion logic uses absolute path usually, but sometimes relative.
        # We check our normalized set.

        target_doc_id = None
        if pdf_path in ingested_docs:
            target_doc_id = pdf_path
        else:
            # Try approximate match (basename)
            pdf_basename = os.path.basename(pdf_path)
            for doc in ingested_docs:
                if doc.endswith(pdf_basename):
                    target_doc_id = doc
                    break

        if not target_doc_id:
            continue  # Skip non-ingested

        # Load Questions
        with open(qa_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    query = item["question"]

                    # Classify query complexity
                    ling, sem, mod, ctx = complexity_classifier.classify_complexity_4d(
                        query
                    )
                    is_complex = (
                        "High" in [ling, sem, mod, ctx]
                        or sum(1 for x in [ling, sem, mod, ctx] if x == "Medium") >= 2
                    )
                    complexity_label = "Complex" if is_complex else "Simple"

                    # For retrieval eval in single-doc QA, relevant_doc IS the target_doc_id
                    relevant_ids = {target_doc_id}

                    # Run Systems
                    for sys_name in ["vector_only", "fixed_rrf", "qalf"]:
                        try:
                            # Execute retrieval
                            if sys_name == "vector_only":
                                results = registry.run_vector_only(query)
                            elif sys_name == "fixed_rrf":
                                results = registry.run_fixed_rrf(query)
                            elif sys_name == "qalf":
                                results = registry.run_qalf(query)

                            retrieved_ids = get_retrieved_doc_ids(
                                results, neo4j_manager
                            )

                            # DEBUG: Log if retrieval set is empty or mismatch
                            # if not retrieved_ids:
                            #    logger.debug(f"Query '{query}' returned no results for {sys_name}")

                            # Calculate Metrics
                            ndcg_10 = ndcg_at_k(retrieved_ids, relevant_ids, k=10)
                            recall_10 = recall_at_k(retrieved_ids, relevant_ids, k=10)
                            accuracy_1 = hit_rate(retrieved_ids, relevant_ids, k=1)

                            all_results.append(
                                {
                                    "Directory": os.path.basename(subdir),
                                    "Query_ID": f"{os.path.basename(subdir)}_{line_idx}",
                                    "System": sys_name,
                                    "NDCG@10": ndcg_10,
                                    "Recall@10": recall_10,
                                    "Accuracy@1": accuracy_1,
                                    "Complexity": complexity_label,
                                    "Ling_Comp": ling,
                                    "Sem_Comp": sem,
                                    "Mod_Comp": mod,
                                    "Ctx_Comp": ctx,
                                }
                            )

                        except Exception as e:
                            logger.error(
                                f"Error running {sys_name} for query in {subdir}: {e}"
                            )

                except json.JSONDecodeError:
                    continue

    # Save and Summarize
    if not all_results:
        logger.warning("No evaluation results generated.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_results.csv", index=False)

    # Pivot for summary table
    summary = df.groupby("System")[["NDCG@10", "Recall@10", "Accuracy@1"]].mean()
    print("\n=== Overall Evaluation Results ===")
    print(summary)

    # Complexity breakdown
    complexity_summary = df.groupby(["System", "Complexity"])[
        ["NDCG@10", "Recall@10", "Accuracy@1"]
    ].mean()
    print("\n=== Performance by Complexity ===")
    print(complexity_summary)

    print("\nResults saved to evaluation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_bench_dir", default="data/raw/DocBench", help="Path to DocBench data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of directories to evaluate",
    )
    args = parser.parse_args()

    run_evaluation(args.doc_bench_dir)
