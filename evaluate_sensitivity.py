import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C
from metrics import ndcg_at_k, recall_at_k

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sensitivity_analysis(target_dir: str, betas: List[float]):
    logger.info("Initializing sensitivity analysis...")

    # Connect to Neo4j
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    # Initialize Systems
    registry = SystemRegistry(neo4j_manager)

    # For sensitivity, we'll use a single directory/query to see the effect
    qa_files = glob.glob(os.path.join(target_dir, "*_qa.jsonl"))
    pdf_files = glob.glob(os.path.join(target_dir, "*.pdf"))

    if not qa_files or not pdf_files:
        logger.error(f"Missing files in {target_dir}")
        return

    pdf_path = os.path.normpath(pdf_files[0])
    qa_path = qa_files[0]
    relevant_ids = {pdf_path}

    with open(qa_path, "r", encoding="utf-8") as f:
        line = f.readline()
        item = json.loads(line)
        query = item["question"]

    sensitivity_results = []

    for beta in tqdm(betas, desc="Varying Beta"):
        # Update registry's QALF beta manually
        registry.qalf.fusion.beta = beta

        # Execute retrieval
        results = registry.run_qalf(query)

        # Extract IDs
        retrieved_ids = []
        for res in results:
            d_id = res.get("id") or res.get("doc_id") or res.get("source") or ""
            retrieved_ids.append(os.path.normpath(d_id))

        # Calculate Utility (NDCG)
        ndcg_10 = ndcg_at_k(retrieved_ids, relevant_ids, k=10)

        # We also want to see the effect on the poisonous document's score if possible
        # For simplicity, we'll just track NDCG here, and separately discuss robustness

        sensitivity_results.append(
            {
                "Beta": beta,
                "NDCG@10": ndcg_10,
                "Top_Doc_ID": results[0]["doc_id"] if results else "None",
                "Top_Doc_Consensus": results[0].get("consensus", 0) if results else 0,
            }
        )

    # Save results
    df = pd.DataFrame(sensitivity_results)
    print("\n=== Sensitivity Analysis Summary ===")
    print(df)

    df.to_csv("sensitivity_results.csv", index=False)
    logger.info("Sensitivity results saved to sensitivity_results.csv")

    neo4j_manager.close()


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw/DocBench/P19-1598")
    args = parser.parse_args()

    betas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    run_sensitivity_analysis(args.dir, betas)
