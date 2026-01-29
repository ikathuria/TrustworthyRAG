import os
import json
import glob
import time
import pandas as pd
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_efficiency_evaluation(doc_bench_dir: str, limit: int = 10):
    logger.info("Initializing efficiency evaluation...")

    # Connect to Neo4j
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    # Initialize Systems
    registry = SystemRegistry(neo4j_manager)

    # Get available directories
    subdirs = glob.glob(os.path.join(doc_bench_dir, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    if limit:
        subdirs = subdirs[:limit]

    efficiency_results = []

    for subdir in tqdm(subdirs, desc="Evaluating Efficiency"):
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))
        if not qa_files:
            continue

        qa_path = qa_files[0]

        with open(qa_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue

            # Take only the first query per directory for efficiency
            line = lines[0]
            try:
                item = json.loads(line)
                query = item["question"]

                print(f"\nProcessing query from {os.path.basename(subdir)}")

                # Evaluate each system for latency
                for sys_name in ["vector_only", "fixed_rrf", "qalf"]:
                    start_time = time.time()

                    if sys_name == "vector_only":
                        results = registry.run_vector_only(query)
                        modalities_used = 1
                    elif sys_name == "fixed_rrf":
                        results = registry.run_fixed_rrf(query)
                        modalities_used = 3
                    elif sys_name == "qalf":
                        results = registry.run_qalf(query)
                        if results:
                            # QALF returns formatted results with 'modalities' key
                            modalities_used = len(results[0].get("modalities", []))
                        else:
                            modalities_used = 0

                    latency_ms = (time.time() - start_time) * 1000

                    efficiency_results.append(
                        {
                            "System": sys_name,
                            "Latency_ms": latency_ms,
                            "Modalities_Used": modalities_used,
                            "Query_Length": len(query.split()),
                        }
                    )
                    print(
                        f"  {sys_name}: {latency_ms:.2f}ms, modalities: {modalities_used}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating query: {e}")

    # Process and save results
    if efficiency_results:
        df = pd.DataFrame(efficiency_results)
        summary = df.groupby("System").agg(
            {
                "Latency_ms": ["mean", "median", "std"],
                "Modalities_Used": ["mean", "max"],
            }
        )

        print("\n=== Efficiency Evaluation Summary ===")
        print(summary)

        df.to_csv("efficiency_results.csv", index=False)
        logger.info("Efficiency results saved to efficiency_results.csv")
    else:
        logger.warning("No results generated.")

    neo4j_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_bench_dir", default="data/raw/DocBench")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    run_efficiency_evaluation(args.doc_bench_dir, args.limit)
