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


def run_limited_test():
    doc_bench_dir = "data/raw/DocBench"
    limit = 2

    logger.info(f"Starting limited efficiency test (limit={limit})...")

    # Connect to Neo4j
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    registry = SystemRegistry(neo4j_manager)
    subdirs = glob.glob(os.path.join(doc_bench_dir, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)][:limit]

    efficiency_results = []

    for subdir in subdirs:
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))
        if not qa_files:
            continue

        with open(qa_files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue

            query = json.loads(lines[0])["question"]
            print(f"\nQuery: {query[:60]}...")

            for sys_name in ["vector_only", "fixed_rrf", "qalf"]:
                start = time.time()
                try:
                    results = registry.get_system(sys_name)(query)
                    duration = (time.time() - start) * 1000

                    modalities = 1
                    if sys_name == "fixed_rrf":
                        modalities = 3
                    elif sys_name == "qalf" and results:
                        modalities = len(results[0].get("modalities", []))

                    efficiency_results.append(
                        {
                            "System": sys_name,
                            "Latency_ms": duration,
                            "Modalities_Used": modalities,
                        }
                    )
                    print(f"  {sys_name}: {duration:.1f}ms")
                except Exception as e:
                    print(f"  {sys_name} failed: {e}")

    if efficiency_results:
        df = pd.DataFrame(efficiency_results)
        print("\nSummary:")
        print(df.groupby("System")["Latency_ms"].mean())
        df.to_csv("efficiency_results.csv", index=False)

    neo4j_manager.close()


if __name__ == "__main__":
    run_limited_test()
