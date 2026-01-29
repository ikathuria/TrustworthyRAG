import os
import json
import glob
import time
import pandas as pd

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C
from metrics import ndcg_at_k


def run_combined_minimal_test():
    print("RUNNING COMBINED MINIMAL TEST")
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )
    registry = SystemRegistry(neo4j_manager)

    target_dir = "data/raw/DocBench/P19-1598"
    qa_file = glob.glob(os.path.join(target_dir, "*_qa.jsonl"))[0]
    pdf_path = os.path.normpath(glob.glob(os.path.join(target_dir, "*.pdf"))[0])

    with open(qa_file, "r", encoding="utf-8") as f:
        query = json.loads(f.readline())["question"]

    print(f"QUERY: {query[:50]}...")

    # 1. Efficiency
    print("\n--- EFFICIENCY ---")
    results_eff = []
    for sys in ["vector_only", "fixed_rrf", "qalf"]:
        start = time.time()
        res = registry.get_system(sys)(query)
        duration = (time.time() - start) * 1000
        mods = 1
        if sys == "fixed_rrf":
            mods = 3
        elif sys == "qalf" and res:
            mods = len(res[0].get("modalities", []))
        print(f"{sys}: {duration:.1f}ms, mods: {mods}")
        results_eff.append({"System": sys, "Latency_ms": duration, "Mods": mods})

    # 2. Sensitivity
    print("\n--- SENSITIVITY ---")
    results_sens = []
    for beta in [0.0, 0.5, 1.0]:
        registry.qalf.fusion.beta = beta
        res = registry.run_qalf(query)
        ids = [os.path.normpath(r.get("doc_id") or "") for r in res]
        ndcg = ndcg_at_k(ids, {pdf_path}, k=10)
        print(f"Beta {beta}: NDCG {ndcg:.3f}")
        results_sens.append({"Beta": beta, "NDCG": ndcg})

    neo4j_manager.close()


if __name__ == "__main__":
    run_combined_minimal_test()
