"""
Adversarial evaluation script for QALF.
Evaluates robustness against Localized (LPA) and Globalized (GPA) Poisoning Attacks.
"""

import json
import logging
from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.systems import SystemRegistry
import src.utils.constants as C

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_adversarial_evaluation(output_dir: str = "data/results", top_k: int = 10):
    try:
        # Look for metadata in standard locations
        metadata_path = "adversarial_metadata.json"
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join("data", "adversarial_metadata.json")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    registry = SystemRegistry(neo4j_manager)

    # We test the poisoned query
    item = metadata["poisoned_queries"][0]
    query = item["query"]
    target_id = item["target_doc_id"]
    gold_id = item["gold_doc_ids"][0]

    print(f"Testing Adversarial Robustness")
    print(f"Query: {query}")
    print(f"Target (Poisoned): {target_id}")
    print(f"Gold: {gold_id}")

    systems = ["vector_only", "fixed_rrf", "qalf"]

    final_results = {}

    for name in systems:
        func = registry.get_system(name)
        system_result = func(query, top_k=top_k)
        results = system_result["results"]

        # Find ranks and details
        target_rank = -1
        gold_rank = -1
        target_consensus = -1
        gold_consensus = -1

        for i, res in enumerate(results):
            # Check all possible ID keys
            d_id = res.get("id") or res.get("doc_id") or res.get("source") or ""

            # Key for consensus is "consensus" in QALF formatted results
            c_score = res.get("consensus", -1)

            if target_id in d_id:
                target_rank = i + 1
                target_consensus = c_score

            if gold_id in d_id:
                gold_rank = i + 1
                gold_consensus = c_score

        print(
            f"[{name}] Target Rank: {target_rank} (Consens: {target_consensus:.2f}), Gold Rank: {gold_rank} (Consens: {gold_consensus:.2f})"
        )
        final_results[name] = {
            "target_rank": target_rank,
            "target_consensus": target_consensus,
            "gold_rank": gold_rank,
            "gold_consensus": gold_consensus,
        }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "adversarial_verification_results.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Adversarial results saved to {output_path}")
    neo4j_manager.close()


if __name__ == "__main__":
    run_adversarial_evaluation()
