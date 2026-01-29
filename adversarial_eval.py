import json
import logging
from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_adversarial_eval():
    with open("adversarial_metadata.json", "r") as f:
        metadata = json.load(f)

    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    registry = SystemRegistry(neo4j_manager)

    systems = ["vector_only", "fixed_rrf", "qalf"]
    results = []

    for item in metadata["poisoned_queries"]:
        query = item["query"]
        poisoned_doc = item.get("target_doc_id")
        poisoned_ent = item.get("target_ent_id")
        gold_docs = item["gold_doc_ids"]

        print(f"\nQuery: {query}")

        for sys_name in systems:
            sys_func = registry.get_system(sys_name)
            retrieved = sys_func(query)

            # Check if poisoned doc or entity is in top results
            top_ids = []
            for r in retrieved[:3]:
                # Collect all possible IDs/sources from result
                ids = [r.get("doc_id"), r.get("id"), r.get("source")]
                top_ids.extend([str(i) for i in ids if i])

            is_poisoned = False
            if poisoned_doc:
                is_poisoned = any(poisoned_doc in doc for doc in top_ids)
            if poisoned_ent:
                # Also check if the result metadata contains the poisoned entity
                # Many systems return entities in metadata
                is_poisoned = is_poisoned or any(
                    poisoned_ent in str(r) for r in retrieved[:3]
                )

            has_gold = any(any(gold in doc for gold in gold_docs) for doc in top_ids)

            print(
                f"[{sys_name}] Poisoned in Top-3: {is_poisoned}, Gold in Top-3: {has_gold}"
            )
            results.append(
                {
                    "query": query,
                    "system": sys_name,
                    "is_poisoned": is_poisoned,
                    "has_gold": has_gold,
                    "top_ids": top_ids,
                }
            )

    with open("adversarial_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    neo4j_manager.close()


if __name__ == "__main__":
    run_adversarial_eval()
