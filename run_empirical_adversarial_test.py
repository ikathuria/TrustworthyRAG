import os
import json
import logging
import pandas as pd
from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import src.utils.constants as C

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_empirical_adversarial_test():
    print("\n" + "=" * 50)
    print("RUNNING EMPIRICAL ADVERSARIAL TEST")
    print("=" * 50)

    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )
    registry = SystemRegistry(neo4j_manager)

    test_queries = [
        {
            "id": "visual_bait",
            "query": "What is the primary evaluation metric for WikiText-2?",
            "expected_poison": "adversarial_poison_visual_bait.pdf",
            "gold_doc": "P19-1598.pdf",  # This is the basename in ingested IDs usually
        },
        {
            "id": "entity_override",
            "query": "Which university is associated with the creation of Linked WikiText-2?",
            "expected_poison_content": "Stanford",
            "gold_doc": "P19-1598.pdf",
        },
    ]

    results_data = []

    for test in test_queries:
        print(f"\nTEST SCENARIO: {test['id']}")
        print(f"Query: {test['query']}")

        for sys_name in ["vector_only", "fixed_rrf", "qalf"]:
            print(f"  Running {sys_name}...")
            results = registry.get_system(sys_name)(test["query"])

            # Extract top doc information
            if results:
                top_res = results[0]
                top_id = (
                    top_res.get("doc_id")
                    or top_res.get("id")
                    or top_res.get("source")
                    or "Unknown"
                )
                top_score = top_res.get("score") or top_res.get("total_score") or 0

                # Check for poison
                is_poisoned = False
                if test["id"] == "visual_bait":
                    is_poisoned = test["expected_poison"] in top_id
                elif test["id"] == "entity_override":
                    # Check if the content (description/result) contains poisoned entity info
                    content = str(top_res).lower()
                    is_poisoned = test["expected_poison_content"].lower() in content

                print(
                    f"    Rank 1: {os.path.basename(top_id)} | Poisoned: {is_poisoned} | Score: {top_score:.4f}"
                )

                results_data.append(
                    {
                        "Scenario": test["id"],
                        "System": sys_name,
                        "Top_Doc": os.path.basename(top_id),
                        "Is_Poisoned": is_poisoned,
                        "Score": top_score,
                    }
                )
            else:
                print("    No results found.")

    # Save summary
    df = pd.DataFrame(results_data)
    print("\n" + "=" * 50)
    print("ADVERSARIAL TEST SUMMARY")
    print("=" * 50)
    print(df)

    df.to_csv("empirical_adversarial_results.csv", index=False)

    neo4j_manager.close()


if __name__ == "__main__":
    run_empirical_adversarial_test()
