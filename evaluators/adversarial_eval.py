"""
Adversarial evaluation script for QALF.
Evaluates robustness against Localized (LPA) and Globalized (GPA) Poisoning Attacks.
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Set
import yaml
import pandas as pd
from tqdm import tqdm

from src.neo4j.neo4j_manager import Neo4jManager
from systems import SystemRegistry
import metrics
import src.utils.constants as C

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_attack_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Load attack metadata.
    Expected format: JSON or CSV mapping query_id -> target_doc_id / target_answer
    """
    metadata = {}
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            metadata = json.load(f)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Assume columns: query_id, target_doc_id
        for _, row in df.iterrows():
            metadata[str(row['query_id'])] = row.to_dict()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
    return metadata

def load_queries(file_path: Path) -> List[Dict[str, Any]]:
    """Load queries from JSONL or CSV file."""
    queries = []
    file_path = Path(file_path)
    
    if file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line))
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        queries = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return queries

def evaluate_adversarial(
    system_func, 
    queries: List[Dict[str, Any]], 
    attack_metadata: Dict[str, Any],
    top_k: int,
    system_name: str,
    mode: str
) -> Dict[str, Any]:
    """Run adversarial evaluation for a single system."""
    logger.info(f"Evaluating system: {system_name} in mode: {mode}")
    
    results = []
    psr_count = 0
    total_targeted = 0
    
    # Metrics accumulators
    recall_sum = 0.0
    
    for q in tqdm(queries, desc=f"Running {system_name} ({mode})"):
        query_text = q.get('query')
        query_id = str(q.get("id", q.get("query_id", "")))
        
        if not query_text:
            continue
            
        # Check if this query is targeted
        target_info = attack_metadata.get(query_id)
        is_targeted = target_info is not None
        
        target_doc_id = None
        if is_targeted:
            # Handle different metadata formats
            if isinstance(target_info, dict):
                target_doc_id = target_info.get('target_doc_id')
            else:
                target_doc_id = target_info # Assume direct mapping if not dict
        
        # Run retrieval
        try:
            retrieved_docs = system_func(query_text, top_k=top_k)
            retrieved_ids = [doc['doc_id'] for doc in retrieved_docs if doc.get('doc_id')]
            
            # Compute PSR (only for targeted queries)
            psr = 0.0
            if is_targeted and target_doc_id:
                psr = metrics.poison_success_rate(retrieved_ids, target_doc_id, k=top_k) # Check if target is in top_k
                if psr > 0:
                    psr_count += 1
                total_targeted += 1
                
            # Compute Recall (if gold standard exists)
            recall = 0.0
            gold_ids = set()
            if 'gold_doc_ids' in q:
                # Parse gold ids similar to retrieval_eval
                if isinstance(q['gold_doc_ids'], str):
                     gold_ids = {x.strip() for x in q['gold_doc_ids'].split(',')}
                elif isinstance(q['gold_doc_ids'], list):
                    gold_ids = set(q['gold_doc_ids'])
                
                recall = metrics.recall_at_k(retrieved_ids, gold_ids, top_k)
                recall_sum += recall

            results.append({
                "query_id": query_id,
                "query": query_text,
                "system": system_name,
                "mode": mode,
                "is_targeted": is_targeted,
                "target_doc_id": target_doc_id,
                "retrieved_ids": retrieved_ids,
                "psr": psr if is_targeted else None,
                f"recall@{top_k}": recall if gold_ids else None
            })
            
        except Exception as e:
            logger.error(f"Error processing query '{query_text}': {e}")
            
    # Compute aggregates
    avg_psr = psr_count / total_targeted if total_targeted > 0 else 0.0
    avg_recall = recall_sum / len(queries) if queries else 0.0
    
    return {
        "system": system_name,
        "mode": mode,
        "metrics": {
            "psr": avg_psr,
            f"recall@{top_k}": avg_recall,
            "targeted_queries": total_targeted
        },
        "detailed_results": results
    }


def main():
    parser = argparse.ArgumentParser(description="QALF Adversarial Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--attack_type", type=str, required=True, choices=["LPA", "GPA"], help="Attack type")
    parser.add_argument("--poison_rate", type=float, default=0.05, help="Poisoning rate (for reporting)")
    parser.add_argument("--systems", nargs="+", default=["all"], help="Systems to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/adversarial", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        # Minimal defaults
        config = {
            "neo4j": {
                "uri": C.NEO4J_URI,
                "username": C.NEO4J_USERNAME,
                "password": C.NEO4J_PASSWORD,
                "database": C.NEO4J_DB
            },
            "datasets": {},
            "evaluation": {"top_k": 10}
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    # Setup Neo4j
    neo4j_config = config.get("neo4j", {})
    neo4j_manager = Neo4jManager(
        uri=neo4j_config.get("uri", C.NEO4J_URI),
        username=neo4j_config.get("username", C.NEO4J_USERNAME),
        password=neo4j_config.get("password", C.NEO4J_PASSWORD),
        database=neo4j_config.get("database", C.NEO4J_DB)
    )
    
    registry = SystemRegistry(neo4j_manager, config)
    
    available_systems = ["vector_only", "fixed_rrf", "qalf"] # Usually subset for adversarial
    systems_to_run = args.systems
    if "all" in systems_to_run:
        systems_to_run = available_systems
        
    # Load data
    # We need both clean and adversarial queries/metadata
    # Assuming config has paths for 'clean_queries', 'adversarial_queries', 'attack_metadata'
    
    clean_queries_path = config.get("datasets", {}).get("clean")
    adv_queries_path = config.get("datasets", {}).get("adversarial")
    metadata_path = config.get("datasets", {}).get("attack_metadata")
    
    if not (clean_queries_path and adv_queries_path and metadata_path):
        logger.error("Missing dataset paths in config (clean, adversarial, or attack_metadata)")
        # For robustness, we might want to allow running just one mode, but typically we want comparison.
        # Let's proceed if we have at least adversarial queries for PSR.
    
    # Load metadata
    try:
        attack_metadata = load_attack_metadata(metadata_path) if metadata_path else {}
        adv_queries = load_queries(adv_queries_path) if adv_queries_path else []
        clean_queries = load_queries(clean_queries_path) if clean_queries_path else []
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    top_k = config.get("evaluation", {}).get("top_k", 10)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    summary_metrics = []
    
    for sys_name in systems_to_run:
        try:
            system_func = registry.get_system(sys_name)
            
            # 1. Evaluate on Poisoned Data (Adversarial Queries)
            # Note: The DB state (clean vs poisoned) is external. 
            # The user must ensure the DB is in the correct state or we assume the DB contains poisoned data.
            # The script calculates PSR based on the current DB state.
            
            logger.info(f"--- Running {sys_name} on Adversarial Queries ---")
            adv_results = evaluate_adversarial(
                system_func, adv_queries, attack_metadata, top_k, sys_name, "poisoned"
            )
            
            # 2. Evaluate on Clean Data (Reference)
            # Ideally we run this on clean DB, but if we only have one DB instance, 
            # we might be running clean queries on poisoned DB to check utility drop.
            logger.info(f"--- Running {sys_name} on Clean Queries ---")
            clean_results = evaluate_adversarial(
                system_func, clean_queries, {}, top_k, sys_name, "clean_ref"
            )
            
            # Calculate Robustness Metrics
            psr = adv_results["metrics"]["psr"]
            clean_recall = clean_results["metrics"][f"recall@{top_k}"]
            poisoned_recall = adv_results["metrics"][f"recall@{top_k}"] # Recall on adv queries (might be same as clean queries)
            
            recall_drop = metrics.retrieval_recall_drop(clean_recall, poisoned_recall)
            
            summary_metrics.append({
                "system": sys_name,
                "attack_type": args.attack_type,
                "poison_rate": args.poison_rate,
                "psr": psr,
                "clean_recall": clean_recall,
                "poisoned_recall": poisoned_recall,
                "recall_drop": recall_drop
            })
            
            # Save detailed
            pd.DataFrame(adv_results["detailed_results"]).to_csv(output_dir / f"{sys_name}_{args.attack_type}_detailed.csv", index=False)
            
        except Exception as e:
            logger.error(f"Failed to run system {sys_name}: {e}")
            
    # Save summary
    if summary_metrics:
        summary_df = pd.DataFrame(summary_metrics)
        print("\n=== Adversarial Evaluation Summary ===")
        print(summary_df.to_markdown(index=False))
        summary_df.to_csv(output_dir / f"summary_{args.attack_type}.csv", index=False)
        
    neo4j_manager.close()

if __name__ == "__main__":
    main()
