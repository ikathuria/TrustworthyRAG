"""
Retrieval evaluation script for QALF.
Evaluates retrieval effectiveness on clean data.
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


def parse_gold_evidence(gold_field: Any) -> Set[str]:
    """Parse gold evidence into a set of document IDs."""
    if isinstance(gold_field, str):
        # Assume comma-separated string or single ID
        return {x.strip() for x in gold_field.split(',')}
    elif isinstance(gold_field, list):
        return set(gold_field)
    else:
        return set()


def evaluate_system(
    system_func, 
    queries: List[Dict[str, Any]], 
    top_k: int,
    system_name: str
) -> Dict[str, Any]:
    """Run evaluation for a single system."""
    logger.info(f"Evaluating system: {system_name}")
    
    results = []
    metrics_sum = {
        f"ndcg@{top_k}": 0.0,
        f"recall@{top_k}": 0.0,
        "mrr": 0.0
    }
    
    for q in tqdm(queries, desc=f"Running {system_name}"):
        query_text = q.get('query')
        if not query_text:
            continue
            
        # Get gold standard
        # Supports 'gold_doc_ids', 'gold_answer' (if it contains IDs), or 'relevant_docs'
        gold_ids = set()
        if 'gold_doc_ids' in q:
            gold_ids = parse_gold_evidence(q['gold_doc_ids'])
        elif 'relevant_docs' in q:
            gold_ids = parse_gold_evidence(q['relevant_docs'])
            
        # Run retrieval
        try:
            retrieved_docs = system_func(query_text, top_k=top_k)
            retrieved_ids = [doc['doc_id'] for doc in retrieved_docs if doc.get('doc_id')]
            
            # Compute metrics
            ndcg = metrics.ndcg_at_k(retrieved_ids, gold_ids, top_k)
            recall = metrics.recall_at_k(retrieved_ids, gold_ids, top_k)
            mrr_score = metrics.mrr(retrieved_ids, gold_ids)
            
            metrics_sum[f"ndcg@{top_k}"] += ndcg
            metrics_sum[f"recall@{top_k}"] += recall
            metrics_sum["mrr"] += mrr_score
            
            results.append({
                "query_id": q.get("id", q.get("query_id", "")),
                "query": query_text,
                "system": system_name,
                "retrieved_ids": retrieved_ids,
                "gold_ids": list(gold_ids),
                f"ndcg@{top_k}": ndcg,
                f"recall@{top_k}": recall,
                "mrr": mrr_score
            })
            
        except Exception as e:
            logger.error(f"Error processing query '{query_text}': {e}")
            
    # Compute averages
    num_queries = len(results)
    avg_metrics = {k: v / num_queries for k, v in metrics_sum.items()} if num_queries > 0 else metrics_sum
    
    return {
        "system": system_name,
        "metrics": avg_metrics,
        "detailed_results": results
    }


def main():
    parser = argparse.ArgumentParser(description="QALF Retrieval Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="clean", help="Dataset key in config (e.g., 'clean')")
    parser.add_argument("--systems", nargs="+", default=["all"], help="Systems to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to default values if config doesn't exist
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        config = {
            "neo4j": {
                "uri": C.NEO4J_URI,
                "username": C.NEO4J_USERNAME,
                "password": C.NEO4J_PASSWORD,
                "database": C.NEO4J_DB
            },
            "datasets": {
                "clean": "data/clean_queries.jsonl" # Placeholder
            },
            "evaluation": {
                "top_k": 10
            }
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
    
    # Initialize Systems
    registry = SystemRegistry(neo4j_manager, config)
    
    # Determine systems to run
    available_systems = [
        "vector_only", "keyword_only", "graph_only", 
        "fixed_rrf", "native_hybrid", "qalf"
    ]
    systems_to_run = args.systems
    if "all" in systems_to_run:
        systems_to_run = available_systems
        
    # Load data
    dataset_path = config.get("datasets", {}).get(args.dataset)
    if not dataset_path:
        logger.error(f"Dataset '{args.dataset}' not found in config.")
        return
        
    logger.info(f"Loading queries from {dataset_path}")
    try:
        queries = load_queries(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        return
        
    top_k = config.get("evaluation", {}).get("top_k", 10)
    
    # Run evaluation
    all_metrics = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for sys_name in systems_to_run:
        try:
            system_func = registry.get_system(sys_name)
            result = evaluate_system(system_func, queries, top_k, sys_name)
            
            all_metrics.append({
                "system": sys_name,
                **result["metrics"]
            })
            
            # Save detailed results
            detailed_df = pd.DataFrame(result["detailed_results"])
            detailed_df.to_csv(output_dir / f"{sys_name}_detailed.csv", index=False)
            
        except Exception as e:
            logger.error(f"Failed to run system {sys_name}: {e}")
            
    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        print("\n=== Evaluation Summary ===")
        print(summary_df.to_markdown(index=False))
        summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
        
    neo4j_manager.close()

if __name__ == "__main__":
    main()
