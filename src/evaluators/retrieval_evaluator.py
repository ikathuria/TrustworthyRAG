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
from src.utils.systems import SystemRegistry
import src.utils.metrics
import src.utils.constants as C


class RetrievalEvaluator:

    def __init__(
        self,
        config: Dict[str, Any] = "config.yaml",
        dataset: str = "clean",
        systems: List[str] = ["all"],
        output_dir: str = "results"
    ):
        self.config = config
        self.dataset = dataset
        self.systems = systems
        self.output_dir = output_dir
        self._logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logger for the class"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_queries(self,file_path: Path) -> List[Dict[str, Any]]:
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
        self,
        system_func, 
        queries: List[Dict[str, Any]], 
        top_k: int,
        system_name: str
    ) -> Dict[str, Any]:
        """Run evaluation for a single system."""
        self._logger.info(f"Evaluating system: {system_name}")
        
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
                gold_ids = self.parse_gold_evidence(q['gold_doc_ids'])
            elif 'relevant_docs' in q:
                gold_ids = self.parse_gold_evidence(q['relevant_docs'])
                
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
                self._logger.error(f"Error processing query '{query_text}': {e}")
                
        # Compute averages
        num_queries = len(results)
        avg_metrics = {k: v / num_queries for k, v in metrics_sum.items()} if num_queries > 0 else metrics_sum
        
        return {
            "system": system_name,
            "metrics": avg_metrics,
            "detailed_results": results
        }

    def main(self):
        # Load config
        config_path = Path(self.config)
        if not config_path.exists():
            # Fallback to default values if config doesn't exist
            self._logger.warning(f"Config file {config_path} not found. Using defaults.")
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
        systems_to_run = self.systems
        if "all" in systems_to_run:
            systems_to_run = available_systems
            
        # Load data
        dataset_path = config.get("datasets", {}).get(self.dataset)
        if not dataset_path:
            self._logger.error(f"Dataset '{self.dataset}' not found in config.")
            return
            
        self._logger.info(f"Loading queries from {dataset_path}")
        try:
            queries = self.load_queries(dataset_path)
        except Exception as e:
            self._logger.error(f"Failed to load queries: {e}")
            return
            
        top_k = config.get("evaluation", {}).get("top_k", 10)
        
        # Run evaluation
        all_metrics = []
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for sys_name in systems_to_run:
            try:
                system_func = registry.get_system(sys_name)
                result = self.evaluate_system(system_func, queries, top_k, sys_name)
                
                all_metrics.append({
                    "system": sys_name,
                    **result["metrics"]
                })
                
                # Save detailed results
                detailed_df = pd.DataFrame(result["detailed_results"])
                detailed_df.to_csv(output_dir / f"{sys_name}_detailed.csv", index=False)
                
            except Exception as e:
                self._logger.error(f"Failed to run system {sys_name}: {e}")
                
        # Save summary
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            print("\n=== Evaluation Summary ===")
            print(summary_df.to_markdown(index=False))
            summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
            
        neo4j_manager.close()

if __name__ == "__main__":
    main()
