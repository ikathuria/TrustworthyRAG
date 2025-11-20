"""
End-to-End QALF Pipeline.
Complete Query-Adaptive Learned Fusion system using Neo4j.
"""

from typing import List, Dict, Any, Tuple
import logging
import yaml

from src.qalf.query_complexity import QueryComplexityClassifier
from src.qalf.query_intent import QueryIntentClassifier
from src.retriever.neo4j_retriever import Neo4jMultiModalRetriever
from src.retriever.qalf_fusion import QALFFusion
from configs.routing_table import route_to_modalities
from configs.alpha_weights import get_alpha_weights


class QALFPipeline:
    """
    Complete QALF pipeline implementing:
    1. Query complexity classification (4D)
    2. Intent classification
    3. Adaptive routing
    4. Multi-modal retrieval (Neo4j)
    5. Consensus-based fusion
    6. Adaptive weight computation
    """

    def __init__(
        self,
        neo4j_manager,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        config_path: str = "configs/hyperparameters.yaml"
    ):
        """
        Initialize QALF pipeline.
        
        Args:
            neo4j_manager: Neo4jManager instance
            embedding_model: Sentence transformer model name
            embedding_dim: Embedding dimension
            config_path: Path to hyperparameters YAML
        """
        self.neo4j_manager = neo4j_manager
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        self._logger = self._setup_logging()
        
        # Load hyperparameters
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.complexity_classifier = QueryComplexityClassifier()
        self.intent_classifier = QueryIntentClassifier()
        self.retriever = Neo4jMultiModalRetriever(
            neo4j_manager,
            embedding_model,
            embedding_dim
        )
        self.fusion = QALFFusion(
            k=self.config.get("rrf", {}).get("k", 60),
            beta=self.config.get("consensus", {}).get("beta", 0.5)
        )
        
        self._logger.info("QALF Pipeline initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_config(self, config_path: str) -> dict:
        """Load hyperparameters from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                "rrf": {"k": 60},
                "consensus": {"beta": 0.5},
                "retrieval": {"top_k": 10, "top_k_final": 10}
            }

    def setup_indexes(self):
        """Setup Neo4j indexes (one-time operation)"""
        self.neo4j_manager.setup_indexes(self.embedding_dim)
        self._logger.info("✅ Neo4j indexes setup complete")

    def qalf_retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Complete QALF retrieval pipeline.
        
        Args:
            query: User query string
            top_k: Number of final results (defaults to config)
        
        Returns:
            List of ranked documents with doc_id, title, score, and metadata
        """
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k_final", 10)
        
        retrieval_top_k = self.config.get("retrieval", {}).get("top_k", 10)
        
        # Step 1: Analyze query
        complexity = self.complexity_classifier.classify_complexity_4d(query)
        intent = self.intent_classifier.classify(query)
        
        self._logger.info(
            f"Query analysis - Complexity: {complexity}, Intent: {intent}"
        )
        
        # Step 2: Route to modalities
        active_modalities = route_to_modalities(complexity, intent)
        self._logger.info(f"Active modalities: {active_modalities}")
        
        # Step 3: Generate query representations
        query_embedding = self.retriever.get_embedding(query)
        query_entities = self.retriever.extract_entities(query)
        query_string = query  # For full-text search
        
        # Step 4: Parallel retrieval from Neo4j
        retrieval_results = {}
        
        if "vector" in active_modalities:
            retrieval_results["vector"] = self.retriever.retrieve_vector(
                query_embedding,
                retrieval_top_k
            )
            self._logger.debug(
                f"Vector retrieval: {len(retrieval_results['vector'])} results"
            )
        
        if "keyword" in active_modalities:
            retrieval_results["keyword"] = self.retriever.retrieve_keyword(
                query_string,
                retrieval_top_k
            )
            self._logger.debug(
                f"Keyword retrieval: {len(retrieval_results['keyword'])} results"
            )
        
        if "graph" in active_modalities:
            retrieval_results["graph"] = self.retriever.retrieve_graph(
                query_entities,
                retrieval_top_k
            )
            self._logger.debug(
                f"Graph retrieval: {len(retrieval_results['graph'])} results"
            )
        
        # Step 5: Compute consensus scores
        consensus_scores = self.fusion.compute_consensus_scores(retrieval_results)
        self._logger.debug(
            f"Consensus scores computed for {len(consensus_scores)} documents"
        )
        
        # Step 6: Adaptive weight computation
        alpha_intent = get_alpha_weights(intent)
        weights = self.fusion.compute_adaptive_weights(
            retrieval_results,
            consensus_scores,
            alpha_intent
        )
        self._logger.debug(f"Adaptive weights: {weights}")
        
        # Step 7: Fuse with consensus
        ranked_results = self.fusion.fuse_with_consensus(
            retrieval_results,
            weights
        )
        
        # Step 8: Format and return top-K
        formatted_results = []
        for rank, (doc_id, score) in enumerate(ranked_results[:top_k], 1):
            # Find document metadata from retrieval results
            doc_metadata = self._find_doc_metadata(doc_id, retrieval_results)
            
            formatted_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": doc_metadata.get("title", doc_id),
                "score": score,
                "consensus": consensus_scores.get(doc_id, 0.0),
                "complexity": complexity,
                "intent": intent,
                "modalities": active_modalities
            })
        
        return formatted_results

    def _find_doc_metadata(
        self,
        doc_id: str,
        retrieval_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Find document metadata from retrieval results"""
        for modality, docs in retrieval_results.items():
            for doc in docs:
                if doc.get("doc_id") == doc_id:
                    return doc
        return {}

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Main retrieval method (alias for qalf_retrieve with dict output).
        
        Args:
            query: User query string
            top_k: Number of final results
        
        Returns:
            Dictionary with results and metadata
        """
        results = self.qalf_retrieve(query, top_k)
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "success": len(results) > 0
        }

