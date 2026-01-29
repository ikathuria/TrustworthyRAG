"""
System wrappers and registry for QALF evaluation.
"""

from typing import List, Dict, Any, Protocol
import logging
from src.retriever.qalf_pipeline import QALFPipeline
from src.neo4j.neo4j_manager import Neo4jManager
from src.retriever.neo4j_retriever import Neo4jMultiModalRetriever
from src.retriever.qalf_fusion import QALFFusion

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalSystem(Protocol):
    def __call__(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]: ...


class SystemRegistry:
    def __init__(self, neo4j_manager: Neo4jManager, config: Dict[str, Any] = None):
        self.neo4j_manager = neo4j_manager
        self.config = config or {}

        # Initialize components
        self.retriever = Neo4jMultiModalRetriever(
            neo4j_manager=self.neo4j_manager,
            embedding_model=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_dim=self.config.get("embedding_dim", 384),
        )

        # Initialize QALF pipeline
        self.qalf = QALFPipeline(
            neo4j_manager=self.neo4j_manager,
            embedding_model=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_dim=self.config.get("embedding_dim", 384),
            enable_generator=False,
        )

        # Initialize Fusion for fixed RRF
        self.fusion = QALFFusion(k=60)

    def run_vector_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run vector-only retrieval."""
        embedding = self.retriever.get_embedding(query)
        results = self.retriever.retrieve_vector(embedding, top_k)
        return self._format_results(results, "vector_only")

    def run_keyword_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run keyword-only retrieval."""
        results = self.retriever.retrieve_keyword(query, top_k)
        return self._format_results(results, "keyword_only")

    def run_graph_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run graph-only retrieval."""
        entities = self.retriever.extract_entities(query)
        results = self.retriever.retrieve_graph(entities, top_k, query_text=query)
        return self._format_results(results, "graph_only")

    def run_fixed_rrf(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run fixed RRF fusion over all modalities."""
        # 1. Get results from all modalities
        embedding = self.retriever.get_embedding(query)
        entities = self.retriever.extract_entities(query)

        results_map = {
            "vector": self.retriever.retrieve_vector(embedding, top_k),
            "keyword": self.retriever.retrieve_keyword(query, top_k),
            "graph": self.retriever.retrieve_graph(entities, top_k, query_text=query),
        }

        # 2. Fuse with RRF (equal weights implicitly via RRF rank)
        # Note: QALF Fusion supports weighted RRF, but here we just want simple RRF.
        # We can use the fusion component but with equal weights or just rely on RRF logic.
        # The QALF fusion component's `fuse_with_consensus` uses weights.
        # For "fixed RRF", we usually mean standard RRF without adaptive weights.

        # Let's implement a simple RRF here or reuse QALF fusion with equal weights.
        # Reusing QALF fusion with equal weights (1.0) for simplicity.
        weights = {"vector": 1.0, "keyword": 1.0, "graph": 1.0}

        # We need consensus scores for the fusion method signature, but for fixed RRF
        # we might not want consensus boosting.
        # However, the user asked for "fixed_rrf – static RRF fusion over all active modalities".
        # Let's assume standard RRF.

        fused_results = self._simple_rrf(results_map, k=60)
        return self._format_results_from_tuples(fused_results[:top_k], "fixed_rrf")

    def run_native_hybrid(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Run 'native' hybrid query (Vector + Fulltext in one Cypher).
        This approximates a standard hybrid search often found in vector DBs.
        """
        embedding = self.retriever.get_embedding(query)

        # Cypher query combining vector and fulltext
        cypher = """
        CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
        YIELD node, score
        WITH node, score, 1.0 as weight
        RETURN node.id as doc_id, 
               node.title as title, 
               score * weight as final_score
        UNION
        CALL db.index.fulltext.queryNodes('documentFulltext', $query)
        YIELD node, score
        WITH node, score, 1.0 as weight
        RETURN node.id as doc_id, 
               node.title as title, 
               score * weight as final_score
        """
        # Note: The above is a naive union. A true native hybrid usually does RRF or score fusion inside the DB.
        # Since Neo4j doesn't have a single "hybrid" call that does both automatically and returns fused results
        # without user-defined logic, we will implement a client-side fusion of the two calls,
        # effectively similar to fixed_rrf but only vector+keyword.

        results_map = {
            "vector": self.retriever.retrieve_vector(embedding, top_k),
            "keyword": self.retriever.retrieve_keyword(query, top_k),
        }
        fused_results = self._simple_rrf(results_map, k=60)
        return self._format_results_from_tuples(fused_results[:top_k], "native_hybrid")

    def run_qalf(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run full QALF pipeline."""
        return self.qalf.qalf_retrieve(query, top_k)

    def run_adaptive_fixed_weights(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Run adaptive routing but with fixed (equal) RRF weights."""
        # 1. Get routing from QALF
        ling, sem, mod, ctx = self.qalf.complexity_classifier.classify_complexity_4d(
            query
        )
        intent = self.qalf.intent_classifier.classify(query)

        # This mirrors QALF logic but we do manual retrieval/fusion
        from configs.routing_table import route_to_modalities

        active_modalities = route_to_modalities((ling, sem, mod, ctx), intent)

        results_map = {}
        if "vector" in active_modalities:
            embedding = self.retriever.get_embedding(query)
            results_map["vector"] = self.retriever.retrieve_vector(embedding, top_k)
        if "keyword" in active_modalities:
            results_map["keyword"] = self.retriever.retrieve_keyword(query, top_k)
        if "graph" in active_modalities:
            entities = self.retriever.extract_entities(query)
            results_map["graph"] = self.retriever.retrieve_graph(
                entities, top_k, query_text=query
            )

        # 2. Fuse with simple RRF (equal weights)
        fused_results = self._simple_rrf(results_map, k=60)
        return self._format_results_from_tuples(fused_results[:top_k], "adaptive_fixed")

    def _format_results(
        self, results: List[Dict[str, Any]], system_name: str
    ) -> List[Dict[str, Any]]:
        """Standardize output format."""
        formatted = []
        for i, res in enumerate(results):
            formatted.append(
                {
                    "doc_id": res.get("doc_id"),
                    "score": res.get("score"),
                    "rank": i + 1,
                    "system": system_name,
                    "title": res.get("title", ""),
                }
            )
        return formatted

    def _format_results_from_tuples(
        self, results: List[Any], system_name: str
    ) -> List[Dict[str, Any]]:
        """Standardize output format from (doc_id, score) tuples."""
        formatted = []
        for i, (doc_id, score) in enumerate(results):
            formatted.append(
                {"doc_id": doc_id, "score": score, "rank": i + 1, "system": system_name}
            )
        return formatted

    def _simple_rrf(
        self, results_map: Dict[str, List[Dict[str, Any]]], k: int = 60
    ) -> List[Any]:
        """Simple RRF implementation."""
        scores = {}
        for source, items in results_map.items():
            for rank, item in enumerate(items):
                doc_id = item.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank + 1)

        # Sort by score descending
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def get_system(self, name: str) -> RetrievalSystem:
        """Get system by name."""
        systems = {
            "vector_only": self.run_vector_only,
            "keyword_only": self.run_keyword_only,
            "graph_only": self.run_graph_only,
            "fixed_rrf": self.run_fixed_rrf,
            "native_hybrid": self.run_native_hybrid,
            "qalf": self.run_qalf,
            "adaptive_fixed": self.run_adaptive_fixed_weights,
        }
        if name not in systems:
            raise ValueError(
                f"Unknown system: {name}. Available: {list(systems.keys())}"
            )
        return systems[name]
