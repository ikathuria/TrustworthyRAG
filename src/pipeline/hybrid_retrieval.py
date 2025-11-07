from typing import List, Dict, Any, Optional
import logging

from src.graph_ingestion import Neo4jManager

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph


class HybridRetriever:
    """Hybrid retrieval combining graph traversal and Neo4j native vector search"""

    def __init__(self, neo4j_manager: Neo4jManager, config: Dict[str, Any]):
        self.neo4j_manager = neo4j_manager
        self.config = config
        self._logger = self._setup_logging()

        self.graph = None
        self.embeddings = None

        self._initialize_components()

    def _setup_logging(self) -> logging.Logger:
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

    def _initialize_components(self):
        embed_model_name = self.config.get(
            "embedding_model", "all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

        self.graph = Neo4jGraph(
            url=self.neo4j_manager.uri,
            username=self.neo4j_manager.username,
            password=self.neo4j_manager.password,
            database=self.neo4j_manager.database,
        )
        self._logger.info("HybridRetriever components initialized")

    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
        if method == "graph":
            return self._graph_retrieval(query, top_k)
        elif method == "vector":
            return self._vector_retrieval(query, top_k)
        else:
            graph_results = self._graph_retrieval(query, top_k)
            vector_results = self._vector_retrieval(query, top_k)
            fused = self._fuse_results(graph_results, vector_results)
            return fused

    def _graph_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            cypher_query = """
                CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
                RETURN node.text as entity, node.type as type, score
                ORDER BY score DESC
                LIMIT $top_k
            """
            results = self.neo4j_manager.query_graph(
                cypher_query, {"query": query, "top_k": top_k}
            )
            return {
                "method": "graph",
                "results": results,
                "count": len(results),
                "success": True,
            }
        except Exception as e:
            self._logger.error(f"Graph retrieval failed: {e}")
            return {"method": "graph", "success": False, "error": str(e)}

    def _vector_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            # Embed query
            query_embedding = self.embeddings.embed_query(query)

            # Convert embedding to Python list or array format for Neo4j
            query_emb_list = query_embedding.tolist()

            # Note: Neo4j GDS does not accept query vector inside config map in all versions.
            # Use a prefiltered approach or create an embedding node beforehand and compare.
            # Here is a simplified approach querying all entity embeddings with cosine similarity >= threshold

            cypher_query = """
                WITH $query_emb AS queryEmb
                MATCH (e:Entity)
                WHERE exists(e.embedding)
                WITH e,
                    gds.alpha.similarity.cosine(e.embedding, queryEmb) AS similarity
                WHERE similarity >= 0.7
                RETURN e.id AS id, e.text AS entity, e.type AS type, similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """

            params = {
                "query_emb": query_emb_list,
                "top_k": top_k
            }

            results = self.neo4j_manager.query_graph(cypher_query, params)
            processed_results = [
                {
                    "id": r["id"],
                    "entity": r["entity"],
                    "type": r["type"],
                    "similarity_score": r["similarity"],
                }
                for r in results
            ]

            return {
                "method": "vector",
                "results": processed_results,
                "count": len(processed_results),
                "success": True,
            }
        except Exception as e:
            self._logger.error(f"Vector retrieval failed: {e}")
            return {"method": "vector", "success": False, "error": str(e)}

    def _fuse_results(
        self, graph_results: Dict[str, Any], vector_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        combined = []
        seen_texts = set()

        if graph_results.get("success") and graph_results.get("results"):
            for r in graph_results["results"]:
                text = r.get("entity") or r.get("content") or ""
                if text and text not in seen_texts:
                    combined.append({"source": "graph", "text": text, **r})
                    seen_texts.add(text)

        if vector_results.get("success") and vector_results.get("results"):
            for r in vector_results["results"]:
                text = r.get("entity") or r.get("content") or ""
                if text and text not in seen_texts:
                    combined.append({"source": "vector", "text": text, **r})
                    seen_texts.add(text)

        # TODO: Add weighted scoring and sorting by confidence/similarity

        success = bool(combined)
        return {"method": "hybrid", "results": combined, "count": len(combined), "success": success}
