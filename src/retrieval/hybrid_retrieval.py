from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from src.neo4j.neo4j_manager import Neo4jManager


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Graph traversal (relationships, multi-hop)
    2. Vector similarity search (text chunks, images, tables, entities)
    3. Fulltext search (keyword matching)
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize HybridRetriever with Neo4j connection and embedding model.
        
        Args:
            neo4j_manager: Neo4jManager instance for database operations
            embedding_model: HuggingFace model name for embeddings
        """
        self.neo4j_manager = neo4j_manager
        self.embedding_model_name = embedding_model
        self._logger = self._setup_logging()

        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self._logger.info(f"Loaded embedding model: {embedding_model}")

        # Ensure indexes exist
        self._ensure_indexes()

    def _setup_logging(self) -> logging.Logger:
        """Setup logger for the class"""
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

    def _ensure_indexes(self):
        """Ensure required indexes exist for retrieval"""
        try:
            # Check for fulltext index on entities
            fulltext_query = """
            CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
            FOR (e:Entity)
            ON EACH [e.text, e.type]
            """
            self.neo4j_manager.query_graph(fulltext_query)

            # Check for fulltext index on chunks
            chunk_fulltext_query = """
            CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
            FOR (c:Chunk)
            ON EACH [c.content]
            """
            self.neo4j_manager.query_graph(chunk_fulltext_query)

            self._logger.info("Fulltext indexes checked/created")
        except Exception as e:
            self._logger.warning(f"Index creation warning: {str(e)}")

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant content using specified method.
        
        Args:
            query: User query string
            method: "hybrid", "graph", "vector", or "fulltext"
            top_k: Number of results to return
            weights: Optional retrieval weights {"vector": 0.5, "graph": 0.3, "fulltext": 0.2}
            
        Returns:
            Dict with results, method used, and metadata
        """
        if weights is None:
            weights = {"vector": 0.5, "graph": 0.3, "fulltext": 0.2}

        if method == "graph":
            return self._graph_retrieval(query, top_k)
        elif method == "vector":
            return self._vector_retrieval(query, top_k)
        elif method == "fulltext":
            return self._fulltext_retrieval(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieval(query, top_k, weights)

    def _vector_retrieval(
        self,
        query: str,
        top_k: int = 5,
        modalities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Vector similarity search across multimodal content.
        
        Args:
            query: Search query
            top_k: Number of results per modality
            modalities: List of modalities to search ["chunk", "entity", "image", "table"]
            
        Returns:
            Dict with vector search results
        """
        if modalities is None:
            modalities = ["chunk", "entity"]

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()

            all_results = []

            # Search Chunks (text content)
            if "chunk" in modalities:
                chunk_results = self._vector_search_by_label(
                    "Chunk",
                    "text_chunk_vector_index",
                    query_embedding,
                    top_k
                )
                all_results.extend(chunk_results)

            # Search Entities
            if "entity" in modalities:
                entity_results = self._vector_search_by_label(
                    "Entity",
                    "entity_vector_index",
                    query_embedding,
                    top_k
                )
                all_results.extend(entity_results)

            # Search Images (by captions)
            if "image" in modalities:
                image_results = self._vector_search_by_label(
                    "Image",
                    "image_vector_index",
                    query_embedding,
                    top_k
                )
                all_results.extend(image_results)

            # Search Tables
            if "table" in modalities:
                table_results = self._vector_search_by_label(
                    "Table",
                    "table_vector_index",
                    query_embedding,
                    top_k
                )
                all_results.extend(table_results)

            # Sort by score and limit
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return {
                "method": "vector",
                "results": all_results[:top_k * 2],  # Return more for fusion
                "count": len(all_results[:top_k * 2]),
                "success": True
            }

        except Exception as e:
            self._logger.error(f"Vector retrieval failed: {str(e)}")
            return {
                "method": "vector",
                "results": [],
                "count": 0,
                "success": False,
                "error": str(e)
            }

    def _vector_search_by_label(
        self,
        label: str,
        index_name: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search vector index for specific node label.
        
        Args:
            label: Node label (Chunk, Entity, Image, Table)
            index_name: Vector index name
            query_embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of results with score and properties
        """
        try:
            # Use Neo4j native vector search
            cypher_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            RETURN 
                node.id AS id,
                node.content AS content,
                node.text AS text,
                node.caption AS caption,
                node.type AS type,
                node.modality AS modality,
                labels(node)[0] AS label,
                score
            ORDER BY score DESC
            """

            params = {
                "index_name": index_name,
                "top_k": top_k,
                "query_embedding": query_embedding
            }

            results = self.neo4j_manager.query_graph(cypher_query, params)

            # Format results
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "id": r.get("id"),
                    "content": r.get("content") or r.get("text") or r.get("caption", ""),
                    "type": r.get("type"),
                    "modality": r.get("modality", label.lower()),
                    "label": r.get("label", label),
                    "score": r.get("score", 0.0),
                    "source": "vector"
                })

            return formatted_results

        except Exception as e:
            self._logger.warning(f"Vector search failed for {label}: {str(e)}")
            return []

    def _graph_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Graph-based retrieval using relationship traversal.
        Finds entities and their connected context.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dict with graph traversal results
        """
        try:
            # Multi-hop graph query to find related entities and chunks
            cypher_query = """
            // Find entities matching query terms
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query) 
            YIELD node AS entity, score
            
            // Get chunks/images/tables these entities were extracted from
            MATCH (entity)-[:EXTRACTED_FROM]->(source)
            
            // Get the document
            OPTIONAL MATCH (source)-[:IN_DOCUMENT]->(doc:Document)
            
            // Get other entities from same source (related context)
            OPTIONAL MATCH (source)<-[:EXTRACTED_FROM]-(related:Entity)
            WHERE related <> entity
            
            WITH entity, source, doc, score, 
                 collect(DISTINCT related.text)[..5] AS related_entities
            
            RETURN 
                entity.id AS entity_id,
                entity.text AS entity,
                entity.type AS entity_type,
                source.content AS source_content,
                source.modality AS modality,
                labels(source)[0] AS source_type,
                doc.source_file AS document,
                related_entities,
                score
            ORDER BY score DESC
            LIMIT $top_k
            """

            params = {"query": query, "top_k": top_k}
            results = self.neo4j_manager.query_graph(cypher_query, params)

            # Format results
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "entity_id": r.get("entity_id"),
                    "entity": r.get("entity"),
                    "entity_type": r.get("entity_type"),
                    # Truncate long content
                    "content": r.get("source_content", "")[:500],
                    "modality": r.get("modality"),
                    "source_type": r.get("source_type"),
                    "document": r.get("document"),
                    "related_entities": r.get("related_entities", []),
                    "score": r.get("score", 0.0),
                    "source": "graph"
                })

            return {
                "method": "graph",
                "results": formatted_results,
                "count": len(formatted_results),
                "success": True
            }

        except Exception as e:
            self._logger.error(f"Graph retrieval failed: {str(e)}")
            return {
                "method": "graph",
                "results": [],
                "count": 0,
                "success": False,
                "error": str(e)
            }

    def _fulltext_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Keyword-based fulltext search on chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dict with fulltext search results
        """
        try:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
            YIELD node, score
            RETURN 
                node.id AS id,
                node.content AS content,
                node.chunk_index AS chunk_index,
                node.modality AS modality,
                score
            ORDER BY score DESC
            LIMIT $top_k
            """

            params = {"query": query, "top_k": top_k}
            results = self.neo4j_manager.query_graph(cypher_query, params)

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "id": r.get("id"),
                    "content": r.get("content", ""),
                    "chunk_index": r.get("chunk_index"),
                    "modality": r.get("modality", "text"),
                    "score": r.get("score", 0.0),
                    "source": "fulltext"
                })

            return {
                "method": "fulltext",
                "results": formatted_results,
                "count": len(formatted_results),
                "success": True
            }

        except Exception as e:
            self._logger.error(f"Fulltext retrieval failed: {str(e)}")
            return {
                "method": "fulltext",
                "results": [],
                "count": 0,
                "success": False,
                "error": str(e)
            }

    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int = 5,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval combining vector, graph, and fulltext search with weighted fusion.
        
        Args:
            query: Search query
            top_k: Number of final results
            weights: Retrieval method weights
            
        Returns:
            Dict with fused results from all methods
        """
        if weights is None:
            weights = {"vector": 0.5, "graph": 0.3, "fulltext": 0.2}

        # Retrieve from each method
        vector_results = self._vector_retrieval(query, top_k=top_k * 2)
        graph_results = self._graph_retrieval(query, top_k=top_k * 2)
        fulltext_results = self._fulltext_retrieval(query, top_k=top_k * 2)

        # Fuse results with weighted scoring
        fused = self._weighted_fusion(
            vector_results,
            graph_results,
            fulltext_results,
            weights,
            top_k
        )

        return fused

    def _weighted_fusion(
        self,
        vector_results: Dict[str, Any],
        graph_results: Dict[str, Any],
        fulltext_results: Dict[str, Any],
        weights: Dict[str, float],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Fuse results from multiple retrieval methods using weighted scoring.
        Implements Reciprocal Rank Fusion (RRF) with weighting.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph traversal
            fulltext_results: Results from fulltext search
            weights: Method weights
            top_k: Number of final results
            
        Returns:
            Dict with fused and ranked results
        """
        # Collect all unique items with their scores
        item_scores = {}
        k = 60  # RRF constant

        # Process vector results
        if vector_results.get("success") and vector_results.get("results"):
            for rank, item in enumerate(vector_results["results"], start=1):
                item_id = item.get("id") or item.get("content", "")[:50]
                if item_id not in item_scores:
                    item_scores[item_id] = {
                        "item": item, "score": 0.0, "sources": []}

                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                item_scores[item_id]["score"] += weights.get(
                    "vector", 0.5) * rrf_score
                item_scores[item_id]["sources"].append("vector")

        # Process graph results
        if graph_results.get("success") and graph_results.get("results"):
            for rank, item in enumerate(graph_results["results"], start=1):
                item_id = item.get("entity_id") or item.get("entity", "")
                if item_id not in item_scores:
                    item_scores[item_id] = {
                        "item": item, "score": 0.0, "sources": []}

                rrf_score = 1.0 / (k + rank)
                item_scores[item_id]["score"] += weights.get(
                    "graph", 0.3) * rrf_score
                item_scores[item_id]["sources"].append("graph")

        # Process fulltext results
        if fulltext_results.get("success") and fulltext_results.get("results"):
            for rank, item in enumerate(fulltext_results["results"], start=1):
                item_id = item.get("id") or item.get("content", "")[:50]
                if item_id not in item_scores:
                    item_scores[item_id] = {
                        "item": item, "score": 0.0, "sources": []}

                rrf_score = 1.0 / (k + rank)
                item_scores[item_id]["score"] += weights.get(
                    "fulltext", 0.2) * rrf_score
                item_scores[item_id]["sources"].append("fulltext")

        # Sort by fused score
        sorted_items = sorted(
            item_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        # Format results
        final_results = []
        for item_data in sorted_items:
            result = item_data["item"].copy()
            result["fusion_score"] = item_data["score"]
            result["retrieval_sources"] = list(set(item_data["sources"]))
            final_results.append(result)

        return {
            "method": "hybrid",
            "results": final_results,
            "count": len(final_results),
            "success": True,
            "weights_used": weights
        }
