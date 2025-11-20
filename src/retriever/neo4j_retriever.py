"""
Unified Neo4j Retriever for QALF.
Implements all three retrieval modalities (vector, graph, keyword) using a single Neo4j database.
"""

from typing import List, Dict, Any, Optional
import logging

from src.neo4j.neo4j_manager import Neo4jManager
from sentence_transformers import SentenceTransformer


class Neo4jMultiModalRetriever:
    """
    Unified retriever for vector, graph, and keyword search using Neo4j.
    All three modalities query the same Neo4j database.
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ):
        """
        Initialize the unified Neo4j retriever.
        
        Args:
            neo4j_manager: Neo4jManager instance
            embedding_model: Name of sentence transformer model
            embedding_dim: Dimension of embeddings
        """
        self.neo4j_manager = neo4j_manager
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        
        self._logger = self._setup_logging()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self._logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            self._logger.error(f"Failed to load embedding model: {e}")
            raise

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

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
        
        Returns:
            List of floats representing the embedding
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entity names from query using spaCy NER.
        
        Args:
            query: Input query
        
        Returns:
            List of entity names
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)
            entities = [ent.text for ent in doc.ents]
            return entities
        except Exception as e:
            self._logger.warning(f"Entity extraction failed: {e}")
            return []

    def retrieve_vector(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search via Neo4j HNSW index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of documents with doc_id, title, and score
        """
        try:
            # Neo4j 5.x+ vector search syntax
            # Note: Syntax may vary by Neo4j version
            cypher_query = """
            CALL db.index.vector.queryNodes('document-embeddings', $top_k, $query_embedding)
            YIELD node, score
            RETURN node.id AS doc_id, 
                   COALESCE(node.title, node.id) AS title,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = self.neo4j_manager.query_graph(
                cypher_query,
                {
                    "query_embedding": query_embedding,
                    "top_k": top_k
                }
            )
            
            # Ensure consistent format
            formatted_results = []
            for record in results:
                formatted_results.append({
                    "doc_id": record.get("doc_id", ""),
                    "title": record.get("title", ""),
                    "score": float(record.get("score", 0.0))
                })
            
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Vector retrieval failed: {e}")
            # Fallback: manual cosine similarity if vector index not available
            return self._fallback_vector_search(query_embedding, top_k)

    def _fallback_vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback vector search using manual cosine similarity.
        Used when vector index is not available.
        """
        try:
            cypher_query = """
            MATCH (doc:Document)
            WHERE exists(doc.embedding) AND size(doc.embedding) = $dim
            WITH doc, 
                 gds.similarity.cosine(doc.embedding, $query_embedding) AS score
            WHERE score > 0.5
            RETURN doc.id AS doc_id,
                   COALESCE(doc.title, doc.id) AS title,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = self.neo4j_manager.query_graph(
                cypher_query,
                {
                    "query_embedding": query_embedding,
                    "dim": len(query_embedding),
                    "top_k": top_k
                }
            )
            
            formatted_results = []
            for record in results:
                formatted_results.append({
                    "doc_id": record.get("doc_id", ""),
                    "title": record.get("title", ""),
                    "score": float(record.get("score", 0.0))
                })
            
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Fallback vector search failed: {e}")
            return []

    def retrieve_keyword(
        self,
        query_string: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Full-text keyword search via Neo4j Lucene index.
        
        Args:
            query_string: Query text
            top_k: Number of results to return
        
        Returns:
            List of documents with doc_id, title, and score
        """
        try:
            cypher_query = """
            CALL db.index.fulltext.queryNodes("documentFulltext", $query_string, {limit: $top_k})
            YIELD node, score
            RETURN node.id AS doc_id,
                   COALESCE(node.title, node.id) AS title,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = self.neo4j_manager.query_graph(
                cypher_query,
                {
                    "query_string": query_string,
                    "top_k": top_k
                }
            )
            
            formatted_results = []
            for record in results:
                formatted_results.append({
                    "doc_id": record.get("doc_id", ""),
                    "title": record.get("title", ""),
                    "score": float(record.get("score", 0.0))
                })
            
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Keyword retrieval failed: {e}")
            return []

    def retrieve_graph(
        self,
        query_entities: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Graph traversal retrieval via Neo4j Cypher.
        
        Args:
            query_entities: List of entity names extracted from query
            top_k: Number of results to return
        
        Returns:
            List of documents with doc_id, title, and score
        """
        if not query_entities:
            return []
        
        try:
            cypher_query = """
            MATCH (entity:Entity)
            WHERE entity.name IN $query_entities OR entity.text IN $query_entities
            MATCH (entity)-[:RELATED_TO|MENTIONS|CONTAINS*1..2]-(doc:Document)
            RETURN DISTINCT doc.id AS doc_id,
                   COALESCE(doc.title, doc.id) AS title,
                   COUNT(*) AS score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = self.neo4j_manager.query_graph(
                cypher_query,
                {
                    "query_entities": query_entities,
                    "top_k": top_k
                }
            )
            
            formatted_results = []
            for record in results:
                formatted_results.append({
                    "doc_id": record.get("doc_id", ""),
                    "title": record.get("title", ""),
                    "score": float(record.get("score", 0.0))
                })
            
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Graph retrieval failed: {e}")
            return []

