"""
Unified Neo4j Retriever for QALF.
Implements all three retrieval modalities (vector, graph, keyword) using a single Neo4j database.
"""

from typing import List, Dict, Any, Optional
import logging
import time

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
        embedding_dim: int = 384,
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
        start_time = time.time()
        self._logger.debug(
            f"📊 Generating embedding for text (length: {len(text)} chars)"
        )
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        elapsed_time = time.time() - start_time
        self._logger.debug(
            f"✅ Embedding generated in {elapsed_time:.3f}s (dim: {len(embedding)})"
        )
        return embedding.tolist()

    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entity names from query using spaCy NER.
        Enhanced with fallback heuristics for better entity detection.

        Args:
            query: Input query

        Returns:
            List of entity names
        """
        start_time = time.time()
        self._logger.debug(f"🔎 Extracting entities from query: '{query[:100]}...'")
        entities = []

        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)
            entities = [ent.text for ent in doc.ents]

            # Fallback: extract potential entities if NER found nothing
            if not entities:
                # Look for quoted entities
                import re

                quoted = re.findall(r'"([^"]+)"', query)
                entities.extend(quoted)

                # Look for capitalized words (potential proper nouns)
                # Skip common question words
                common_words = {
                    "who",
                    "what",
                    "when",
                    "where",
                    "why",
                    "how",
                    "is",
                    "are",
                    "was",
                    "were",
                    "the",
                    "a",
                    "an",
                }
                words = query.split()
                capitalized = [
                    w.strip(".,!?")
                    for w in words
                    if w[0].isupper() and w.lower() not in common_words
                ]
                entities.extend(capitalized)

                # Remove duplicates while preserving order
                seen = set()
                entities = [
                    e
                    for e in entities
                    if e.lower() not in seen and not seen.add(e.lower())
                ]

            elapsed_time = time.time() - start_time
            self._logger.debug(
                f"✅ Extracted {len(entities)} entities in {elapsed_time:.3f}s: {entities}"
            )
            return entities
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.warning(
                f"⚠️  Entity extraction failed after {elapsed_time:.3f}s: {e}"
            )
            # Last resort: extract any capitalized words
            import re

            capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
            common_words = {
                "Who",
                "What",
                "When",
                "Where",
                "Why",
                "How",
                "The",
                "A",
                "An",
            }
            entities = [e for e in capitalized if e not in common_words]
            if entities:
                self._logger.debug(f"   Using fallback extraction: {entities}")
            return entities

    def retrieve_vector(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search via Neo4j HNSW index.
        Searches Chunk and Table nodes (which have embeddings) and aggregates by Document.
        Now includes multimodal search (text chunks + tables).

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of documents with doc_id, title, and score
        """
        start_time = time.time()
        self._logger.info(
            f"🔍 [VECTOR] Starting multimodal vector similarity search (top_k={top_k}, embedding_dim={len(query_embedding)})"
        )
        try:
            # Search both Chunk nodes (text) and Table nodes (tables) with embeddings
            # Use separate queries and combine in Python (more reliable than UNION ALL)
            # First, search text chunks
            chunk_query = """
            CALL db.index.vector.queryNodes('text_chunk_vector_index', $top_k * 3, $query_embedding)
            YIELD node AS chunk, score
            MATCH (chunk)-[:IN_DOCUMENT]->(doc:Document)
            WITH doc, MAX(score) AS max_score, AVG(score) AS avg_score, COUNT(chunk) AS chunk_count
            RETURN doc.id AS doc_id, 
                   COALESCE(doc.source_file, doc.id) AS title,
                   max_score AS score,
                   avg_score,
                   chunk_count,
                   'text' AS modality
            ORDER BY max_score DESC
            LIMIT $top_k
            """

            chunk_results = self.neo4j_manager.query_graph(
                chunk_query, {"query_embedding": query_embedding, "top_k": top_k}
            )

            # Then, search tables (if index exists)
            table_results = []
            try:
                table_query = """
                CALL db.index.vector.queryNodes('table_vector_index', $top_k * 3, $query_embedding)
                YIELD node AS table, score
                MATCH (table)-[:IN_DOCUMENT]->(doc:Document)
                WITH doc, MAX(score) AS max_score, AVG(score) AS avg_score, COUNT(table) AS table_count
                RETURN doc.id AS doc_id, 
                       COALESCE(doc.source_file, doc.id) AS title,
                       max_score AS score,
                       avg_score,
                       table_count,
                       'table' AS modality
                ORDER BY max_score DESC
                LIMIT $top_k
                """
                table_results = self.neo4j_manager.query_graph(
                    table_query, {"query_embedding": query_embedding, "top_k": top_k}
                )
            except Exception as table_error:
                self._logger.debug(
                    f"   Table index not available or error: {table_error}"
                )

            # Combine and aggregate results by document
            doc_scores = {}
            for record in chunk_results:
                doc_id = record.get("doc_id", "")
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc_id": doc_id,
                        "title": record.get("title", ""),
                        "scores": [],
                        "modalities": set(),
                    }
                doc_scores[doc_id]["scores"].append(record.get("score", 0.0))
                doc_scores[doc_id]["modalities"].add("text")

            for record in table_results:
                doc_id = record.get("doc_id", "")
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc_id": doc_id,
                        "title": record.get("title", ""),
                        "scores": [],
                        "modalities": set(),
                    }
                doc_scores[doc_id]["scores"].append(record.get("score", 0.0))
                doc_scores[doc_id]["modalities"].add("table")

            # Aggregate scores: take max score, boost if multiple modalities
            formatted_results = []
            for doc_id, data in doc_scores.items():
                max_score = max(data["scores"])
                modality_boost = (
                    1.2 if len(data["modalities"]) > 1 else 1.0
                )  # Boost multimodal matches
                final_score = max_score * modality_boost
                formatted_results.append(
                    {"doc_id": doc_id, "title": data["title"], "score": final_score}
                )

            # Sort by score and limit
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            formatted_results = formatted_results[:top_k]

            elapsed_time = time.time() - start_time
            self._logger.info(
                f"✅ [VECTOR] Retrieved {len(formatted_results)} documents in {elapsed_time:.3f}s"
            )
            if formatted_results:
                top_score = formatted_results[0].get("score", 0.0)
                modalities_found = doc_scores.get(
                    formatted_results[0].get("doc_id", ""), {}
                ).get("modalities", set())
                self._logger.debug(
                    f"   Top score: {top_score:.4f}, modalities: {modalities_found}"
                )

            return formatted_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.warning(
                f"⚠️  [VECTOR] Multimodal vector retrieval failed after {elapsed_time:.3f}s: {e}"
            )
            self._logger.info("   Attempting text-only vector search...")
            # Fallback: text-only search if table index doesn't exist
            try:
                cypher_query = """
                CALL db.index.vector.queryNodes('text_chunk_vector_index', $top_k * 3, $query_embedding)
                YIELD node AS chunk, score
                MATCH (chunk)-[:IN_DOCUMENT]->(doc:Document)
                WITH doc, MAX(score) AS max_score, AVG(score) AS avg_score, COUNT(chunk) AS chunk_count
                RETURN doc.id AS doc_id, 
                       COALESCE(doc.source_file, doc.id) AS title,
                       max_score AS score
                ORDER BY max_score DESC
                LIMIT $top_k
                """
                results = self.neo4j_manager.query_graph(
                    cypher_query, {"query_embedding": query_embedding, "top_k": top_k}
                )
                formatted_results = []
                for record in results:
                    formatted_results.append(
                        {
                            "doc_id": record.get("doc_id", ""),
                            "title": record.get("title", ""),
                            "score": float(record.get("score", 0.0)),
                        }
                    )
                self._logger.info(
                    f"✅ [VECTOR-FALLBACK] Retrieved {len(formatted_results)} documents (text-only)"
                )
                return formatted_results
            except Exception as e2:
                self._logger.warning(f"⚠️  [VECTOR] Fallback also failed: {e2}")
                return self._fallback_vector_search(query_embedding, top_k)

    def _fallback_vector_search(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback vector search using manual cosine similarity on Chunk nodes.
        Used when vector index is not available.
        """
        start_time = time.time()
        self._logger.info(
            f"🔄 [VECTOR-FALLBACK] Using fallback cosine similarity search (top_k={top_k})"
        )
        try:
            # Search Chunk nodes and aggregate to Documents
            cypher_query = """
            MATCH (chunk:Chunk)
            WHERE chunk.embedding IS NOT NULL AND size(chunk.embedding) = $dim
            WITH chunk, 
                 gds.similarity.cosine(chunk.embedding, $query_embedding) AS score
            WHERE score > 0.3
            MATCH (chunk)-[:IN_DOCUMENT]->(doc:Document)
            WITH doc, MAX(score) AS max_score, AVG(score) AS avg_score, COUNT(chunk) AS chunk_count
            RETURN doc.id AS doc_id,
                   COALESCE(doc.source_file, doc.id) AS title,
                   max_score AS score
            ORDER BY max_score DESC, chunk_count DESC
            LIMIT $top_k
            """

            results = self.neo4j_manager.query_graph(
                cypher_query,
                {
                    "query_embedding": query_embedding,
                    "dim": len(query_embedding),
                    "top_k": top_k,
                },
            )

            formatted_results = []
            for record in results:
                formatted_results.append(
                    {
                        "doc_id": record.get("doc_id", ""),
                        "title": record.get("title", ""),
                        "score": float(record.get("score", 0.0)),
                    }
                )

            elapsed_time = time.time() - start_time
            self._logger.info(
                f"✅ [VECTOR-FALLBACK] Retrieved {len(formatted_results)} documents in {elapsed_time:.3f}s"
            )

            return formatted_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.error(
                f"❌ [VECTOR-FALLBACK] Fallback search failed after {elapsed_time:.3f}s: {e}"
            )
            return []

    def retrieve_keyword(
        self, query_string: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Full-text keyword search via Neo4j Lucene index.
        Searches Chunk nodes and aggregates by Document.

        Args:
            query_string: Query text
            top_k: Number of results to return

        Returns:
            List of documents with doc_id, title, and score
        """
        start_time = time.time()
        self._logger.info(
            f"🔍 [KEYWORD] Starting full-text keyword search (top_k={top_k}, query: '{query_string[:100]}...')"
        )
        try:
            # Try searching Chunk fulltext index first (chunk_fulltext from setup_indexes)
            # If that doesn't exist, fall back to searching Chunk content directly
            cypher_query = """
            CALL db.index.fulltext.queryNodes("chunk_fulltext", $query_string, {limit: $top_k * 3})
            YIELD node AS chunk, score
            MATCH (chunk)-[:IN_DOCUMENT]->(doc:Document)
            WITH doc, MAX(score) AS max_score, AVG(score) AS avg_score, COUNT(chunk) AS chunk_count
            RETURN doc.id AS doc_id,
                   COALESCE(doc.source_file, doc.id) AS title,
                   max_score AS score
            ORDER BY max_score DESC, chunk_count DESC
            LIMIT $top_k
            """

            results = self.neo4j_manager.query_graph(
                cypher_query, {"query_string": query_string, "top_k": top_k}
            )

            # If no results, try direct text search on Chunk content
            if not results:
                self._logger.debug(
                    "   Fulltext index search returned no results, trying direct text search..."
                )
                cypher_query = """
                MATCH (chunk:Chunk)
                WHERE chunk.content CONTAINS $query_string
                MATCH (chunk)-[:IN_DOCUMENT]->(doc:Document)
                WITH doc, COUNT(chunk) AS chunk_count
                RETURN doc.id AS doc_id,
                       COALESCE(doc.source_file, doc.id) AS title,
                       chunk_count AS score
                ORDER BY chunk_count DESC
                LIMIT $top_k
                """
                results = self.neo4j_manager.query_graph(
                    cypher_query, {"query_string": query_string, "top_k": top_k}
                )

            formatted_results = []
            for record in results:
                formatted_results.append(
                    {
                        "doc_id": record.get("doc_id", ""),
                        "title": record.get("title", ""),
                        "score": float(record.get("score", 0.0)),
                    }
                )

            elapsed_time = time.time() - start_time
            self._logger.info(
                f"✅ [KEYWORD] Retrieved {len(formatted_results)} documents in {elapsed_time:.3f}s"
            )
            if formatted_results:
                top_score = formatted_results[0].get("score", 0.0)
                self._logger.debug(f"   Top score: {top_score:.4f}")

            return formatted_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.error(
                f"❌ [KEYWORD] Keyword retrieval failed after {elapsed_time:.3f}s: {e}"
            )
            return []

    def retrieve_graph(
        self, query_entities: List[str], top_k: int = 10, query_text: str = None
    ) -> List[Dict[str, Any]]:
        """
        Graph traversal retrieval via Neo4j Cypher.
        Enhanced to search for specific relationship types and handle relationship queries.

        Args:
            query_entities: List of entity names extracted from query
            top_k: Number of results to return
            query_text: Original query text (for relationship detection)

        Returns:
            List of documents with doc_id, title, and score
        """
        start_time = time.time()

        # Detect relationship keywords in query
        relationship_keywords = []
        if query_text:
            query_lower = query_text.lower()
            relationship_patterns = {
                "CREATED": ["created", "creator", "made", "built", "developed"],
                "OWNS": ["owns", "owner", "belongs to"],
                "WROTE": ["wrote", "author", "written by"],
                "DESIGNED": ["designed", "designer"],
                "INVENTED": ["invented", "inventor"],
            }
            for rel_type, keywords in relationship_patterns.items():
                if any(kw in query_lower for kw in keywords):
                    relationship_keywords.append(rel_type)

        # If no entities but we have relationship keywords, try to find entities from query
        if not query_entities and query_text:
            # Extract potential entity names (capitalized words, quoted phrases)
            import re

            # Look for quoted entities
            quoted = re.findall(r'"([^"]+)"', query_text)
            # Look for capitalized words (potential proper nouns)
            capitalized = re.findall(r"\b[A-Z][a-z]+\b", query_text)
            query_entities = quoted + capitalized
            # Remove common words
            common_words = {
                "who",
                "what",
                "when",
                "where",
                "why",
                "how",
                "created",
                "created by",
            }
            query_entities = [
                e for e in query_entities if e.lower() not in common_words
            ]

        if not query_entities:
            self._logger.info(
                f"🔍 [GRAPH] No entities extracted, skipping graph retrieval"
            )
            return []

        self._logger.info(
            f"🔍 [GRAPH] Starting graph traversal search (top_k={top_k}, entities: {query_entities}, relationships: {relationship_keywords})"
        )
        try:
            # Build relationship pattern - include specific relationship types if detected
            # Use a more flexible approach: search for entities and their relationships
            # Boost score if specific relationship types are found
            if relationship_keywords:
                # Search for entities and check if they have relationships matching the keywords
                # Use CASE to check relationship types dynamically
                cypher_query = """
                MATCH (entity:Entity)
                WHERE entity.name IN $query_entities OR entity.text IN $query_entities OR entity.id IN $query_entities
                OPTIONAL MATCH (entity)-[r]->(related:Entity)
                WITH entity, related, r, type(r) AS rel_type,
                     CASE 
                         WHEN related IS NOT NULL AND r IS NOT NULL AND 
                              (rel_type IN $relationship_types OR 
                               rel_type CONTAINS 'CREATED' OR 
                               rel_type CONTAINS 'CREATED_BY' OR
                               rel_type CONTAINS 'DEVELOPED' OR
                               rel_type CONTAINS 'MADE')
                         THEN 2.5
                         WHEN related IS NOT NULL AND r IS NOT NULL 
                         THEN 1.5
                         ELSE 1.0 
                     END AS relationship_boost
                MATCH (entity)-[:EXTRACTED_FROM]->(source)
                MATCH (source)-[:IN_DOCUMENT]->(doc:Document)
                OPTIONAL MATCH (related)-[:EXTRACTED_FROM]->(source2)
                OPTIONAL MATCH (source2)-[:IN_DOCUMENT]->(doc2:Document)
                WITH DISTINCT doc, doc2, entity, related, relationship_boost
                WITH COALESCE(doc, doc2) AS final_doc, SUM(relationship_boost) AS score
                WHERE final_doc IS NOT NULL
                RETURN final_doc.id AS doc_id,
                       COALESCE(final_doc.source_file, final_doc.id) AS title,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """

                results = self.neo4j_manager.query_graph(
                    cypher_query,
                    {
                        "query_entities": query_entities,
                        "relationship_types": relationship_keywords,
                        "top_k": top_k,
                    },
                )
            else:
                # Generic relationship search (fallback) - search for any relationships
                cypher_query = """
                MATCH (entity:Entity)
                WHERE entity.name IN $query_entities OR entity.text IN $query_entities
                OPTIONAL MATCH (entity)-[r]->(related:Entity)
                WITH entity, related, r,
                     CASE WHEN related IS NOT NULL AND r IS NOT NULL THEN 1.5 ELSE 1.0 END AS relationship_boost
                MATCH (entity)-[:EXTRACTED_FROM]->(source)
                MATCH (source)-[:IN_DOCUMENT]->(doc:Document)
                OPTIONAL MATCH (related)-[:EXTRACTED_FROM]->(source2)
                OPTIONAL MATCH (source2)-[:IN_DOCUMENT]->(doc2:Document)
                WITH DISTINCT doc, doc2, entity, related, relationship_boost
                WITH COALESCE(doc, doc2) AS final_doc, SUM(relationship_boost) AS score
                WHERE final_doc IS NOT NULL
                RETURN final_doc.id AS doc_id,
                       COALESCE(final_doc.source_file, final_doc.id) AS title,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """

                results = self.neo4j_manager.query_graph(
                    cypher_query, {"query_entities": query_entities, "top_k": top_k}
                )

            formatted_results = []
            for record in results:
                formatted_results.append(
                    {
                        "doc_id": record.get("doc_id", ""),
                        "title": record.get("title", ""),
                        "score": float(record.get("score", 0.0)),
                    }
                )

            elapsed_time = time.time() - start_time
            self._logger.info(
                f"✅ [GRAPH] Retrieved {len(formatted_results)} documents in {elapsed_time:.3f}s"
            )
            if formatted_results:
                top_score = formatted_results[0].get("score", 0.0)
                self._logger.debug(f"   Top score: {top_score:.4f}")

            return formatted_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.error(
                f"❌ [GRAPH] Graph retrieval failed after {elapsed_time:.3f}s: {e}"
            )
            return []
