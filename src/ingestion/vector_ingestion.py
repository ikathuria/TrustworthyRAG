from typing import List, Dict, Any

from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging

from src.ingestion.neo4j_manager import Neo4jManager
from src.utils.base_extractor import Entity, Relation


class VectorDBManager(Neo4jManager):
    """Manages Neo4j vector database operations for cybersecurity knowledge graph"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize GraphDBManager with connection parameters"""
        super().__init__(uri, username, password, database)
        self._initialize_vector_index()

    def _initialize_vector_index(self):
        """Create GDS vector similarity index on Entity.embedding"""
        index_query = """
        CALL gds.index.drop('entityEmbeddingIndex') YIELD name
        """
        create_index_query = """
        CALL gds.index.createVectorSimilarityIndex(
            'entityEmbeddingIndex',
            'Entity',
            'embedding',
            null
        )
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Drop outstanding old indexes if exists
                try:
                    session.run(index_query)
                except Exception:
                    pass
                # Create new index
                session.run(create_index_query)
            self._logger.info("Vector similarity index created in Neo4j (GDS)")
        except Exception as e:
            self._logger.error(f"Error creating vector index: {e}")
