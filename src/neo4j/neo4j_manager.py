from typing import List, Dict, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging
from src.utils.base_extractor import Entity, Relation
import src.utils.constants as C


class Neo4jManager:
    """Manages Neo4j database operations for knowledge graph"""

    def __init__(
        self,
        uri: str = C.NEO4J_URI,
        username: str = C.NEO4J_USERNAME,
        password: str = C.NEO4J_PASSWORD,
        database: str = C.NEO4J_DB
    ):
        """Initialize Neo4jManager with connection parameters
        
        Args:
            uri (str): Neo4j URI
            username (str): Neo4j username
            password (str): Neo4j password
            database (str): Neo4j database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._logger = self._setup_logging()
        self.driver = None
        self._connect()

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

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self._logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            self._logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def query_graph(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as list of dicts.
        
        Args:
            query: Cypher query string
            params: Query parameters (optional)
            
        Returns:
            List of result records as dictionaries
        """
        if params is None:
            params = {}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [dict(record) for record in result]
        except Exception as e:
            self._logger.error(f"Query execution failed: {str(e)}")
            self._logger.debug(f"Query: {query}")
            self._logger.debug(f"Params: {params}")
            raise

    def execute_write(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a write transaction.
        
        Args:
            query: Cypher query string
            params: Query parameters (optional)
            
        Returns:
            Query result
        """
        if params is None:
            params = {}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.write_transaction(
                    lambda tx: tx.run(query, params)
                )
                return result
        except Exception as e:
            self._logger.error(f"Write transaction failed: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        queries = {
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relations': "MATCH ()-[r]->() RETURN count(r) as count",
            'total_chunks': "MATCH (c:Chunk) RETURN count(c) as count",
            'total_documents': "MATCH (d:Document) RETURN count(d) as count",
            'entity_types': """
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
                LIMIT 10
            """,
            'relation_types': """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """
        }

        stats = {}
        for key, query in queries.items():
            try:
                result = self.query_graph(query)
                if key in ['total_entities', 'total_relations', 'total_chunks', 'total_documents']:
                    stats[key] = result[0]['count'] if result else 0
                else:
                    stats[key] = result
            except Exception as e:
                stats[key] = 0 if key.startswith('total') else []
                self._logger.debug(f"Error getting stats for {key}: {str(e)}")

        return stats

    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        try:
            self.query_graph("MATCH (n) DETACH DELETE n")
            self._logger.warning(
                "Database cleared - all nodes and relationships deleted")
        except Exception as e:
            self._logger.error(f"Failed to clear database: {str(e)}")
            raise

    def create_constraints(self):
        """Create uniqueness constraints for key node types"""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT table_id IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE"
        ]

        for constraint in constraints:
            try:
                self.query_graph(constraint)
                self._logger.info(
                    f"Constraint created: {constraint.split()[2]}")
            except Exception as e:
                self._logger.debug(f"Constraint might already exist: {str(e)}")

    def create_indexes(self):
        """Create indexes for performance"""
        indexes = [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX chunk_modality IF NOT EXISTS FOR (c:Chunk) ON (c.modality)",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.text, e.type]",
            "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content]"
        ]

        for index in indexes:
            try:
                self.query_graph(index)
                self._logger.info(f"Index created: {index.split()[2]}")
            except Exception as e:
                self._logger.debug(f"Index might already exist: {str(e)}")

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            self._logger.info("Neo4j connection closed")
