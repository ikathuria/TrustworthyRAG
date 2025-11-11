from typing import List, Dict, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging

from src.utils.base_extractor import Entity, Relation
import src.utils.constants as C


class Neo4jManager:
    """Manages Neo4j database operations for cybersecurity knowledge graph"""

    def __init__(
            self, uri: str = C.NEO4J_URI,
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

            # test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")

            self._logger.info(f"Connected to Neo4j at {self.uri}")

        except (ServiceUnavailable, AuthError) as e:
            self._logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        queries = {
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relations': "MATCH ()-[r]->() RETURN count(r) as count",
            'entity_types': """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """,
            'relation_types': """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """
        }

        stats = {}
        for key, query in queries.items():
            try:
                result = self.query_graph(query)
                if key in ['total_entities', 'total_relations']:
                    stats[key] = result[0]['count'] if result else 0
                else:
                    stats[key] = result
            except Exception as e:
                stats[key] = 0 if key.startswith('total') else []
                self._logger.debug(f"Error getting stats for {key}: {str(e)}")

        return stats

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            self._logger.info("Neo4j connection closed")
