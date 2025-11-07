from typing import List, Dict, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging

from src.utils.base_extractor import Entity, Relation


class Neo4jManager:
    """Manages Neo4j database operations for cybersecurity knowledge graph"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        self._logger = self._setup_logging()
        self.driver = None

        self._connect()
        self._initialize_schema()
        self._initialize_vector_index()

    def _setup_logging(self) -> logging.Logger:
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

    def _initialize_schema(self):
        """Initialize cybersecurity-specific graph schema"""
        schema_queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (c:CVE) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)",
            "CREATE INDEX malware_name IF NOT EXISTS FOR (m:MALWARE) ON (m.name)",
            "CREATE INDEX org_name IF NOT EXISTS FOR (o:ORGANIZATION) ON (o.name)",
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.text, e.name]"
        ]

        with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as e:
                    self._logger.debug(f"Schema note: {e}")

        self._logger.info("Neo4j schema initialized")

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

    def ingest_entities_with_embedding(self, entities: List[Entity], embedding_dict: Dict[str, List[float]]):
        session = self.driver.session(database=self.database)
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        SET e.text = entity.text,
            e.type = entity.type,
            e.embedding = entity.embedding
        """

        data = []
        for entity in entities:
            emb = embedding_dict.get(entity.text.lower())
            if emb:
                data.append({
                    "id": f"{entity.label}_{entity.text.lower()}",
                    "text": entity.text,
                    "type": entity.label,
                    "embedding": emb,
                })

        session.run(query, entities=data)
        session.close()

    def ingest_entities(self, entities: List[Entity], batch_size: int = 100) -> Dict[str, int]:
        """Ingest entities into Neo4j with batching"""
        stats = {'created': 0, 'updated': 0, 'errors': 0}

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_stats = self._create_entities_batch(batch)
            stats['created'] += batch_stats['created']
            stats['errors'] += batch_stats['errors']

        self._logger.info(f"Ingested {stats['created']} entities")
        return stats

    def _create_entities_batch(self, entities: List[Entity]) -> Dict[str, int]:
        """Create a batch of entities"""
        stats = {'created': 0, 'errors': 0}

        query = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        SET e.text = entity.text,
            e.type = entity.type,
            e.confidence = entity.confidence,
            e.start_pos = entity.start_pos,
            e.end_pos = entity.end_pos,
            e.source_doc = entity.source_doc,
            e.created_at = datetime(),
            e.updated_at = datetime()

        WITH e, entity
        CALL apoc.create.addLabels(e, [entity.type]) YIELD node
        RETURN count(node) as created
        """

        query_no_apoc = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        SET e.text = entity.text,
            e.type = entity.type,
            e.confidence = entity.confidence,
            e.start_pos = entity.start_pos,
            e.end_pos = entity.end_pos,
            e.source_doc = entity.source_doc,
            e.created_at = datetime(),
            e.updated_at = datetime()

        FOREACH (ignore IN CASE WHEN entity.type = 'MALWARE' THEN [1] ELSE [] END |
            SET e:MALWARE)
        FOREACH (ignore IN CASE WHEN entity.type = 'ORGANIZATION' THEN [1] ELSE [] END |
            SET e:ORGANIZATION)
        FOREACH (ignore IN CASE WHEN entity.type = 'PERSON' THEN [1] ELSE [] END |
            SET e:PERSON)
        FOREACH (ignore IN CASE WHEN entity.type = 'CVE' THEN [1] ELSE [] END |
            SET e:CVE)
        FOREACH (ignore IN CASE WHEN entity.type = 'VULNERABILITY' THEN [1] ELSE [] END |
            SET e:VULNERABILITY)
        FOREACH (ignore IN CASE WHEN entity.type = 'SYSTEM' THEN [1] ELSE [] END |
            SET e:SYSTEM)
        FOREACH (ignore IN CASE WHEN entity.type = 'IP' THEN [1] ELSE [] END |
            SET e:IP)
        FOREACH (ignore IN CASE WHEN entity.type = 'URL' THEN [1] ELSE [] END |
            SET e:URL)
        FOREACH (ignore IN CASE WHEN entity.type = 'DOMAIN' THEN [1] ELSE [] END |
            SET e:DOMAIN)

        RETURN count(e) as created
        """

        entity_data = []
        for entity in entities:
            normalized_text = entity.text.lower().strip()
            entity_id = f"{entity.label}_{normalized_text}"

            entity_data.append({
                'id': entity_id,
                'text': entity.text,
                'type': entity.label,
                'confidence': entity.confidence,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'source_doc': entity.source_doc[:100] if entity.source_doc else ""
            })

        try:
            with self.driver.session(database=self.database) as session:
                try:
                    result = session.run(query, entities=entity_data)
                    stats['created'] = result.single()['created']
                except:
                    result = session.run(query_no_apoc, entities=entity_data)
                    stats['created'] = result.single()['created']
        except Exception as e:
            self._logger.error(f"Error creating entities: {str(e)}")
            stats['errors'] = len(entity_data)

        return stats

    def ingest_relations(self, relations: List[Relation], batch_size: int = 100) -> Dict[str, int]:
        """Ingest relationships into Neo4j"""
        stats = {'created': 0, 'errors': 0}

        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            batch_stats = self._create_relations_batch(batch)
            stats['created'] += batch_stats['created']
            stats['errors'] += batch_stats['errors']

        self._logger.info(f"Ingested {stats['created']} relations")
        return stats

    def _create_relations_batch(self, relations: List[Relation]) -> Dict[str, int]:
        """Create a batch of relationships"""
        stats = {'created': 0, 'errors': 0}

        relations_by_type = {}
        for rel in relations:
            rel_type = rel.relation_type
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            relations_by_type[rel_type].append(rel)

        for rel_type, rel_batch in relations_by_type.items():
            try:
                created = self._create_single_type_relations(
                    rel_type, rel_batch)
                stats['created'] += created
            except Exception as e:
                self._logger.error(
                    f"Error creating {rel_type} relations: {str(e)}")
                stats['errors'] += len(rel_batch)

        return stats

    def _create_single_type_relations(self, relation_type: str, relations: List[Relation]) -> int:
        """Create relationships of a single type"""

        safe_rel_type = relation_type.replace(
            ' ', '_').replace('-', '_').upper()

        query = f"""
        UNWIND $relations as rel
        MATCH (head:Entity {{id: rel.head_id}})
        MATCH (tail:Entity {{id: rel.tail_id}})
        MERGE (head)-[r:{safe_rel_type}]->(tail)
        SET r.confidence = rel.confidence,
            r.context = rel.context,
            r.created_at = datetime(),
            r.updated_at = datetime()
        RETURN count(r) as created
        """

        relation_data = []
        for rel in relations:
            head_id = f"{rel.head_entity.label}_{rel.head_entity.text.lower().strip()}"
            tail_id = f"{rel.tail_entity.label}_{rel.tail_entity.text.lower().strip()}"

            relation_data.append({
                'head_id': head_id,
                'tail_id': tail_id,
                'confidence': rel.confidence,
                'context': rel.context[:200] if rel.context else ""
            })

        with self.driver.session(database=self.database) as session:
            result = session.run(query, relations=relation_data)
            return result.single()['created']

    def query_graph(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self._logger.error(f"Query failed: {str(e)}")
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
            except:
                stats[key] = 0 if key.startswith('total') else []

        return stats

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            self._logger.info("Neo4j connection closed")
