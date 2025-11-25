import traceback
from typing import List, Dict, Any, Optional
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_ollama import OllamaLLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.base_extractor import Entity, Relation
from src.utils.base_parser import ParsedContent


class GraphDBManager(Neo4jManager):
    """Manages Neo4j database operations for multimodal knowledge graph with LLM-driven extraction"""

    def __init__(
        self,
        llm: OllamaLLM,
        uri: str,
        username: str,
        password: str,
        database: str,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None
    ):
        """Initialize GraphDBManager with connection parameters and LLM
        
        Args:
            llm: LangChain LLM instance for entity/relationship extraction
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            allowed_nodes: List of allowed node types (empty = let LLM decide)
            allowed_relationships: List of allowed relationship types (empty = let LLM decide)
        """
        super().__init__(uri, username, password, database)
        self.llm = llm
        self.allowed_nodes = allowed_nodes or []
        self.allowed_relationships = allowed_relationships or []
        self.initialize_schema()

    def initialize_schema(self):
        """Initialize graph transformer for LLM-based extraction"""
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships,
            strict_mode=False,  # Allow flexible schema
        )
        self._logger.info("LLM Graph Transformer initialized")

        # Create constraints for unique entities
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT table_id IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE"
        ]

        for constraint in constraints:
            try:
                with self.driver.session(database=self.database) as session:
                    session.run(constraint)
                self._logger.info(
                    f"Constraint created: {constraint.split()[2]}")
            except Exception as e:
                self._logger.debug(f"Constraint might already exist: {str(e)}")

    def ingest_parsed_content_multimodal(self, parsed_content: ParsedContent) -> Dict[str, Any]:
        """
        Ingest multimodal parsed content (text, images, tables) into Neo4j graph.
        Uses LLM to extract entities and relationships from all modalities.
        
        Args:
            parsed_content: ParsedContent object with text, images, and tables
            
        Returns:
            Dict with ingestion statistics
        """
        stats = {
            'documents': 0,
            'text_chunks': 0,
            'images': 0,
            'tables': 0,
            'entities': 0,
            'relationships': 0
        }

        doc_id = self._create_document_node(parsed_content)
        stats['documents'] = 1

        # Process text chunks
        if parsed_content.text and parsed_content.text.get('content'):
            text_stats = self._ingest_text_chunks(
                text_content=parsed_content.text['content'],
                doc_id=doc_id,
                source_file=parsed_content.source_file
            )
            stats['text_chunks'] += text_stats['chunks']
            stats['entities'] += text_stats['entities']
            stats['relationships'] += text_stats['relationships']

        # Process images
        if parsed_content.images:
            image_stats = self._ingest_images(
                images=parsed_content.images,
                doc_id=doc_id,
                source_file=parsed_content.source_file
            )
            stats['images'] += image_stats['images']
            stats['entities'] += image_stats['entities']
            stats['relationships'] += image_stats['relationships']

        # Process tables
        if parsed_content.tables:
            table_stats = self._ingest_tables(
                tables=parsed_content.tables,
                doc_id=doc_id,
                source_file=parsed_content.source_file
            )
            stats['tables'] += table_stats['tables']
            stats['entities'] += table_stats['entities']
            stats['relationships'] += table_stats['relationships']

        self._logger.info(
            f"Multimodal ingestion complete for {parsed_content.source_file}: {stats}")
        return stats

    def _create_document_node(self, parsed_content: ParsedContent) -> str:
        """Create a Document node in Neo4j"""
        doc_id = parsed_content.source_file
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.source_file = $source_file,
            d.properties = $properties
        RETURN d.id as id
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                doc_id=doc_id,
                source_file=parsed_content.source_file,
                properties=str(parsed_content.metadata)
            )
            return result.single()['id']

    def _ingest_text_chunks(
        self,
        text_content: str,
        doc_id: str,
        source_file: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, int]:
        """
        Ingest text chunks and extract entities/relationships using LLM.
        
        Args:
            text_content: Full text content
            doc_id: Document ID to link chunks
            source_file: Source file name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict with chunk, entity, and relationship counts
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text_content)

        stats = {'chunks': 0, 'entities': 0, 'relationships': 0}

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_text_chunk_{idx}"

            # Create Chunk node
            self._create_chunk_node(
                chunk_id=chunk_id,
                content=chunk_text,
                modality='text',
                doc_id=doc_id,
                chunk_index=idx
            )
            stats['chunks'] += 1

            # Extract entities and relationships using LLM
            doc = Document(
                page_content=chunk_text,
                properties={'source': source_file,
                          'chunk_index': idx, 'modality': 'text'}
            )

            try:
                graph_docs = self.graph_transformer.convert_to_graph_documents([doc])

                if graph_docs and len(graph_docs) > 0:
                    # Ingest entities
                    for node in graph_docs[0].nodes:
                        entity = Entity(
                            text=node.id,
                            type=node.type,
                            properties=node.properties if hasattr(
                                node, 'properties') else {}
                        )
                        self._create_entity_node(entity, chunk_id)
                        stats['entities'] += 1

                    # Ingest relationships
                    for rel in graph_docs[0].relationships:
                        relation = Relation(
                            source=rel.source.id,
                            target=rel.target.id,
                            type=rel.type,
                            properties=rel.properties if hasattr(
                                rel, 'properties') else {}
                        )
                        self._create_relationship(relation)
                        stats['relationships'] += 1

            except Exception as e:
                self._logger.warning(
                    f"LLM extraction failed for chunk {idx}: {str(e)}")
                traceback.print_exc()

        return stats

    def _ingest_images(
        self,
        images: List[Dict[str, Any]],
        doc_id: str,
        source_file: str
    ) -> Dict[str, int]:
        """
        Ingest images and extract entities/relationships from captions using LLM.
        
        Args:
            images: List of image dictionaries with caption, path, bbox
            doc_id: Document ID to link images
            source_file: Source file name
            
        Returns:
            Dict with image, entity, and relationship counts
        """
        stats = {'images': 0, 'entities': 0, 'relationships': 0}

        for idx, img_data in enumerate(images):
            image_id = f"{doc_id}_image_{idx}"
            caption = img_data.get('caption', f'Image {idx}')
            path = img_data.get('path', '')

            # Create Image node
            query = """
            MERGE (i:Image {id: $image_id})
            SET i.caption = $caption,
                i.path = $path,
                i.bbox = $bbox,
                i.page = $page,
                i.has_pixels = $has_pixels,
                i.modality = 'image'
            WITH i
            MATCH (d:Document {id: $doc_id})
            MERGE (i)-[:IN_DOCUMENT]->(d)
            RETURN i.id as id
            """

            with self.driver.session(database=self.database) as session:
                session.run(
                    query,
                    image_id=image_id,
                    caption=caption,
                    path=path,
                    bbox=str(img_data.get('bbox', [])),
                    page=img_data.get('page', 1),
                    has_pixels=img_data.get('has_pixels', False),
                    doc_id=doc_id
                )
            stats['images'] += 1

            # Extract entities from caption using LLM
            if caption:
                doc = Document(
                    page_content=caption,
                    properties={'source': source_file,
                              'modality': 'image', 'image_id': image_id}
                )

                try:
                    graph_docs = self.graph_transformer.convert_to_graph_documents([doc])

                    if graph_docs and len(graph_docs) > 0:
                        for node in graph_docs[0].nodes:
                            entity = Entity(
                                text=node.id,
                                type=node.type,
                                properties=node.properties if hasattr(
                                    node, 'properties') else {}
                            )
                            self._create_entity_node(entity, image_id)
                            stats['entities'] += 1

                        for rel in graph_docs[0].relationships:
                            relation = Relation(
                                source=rel.source.id,
                                target=rel.target.id,
                                type=rel.type,
                                properties=rel.properties if hasattr(
                                    rel, 'properties') else {}
                            )
                            self._create_relationship(relation)
                            stats['relationships'] += 1

                except Exception as e:
                    self._logger.warning(
                        f"LLM extraction failed for image {idx}: {str(e)}")

        return stats

    def _ingest_tables(
        self,
        tables: List[Dict[str, Any]],
        doc_id: str,
        source_file: str
    ) -> Dict[str, int]:
        """
        Ingest tables and extract entities/relationships from content using LLM.
        
        Args:
            tables: List of table dictionaries with content, bbox, page
            doc_id: Document ID to link tables
            source_file: Source file name
            
        Returns:
            Dict with table, entity, and relationship counts
        """
        stats = {'tables': 0, 'entities': 0, 'relationships': 0}

        for idx, table_data in enumerate(tables):
            table_id = f"{doc_id}_table_{idx}"
            content = table_data.get('content', '')
            
            # Extract table identifier (e.g., "Table 3", "Table 5") from content
            import re
            table_identifier = None
            # Look for patterns like "Table 3", "Table 5", "Table III", etc.
            table_patterns = [
                r'Table\s+(\d+)',  # "Table 3", "Table 5"
                r'Table\s+([IVX]+)',  # "Table III", "Table V"
                r'TABLE\s+(\d+)',  # "TABLE 3"
            ]
            for pattern in table_patterns:
                match = re.search(pattern, content[:500], re.IGNORECASE)  # Check first 500 chars
                if match:
                    table_identifier = f"Table {match.group(1)}"
                    break
            
            # If no identifier found, use index-based identifier
            if not table_identifier:
                table_identifier = f"Table {idx + 1}"

            # Create Table node
            query = """
            MERGE (t:Table {id: $table_id})
            SET t.content = $content,
                t.bbox = $bbox,
                t.page = $page,
                t.modality = 'table',
                t.table_identifier = $table_identifier
            WITH t
            MATCH (d:Document {id: $doc_id})
            MERGE (t)-[:IN_DOCUMENT]->(d)
            RETURN t.id as id
            """

            with self.driver.session(database=self.database) as session:
                session.run(
                    query,
                    table_id=table_id,
                    content=content,
                    bbox=str(table_data.get('bbox', [])),
                    page=table_data.get('page', 1),
                    table_identifier=table_identifier,
                    doc_id=doc_id
                )
            stats['tables'] += 1

            # Extract entities from table content using LLM
            if content:
                # Use full table content for better extraction (but limit to reasonable size for LLM)
                # Store full content in node, use full content for entity extraction
                table_description = f"Table content: {content[:2000]}"  # Increased from 500 to 2000

                doc = Document(
                    page_content=table_description,
                    properties={'source': source_file,
                              'modality': 'table', 'table_id': table_id}
                )

                try:
                    graph_docs = self.graph_transformer.convert_to_graph_documents([doc])

                    if graph_docs and len(graph_docs) > 0:
                        for node in graph_docs[0].nodes:
                            entity = Entity(
                                text=node.id,
                                type=node.type,
                                properties=node.properties if hasattr(
                                    node, 'properties') else {}
                            )
                            self._create_entity_node(entity, table_id)
                            stats['entities'] += 1

                        for rel in graph_docs[0].relationships:
                            relation = Relation(
                                source=rel.source.id,
                                target=rel.target.id,
                                type=rel.type,
                                properties=rel.properties if hasattr(
                                    rel, 'properties') else {}
                            )
                            self._create_relationship(relation)
                            stats['relationships'] += 1

                except Exception as e:
                    self._logger.warning(
                        f"LLM extraction failed for table {idx}: {str(e)}")

        return stats

    def _create_chunk_node(
        self,
        chunk_id: str,
        content: str,
        modality: str,
        doc_id: str,
        chunk_index: int,
        page: Optional[int] = None
    ):
        """Create a Chunk node and link to Document"""
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.content = $content,
            c.modality = $modality,
            c.chunk_index = $chunk_index,
            c.page = COALESCE($page, 1)
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (c)-[:IN_DOCUMENT]->(d)
        RETURN c.id as id
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                chunk_id=chunk_id,
                content=content,
                modality=modality,
                chunk_index=chunk_index,
                page=page,
                doc_id=doc_id
            )

    def _create_entity_node(self, entity: Entity, source_id: str):
        """Create an Entity node and link to source (Chunk/Image/Table)"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.text = $text,
            e.type = $type,
            e.properties = $properties
        WITH e
        MATCH (s {id: $source_id})
        MERGE (e)-[:EXTRACTED_FROM]->(s)
        RETURN e.id as id
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                entity_id=entity.text.lower().replace(' ', '_'),
                text=entity.text,
                type=entity.type,
                properties=str(entity.properties),
                source_id=source_id
            )

    def _create_relationship(self, relation: Relation):
        """Create a relationship between two entities"""
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation.type}]->(target)
        SET r.properties = $properties
        RETURN r
        """

        with self.driver.session(database=self.database) as session:
            try:
                session.run(
                    query,
                    source_id=relation.source.lower().replace(' ', '_'),
                    target_id=relation.target.lower().replace(' ', '_'),
                    properties=str(relation.properties)
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to create relationship: {str(e)}")

    def ingest_batch_parsed_content(
        self,
        parsed_contents: List[ParsedContent]
    ) -> Dict[str, int]:
        """
        Batch ingest multiple ParsedContent objects.
        
        Args:
            parsed_contents: List of ParsedContent objects
            
        Returns:
            Dict with total ingestion statistics
        """
        total_stats = {
            'documents': 0,
            'text_chunks': 0,
            'images': 0,
            'tables': 0,
            'entities': 0,
            'relationships': 0
        }

        for parsed_content in parsed_contents:
            try:
                stats = self.ingest_parsed_content_multimodal(parsed_content)
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
            except Exception as e:
                self._logger.error(
                    f"Failed to ingest {parsed_content.source_file}: {str(e)}")

        self._logger.info(f"Batch ingestion complete: {total_stats}")
        return total_stats

    def query_graph(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
