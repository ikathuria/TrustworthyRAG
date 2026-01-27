from typing import List, Dict, Any, Optional
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import logging
import numpy as np
from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.base_parser import ParsedContent


class VectorDBManager(Neo4jManager):
    """Manages Neo4j vector database operations for multimodal embeddings"""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        text_embedding_model: str = "all-MiniLM-L6-v2",
        image_embedding_model: str = None,  # For OpenCLIP or similar
    ):
        """Initialize VectorDBManager with connection parameters and embedding models

        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            text_embedding_model: HuggingFace model name for text embeddings
            image_embedding_model: Model for image embeddings (optional, for future use)
        """
        super().__init__(uri, username, password, database)

        # Initialize embedding models
        self.text_embedder = HuggingFaceEmbeddings(model_name=text_embedding_model)
        self.text_model = SentenceTransformer(text_embedding_model)

        # Initialize vector stores for different modalities
        self._initialize_vector_stores()
        self._create_vector_indexes()

    def _initialize_vector_stores(self):
        """Initialize Neo4jVector stores for different modalities"""
        try:
            # Text chunks vector store
            self.text_vector_store = Neo4jVector(
                embedding=self.text_embedder,
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="text_chunk_vector_index",
                node_label="Chunk",
                text_node_property="content",
                embedding_node_property="embedding",
            )

            # Image vector store (captions)
            self.image_vector_store = Neo4jVector(
                embedding=self.text_embedder,  # Use text embedder for captions
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="image_vector_index",
                node_label="Image",
                text_node_property="caption",
                embedding_node_property="embedding",
            )

            # Table vector store (content summaries)
            self.table_vector_store = Neo4jVector(
                embedding=self.text_embedder,
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="table_vector_index",
                node_label="Table",
                text_node_property="content",
                embedding_node_property="embedding",
            )

            # Entity vector store
            self.entity_vector_store = Neo4jVector(
                embedding=self.text_embedder,
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="entity_vector_index",
                node_label="Entity",
                text_node_property="text",
                embedding_node_property="embedding",
            )

            self._logger.info("Vector stores initialized for all modalities")

        except Exception as e:
            self._logger.error(f"Error initializing vector stores: {str(e)}")
            raise

    def _create_vector_indexes(self):
        """Create vector similarity indexes in Neo4j"""
        indexes = [
            {
                "name": "text_chunk_vector_index",
                "label": "Chunk",
                "property": "embedding",
                "dimensions": 384,  # all-MiniLM-L6-v2 dimension
                "similarity": "cosine",
            },
            {
                "name": "image_vector_index",
                "label": "Image",
                "property": "embedding",
                "dimensions": 384,
                "similarity": "cosine",
            },
            {
                "name": "table_vector_index",
                "label": "Table",
                "property": "embedding",
                "dimensions": 384,
                "similarity": "cosine",
            },
            {
                "name": "entity_vector_index",
                "label": "Entity",
                "property": "embedding",
                "dimensions": 384,
                "similarity": "cosine",
            },
        ]

        for idx_config in indexes:
            try:
                # Drop existing index if exists
                drop_query = f"DROP INDEX {idx_config['name']} IF EXISTS"
                with self.driver.session(database=self.database) as session:
                    session.run(drop_query)

                # Create new vector index
                create_query = f"""
                CREATE VECTOR INDEX {idx_config['name']} IF NOT EXISTS
                FOR (n:{idx_config['label']})
                ON n.{idx_config['property']}
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {idx_config['dimensions']},
                        `vector.similarity_function`: '{idx_config['similarity']}'
                    }}
                }}
                """

                with self.driver.session(database=self.database) as session:
                    session.run(create_query)
                    self._logger.info(f"Created vector index: {idx_config['name']}")

            except Exception as e:
                self._logger.warning(
                    f"Index {idx_config['name']} may already exist or error: {str(e)}"
                )

    def embed_and_store_parsed_content(
        self, parsed_content: ParsedContent
    ) -> Dict[str, int]:
        """
        Generate embeddings for multimodal content and store in Neo4j.
        This should be called AFTER graph ingestion to add embeddings to existing nodes.

        Args:
            parsed_content: ParsedContent object with text, images, and tables

        Returns:
            Dict with embedding counts per modality
        """
        stats = {
            "text_embeddings": 0,
            "image_embeddings": 0,
            "table_embeddings": 0,
            "entity_embeddings": 0,
        }

        # Normalize doc_id for consistency
        doc_id = parsed_content.source_file.replace("\\", "/")

        # Embed text chunks
        if parsed_content.text and parsed_content.text.get("content"):
            stats["text_embeddings"] = self._embed_text_chunks(
                doc_id=doc_id, text_content=parsed_content.text["content"]
            )

        # Embed image captions
        if parsed_content.images:
            stats["image_embeddings"] = self._embed_images(
                doc_id=doc_id, images=parsed_content.images
            )

        # Embed table content
        if parsed_content.tables:
            stats["table_embeddings"] = self._embed_tables(
                doc_id=doc_id, tables=parsed_content.tables
            )

        # Embed entities extracted from this document
        stats["entity_embeddings"] = self._embed_entities(doc_id=doc_id)

        self._logger.info(f"Embedding complete for {doc_id}: {stats}")
        return stats

    def _embed_text_chunks(
        self, doc_id: str, text_content: str, chunk_size: int = 1000
    ) -> int:
        """Generate and store embeddings for text chunks"""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200
        )
        chunks = splitter.split_text(text_content)

        count = 0
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_text_chunk_{idx}"

            # Generate embedding
            embedding = self.text_model.encode(chunk_text).tolist()

            # Update existing Chunk node with embedding
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            SET c.embedding = $embedding
            RETURN c.id as id
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, chunk_id=chunk_id, embedding=embedding)
                if result.single():
                    count += 1

        return count

    def _embed_images(self, doc_id: str, images: List[Dict[str, Any]]) -> int:
        """Generate and store embeddings for image captions"""
        count = 0
        for idx, img_data in enumerate(images):
            image_id = f"{doc_id}_image_{idx}"
            caption = img_data.get("caption", f"Image {idx}")

            # Generate embedding from caption
            embedding = self.text_model.encode(caption).tolist()

            # Update existing Image node with embedding
            query = """
            MATCH (i:Image {id: $image_id})
            SET i.embedding = $embedding
            RETURN i.id as id
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, image_id=image_id, embedding=embedding)
                if result.single():
                    count += 1

        return count

    def _embed_tables(self, doc_id: str, tables: List[Dict[str, Any]]) -> int:
        """Generate and store embeddings for table content"""
        count = 0
        for idx, table_data in enumerate(tables):
            table_id = f"{doc_id}_table_{idx}"
            content = table_data.get("content", "")

            # Generate embedding from table content (truncate if too long)
            content_summary = content[:500]
            embedding = self.text_model.encode(content_summary).tolist()

            # Update existing Table node with embedding
            query = """
            MATCH (t:Table {id: $table_id})
            SET t.embedding = $embedding
            RETURN t.id as id
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, table_id=table_id, embedding=embedding)
                if result.single():
                    count += 1

        return count

    def _embed_entities(self, doc_id: str) -> int:
        """Generate and store embeddings for entities extracted from document"""
        # Find all entities linked to this document
        query = """
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(s)-[:IN_DOCUMENT]->(d:Document {id: $doc_id})
        WHERE e.embedding IS NULL
        RETURN e.id as id, e.text as text, e.type as type
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, doc_id=doc_id)
            entities = [record.data() for record in result]

        count = 0
        for entity in entities:
            # Generate embedding from entity text and type
            text_to_embed = f"{entity['text']} ({entity['type']})"
            embedding = self.text_model.encode(text_to_embed).tolist()

            # Update entity with embedding
            update_query = """
            MATCH (e:Entity {id: $entity_id})
            SET e.embedding = $embedding
            RETURN e.id as id
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(
                    update_query, entity_id=entity["id"], embedding=embedding
                )
                if result.single():
                    count += 1

        return count

    def similarity_search_multimodal(
        self, query: str, modality: str = "all", k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search across modalities.

        Args:
            query: Search query text
            modality: One of ["text", "image", "table", "entity", "all"]
            k: Number of results to return

        Returns:
            List of results with content and metadata
        """
        results = []

        if modality in ["text", "all"]:
            text_results = self.text_vector_store.similarity_search(query, k=k)
            results.extend(
                [
                    {
                        "modality": "text",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in text_results
                ]
            )

        if modality in ["image", "all"]:
            image_results = self.image_vector_store.similarity_search(query, k=k)
            results.extend(
                [
                    {
                        "modality": "image",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in image_results
                ]
            )

        if modality in ["table", "all"]:
            table_results = self.table_vector_store.similarity_search(query, k=k)
            results.extend(
                [
                    {
                        "modality": "table",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in table_results
                ]
            )

        if modality in ["entity", "all"]:
            entity_results = self.entity_vector_store.similarity_search(query, k=k)
            results.extend(
                [
                    {
                        "modality": "entity",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in entity_results
                ]
            )

        return results

    def get_existing_graph_vectorstore(
        self,
        node_label: str = "Chunk",
        text_properties: List[str] = None,
        embedding_property: str = "embedding",
    ) -> Neo4jVector:
        """
        Create a vector store from existing graph nodes.
        Useful for querying nodes that already have embeddings.

        Args:
            node_label: Label of nodes to query
            text_properties: Properties to use as text
            embedding_property: Property containing embeddings

        Returns:
            Neo4jVector instance
        """
        text_properties = text_properties or ["content"]

        return Neo4jVector.from_existing_graph(
            embedding=self.text_embedder,
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property=embedding_property,
        )

    def batch_embed_parsed_contents(
        self, parsed_contents: List[ParsedContent]
    ) -> Dict[str, int]:
        """
        Batch embed multiple ParsedContent objects.

        Args:
            parsed_contents: List of ParsedContent objects

        Returns:
            Dict with total embedding counts
        """
        total_stats = {
            "text_embeddings": 0,
            "image_embeddings": 0,
            "table_embeddings": 0,
            "entity_embeddings": 0,
        }

        for parsed_content in parsed_contents:
            try:
                stats = self.embed_and_store_parsed_content(parsed_content)
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
            except Exception as e:
                self._logger.error(
                    f"Failed to embed {parsed_content.source_file}: {str(e)}"
                )

        self._logger.info(f"Batch embedding complete: {total_stats}")
        return total_stats
