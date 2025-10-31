from typing import List, Dict, Any, Optional
import logging
from src.knowledge_graph.neo4j_manager import Neo4jManager

# LangChain imports
try:
    from langchain_neo4j import Neo4jGraph
    from langchain_community.vectorstores import Neo4jVector
    from langchain_ollama.llms import OllamaLLM
    from langchain_ollama import OllamaEmbeddings
    from langchain.chains import GraphCypherQAChain
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain langchain-community")


class HybridRetriever:
    """Hybrid retrieval combining graph traversal and vector search"""

    def __init__(self, neo4j_manager: Neo4jManager, config: Dict[str, Any]):
        self.neo4j_manager = neo4j_manager
        self.config = config
        self._logger = self._setup_logging()

        # Initialize components
        self.llm = None
        self.embeddings = None
        self.graph = None
        self.vector_store = None
        self.cypher_chain = None

        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain()
        else:
            self._logger.warning(
                "LangChain not available. Using basic retrieval only.")

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

    def _initialize_langchain(self):
        """Initialize LangChain components with Ollama"""
        try:
            # Initialize Ollama LLM
            model_name = self.config.get('model_name', 'llama3.1:8b')
            base_url = self.config.get(
                'ollama_base_url', 'http://localhost:11434')

            self.llm = OllamaLLM(
                model=model_name,
                base_url=base_url,
                temperature=0
            )

            # Initialize Ollama embeddings
            embedding_model = self.config.get(
                'embedding_model', 'nomic-embed-text')
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=base_url
            )

            # Initialize Neo4j graph connection
            self.graph = Neo4jGraph(
                url=self.neo4j_manager.uri,
                username=self.neo4j_manager.username,
                password=self.neo4j_manager.password,
                database=self.neo4j_manager.database,
                enhanced_schema=False
            )

            # Initialize Cypher QA chain
            self._initialize_cypher_chain()

            # Initialize vector store (optional)
            self._initialize_vector_store()

            self._logger.info(
                f"LangChain components initialized with Ollama model: {model_name}")

        except Exception as e:
            self._logger.error(
                f"Failed to initialize LangChain with Ollama: {str(e)}")
            self._logger.info("Make sure Ollama is running: ollama serve")

    def _initialize_cypher_chain(self):
        """Initialize Graph Cypher QA Chain with cybersecurity-focused prompt"""

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are a cybersecurity analyst with access to a Neo4j knowledge graph containing threat intelligence.
The graph contains entities like MALWARE, ORGANIZATION, PERSON, VULNERABILITY, CVE, IP, URL, etc.

Schema:
{schema}

Question: {question}

Generate a Cypher query to answer the question. Guidelines:
1. Use MATCH to find relevant nodes
2. Use OPTIONAL MATCH for relationships that might not exist
3. Filter by entity labels (MALWARE, ORGANIZATION, CVE, etc.)
4. Return meaningful properties like text, type, confidence
5. Limit to top 10 results
6. Use ORDER BY for most relevant results first

Only return the Cypher query, nothing else.

Cypher Query:
"""
        )

        try:
            self.cypher_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                return_intermediate_steps=True,
                cypher_prompt=cypher_prompt,
                allow_dangerous_requests=True  # Required for LangChain security
            )
            self._logger.info("Cypher QA chain initialized with Ollama")
        except Exception as e:
            self._logger.warning(f"Could not initialize Cypher chain: {e}")

    def _initialize_vector_store(self):
        """Initialize vector store for semantic search using Ollama embeddings"""
        try:
            # First, pull the embedding model if not available
            import subprocess
            embedding_model = self.config.get(
                'embedding_model', 'nomic-embed-text')

            try:
                subprocess.run(['ollama', 'pull', embedding_model],
                               capture_output=True, timeout=60)
            except:
                self._logger.warning(
                    f"Could not pull {embedding_model}, assuming it exists")

            # Create vector index on Entity nodes
            self.vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=self.neo4j_manager.uri,
                username=self.neo4j_manager.username,
                password=self.neo4j_manager.password,
                index_name="entity_embeddings",
                node_label="Entity",
                text_node_properties=["text", "type"],
                embedding_node_property="embedding"
            )
            self._logger.info(
                "Vector store initialized with Ollama embeddings")
        except Exception as e:
            self._logger.warning(f"Could not initialize vector store: {e}")

    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve context for a query
        
        Args:
            query: User question
            method: "graph", "vector", or "hybrid"
            top_k: Number of results to return
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        if method == "graph":
            return self._graph_retrieval(query, top_k)
        elif method == "vector":
            return self._vector_retrieval(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieval(query, top_k)

    def _graph_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve using graph traversal"""

        if self.cypher_chain:
            try:
                # Use LangChain Cypher chain
                result = self.cypher_chain.invoke({"query": query})

                return {
                    'method': 'graph',
                    'answer': result.get('result', ''),
                    'cypher_query': result.get('intermediate_steps', [{}])[0].get('query', ''),
                    'context': result.get('intermediate_steps', [{}])[0].get('context', []),
                    'success': True
                }
            except Exception as e:
                self._logger.error(f"Cypher chain failed: {e}")

        # Fallback: Use predefined patterns
        return self._pattern_based_retrieval(query, top_k)

    def _pattern_based_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Fallback retrieval using keyword patterns"""

        query_lower = query.lower()

        # CREATOR/AUTHOR QUESTIONS - FIXED QUERY
        if any(word in query_lower for word in ['who created', 'who authored', 'who developed', 'creator of', 'author of']):
            entity_query = query_lower
            for word in ['who', 'created', 'authored', 'developed', 'is', 'the', 'what', '?']:
                entity_query = entity_query.replace(word, '')
            entity_name = entity_query.strip()

            cypher_query = """
            MATCH (entity:Entity)
            WHERE toLower(entity.text) CONTAINS $entity_name
            
            WITH entity
            ORDER BY entity.confidence DESC
            LIMIT $top_k
            
            OPTIONAL MATCH (creator)-[r:CREATED_BY]->(entity)
            WHERE creator.type IN ['PERSON', 'ORGANIZATION']
            
            RETURN entity.text as entity_name,
                entity.type as entity_type,
                entity.confidence as confidence,
                collect(DISTINCT creator.text) as creators,
                collect(DISTINCT creator.type) as creator_types
            """

            try:
                results = self.neo4j_manager.query_graph(
                    cypher_query,
                    {'top_k': top_k, 'entity_name': entity_name}
                )

                return {
                    'method': 'graph_creator_query',
                    'results': results,
                    'cypher_query': cypher_query,
                    'count': len(results),
                    'success': True
                }
            except Exception as e:
                self._logger.error(f"Creator query failed: {e}")

        # WHAT IS X QUESTIONS - FIXED QUERY
        elif any(word in query_lower for word in ['what is', 'describe', 'tell me about']):
            entity_query = query_lower
            for word in ['what', 'is', 'describe', 'tell', 'me', 'about', '?']:
                entity_query = entity_query.replace(word, '')
            entity_name = entity_query.strip()

            cypher_query = """
            MATCH (entity:Entity)
            WHERE toLower(entity.text) CONTAINS $entity_name
            
            WITH entity
            ORDER BY entity.confidence DESC
            LIMIT $top_k
            
            OPTIONAL MATCH (entity)-[r]-(related)
            WHERE related IS NOT NULL
            
            RETURN entity.text as entity_name,
                entity.type as entity_type,
                entity.confidence as confidence,
                collect(DISTINCT {
                    rel_type: type(r),
                    related_entity: related.text,
                    related_type: related.type
                }) as relationships
            """

            try:
                results = self.neo4j_manager.query_graph(
                    cypher_query,
                    {'top_k': top_k, 'entity_name': entity_name}
                )

                return {
                    'method': 'graph_entity_query',
                    'results': results,
                    'cypher_query': cypher_query,
                    'count': len(results),
                    'success': True
                }
            except Exception as e:
                self._logger.error(f"Entity query failed: {e}")

        # MALWARE QUESTIONS - IMPROVED
        elif any(word in query_lower for word in ['malware', 'trojan', 'backdoor', 'rootkit']):
            cypher_query = """
            MATCH (m:MALWARE)
            RETURN DISTINCT m.text as malware, m.confidence as confidence
            ORDER BY m.confidence DESC, m.text
            LIMIT $top_k
            """
        elif any(word in query_lower for word in ['organization', 'group', 'team']):
            cypher_query = """
            MATCH (o:ORGANIZATION)
            OPTIONAL MATCH (o)-[r]-(related)
            RETURN o.text as organization, type(r) as relationship,
                   related.text as related_entity, related.type as entity_type
            LIMIT $top_k
            """
        elif any(word in query_lower for word in ['cve', 'vulnerability', 'exploit']):
            cypher_query = """
            MATCH (v)
            WHERE v:CVE OR v:VULNERABILITY
            OPTIONAL MATCH (v)-[r]-(related)
            RETURN v.text as vulnerability, type(r) as relationship,
                   related.text as related_entity
            LIMIT $top_k
            """
        else:
            # Generic full-text search
            cypher_query = """
            CALL db.index.fulltext.queryNodes('entity_search', $query)
            YIELD node, score
            OPTIONAL MATCH (node)-[r]-(related)
            RETURN node.text as entity, node.type as type, score,
                   type(r) as relationship, related.text as related_entity
            ORDER BY score DESC
            LIMIT $top_k
            """

        try:
            results = self.neo4j_manager.query_graph(
                cypher_query,
                {'top_k': top_k, 'query': query}
            )

            return {
                'method': 'graph_pattern',
                'results': results,
                'cypher_query': cypher_query,
                'count': len(results),
                'success': True
            }
        except Exception as e:
            self._logger.error(f"Pattern retrieval failed: {e}")
            return {'method': 'graph_pattern', 'success': False, 'error': str(e)}

    def _vector_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve using vector similarity search"""

        if not self.vector_store:
            return {
                'method': 'vector',
                'success': False,
                'error': 'Vector store not initialized'
            }

        try:
            docs = self.vector_store.similarity_search_with_score(
                query, k=top_k)

            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                })

            return {
                'method': 'vector',
                'results': results,
                'count': len(results),
                'success': True
            }

        except Exception as e:
            self._logger.error(f"Vector retrieval failed: {e}")
            return {'method': 'vector', 'success': False, 'error': str(e)}

    def _hybrid_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Combine graph and vector retrieval"""

        graph_results = self._graph_retrieval(query, top_k)
        vector_results = self._vector_retrieval(query, top_k)

        return {
            'method': 'hybrid',
            'graph': graph_results,
            'vector': vector_results,
            'success': graph_results.get('success', False) or vector_results.get('success', False)
        }

    def answer_question(self, question: str, method: str = "hybrid") -> str:
        """Answer question using Ollama with better formatting"""

        context = self.retrieve(question, method=method)

        if not context.get('success', False):
            return "No relevant information found."

        # Special handling for creator questions
        if context.get('method') == 'graph_creator_query' and context.get('results'):
            result = context['results'][0]
            if result.get('creators'):
                # Filter None values
                creators = [c for c in result['creators'] if c]
                if creators:
                    entity_name = result.get('entity_name', 'the entity')
                    return f"{entity_name} was created by {', '.join(creators)}."

        # Special handling for "what is" questions
        if context.get('method') == 'graph_entity_query' and context.get('results'):
            result = context['results'][0]
            entity_name = result.get('entity_name', '')
            entity_type = result.get('entity_type', '')
            relationships = result.get('relationships', [])

            answer_parts = [f"{entity_name} is a {entity_type}."]

            # Add relationship info
            creators = [r for r in relationships if r.get(
                'rel_type') == 'CREATED_BY' and r.get('related_entity')]
            if creators:
                answer_parts.append(
                    f"It was created by {', '.join([c['related_entity'] for c in creators])}.")

            exploits = [r for r in relationships if r.get(
                'rel_type') == 'EXPLOITS']
            if exploits:
                targets = [e['related_entity']
                        for e in exploits if e.get('related_entity')]
                if targets:
                    answer_parts.append(f"It exploits {', '.join(targets[:3])}.")

            return " ".join(answer_parts)

        # Otherwise use LLM
        formatted_context = self._format_context(context)

        if not self.llm:
            return formatted_context

        try:
            prompt = f"""You are a cybersecurity analyst. Answer based on the knowledge graph data.

    Context:
    {formatted_context}

    Question: {question}

    Provide a clear, factual answer. If multiple creators/authors are mentioned, list them.

    Answer:"""

            response = self.llm.invoke(prompt)
            return response

        except Exception as e:
            self._logger.error(f"Answer generation failed: {e}")
            return f"Context retrieved:\n{formatted_context}"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format retrieved context for LLM consumption"""

        formatted = []

        if context['method'] == 'hybrid':
            # Format graph results
            if context['graph'].get('success'):
                formatted.append("Graph Information:")
                if 'results' in context['graph']:
                    for result in context['graph']['results']:
                        formatted.append(f"  - {result}")

            # Format vector results
            if context['vector'].get('success'):
                formatted.append("\nSemantic Search Results:")
                for result in context['vector'].get('results', []):
                    formatted.append(
                        f"  - {result['content']} (score: {result['similarity_score']:.3f})")

        elif 'results' in context:
            for result in context['results']:
                formatted.append(f"  - {result}")

        return "\n".join(formatted)
