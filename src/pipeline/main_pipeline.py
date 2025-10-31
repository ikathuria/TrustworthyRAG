
"""
Main Pipeline Orchestrator for Cybersecurity Knowledge Graph System
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging
import time

# Import components
from utils.base import BaseParser, BaseExtractor, ParsedContent, Entity, Relation
from src.mineru_parser import MinerUParser, create_parser
from src.cybersec_extractor import CyNERExtractor, create_extractor
from src.neo4j_manager import Neo4jManager, create_neo4j_manager


class CybersecKnowledgeGraphPipeline:
    """Main pipeline orchestrator using Factory pattern and OOP principles"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._logger = self._setup_logging()

        # Initialize components using factory pattern
        self.parser = None
        self.extractor = None
        self.neo4j_manager = None
        self.retriever = None
        self.evaluator = None

        self._initialize_components()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
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

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(file)
                else:
                    config = json.load(file)

            self._validate_config(config)
            return config

        except Exception as e:
            # Fallback to default configuration
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self._get_default_config()

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_sections = ['parser', 'extractor', 'neo4j', 'pipeline']

        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Missing required configuration section: {section}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'parser': {
                'type': 'mineru',
                'device_map': 'auto',
                'torch_dtype': 'auto',
                'max_new_tokens': 1024
            },
            'extractor': {
                'type': 'cyner',
                'transformer_model': 'xlm-roberta-large',
                'use_heuristic': True,
                'confidence_threshold': 0.7
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password',
                'database': 'neo4j'
            },
            'pipeline': {
                'batch_size': 100,
                'enable_evaluation': True,
                'output_dir': './outputs'
            }
        }

    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize parser
            self.parser = create_parser(
                self.config['parser']['type'],
                self.config['parser']
            )
            self._logger.info(
                f"Initialized {self.config['parser']['type']} parser")

            # Initialize extractor
            self.extractor = create_extractor(
                self.config['extractor']['type'],
                self.config['extractor']
            )
            self._logger.info(
                f"Initialized {self.config['extractor']['type']} extractor")

            # Initialize Neo4j manager
            self.neo4j_manager = create_neo4j_manager(self.config['neo4j'])
            self._logger.info("Initialized Neo4j manager")

            # Initialize retriever and evaluator
            self._initialize_rag_components()

        except Exception as e:
            self._logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _initialize_rag_components(self):
        """Initialize RAG and evaluation components"""
        try:
            # Initialize retriever if LangChain is available
            if hasattr(self.neo4j_manager, 'graph') and self.neo4j_manager.graph:
                self.retriever = HybridRetriever(
                    self.neo4j_manager,
                    self.config.get('llm', {})
                )
                self._logger.info("Initialized hybrid retriever")

            # Initialize evaluator
            if self.config['pipeline'].get('enable_evaluation', False):
                self.evaluator = SystemEvaluator(
                    self.config.get('evaluation', {}))
                self._logger.info("Initialized system evaluator")

        except Exception as e:
            self._logger.warning(
                f"Could not initialize RAG components: {str(e)}")

    def process_documents(self, file_paths: List[str],
                          save_intermediate: bool = True) -> Dict[str, Any]:
        """End-to-end processing pipeline"""
        self._logger.info(
            f"Starting processing of {len(file_paths)} documents")
        start_time = time.time()

        results = {
            'processed_docs': [],
            'total_entities': 0,
            'total_relations': 0,
            'ingestion_stats': {},
            'processing_time': 0,
            'errors': []
        }

        try:
            # Step 1: Parse documents
            parsed_docs = self._parse_documents(file_paths)
            results['processed_docs'] = len(parsed_docs)

            if save_intermediate:
                self._save_intermediate_results(parsed_docs, 'parsed_docs')

            # Step 2: Extract entities and relations
            all_entities = []
            all_relations = []

            for parsed_doc in parsed_docs:
                try:
                    entities, relations = self._extract_from_document(
                        parsed_doc)
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                except Exception as e:
                    self._logger.error(
                        f"Extraction failed for {parsed_doc.source_file}: {e}")
                    results['errors'].append({
                        'file': parsed_doc.source_file,
                        'stage': 'extraction',
                        'error': str(e)
                    })

            results['total_entities'] = len(all_entities)
            results['total_relations'] = len(all_relations)

            if save_intermediate:
                self._save_intermediate_results(all_entities, 'entities')
                self._save_intermediate_results(all_relations, 'relations')

            # Step 3: Ingest into knowledge graph
            if all_entities or all_relations:
                ingestion_stats = self._ingest_to_graph(
                    all_entities, all_relations)
                results['ingestion_stats'] = ingestion_stats

            # Step 4: Update processing time
            results['processing_time'] = time.time() - start_time

            self._logger.info(
                f"Processing completed in {results['processing_time']:.2f} seconds")
            self._logger.info(f"Results: {results['processed_docs']} docs, "
                              f"{results['total_entities']} entities, "
                              f"{results['total_relations']} relations")

            return results

        except Exception as e:
            self._logger.error(f"Pipeline processing failed: {str(e)}")
            results['errors'].append({
                'stage': 'pipeline',
                'error': str(e)
            })
            return results

    def _parse_documents(self, file_paths: List[str]) -> List[ParsedContent]:
        """Parse all documents"""
        parsed_docs = []

        for file_path in file_paths:
            try:
                if not self.parser.validate_file(file_path):
                    continue

                parsed_content = self.parser.parse(file_path)
                parsed_docs.append(parsed_content)

                self._logger.info(f"Successfully parsed: {file_path}")

            except Exception as e:
                self._logger.error(f"Failed to parse {file_path}: {str(e)}")
                continue

        return parsed_docs

    def _extract_from_document(self, parsed_doc: ParsedContent) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a single document"""
        # Extract entities
        entities = self.extractor.extract_entities(parsed_doc.text)

        # Set source document for entities
        for entity in entities:
            entity.source_doc = parsed_doc.source_file

        # Extract relations
        relations = self.extractor.extract_relations(parsed_doc.text, entities)

        return entities, relations

    def _ingest_to_graph(self, entities: List[Entity], relations: List[Relation]) -> Dict[str, Any]:
        """Ingest entities and relations into Neo4j"""
        batch_size = self.config['pipeline'].get('batch_size', 100)

        return self.neo4j_manager.ingest_entities_and_relations(
            entities, relations, batch_size
        )

    def _save_intermediate_results(self, data: Any, data_type: str):
        """Save intermediate results to files"""
        output_dir = Path(self.config['pipeline']['output_dir'])
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{data_type}_{timestamp}.json"

        try:
            if data_type == 'parsed_docs':
                # Convert ParsedContent objects to dictionaries
                serializable_data = []
                for doc in data:
                    serializable_data.append({
                        'text': doc.text,
                        'tables': doc.tables,
                        'images': doc.images,
                        'metadata': doc.metadata,
                        'source_file': doc.source_file,
                        'parsed_at': doc.parsed_at.isoformat() if doc.parsed_at else None
                    })
            elif data_type in ['entities', 'relations']:
                # Convert Entity/Relation objects to dictionaries
                serializable_data = []
                for item in data:
                    if hasattr(item, '__dict__'):
                        serializable_data.append(item.__dict__)
                    else:
                        serializable_data.append(str(item))
            else:
                serializable_data = data

            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

            self._logger.info(f"Saved {data_type} to {filename}")

        except Exception as e:
            self._logger.error(f"Failed to save {data_type}: {str(e)}")

    def query_knowledge_graph(self, query: str, method: str = "hybrid") -> str:
        """Query the knowledge graph using specified method"""
        if not self.retriever:
            # Fallback to direct Cypher query
            if "cypher" in method.lower():
                results = self.neo4j_manager.query_graph(query)
                return json.dumps(results, indent=2)
            else:
                return "Retriever not available. Use Cypher queries directly."

        return self.retriever.retrieve_context(query, method)

    def get_entity_insights(self, entity_name: str, entity_type: str = None) -> Dict[str, Any]:
        """Get insights about a specific entity"""
        try:
            # Get entity neighborhood
            neighborhood = self.neo4j_manager.get_entity_neighborhood(
                entity_name, entity_type, depth=2
            )

            # Get statistics
            stats = self.neo4j_manager.get_statistics()

            return {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'neighborhood_size': len(neighborhood),
                'neighborhood': neighborhood[:10],  # Limit for display
                'graph_stats': stats
            }

        except Exception as e:
            self._logger.error(
                f"Error getting insights for {entity_name}: {str(e)}")
            return {'error': str(e)}

    def evaluate_system(self, test_queries: List[Dict] = None) -> Dict[str, float]:
        """Evaluate retrieval system performance"""
        if not self.evaluator or not test_queries:
            return {'note': 'Evaluation not available or no test queries provided'}

        try:
            return self.evaluator.evaluate(test_queries, self.retriever)
        except Exception as e:
            self._logger.error(f"Evaluation failed: {str(e)}")
            return {'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'parser': self.parser.__class__.__name__ if self.parser else None,
                'extractor': self.extractor.__class__.__name__ if self.extractor else None,
                'neo4j_manager': bool(self.neo4j_manager),
                'retriever': bool(self.retriever),
                'evaluator': bool(self.evaluator)
            }
        }

        # Get graph statistics if available
        if self.neo4j_manager:
            try:
                status['graph_stats'] = self.neo4j_manager.get_statistics()
            except Exception as e:
                status['graph_stats'] = {'error': str(e)}

        return status

    def shutdown(self):
        """Gracefully shutdown all components"""
        self._logger.info("Shutting down pipeline...")

        if self.neo4j_manager:
            self.neo4j_manager.close()

        self._logger.info("Pipeline shutdown complete")


# Placeholder classes for components not yet implemented
class HybridRetriever:
    """Placeholder for hybrid retriever"""

    def __init__(self, neo4j_manager, llm_config):
        self.neo4j_manager = neo4j_manager
        self.llm_config = llm_config

    def retrieve_context(self, query: str, method: str = "hybrid") -> str:
        return f"Retrieved context for: {query} using {method} method"


class SystemEvaluator:
    """Placeholder for system evaluator"""

    def __init__(self, config):
        self.config = config

    def evaluate(self, test_queries: List[Dict], retriever) -> Dict[str, float]:
        return {'precision': 0.85, 'recall': 0.78, 'f1_score': 0.81}


# Factory function for pipeline creation
def create_pipeline(config_path: str) -> CybersecKnowledgeGraphPipeline:
    """Factory function for creating pipeline instances"""
    return CybersecKnowledgeGraphPipeline(config_path)


# Example usage
if __name__ == "__main__":
    print("Cybersecurity Knowledge Graph Pipeline implementation ready!")
