import os
from src.graph_ingestion import Neo4jManager
from src.utils.schema_loader import DomainSchema
from src.pipeline.hybrid_extractor import HybridEntityExtractor
from src.multimodal_grounding import MultimodalGrounder
from src.preprocessing.document_parser import MinerUParser
import src.utils.constants as const


def run_pipeline(pdf_path: str, neo4j_config: dict, schema_path: str):
    # Step 1: Parse PDF document (using MinerU parser)
    parser = MinerUParser()
    parsed_doc = parser.parse(pdf_path)

    # Step 2: Load domain schema
    domain_schema = DomainSchema(schema_file=schema_path)

    # Step 3: Initialize Neo4jManager
    neo4j_manager = Neo4jManager(
        uri=neo4j_config['uri'],
        username=neo4j_config['username'],
        password=neo4j_config['password'],
        database=neo4j_config.get('database', 'neo4j')
    )

    # Step 4: Extract entities and relations via Hybrid Extractor (pattern + LLM)
    extractor = HybridEntityExtractor(
        config={"spacy_model": "en_core_web_lg",
                "use_llm_entities": True, "use_llm_relations": True},
        domain_schema=domain_schema,
        llm=None
    )
    entities = extractor.extract_entities(parsed_doc.text)
    relations = extractor.extract_relations(parsed_doc.text, entities)

    # Step 5: Ingest entities and relations into Neo4j
    neo4j_manager.ingest_entities(entities)
    neo4j_manager.ingest_relations(relations)

    # Step 6: Multimodal Grounding - embed images and link to entities
    multimodal_grounder = MultimodalGrounder(
        neo4j_manager=neo4j_manager)

    # Gather image files and captions extracted in parsing step
    image_paths = [img['path'] for img in parsed_doc.images]
    caption_map = {img['path']: img.get('caption', '')
                   for img in parsed_doc.images}

    multimodal_grounder.process(entities, image_paths, caption_map)

    # Done
    print("Pipeline completed successfully.")

    # Optional: Get stats
    stats = neo4j_manager.get_statistics()
    print(f"Neo4j Stats: {stats}")

    # Close Neo4j session
    neo4j_manager.close()


if __name__ == "__main__":
    # Configuration
    pdf_file_path = const.TEST_PDF
    neo4j_config = {
        "uri": const.NEO4J_URI,
        "username": const.NEO4J_USERNAME,
        "password": const.NEO4J_PASSWORD
    }
    domain_schema_path = const.CONFIG_DIR + const.SCHEMA_FILE

    run_pipeline(pdf_file_path, neo4j_config, domain_schema_path)
