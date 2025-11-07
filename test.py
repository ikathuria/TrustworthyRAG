from src.preprocessing.document_parser import MinerUParser
from src.cybersec_extractor import CybersecEntityExtractor

# Parse document
parser_config = {"dtype": "auto", "device_map": "auto"}
parser = MinerUParser(parser_config)
parsed_doc = parser.parse("data/WickedRose_andNCPH-6.pdf")

extractor_config = {
    "spacy_model": "en_core_web_lg"
}
extractor = CybersecEntityExtractor(extractor_config)

# Extract from your parsed document
entities = extractor.extract_entities(parsed_doc.text)
relations = extractor.extract_relations(parsed_doc.text, entities)

print(f"Found {len(entities)} entities and {len(relations)} relations")
