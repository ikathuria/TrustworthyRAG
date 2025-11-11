"""Constants used across the project."""

CONFIG_DIR = "configs/"
SCHEMA_FILE = "cybersecurity.yml"

DATA_DIR = "data/"
RAW_DATA_DIR = DATA_DIR + "raw/"
PROCESSED_DATA_DIR = DATA_DIR + "processed/"
EXTRACTED_TEXT_DIR = DATA_DIR + "extracted_text/"
EXTRACTED_IMAGES_DIR = DATA_DIR + "extracted_images/"

TEST_PDF = DATA_DIR + "WickedRose_andNCPH.pdf"

# MinerU2.5 - Vision-Language Model
MINERU_MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"

# Transformers - Text Embedding
TRANSFORMER_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# NEO4J - Graph Database
NEO4J_DB = "neo4j"
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "test1234"

# Spacy
SPACY_MODEL = "en_core_web_sm"

# Ollama - RAG Retrieval
OLLAMA_URI = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
