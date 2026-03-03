"""Constants used across the project."""
import os
from dotenv import load_dotenv

load_dotenv()


CONFIG_DIR = "configs/"
SCHEMA_FILE = "cybersecurity.yml"

DATA_DIR = "data/"
RAW_DATA_DIR = DATA_DIR + "raw/"
PROCESSED_DATA_DIR = DATA_DIR + "processed/"
EXTRACTED_TEXT_DIR = DATA_DIR + "extracted_text/"
EXTRACTED_IMAGES_DIR = DATA_DIR + "extracted_images/"

DOC_BENCHMARK_DIR = DATA_DIR + "DocBench/"

TEST_PDF = RAW_DATA_DIR + "WickedRose_andNCPH.pdf"
GROUND_TRUTH_PATH = DOC_BENCHMARK_DIR + "0/0_qa.jsonl"

# MinerU2.5 - Vision-Language Model
MINERU_MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"

# Transformers - Text Embedding
TRANSFORMER_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Spacy
SPACY_MODEL = "en_core_web_sm"

# Ollama - RAG Retrieval
OLLAMA_URI = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
GENERATOR_MODEL = "llama3.1:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# NEO4J - Graph Database
# Create a .env file in the src/utils/ directory with the following variables:
# NEO4J_URI=neo4j://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your_password
NEO4J_DB = "neo4j"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
