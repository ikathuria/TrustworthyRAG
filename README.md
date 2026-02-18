# TrustworthyRAG
> Robust and Explainable Retrieval-Augmented Generation (RAG) for Cybersecurity under Adversarial Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

## 📖 Overview

TrustworthyRAG is a research project exploring the vulnerabilities of Retrieval-Augmented Generation (RAG) systems to **adversarial attacks** in the cybersecurity domain. The project focuses on:

- **Adversarial Attacks:** Prompt injections, retrieval poisoning, and backdoors targeting LLM-based RAG pipelines.
- **Explainability:** Techniques to trace which retrieved documents influence model outputs.
- **QALF Architecture:** A query-adaptive fusion mechanism that provides inherent resistance to single-modality poisoning.

## 🚀 QALF (Query-Adaptive Learned Fusion)

**QALF** is a lightweight architecture for adversarial-aware multimodal retrieval-augmented generation (RAG). It combines vector, graph, and keyword retrieval using a **single Neo4j 5.x+ database**.

### Key Features
- **4D Query Dynamic Analysis**: Linguistic, Semantic, Modality, and Contextual complexity classification.
- **Intent-Based Routing**: Selects active modalities based on 7 intent categories.
- **Consensus-Based Fusion**: Adaptive weighting based on cross-modality agreement to mitigate biased or poisoned results.
- **Integrated Generator**: RAG pipeline with source-grounded answer generation using Llama 3.

## 📂 Repository Structure

```
TrustworthyRAG/
├── data/               # Raw and processed datasets (DocBench)
├── src/                # Core implementation
│   ├── evaluators/     # Specialized evaluation modules
│   ├── generator/      # Llama-3 RAG generator logic
│   ├── neo4j/          # Neo4j management and ingestion
│   ├── preprocessing/  # Document parsing and chunking
│   ├── retriever/      # QALF pipeline (retriever + fusion)
│   ├── utils/          # Constants, metrics, and system wrappers
│   └── main.py         # Ingestion and query entry point
├── evaluate.py         # Central evaluation interface
└── requirements.txt
```

## ⚙️ Setup Instructions

### 1. Requirements
- **Neo4j 5.x+**: Running locally or via Docker.
- **Ollama**: Running locally for LLM generation and evaluation.

### 2. Install Dependencies
```bash
git clone https://github.com/yourusername/TrustworthyRAG.git
cd TrustworthyRAG
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Model Setup (Ollama)
Ensure the required models are pulled:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 4. Database Setup
Update `src/utils/constants.py` with your Neo4j credentials or set environment variables:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## 🛠️ Usage

### Data Ingestion
Ingest PDFs into the Neo4j graph/vector store:
```bash
python main.py --mode ingest --files "data/raw/DocBench/0/*.pdf"
```

### Retrieval & Querying
Run a single QALF query:
```bash
python main.py --mode query --query "What is the primary challenge addressed by BERT?"
```

## 📊 Evaluation

The `evaluate.py` script provides a unified interface for various evaluation metrics and scenarios.

| Mode | Description |
|------|-------------|
| `retrieval` | Standard NDCG@10 and Recall@10 evaluation on DocBench. |
| `adversarial` | Multi-modal consensus check against poisoned injections. |
| `generator` | LLM-based accuracy scoring (0/1) with domain-specific results. |
| `ablation` | Comparison across systems (Vector-only, RRF, QALF). |
| `efficiency` | Latency and modality usage analysis. |
| `sensitivity` | Analysis of QALF performance across different beta parameters. |
| `significance`| T-test for statistical significance of QALF vs Vector Baseline. |
| `ragas` | Prepares dataset samples for RAGAS evaluation. |

**Example Command:**
```bash
# Run full retrieval evaluation on first 5 DocBench directories
python evaluate.py --mode retrieval --limit 5

# Run statistical significance analysis
python evaluate.py --mode significance

# Run generator accuracy evaluation with LaTeX table output
python evaluate.py --mode generator --limit 10
```

Results are saved to `data/results/` for all modes.
