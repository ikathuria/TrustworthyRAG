# TrustworthyRAG
> Robust and Explainable Retrieval-Augmented Generation (RAG) for Cybersecurity under Adversarial Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

## 📖 Overview

TrustworthyRAG is a research project exploring the vulnerabilities of Retrieval-Augmented Generation (RAG) systems to **adversarial attacks** in the cybersecurity domain. The project focuses on:

- **Adversarial Attacks:** Prompt injections, retrieval poisoning, and backdoors targeting LLM-based RAG pipelines.
- **Explainability:** Techniques to trace which retrieved documents influence model outputs.
- **Defense & Recommendation:** Filtering, trust scoring, and recommender-style strategies to mitigate attacks.

This project forms the basis of a **Master's thesis** aimed at building robust and transparent RAG systems in cybersecurity.

## 🚀 QALF (Query-Adaptive Learned Fusion)

**QALF** is a lightweight architecture for adversarial-aware multimodal retrieval-augmented generation (RAG) implemented in this project. QALF combines three orthogonal retrieval mechanisms (vector, graph, keyword) using a **single Neo4j 5.x+ database** to achieve both performance improvement and adversarial robustness.

### Key Features

- **4D Query Complexity Classification**: Linguistic, Semantic, Modality, and Contextual complexity analysis
- **Intent-Based Routing**: 7 intent categories (factual, comparative, temporal, causal, definitional, visual/tabular, multi-hop)
- **Consensus-Based Fusion**: Adaptive weighting based on cross-modality agreement
- **Inherent Adversarial Defense**: Documents retrieved by multiple modalities receive higher weights
- **Unified Neo4j Architecture**: All three modalities (vector, graph, keyword) in a single database

### Architecture

```
QALF Pipeline:
1. Query Analysis → Complexity (4D) + Intent (7 categories)
2. Adaptive Routing → Select active modalities
3. Multi-Modal Retrieval → Vector, Graph, Keyword (all from Neo4j)
4. Consensus Computation → Cross-modality agreement scores
5. Adaptive Weighting → w_i = α_intent × (1 - β × (1 - Consensus_mean_i))
6. Weighted RRF Fusion → Score_QALF(d) = Σ (w_i × RRF_i(d))
```

See `examples/qalf_example.py` for usage examples.

## Key Features

- Modular RAG pipeline with flexible retriever + generator integration.
- Adversarial attack scenarios for testing vulnerabilities.
- Explainability modules to track document influence on outputs.
- Defense strategies including filtering, trust scoring, and re-ranking.
- Structured for reproducibility of experiments and easy paper integration.

## Project Goals

1. **Evaluate Vulnerabilities:** Systematically assess how different adversarial attacks impact RAG performance in cybersecurity tasks.
2. **Develop Explainability Tools:** Create methods to trace and visualize the influence of retrieved documents on generated responses.
3. **Propose Defense Mechanisms:** Implement and evaluate strategies to defend against identified attacks.

## 📂 Repository Structure

```
TrustworthyRAG/
├── data/               # Raw and processed cybersecurity datasets
├── src/                # Source code
│   ├── neo4j/          # Graph and vector ingestion
│   │   ├── graph_ingestion.py
│   │   ├── vector_ingestion.py
│   │   └── neo4j_manager.py
│   ├── preprocessing/  # Document parsing
│   │   └── document_parser.py
│   ├── qalf/           # Query analysis (complexity & intent)
│   │   ├── query_complexity.py
│   │   └── query_intent.py
│   ├── retriever/      # Retrieval and generation pipeline
│   │   ├── qalf_pipeline.py
│   │   ├── neo4j_retriever.py
│   │   ├── qalf_fusion.py
│   │   └── rag_generator.py
│   ├── utils/          # Utility functions
│   └── multimodal_grounding.py
├── examples/           # Usage examples
├── evaluate.py         # Evaluation script
├── main.py             # Main entry point
└── requirements.txt
```

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TrustworthyRAG.git
cd TrustworthyRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Start Neo4j:
   - **Option A (Local Instance)**: Make sure your local Neo4j is running
     - Default connection: `neo4j://127.0.0.1:7687`
     - Update connection settings in `src/utils/constants.py` if needed
   - **Option B (Docker)**: Start Neo4j using Docker Compose:
     ```bash
     docker-compose up -d
     ```

4. Ingest Data:
```bash
python main.py --mode ingest --files "data/*.pdf"
```

5. Run Query Interface:
```bash
python main.py --mode query
```

6. Run Evaluation:
```bash
python evaluate.py --ground_truth path/to/ground_truth.jsonl
```

7. Run QALF example:
```bash
python examples/qalf_example.py
```
