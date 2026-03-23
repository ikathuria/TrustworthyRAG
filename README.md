# TrustworthyRAG

Query-Adaptive Learned Fusion (QALF) for trustworthy multimodal retrieval-augmented generation (RAG).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

## Overview

This repository implements the system described in the paper **"Query-Adaptive Learned Fusion for Robust Multimodal Retrieval-Augmented Generation"**.

QALF addresses a key limitation of hybrid RAG systems: static fusion across all queries. Instead, it uses query-aware routing and consensus-aware fusion over a unified Neo4j backend.

Core idea:
- classify each query by **4D complexity** (linguistic, semantic, modality, contextual),
- infer **intent** (8 classes),
- route to the right subset of modalities (**vector**, **keyword**, **graph**),
- fuse results with a consensus-aware weighted RRF to reward cross-modal agreement and downweight suspicious single-modality evidence.

## Paper-Aligned Results

On DocBench:
- **NDCG@10:** `0.488` (QALF) vs `0.423` (Vector-only), a **14%** improvement.
- **Generative Accuracy:** **76.4%** overall on DocBench.
- **Robustness:** consensus-based fusion suppresses poisoned evidence in visual-bait and entity-override case studies.

## What Is Implemented In This Repo

- `main.py`: end-to-end ingestion and interactive query pipeline
  - parsing -> graph ingestion -> vector embedding -> adaptive retrieval (+ optional generation)
- `src/qalf/`
  - `query_complexity.py`: 4D complexity classifier
  - `query_intent.py`: intent classifier (8 categories)
- `configs/routing_table.py`: intent/complexity -> active modality routing
- `configs/alpha_weights.py`: base intent weights for adaptive fusion
- `src/retriever/`
  - `neo4j_retriever.py`: vector, keyword, graph retrieval against one Neo4j database
  - `qalf_fusion.py`: consensus computation + adaptive weighted RRF
  - `qalf_pipeline.py`: orchestrates full QALF retrieval flow
- `src/generator/rag_generator.py`: source-grounded answer generation using Ollama-hosted LLM
- `evaluate.py` + `src/evaluators/`: retrieval, adversarial, ablation, generator, sensitivity, significance, efficiency, RAGAS prep

## Repository Layout

```text
TrustworthyRAG/
├── configs/                # Hyperparameters, routing table, alpha weights
├── data/                   # Raw, processed, and evaluation outputs
├── examples/               # Quick examples and connection checks
├── src/
│   ├── evaluators/         # Evaluation modules
│   ├── generator/          # RAG answer generation
│   ├── neo4j/              # Graph/vector ingestion and DB management
│   ├── preprocessing/      # Document parsing and extraction
│   ├── qalf/               # Query complexity and intent analysis
│   ├── retriever/          # Retrieval + fusion pipeline
│   └── utils/              # Constants and shared utilities
├── evaluate.py             # Evaluation entrypoint
├── main.py                 # Ingestion/query entrypoint
├── paper.tex               # Manuscript
└── requirements.txt
```

## Setup

### 1) Prerequisites

- Python `3.10+`
- Neo4j `5.x+` (local or remote)
- Ollama (required for graph ingestion and generation)

### 2) Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3) Pull Ollama models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 4) Configure environment

Create a `.env` in the project root:

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Running The Pipeline

### Ingest documents

`main.py` supports wildcard ingestion and skips already ingested files.

```bash
python main.py --mode ingest --files "data/raw/DocBench/0/*.pdf"
```

### Start interactive QALF retrieval (+ generation)

```bash
python main.py --mode query
```

In the interactive prompt:
- enter natural-language queries,
- use `stats` to inspect indexed graph stats,
- use `quit`/`exit` to stop.

### Run ingest + query in one flow

```bash
python main.py --mode both --files "data/raw/DocBench/0/*.pdf"
```

## Evaluation

Use `evaluate.py`:

```bash
python evaluate.py --mode retrieval --top_k 10
python evaluate.py --mode adversarial
python evaluate.py --mode generator --limit 10
python evaluate.py --mode all
```

Supported `--mode` values:
- `retrieval`: NDCG@K / Recall@K
- `adversarial`: poisoning robustness checks
- `ablation`: Vector-only vs variants vs QALF
- `efficiency`: latency/modality analysis
- `sensitivity`: beta (`consensus.beta`) sweep
- `ragas`: RAGAS dataset preparation
- `significance`: statistical significance from saved eval csv
- `generator`: LLM-judge generation accuracy
- `all`: run all evaluations

Outputs are saved under `data/results/`.

## Key Hyperparameters

From `configs/hyperparameters.yaml`:
- `rrf.k = 60`
- `consensus.beta = 0.5`
- retrieval defaults: `top_k = 10`, `top_k_final = 10`
- embedding model: `all-MiniLM-L6-v2` (384d)

<!-- ## Citation

If you use this repository, please cite the accompanying paper in `paper.tex`. -->
