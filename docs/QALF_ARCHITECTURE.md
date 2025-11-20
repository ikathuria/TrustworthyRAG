# QALF Architecture Overview

## Introduction

**QALF (Query-Adaptive Learned Fusion)** is a lightweight architecture for adversarial-aware multimodal retrieval-augmented generation (RAG). This document provides an overview of the QALF implementation in the TrustworthyRAG project.

## Core Components

### 1. Query Complexity Classification (4D)

The `QueryComplexityClassifier` analyzes queries along four dimensions:

- **Linguistic Complexity (L)**: Parse tree depth, sentence length → {Low, Medium, High}
- **Semantic Complexity (S)**: Entity count, relationship depth, reasoning steps → {Low, Medium, High}
- **Modality Complexity (M)**: Visual/tabular keywords ("show", "chart", "table") → {Low, Medium, High}
- **Contextual Complexity (C)**: Domain-specificity (general vs. expert-level) → {Low, Medium, High}

**Implementation**: `src/qalf/query_complexity.py`

### 2. Intent Classification

The `QueryIntentClassifier` categorizes queries into 7 intent types:

1. **Factual Lookup**: "Who is the CEO of X?"
2. **Comparative**: "Compare X vs Y"
3. **Temporal**: "Trend in X from 2024 to 2025"
4. **Causal**: "Why does X happen?"
5. **Definitional**: "What is X?"
6. **Visual/Tabular**: "Show me the chart"
7. **Multi-hop**: Complex reasoning queries

**Implementation**: `src/qalf/query_intent.py`

### 3. Adaptive Routing

Based on complexity and intent, the system routes queries to active modalities:

- Simple factual: {Vector}
- Complex factual: {Vector, Keyword}
- Complex multi-hop: {Vector, Graph, Keyword}
- Comparative: {Vector, Graph, Keyword}
- Visual: {Vector, Keyword} or {Vector, Graph, Keyword}

**Implementation**: `configs/routing_table.py`

### 4. Unified Neo4j Retriever

All three retrieval modalities use a single Neo4j 5.x+ database:

- **Vector Search**: HNSW indexes for approximate nearest neighbor
- **Graph Traversal**: Native Cypher pattern matching
- **Keyword Search**: Full-text Lucene indexes (TF-IDF, similar to BM25)

**Implementation**: `src/retriever/neo4j_retriever.py`

### 5. Consensus-Based Fusion

The fusion algorithm computes:

1. **Consensus Score**: `Consensus(d) = (# modalities retrieving d) / (# active modalities)`
2. **Adaptive Weight**: `w_i = α_intent × (1 - β × (1 - Consensus_mean_i))`
3. **Final Score**: `Score_QALF(d) = Σ (w_i × RRF_i(d))`

**Key Property**: Documents retrieved by multiple modalities get higher weights, providing inherent adversarial defense.

**Implementation**: `src/retriever/qalf_fusion.py`

### 6. End-to-End Pipeline

The `QALFPipeline` orchestrates all components:

1. Query analysis (complexity + intent)
2. Adaptive routing
3. Multi-modal retrieval
4. Consensus computation
5. Adaptive weighting
6. Weighted RRF fusion

**Implementation**: `src/retriever/qalf_pipeline.py`

## Configuration Files

- `configs/routing_table.py`: Routing decision table
- `configs/alpha_weights.py`: Base weights (α_intent) per intent
- `configs/hyperparameters.yaml`: RRF constant (k=60), consensus beta (β=0.5), etc.

## Usage Example

```python
from src.neo4j.neo4j_manager import Neo4jManager
from src.retriever.qalf_pipeline import QALFPipeline

# Initialize
neo4j_manager = Neo4jManager()
pipeline = QALFPipeline(neo4j_manager)

# Setup indexes (one-time)
pipeline.setup_indexes()

# Retrieve
results = pipeline.qalf_retrieve("Compare GraphRAG architectures from 2024 to 2025", top_k=10)

for result in results:
    print(f"{result['rank']}. {result['title']} (score: {result['score']:.4f})")
```

## Performance Targets

- **NDCG@10 Improvement**: 8-15% over fixed-weight RRF baselines
- **Robustness Improvement**: 70% improvement against adversarial poisoning (56% → 15-20% attack success)

## Advantages of Neo4j-Only Architecture

✅ Single database to maintain (vs. 3 separate systems)  
✅ Unified query interface (all Cypher)  
✅ Built-in hybrid search (vector + keyword in single query)  
✅ Transactional consistency across all modalities  
✅ Simplified deployment (1 Docker service vs. 3)  
✅ Graph pre-filtering (unique to Neo4j: filter graph first, then vector search)  
✅ Unified data model (same Document nodes for all three modalities)

## References

See `QALF_Complete_Summary_Neo4j.md` for the complete project specification and literature foundation.

