# TrustworthyRAG
> Robust and Explainable Retrieval-Augmented Generation (RAG) for Cybersecurity under Adversarial Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

## 📖 Overview

TrustworthyRAG is a research project exploring the vulnerabilities of Retrieval-Augmented Generation (RAG) systems to **adversarial attacks** in the cybersecurity domain. The project focuses on:

- **Adversarial Attacks:** Prompt injections, retrieval poisoning, and backdoors targeting LLM-based RAG pipelines.  
- **Explainability:** Techniques to trace which retrieved documents influence model outputs.  
- **Defense & Recommendation:** Filtering, trust scoring, and recommender-style strategies to mitigate attacks.  

This project forms the basis of a **Master's thesis** aimed at building robust and transparent RAG systems in cybersecurity.

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

## Publications
TODO: Add links to any published papers or preprints related to this project.

## 📂 Repository Structure

```
TrustworthyRAG/
├── data/ # Raw and processed cybersecurity datasets
|  | raw/ # Original datasets
|  | processed/ # Cleaned and preprocessed data
|  | metadata/ # Dataset metadata and documentation
|  └── adversarial/ # Adversarial examples and attack data
├── src/ # Source code
|  |── data_ingestion/ # Knowledge graph construction and data loading
|  |  | cybersec_data_ingestion.py # Main cybersecurity data ingestion module
|  |  | cwe_data_ingestion.py # CWE data ingestion module
|  |  | cve_data_ingestion.py # CVE data ingestion module
|  |  └── kg_construction.py # Knowledge graph construction module
|  ├── rag_pipeline/ # RAG pipeline implementation
|  |  | retriever.py # Retriever module
|  |  | generator.py # Generator module
|  |  └── pipeline.py # Main RAG pipeline
|  ├── utils/ # Utility functions for data processing, evaluation, etc.
|  ├── attacks/ # Adversarial attack implementations
|  |  | prompt_injection.py # Prompt injection attack
|  |  | retrieval_poisoning.py # Retrieval poisoning attack
|  |  └── backdoors.py # Backdoor attack
|  ├── defenses/ # Defense mechanisms
|  |  | filtering.py # Filtering strategies
|  |  | trust_scoring.py # Trust scoring methods
|  |  └── re_ranking.py # Re-ranking strategies
|  └── explainability/ # Explainability tools
|     | attention_viz.py # Attention visualization
|     └── shap_lime.py # SHAP/LIME explanations
├── experiments/ # Scripts and notebooks for running experiments
├── results/ # Figures, tables, and logs from experiments
├── papers/ # Drafts and notes for research papers
└── docs/ # Additional documentation and slides
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
```

3. Run baseline RAG pipeline:
```bash
python src/rag_pipeline/pipeline.py --config configs/baseline.yaml
```

4. Run adversarial attack scripts:
```bash
python src/attacks/prompt_injection.py
python src/attacks/retrieval_poisoning.py
```

5. Explore results in experiments/notebooks/:
```bash
python src/rag_pipeline/pipeline.py --config configs/baseline.yaml
```


| Paper/Framework/Concept | Primary Focus/Contribution to RAG | URL |
| ----------------------- | --------------------------------- | --- |
| RAG-Fusion: A New Take on Retrieval-Augmented Generation | Fusion technique using Reciprocal Rank Fusion (RRF) and query decomposition to aggregate scores. | https://arxiv.org/abs/2402.03367 |
| Layered Query Retrieval (LQR) | Adaptive RAG framework using semantic rules for multi-hop query complexity classification and routing. | https://www.mdpi.com/2076-3417/14/23/11014 |
| Joint Fusion and Encoding | Multimodal retrieval study advocating for early cross-modal fusion to enhance context interpretation. | https://arxiv.org/html/2502.20008v1 |
| MIntOOD / MMIU Dataset | Multimodal intent classification and OOD detection using weighted feature fusion networks. | https://arxiv.org/abs/2412.12453 |
| Survey of RAG Systems Enhancements | Provides a taxonomy of RAG architectures: retriever-centric, generator-centric, hybrid, and robustness-oriented. | https://arxiv.org/html/2506.00054v1 |
| Hybrid GraphRAG Performance | Empirical study showing that Hybrid GraphRAG significantly outperforms traditional RAG in factual correctness. | https://arxiv.org/abs/2507.03608 |
| RAG-Anything / MinerU | All-in-One Multimodal Document Processing framework for unified handling of complex, mixed-content documents. | https://github.com/HKUDS/RAG-Anything |
| Adaptive-RAG / Query Complexity Models | Defines and categorizes query complexity (Low, Intermediate, High) to enable dynamic retrieval strategy selection. | https://openreview.net/forum?id=JLkgI0h7wy |
| Corrective RAG (CRAG) / Self-RAG | Robustness techniques using retrieval evaluators and reflection tokens for on-demand assessment of document quality. | https://www.pinecone.io/learn/advanced-rag-techniques/ |
| Madam-RAG | Multi-agent debate architecture designed to explicitly resolve inter-context knowledge conflict across retrieved documents.	| https://arxiv.org/html/2504.13079v2 |
| The Role of Syntactic Complexity in IR | Experiments demonstrating that higher linguistic complexity (full sentences) can aid retrieval accuracy by decreasing ambiguity.	| https://pmc.ncbi.nlm.nih.gov/articles/PMC3366494/ |
| Visual Complexity and Learning | Study linking less visually complex page layouts to higher learning success in information retrieval contexts.	| https://arxiv.org/abs/2501.05289 |
| Dynamic Weighting Algorithms | Research on dynamically updating classifier weights (using metrics like G-mean) in ensemble learning to accommodate concept drift.	| https://www.mdpi.com/2227-7390/13/10/5924 |
| Structured Fusion with Evaluation Guidance | Study demonstrating that integrating explicit evaluation guidance into the fusion process improves overall quality and consistency.	| https://arxiv.org/pdf/2509.01053 |
| RARR | A revision-stage framework utilizing a consistency model to detect and correct disagreements (hallucinations) in the generation phase.	| https://www.mdpi.com/2227-7390/13/5/856 |
