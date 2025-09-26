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
