# Evaluation Results Summary

This document summarizes the core performance metrics of the QALF (Query-Adaptive Learned Fusion) architecture as reported in the associated research paper.

## 📊 Retrieval Performance (DocBench)

QALF achieves a significant improvement over standard vector-only baselines by dynamically routing queries to optimal retrieval modalities.

| Metric | Vector-Only Baseline | QALF (Proposed) | Improvement |
|--------|---------------------|-----------------|-------------|
| **NDCG@10** | 0.423 | **0.488** | **+13%** |
| **Recall@10** | 0.599 | **0.594** | -0.8% |
| **Accuracy@1** | 0.265 | **0.383** | **+44.5%** |

> [!NOTE]
> The slight dip in overall Recall (-0.8%) is a deliberate trade-off for higher precision and robustness against adversarial noise.

### Performance by Complexity

| Complexity | Vector NDCG@10 | QALF NDCG@10 | Statistical Significance |
|------------|----------------|--------------|--------------------------|
| **Complex** | 0.591 | **0.672** | $p < 0.001$ |
| **Simple** | 0.254 | **0.303** | $p < 0.001$ |

## 💬 Generative Accuracy

Overall accuracy on the DocBench benchmark (Llama-3.1 8B):

| System | Overall Accuracy (%) |
|--------|----------------------|
| Llama-3 8B (Standalone) | 49.6 |
| MMGR | 61.0 |
| RAGAnything | 63.4 |
| GPT-4 | 69.8 |
| **QALF (Proposed)** | **76.0** |

## 🛡️ Robustness Highlights

- **Visual Bait Defense**: Consensus scores correctly downweigh documents with mismatched cross-modal evidence (e.g., poisoned image-caption pairs).
- **Entity Override Resistance**: Graph-aware routing prevents single-channel entity poisoning from dominating the retrieval set.
