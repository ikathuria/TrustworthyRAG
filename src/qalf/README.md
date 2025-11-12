# Query Adaptive Learned Fusion (QALF)

We have developed a custom Query Adaptive Learned Fusion (QALF) mechanism to enhance the retrieval performance of our multimodal system. QALF dynamically adjusts the contribution of different modalities based on the query characteristics, allowing for more effective information retrieval.

There are three main pillars to QALF:
1. **Query Complexity Classification** - Routes queries to optimal retrievers based on linguistic/semantic/modality/contextual complexity.

2. **Intent-Aware Routing** - Selects retriever weights specific to query intent (factual vs. multi-hop vs. visual).

3. **Consensus-Based Fusion** - Models agreement between modalities; uses consensus distribution to weight fusion parameters.



