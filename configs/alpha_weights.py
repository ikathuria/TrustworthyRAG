"""
Alpha weights (α_intent) for QALF adaptive weight computation.
Base weights for each intent category.
"""

from typing import Dict

# Base weights per intent (α_intent)
# These are used in: w_i = α_intent × (1 - β × (1 - Consensus_mean_i))
ALPHA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "factual_lookup": {
        "vector": 0.7,
        "graph": 0.2,
        "keyword": 0.1
    },
    "comparative": {
        "vector": 0.4,
        "graph": 0.5,
        "keyword": 0.1
    },
    "temporal": {
        "vector": 0.2,
        "graph": 0.7,
        "keyword": 0.1
    },
    "causal": {
        "vector": 0.3,
        "graph": 0.6,
        "keyword": 0.1
    },
    "definitional": {
        "vector": 0.6,
        "graph": 0.2,
        "keyword": 0.2
    },
    "visual_tabular": {
        "vector": 0.3,
        "graph": 0.2,
        "keyword": 0.5
    },
    "multi_hop": {
        "vector": 0.2,
        "graph": 0.7,
        "keyword": 0.1
    },
    "relationship": {
        "vector": 0.3,
        "graph": 0.6,
        "keyword": 0.1
    }
}


def get_alpha_weights(intent: str) -> Dict[str, float]:
    """
    Get base weights (α_intent) for a given intent.
    
    Args:
        intent: Intent category string
    
    Returns:
        Dictionary mapping modality to base weight
    """
    return ALPHA_WEIGHTS.get(intent, {
        "vector": 0.7,
        "graph": 0.2,
        "keyword": 0.1
    })

