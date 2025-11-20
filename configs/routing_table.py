"""
Routing Decision Table for QALF.
Maps (Complexity × Intent) → Active Modalities
"""

from typing import List, Tuple

# Complexity levels
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]

# Intent categories
INTENTS = [
    "factual_lookup",
    "comparative",
    "temporal",
    "causal",
    "definitional",
    "visual_tabular",
    "multi_hop"
]

# Modalities
MODALITIES = ["vector", "graph", "keyword"]


def route_to_modalities(complexity: Tuple[str, str, str, str], intent: str) -> List[str]:
    """
    Routes query to active modalities based on complexity and intent.
    
    Args:
        complexity: Tuple of (linguistic, semantic, modality, contextual)
        intent: Intent category string
    
    Returns:
        List of active modalities: ["vector"], ["vector", "keyword"], 
        or ["vector", "graph", "keyword"]
    """
    linguistic, semantic, modality, contextual = complexity
    
    # Simple factual queries: vector only
    if intent == "factual_lookup" and semantic == "Low" and linguistic == "Low":
        return ["vector"]
    
    # Complex factual: vector + keyword
    if intent == "factual_lookup" and (semantic == "Medium" or semantic == "High"):
        return ["vector", "keyword"]
    
    # Comparative queries: all modalities
    if intent == "comparative":
        return ["vector", "graph", "keyword"]
    
    # Temporal queries: graph + vector
    if intent == "temporal":
        return ["vector", "graph", "keyword"]
    
    # Causal queries: graph + vector
    if intent == "causal":
        return ["vector", "graph", "keyword"]
    
    # Multi-hop queries: all modalities
    if intent == "multi_hop":
        return ["vector", "graph", "keyword"]
    
    # Visual/tabular: vector + keyword (or all if complex)
    if intent == "visual_tabular":
        if modality == "High" or semantic == "High":
            return ["vector", "graph", "keyword"]
        return ["vector", "keyword"]
    
    # Definitional: vector + keyword
    if intent == "definitional":
        return ["vector", "keyword"]
    
    # Default: vector + keyword
    return ["vector", "keyword"]


def get_routing_table() -> dict:
    """
    Returns the full routing table for reference.
    """
    return {
        "factual_lookup": {
            "Low": ["vector"],
            "Medium": ["vector", "keyword"],
            "High": ["vector", "graph", "keyword"]
        },
        "comparative": {
            "all": ["vector", "graph", "keyword"]
        },
        "temporal": {
            "all": ["vector", "graph", "keyword"]
        },
        "causal": {
            "all": ["vector", "graph", "keyword"]
        },
        "multi_hop": {
            "all": ["vector", "graph", "keyword"]
        },
        "visual_tabular": {
            "Low": ["vector", "keyword"],
            "Medium": ["vector", "keyword"],
            "High": ["vector", "graph", "keyword"]
        },
        "definitional": {
            "all": ["vector", "keyword"]
        }
    }

