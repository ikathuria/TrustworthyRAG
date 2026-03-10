"""
Routing Decision Table for QALF.
Maps (Complexity × Intent) → Active Modalities

Complexity: 4D tuple (linguistic, semantic, modality, contextual)
Intent: One of 8 categories from QueryIntentClassifier
"""

from typing import List, Tuple

# Complexity levels (per dimension)
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]

# Intent categories (must match QueryIntentClassifier)
INTENTS = [
    "factual_lookup",
    "relationship",
    "comparative",
    "temporal",
    "causal",
    "definitional",
    "visual_tabular",
    "multi_hop",
]

# Modalities
MODALITIES = ["vector", "graph", "keyword"]


def route_to_modalities(
    complexity: Tuple[str, str, str, str], intent: str
) -> List[str]:
    """
    Routes query to active modalities based on complexity and intent.

    Args:
        complexity: Tuple of (linguistic, semantic, modality, contextual)
        intent: Intent category string from QueryIntentClassifier

    Returns:
        List of active modalities: ["vector"], ["vector", "keyword"],
        ["vector", "graph"], or ["vector", "graph", "keyword"]
    """
    linguistic, semantic, modality, contextual = complexity

    # Relationship: entity-creator links (e.g., "Who created X?")
    # Graph essential; keyword rarely helps for creator queries
    if intent == "relationship":
        return ["vector", "graph"]

    # Factual lookup: complexity-aware
    # Low semantic + Low linguistic = simple factual ("Who won 2024 World Cup?")
    # Medium/High semantic = multi-entity or relational factual
    if intent == "factual_lookup":
        if semantic == "Low" and linguistic == "Low":
            return ["vector"]
        if semantic == "Medium" or semantic == "High":
            return ["vector", "keyword", "graph"]
        # Low semantic, Medium/High linguistic (long but simple query)
        return ["vector", "keyword"]

    # Comparative: compare entities; all modalities help
    if intent == "comparative":
        return ["vector", "graph", "keyword"]

    # Temporal: time-based queries; graph for sequences, keyword for exact dates
    if intent == "temporal":
        return ["vector", "graph", "keyword"]

    # Causal: cause-effect; graph essential for relational chains
    if intent == "causal":
        return ["vector", "graph", "keyword"]

    # Multi-hop: multi-step reasoning; all modalities needed
    if intent == "multi_hop":
        return ["vector", "graph", "keyword"]

    # Visual/tabular: complexity-aware via modality dimension
    # Low modality = simple "show table X"; High = cross-table/cross-figure
    if intent == "visual_tabular":
        if modality == "Low":
            return ["vector", "keyword"]
        return ["vector", "keyword", "graph"]

    # Definitional: exact terms + semantic; graph rarely needed
    if intent == "definitional":
        return ["vector", "keyword"]

    # Default: all modalities for unknown/unclassified
    return ["vector", "keyword", "graph"]


def get_routing_table() -> dict:
    """
    Returns the full routing table for reference.
    Keys: intent → complexity_key → modalities
    Complexity_key: "Low/Low" (factual), "Low"/"Medium"/"High" (visual_tabular), "all"
    """
    return {
        "factual_lookup": {
            "Low (C_L,C_S)": ["vector"],
            "Medium/High semantic": ["vector", "keyword", "graph"],
            "else": ["vector", "keyword"],
        },
        "relationship": {"all": ["vector", "graph"]},
        "comparative": {"all": ["vector", "graph", "keyword"]},
        "temporal": {"all": ["vector", "graph", "keyword"]},
        "causal": {"all": ["vector", "graph", "keyword"]},
        "multi_hop": {"all": ["vector", "graph", "keyword"]},
        "visual_tabular": {
            "Low (C_M)": ["vector", "keyword"],
            "Medium/High (C_M)": ["vector", "keyword", "graph"],
        },
        "definitional": {"all": ["vector", "keyword"]},
    }
