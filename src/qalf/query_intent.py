import re
from typing import Dict


class QueryIntentClassifier:
    """
    Intent-Based Classifier for QALF.
    Classifies queries into 7 intent categories for adaptive routing.
    """

    def __init__(self):
        # 7 Intent Categories as per QALF specification
        self.intent_patterns = {
            "factual_lookup": [
                r"who is", r"what is", r"when did", r"how many",
                r"define", r"explain", r"describe", r"find", r"list"
            ],
            "comparative": [
                r"compare", r"vs", r"versus", r"difference between",
                r"greater", r"less", r"better", r"worse"
            ],
            "temporal": [
                r"trend", r"over time", r"from.*to", r"historical",
                r"before", r"after", r"when", r"since", r"timeline"
            ],
            "causal": [
                r"why", r"how does", r"because", r"caused",
                r"due to", r"reason", r"result"
            ],
            "definitional": [
                r"what is", r"define", r"meaning", r"definition"
            ],
            "visual_tabular": [
                r"show", r"visualize", r"chart", r"graph",
                r"table", r"diagram", r"figure", r"image", r"map"
            ],
            "multi_hop": [
                r"and then", r"after that", r"sequence", r"step",
                r"process", r"how.*and.*how"
            ]
        }

    def classify(self, query: str) -> str:
        """
        Classifies query intent.
        
        Returns:
            One of: factual_lookup, comparative, temporal, causal,
                    definitional, visual_tabular, multi_hop
        """
        query_lower = query.lower()
        
        # Check patterns in priority order (more specific first)
        # Multi-hop patterns (check first as they can overlap)
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["multi_hop"]):
            # Additional check: long queries with multiple clauses
            if ("and" in query_lower and len(query.split()) > 15) or query.count("?") > 1:
                return "multi_hop"
        
        # Visual/tabular patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["visual_tabular"]):
            return "visual_tabular"
        
        # Comparative patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["comparative"]):
            return "comparative"
        
        # Temporal patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["temporal"]):
            return "temporal"
        
        # Causal patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["causal"]):
            return "causal"
        
        # Factual lookup patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["factual_lookup"]):
            return "factual_lookup"
        
        # Definitional patterns (fallback)
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["definitional"]):
            return "definitional"
        
        # Default fallback
        return "factual_lookup"

    def get_routing_weights(self, intent: str) -> Dict[str, float]:
        """
        Legacy method for backward compatibility.
        Returns base weights for intent (used as α_intent in QALF).
        """
        weights = {
            "factual_lookup": {"vector": 0.7, "graph": 0.2, "keyword": 0.1},
            "comparative": {"vector": 0.4, "graph": 0.5, "keyword": 0.1},
            "temporal": {"vector": 0.2, "graph": 0.7, "keyword": 0.1},
            "causal": {"vector": 0.3, "graph": 0.6, "keyword": 0.1},
            "definitional": {"vector": 0.6, "graph": 0.2, "keyword": 0.2},
            "visual_tabular": {"vector": 0.3, "graph": 0.2, "keyword": 0.5},
            "multi_hop": {"vector": 0.2, "graph": 0.7, "keyword": 0.1}
        }
        return weights.get(intent, {"vector": 0.7, "graph": 0.2, "keyword": 0.1})
