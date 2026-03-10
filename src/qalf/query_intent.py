import re
import logging
from typing import Dict


class QueryIntentClassifier:
    """
    Intent-Based Classifier for QALF.
    Classifies queries into 8 intent categories for adaptive routing.
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        # 8 Intent Categories as per QALF specification (routing_table.INTENTS)
        self.intent_patterns = {
            "factual_lookup": [
                r"who is", r"what is", r"when did", r"how many",
                r"define", r"explain", r"describe", r"find", r"list"
            ],
            "relationship": [
                r"who created", r"who owns", r"who developed", r"who made",
                r"who wrote", r"who designed", r"who built", r"who invented",
                r"created by", r"owned by", r"developed by", r"made by",
                r"wrote by", r"designed by", r"built by", r"invented by",
                r"creator of", r"owner of", r"developer of", r"author of"
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
            One of: factual_lookup, relationship, comparative, temporal, causal,
                    definitional, visual_tabular, multi_hop
        """
        import time
        start_time = time.time()
        self._logger.debug(f"Classifying intent for query: '{query[:50]}...'")
        
        query_lower = query.lower()
        
        # Check patterns in priority order (more specific first)
        # Relationship patterns (check early - these are specific)
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns.get("relationship", [])):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'relationship' in {elapsed:.3f}s")
            return "relationship"
        
        # Multi-hop patterns (check first as they can overlap)
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["multi_hop"]):
            # Additional check: long queries with multiple clauses
            if ("and" in query_lower and len(query.split()) > 15) or query.count("?") > 1:
                elapsed = time.time() - start_time
                self._logger.debug(f"Intent classified as 'multi_hop' in {elapsed:.3f}s")
                return "multi_hop"
        
        # Visual/tabular patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["visual_tabular"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'visual_tabular' in {elapsed:.3f}s")
            return "visual_tabular"
        
        # Comparative patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["comparative"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'comparative' in {elapsed:.3f}s")
            return "comparative"
        
        # Temporal patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["temporal"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'temporal' in {elapsed:.3f}s")
            return "temporal"
        
        # Causal patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["causal"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'causal' in {elapsed:.3f}s")
            return "causal"
        
        # Factual lookup patterns
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["factual_lookup"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'factual_lookup' in {elapsed:.3f}s")
            return "factual_lookup"
        
        # Definitional patterns (fallback)
        if any(re.search(pat, query_lower, re.IGNORECASE) 
               for pat in self.intent_patterns["definitional"]):
            elapsed = time.time() - start_time
            self._logger.debug(f"Intent classified as 'definitional' in {elapsed:.3f}s")
            return "definitional"
        
        # Default fallback
        elapsed = time.time() - start_time
        self._logger.debug(f"Intent classified as 'factual_lookup' (default) in {elapsed:.3f}s")
        return "factual_lookup"

    def get_routing_weights(self, intent: str) -> Dict[str, float]:
        """
        Legacy method for backward compatibility.
        Returns base weights for intent (used as α_intent in QALF).
        Prefer configs.alpha_weights.get_alpha_weights() for production.
        """
        weights = {
            "factual_lookup": {"vector": 0.4, "graph": 0.2, "keyword": 0.4},
            "relationship": {"vector": 0.3, "graph": 0.6, "keyword": 0.1},
            "comparative": {"vector": 0.4, "graph": 0.5, "keyword": 0.1},
            "temporal": {"vector": 0.2, "graph": 0.7, "keyword": 0.1},
            "causal": {"vector": 0.3, "graph": 0.6, "keyword": 0.1},
            "definitional": {"vector": 0.4, "graph": 0.2, "keyword": 0.4},
            "visual_tabular": {"vector": 0.3, "graph": 0.2, "keyword": 0.5},
            "multi_hop": {"vector": 0.2, "graph": 0.7, "keyword": 0.1},
        }
        return weights.get(intent, {"vector": 0.4, "graph": 0.2, "keyword": 0.4})
