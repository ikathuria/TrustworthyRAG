import re


class QueryIntentClassifier:
    """
    Classifies intent of query for adaptive retrieval routing.
    """

    intent_keywords = {
        "factual": [r"what is", r"define", r"explain", r"describe", r"find", r"list"],
        "comparison": [r"compare", r"difference", r"vs", r"between", r"greater", r"less"],
        "table": [r"table", r"tabular", r"list", r"spreadsheet", r"sheet"],
        "image": [r"image", r"diagram", r"figure", r"visual", r"chart", r"map", r"graph"],
        "multi-hop": [r"and then", r"after that", r"sequence", r"step", r"process"],
        "temporal": [r"before", r"after", r"when", r"since", r"time", r"timeline"],
        "entity": [r"who", r"name", r"author", r"organization", r"person"],
        "follow-up": [r"as above", r"previous", r"those", r"it", r"refer", r"continue"]
    }

    def classify(self, query):
        for intent, patterns in self.intent_keywords.items():
            for pat in patterns:
                if re.search(pat, query, re.IGNORECASE):
                    return intent
        return "factual"  # Default fallback

    def get_routing_weights(self, intent):
        # Example routing weights (adapt for your fusion engine)
        weights = {
            "factual": {"vector": 0.7, "graph": 0.2, "keyword": 0.1},
            "comparison": {"vector": 0.4, "graph": 0.5, "keyword": 0.1},
            "table": {"vector": 0.2, "graph": 0.1, "keyword": 0.7},
            "image": {"vector": 0.3, "graph": 0.5, "keyword": 0.2},
            "multi-hop": {"vector": 0.2, "graph": 0.7, "keyword": 0.1},
            "temporal": {"vector": 0.2, "graph": 0.7, "keyword": 0.1},
            "entity": {"vector": 0.5, "graph": 0.4, "keyword": 0.1},
            "follow-up": {"vector": 0.2, "graph": 0.2, "keyword": 0.6}
        }
        return weights.get(intent, {"vector": 0.7, "graph": 0.2, "keyword": 0.1})
