import spacy


class QueryComplexityClassifier:
    """
    Scores linguistic, semantic, modality, and contextual complexity of a query.
    """

    def __init__(self, nlp_model="en_core_web_sm"):
        self.nlp = spacy.load(nlp_model)

    def score_linguistic(self, query):
        # Counts named entities, modifiers, and question phrases
        doc = self.nlp(query)
        entity_score = len(doc.ents)
        modifier_score = sum(
            1 for token in doc if token.dep_ in {"amod", "advmod"})
        question_score = 1 if any(token.tag_ == "WP" for token in doc) else 0
        return entity_score + modifier_score + question_score

    def score_semantic(self, query):
        # Looks for logical, comparative, temporal, or multi-hop signals
        doc = self.nlp(query)
        semantic_keywords = {"before", "after", "greater", "less", "compare", "versus", "between",
                             "when", "since", "sequence", "reason", "caused", "due"}
        return sum(token.lemma_ in semantic_keywords for token in doc)

    def score_modality(self, query):
        # Detects reference to images, tables, diagrams, or specific content types
        modality_keywords = {"diagram", "table", "image", "figure", "visual", "chart",
                             "map", "list", "graph"}
        return sum(1 for word in modality_keywords if word in query.lower())

    def score_contextual(self, query):
        # Checks for references to prior context, pronouns, or follow-up queries
        context_phrases = {"as above", "previous",
                           "that", "those", "it", "refer", "continue"}
        return sum(1 for phrase in context_phrases if phrase in query.lower())

    def classify(self, query):
        return {
            "linguistic": self.score_linguistic(query),
            "semantic": self.score_semantic(query),
            "modality": self.score_modality(query),
            "contextual": self.score_contextual(query)
        }

    def total_score(self, query):
        scores = self.classify(query)
        return sum(scores.values())
