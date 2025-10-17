import shap
from lime.lime_text import LimeTextExplainer
import numpy as np


class RAGExplainer:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.lime_explainer = LimeTextExplainer(
            class_names=['Low Risk', 'High Risk'])

    def explain_with_lime(self, query, num_features=10):
        def predict_fn(texts):
            predictions = []
            for text in texts:
                result = self.rag_system.query(text)
                confidence = self._extract_threat_confidence(result['answer'])
                predictions.append([1 - confidence, confidence])
            return np.array(predictions)

        explanation = self.lime_explainer.explain_instance(
            query, predict_fn, num_features=num_features)
        return explanation

    def _extract_threat_confidence(self, answer):
        threat_words = ['critical', 'high', 'severe',
                        'dangerous', 'exploit', 'vulnerability']
        safe_words = ['secure', 'safe', 'protected',
                      'mitigated', 'patched', 'low risk']

        answer_lower = answer.lower()
        threat_score = sum(word in answer_lower for word in threat_words)
        safe_score = sum(word in answer_lower for word in safe_words)

        if threat_score + safe_score == 0:
            return 0.5
        return threat_score / (threat_score + safe_score)
