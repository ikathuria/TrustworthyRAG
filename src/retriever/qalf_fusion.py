"""
Consensus-Based Fusion with Adaptive Weights for QALF.
Implements the core QALF fusion algorithm.
"""

from typing import List, Dict, Any
import logging


class QALFFusion:
    """
    Consensus-based fusion with adaptive weights.
    Implements: Score_QALF(d) = Σ (w_i × RRF_i(d))
    where w_i = α_intent × (1 - β × (1 - Consensus_mean_i))
    """

    def __init__(self, k: int = 60, beta: float = 0.5):
        """
        Initialize QALF fusion.
        
        Args:
            k: RRF constant (default 60)
            beta: Consensus influence factor (0.0 to 1.0)
        """
        self.k = k
        self.beta = beta
        self._logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def compute_consensus_scores(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Compute consensus score for each document.
        Consensus(d) = (# modalities retrieving d in top-K) / (# active modalities)
        
        Args:
            retrieval_results: Dict mapping modality to list of documents
        
        Returns:
            Dict mapping doc_id to consensus score (0.0 to 1.0)
        """
        doc_to_modalities = {}
        num_active_modalities = len(retrieval_results)
        
        if num_active_modalities == 0:
            return {}
        
        # Count how many modalities retrieved each document
        for modality, docs in retrieval_results.items():
            for doc in docs:
                doc_id = doc.get("doc_id", "")
                if doc_id:
                    if doc_id not in doc_to_modalities:
                        doc_to_modalities[doc_id] = 0
                    doc_to_modalities[doc_id] += 1
        
        # Compute consensus: count / num_active_modalities
        consensus = {
            doc_id: count / num_active_modalities
            for doc_id, count in doc_to_modalities.items()
        }
        
        return consensus

    def compute_adaptive_weights(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        consensus_scores: Dict[str, float],
        alpha_intent: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute adaptive weights for each modality.
        w_i = α_intent × (1 - β × (1 - Consensus_mean_i))
        
        Args:
            retrieval_results: Dict mapping modality to list of documents
            consensus_scores: Dict mapping doc_id to consensus score
            alpha_intent: Base weights per modality (α_intent)
        
        Returns:
            Dict mapping modality to adaptive weight
        """
        weights = {}
        
        for modality, docs in retrieval_results.items():
            if not docs:
                # No documents retrieved, use base weight
                weights[modality] = alpha_intent.get(modality, 1.0)
                continue
            
            # Get consensus scores for this modality's documents
            consensus_scores_for_modality = [
                consensus_scores.get(doc.get("doc_id", ""), 0.0)
                for doc in docs
            ]
            
            # Compute mean consensus
            if len(consensus_scores_for_modality) == 0:
                consensus_mean = 0.0
            else:
                consensus_mean = sum(consensus_scores_for_modality) / len(docs)
            
            # Adaptive weight formula
            alpha = alpha_intent.get(modality, 1.0)
            weight = alpha * (1 - self.beta * (1 - consensus_mean))
            weights[modality] = weight
            
            self._logger.debug(
                f"Modality {modality}: α={alpha:.3f}, "
                f"Consensus_mean={consensus_mean:.3f}, "
                f"w={weight:.3f}"
            )
        
        return weights

    def fuse_with_consensus(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float]
    ) -> List[tuple]:
        """
        Fuse results using weighted RRF.
        Score_QALF(d) = Σ (w_i × RRF_i(d))
        
        Args:
            retrieval_results: Dict mapping modality to list of documents
            weights: Dict mapping modality to adaptive weight
        
        Returns:
            List of (doc_id, final_score) tuples, sorted by score descending
        """
        rrf_scores = {}
        
        for modality, docs in retrieval_results.items():
            weight = weights.get(modality, 1.0)
            
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.get("doc_id", "")
                if not doc_id:
                    continue
                
                # Standard RRF formula
                rrf_score = 1.0 / (rank + self.k)
                
                # Weighted RRF
                weighted_rrf = weight * rrf_score
                
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += weighted_rrf
        
        # Sort by final score
        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_results

    def fuse(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        consensus_scores: Dict[str, float],
        alpha_intent: Dict[str, float]
    ) -> List[tuple]:
        """
        Complete fusion pipeline: compute weights and fuse.
        
        Args:
            retrieval_results: Dict mapping modality to list of documents
            consensus_scores: Dict mapping doc_id to consensus score
            alpha_intent: Base weights per modality
        
        Returns:
            List of (doc_id, final_score) tuples, sorted by score descending
        """
        # Compute adaptive weights
        weights = self.compute_adaptive_weights(
            retrieval_results,
            consensus_scores,
            alpha_intent
        )
        
        # Fuse with consensus
        ranked_results = self.fuse_with_consensus(
            retrieval_results,
            weights
        )
        
        return ranked_results

