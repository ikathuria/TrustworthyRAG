"""
Metrics for evaluating retrieval effectiveness and adversarial robustness.
"""

from typing import List, Set, Union, Any, Dict
import numpy as np


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: Set of relevant document IDs (gold standard).
        k: Cutoff rank.
        
    Returns:
        NDCG@K score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
        
    k = min(k, len(retrieved_ids))
    
    dcg = 0.0
    for i in range(k):
        doc_id = retrieved_ids[i]
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)
        
    # Ideal DCG
    idcg = 0.0
    num_relevant = len(relevant_ids)
    for i in range(min(k, num_relevant)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Recall at K (Recall@K).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: Set of relevant document IDs.
        k: Cutoff rank.
        
    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
        
    k = min(k, len(retrieved_ids))
    retrieved_set = set(retrieved_ids[:k])
    
    hits = len(retrieved_set.intersection(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: Set of relevant document IDs.
        
    Returns:
        Reciprocal Rank score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
        
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
            
    return 0.0


def poison_success_rate(
    retrieved_ids: List[str], 
    target_doc_id: str, 
    k: int = 1
) -> float:
    """
    Compute Poison Success Rate (PSR).
    Checks if the target document is present in the top K results.
    
    Args:
        retrieved_ids: List of retrieved document IDs.
        target_doc_id: The attacker's target document ID.
        k: Rank cutoff to check (default 1 for Top-1 attack success).
        
    Returns:
        1.0 if successful, 0.0 otherwise.
    """
    if not target_doc_id:
        return 0.0
        
    top_k = retrieved_ids[:k]
    return 1.0 if target_doc_id in top_k else 0.0


def robustness_ratio(poisoned_metric: float, clean_metric: float) -> float:
    """
    Compute Robustness Ratio = Poisoned Metric / Clean Metric.
    Usually applied to accuracy or retrieval performance.
    
    Args:
        poisoned_metric: Metric value in poisoned setting.
        clean_metric: Metric value in clean setting.
        
    Returns:
        Ratio. Returns 0.0 if clean_metric is 0.
    """
    if clean_metric == 0.0:
        return 0.0
    return poisoned_metric / clean_metric


def retrieval_recall_drop(
    clean_recall: float, 
    poisoned_recall: float
) -> float:
    """
    Compute Retrieval Recall Drop.
    
    Args:
        clean_recall: Recall in clean setting.
        poisoned_recall: Recall in poisoned setting.
        
    Returns:
        Difference (Clean - Poisoned).
    """
    return clean_recall - poisoned_recall
