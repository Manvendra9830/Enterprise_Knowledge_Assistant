"""
Retrieval Evaluation Module.

Calculates Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) 
for the hybrid retrieval pipeline.
"""

def calculate_precision_at_k(retrieved_docs: list[str], expected_docs: list[str], k: int = 5) -> float:
    """Precision@K: Fraction of top-K retrieved docs that are relevant."""
    if not expected_docs:
        return 1.0 if not retrieved_docs else 0.0
    
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc in expected_docs)
    return relevant_retrieved / min(k, len(top_k)) if top_k else 0.0

def calculate_recall_at_k(retrieved_docs: list[str], expected_docs: list[str], k: int = 5) -> float:
    """Recall@K: Fraction of all relevant docs that appear in top-K."""
    if not expected_docs:
        return 1.0 if not retrieved_docs else 0.0
        
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc in expected_docs)
    return relevant_retrieved / len(expected_docs)

def calculate_mrr(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    if not expected_docs:
        return 1.0 if not retrieved_docs else 0.0
        
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in expected_docs:
            return 1.0 / i
    return 0.0
