"""
Enterprise Knowledge Assistant — Evaluation Framework

Runs the test cases through the RAG pipeline and calculates enterprise-grade metrics:
- Retrieval: Precision@K, Recall@K, MRR
- Citation: Accuracy, Completeness

Outputs a comprehensive CSV report.
"""

import json
import csv
import sys
import os
from pathlib import Path
import logging

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.retrieval_eval import calculate_precision_at_k, calculate_recall_at_k, calculate_mrr
from evaluation.citation_eval import evaluate_citation_accuracy, evaluate_citation_completeness

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(api_url: str = "http://localhost:8000/api"):
    """
    Run evaluation against the live API.
    Note: Requires the FastAPI server to be running.
    """
    import httpx
    
    test_cases_path = Path(__file__).parent / "test_cases.json"
    report_path = Path(__file__).parent / "evaluation_report.csv"
    
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)
        
    results = []
    
    logger.info(f"Starting evaluation of {len(test_cases)} test cases...")
    
    with httpx.Client(timeout=30.0) as client:
        for i, case in enumerate(test_cases, 1):
            query = case["question"]
            logger.info(f"\n[{i}/{len(test_cases)}] Testing: '{query}'")
            
            try:
                # Call API
                response = client.post(f"{api_url}/ask", json={"question": query})
                response.raise_for_status()
                data = response.json()
                
                answer = data["answer"]
                sources = data["sources"]
                retrieved_docs = [s["document"] for s in sources]
                
                # Calculate metrics
                expected_docs = case["expected_source_documents"]
                
                precision_5 = calculate_precision_at_k(retrieved_docs, expected_docs, 5)
                recall_5 = calculate_recall_at_k(retrieved_docs, expected_docs, 5)
                mrr = calculate_mrr(retrieved_docs, expected_docs)
                cit_accuracy = evaluate_citation_accuracy(sources, expected_docs)
                
                results.append({
                    "Question": query,
                    "Category": case["category"],
                    "Precision@5": f"{precision_5:.2f}",
                    "Recall@5": f"{recall_5:.2f}",
                    "MRR": f"{mrr:.2f}",
                    "Citation_Accuracy": f"{cit_accuracy:.2f}",
                    "Confidence_Score": f"{data.get('confidence', 0.0):.2f}",
                    "Answer_Source": data.get('answer_source', 'gemini')
                })
                
                logger.info(f"  -> MRR: {mrr:.2f} | Recall@5: {recall_5:.2f} | Confidence: {data.get('confidence', 0.0):.2f}")
                
            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                
    # Write report
    if results:
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nEvaluation complete! Report saved to: {report_path}")

if __name__ == "__main__":
    run_evaluation()
