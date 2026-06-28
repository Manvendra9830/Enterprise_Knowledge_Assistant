"""
Citation Evaluation Module.

Evaluates the accuracy and completeness of the sources cited by the RAG system.
"""

from typing import List, Dict, Any

def evaluate_citation_accuracy(cited_sources: List[Dict[str, Any]], expected_docs: List[str]) -> float:
    """Check if the cited documents are in the expected list."""
    if not expected_docs:
        return 1.0 if not cited_sources else 0.0
        
    if not cited_sources:
        return 0.0
        
    accurate_citations = sum(1 for source in cited_sources if source["document"] in expected_docs)
    return accurate_citations / len(cited_sources)

def evaluate_citation_completeness(cited_sources: List[Dict[str, Any]], expected_docs: List[str]) -> float:
    """Check if all expected documents were cited."""
    if not expected_docs:
        return 1.0 if not cited_sources else 0.0
        
    if not cited_sources:
        return 0.0
        
    cited_doc_names = [source["document"] for source in cited_sources]
    found = sum(1 for doc in expected_docs if doc in cited_doc_names)
    return found / len(expected_docs)
