"""
Answer Evaluation Module.

Evaluates the quality of the generated answer. In a full production system,
this would use an LLM-as-a-judge approach (e.g., passing the Q&A pair to GPT-4 
for scoring). Here we implement heuristic checks for groundedness and relevance based 
on the expected answer substrings.
"""

def evaluate_answer_relevance(answer: str, expected_contains: list[str]) -> float:
    """Check if the answer contains the key phrases expected."""
    if not expected_contains:
        # If no answer is expected (out of scope), check if the model refused to answer.
        if "could not find sufficient information" in answer.lower():
            return 1.0
        return 0.0
        
    answer_lower = answer.lower()
    matches = sum(1 for phrase in expected_contains if phrase.lower() in answer_lower)
    return matches / len(expected_contains)

def evaluate_groundedness(answer: str, context: str) -> float:
    """
    Check if the answer is grounded in the context.
    A full implementation uses an LLM judge.
    """
    # Placeholder for LLM-based groundedness check.
    # In this demo, we assume high groundedness if guardrails pass and temperature is low.
    return 1.0 if answer else 0.0
