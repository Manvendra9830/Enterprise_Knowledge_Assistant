"""
Context Validation (Guardrails) Layer.

Evaluates the retrieved chunks before they are sent to the LLM.
If the retrieval confidence is too low (e.g. out of scope query),
it blocks generation to prevent hallucination.
"""

import logging
from pydantic import BaseModel
from app.models import RankedChunk

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    proceed: bool
    reason: str


class ContextGuardrails:
    """Validates retrieval quality to prevent LLM hallucinations."""

    # EXACT required text for guardrail failure
    REFUSAL_ANSWER = "I don't have enough information in the uploaded documents to answer that question."

    def __init__(self, confidence_threshold: float = 0.60, avg_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.avg_threshold = avg_threshold

    def validate_context(self, ranked_chunks: list[RankedChunk], query: str) -> ValidationResult:
        """
        Validates if the retrieved chunks are relevant enough to answer the query.
        """
        if not ranked_chunks:
            logger.info("Guardrails FAILED: No chunks retrieved")
            return ValidationResult(proceed=False, reason="No chunks retrieved")

        # 1. Best Score Check
        best_score = max((c.reranker_score for c in ranked_chunks), default=0.0)
        
        # 2. Average Top-3 Score Check
        top_3 = ranked_chunks[:3]
        avg_top3 = sum(c.reranker_score for c in top_3) / len(top_3) if top_3 else 0.0
        
        if best_score < self.confidence_threshold:
            msg = f"Guardrails FAILED: best_score ({best_score:.3f}) < threshold ({self.confidence_threshold})"
            logger.warning(msg)
            logger.info(f"guardrail_decision=REJECT | best_score={best_score:.3f} | avg_top3={avg_top3:.3f}")
            return ValidationResult(proceed=False, reason=msg)
            
        if avg_top3 < self.avg_threshold:
            msg = f"Guardrails FAILED: average_top3_score ({avg_top3:.3f}) < threshold ({self.avg_threshold})"
            logger.warning(msg)
            logger.info(f"guardrail_decision=REJECT | best_score={best_score:.3f} | avg_top3={avg_top3:.3f}")
            return ValidationResult(proceed=False, reason=msg)

        logger.info(f"guardrail_decision=ACCEPT | best_score={best_score:.3f} | avg_top3={avg_top3:.3f}")
        return ValidationResult(proceed=True, reason="Context is sufficient")
