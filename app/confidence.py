"""
Confidence Scoring Engine — Calculates a weighted confidence score.

The confidence score is a crucial signal for the frontend to visually indicate
answer reliability to the user. It combines semantic similarity (retrieval score)
with cross-encoder reranker scores.
"""

from __future__ import annotations

from app.models import ConfidenceLevel


class ConfidenceEngine:
    """Calculates weighted confidence scores and maps them to UI bands."""

    def __init__(self, retrieval_weight: float = 0.6, reranker_weight: float = 0.4):
        self.retrieval_weight = retrieval_weight
        self.reranker_weight = reranker_weight

        # Ensure weights sum to 1.0 for predictable behavior
        total = self.retrieval_weight + self.reranker_weight
        if total != 1.0:
            self.retrieval_weight /= total
            self.reranker_weight /= total

    def calculate_confidence(self, retrieval_score: float, reranker_score: float) -> float:
        """
        Calculate the final weighted confidence score.
        Scores should be pre-normalized to [0, 1].
        """
        # Clamp inputs
        retrieval = max(0.0, min(1.0, retrieval_score))
        reranker = max(0.0, min(1.0, reranker_score))

        # Weighted calculation
        confidence = (retrieval * self.retrieval_weight) + (reranker * self.reranker_weight)

        # Final clamp and round
        return round(max(0.0, min(1.0, confidence)), 4)

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """
        Map a continuous confidence score to a discrete label for UI styling.
        Ranges:
        - [0.85, 1.00]: High
        - [0.60, 0.85): Medium
        - [0.35, 0.60): Low
        - [0.00, 0.35): None (usually blocked by guardrails before this)
        """
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE
