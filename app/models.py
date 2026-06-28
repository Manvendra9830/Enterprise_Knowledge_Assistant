"""
Pydantic Models — Request/Response schemas for the Enterprise Knowledge Assistant API.

Defines structured data models for API communication, document processing,
and the RAG pipeline internal data flow.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """Human-readable confidence bands."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# ---------------------------------------------------------------------------
# Internal / Pipeline Models
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """A single chunk produced by the document processor."""
    chunk_id: str
    text: str
    document_name: str
    page_number: int = 1
    chunk_index: int = 0
    file_type: str = "md"
    metadata: dict = Field(default_factory=dict)


class RankedChunk(BaseModel):
    """A chunk enriched with retrieval and re-ranking scores."""
    chunk: DocumentChunk
    retrieval_score: float = 0.0
    reranker_score: float = 0.0
    combined_score: float = 0.0


class ValidationResult(BaseModel):
    """Output of the context validation / guardrail layer."""
    proceed: bool
    reason: str = ""
    best_score: float = 0.0
    avg_score: float = 0.0


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """POST /api/ask — request body."""
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation memory")


class FeedbackRequest(BaseModel):
    """POST /api/feedback — request body."""
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    """A single source citation with excerpt."""
    document: str
    page: int
    excerpt: str = Field("", description="100-200 character excerpt from the source chunk")


class AskResponse(BaseModel):
    """POST /api/ask — response body."""
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.NONE
    session_id: str = ""
    answer_source: str = "gemini"


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""
    name: str
    file_type: str
    chunk_count: int
    ingested_at: str = ""


class HealthResponse(BaseModel):
    """GET /api/health — response body."""
    status: str = "ok"
    documents_loaded: int = 0
    total_chunks: int = 0
    api: str = "healthy"
    chromadb: str = "healthy"
    embedding_model: str = "loaded"
    reranker: str = "loaded"
    gemini: str = "unavailable"
    ollama: str = "unavailable"
    fallback_mode: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FeedbackResponse(BaseModel):
    """POST /api/feedback — response body."""
    status: str = "received"
    message: str = "Thank you for your feedback"
